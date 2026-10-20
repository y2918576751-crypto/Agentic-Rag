from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import load_env


@dataclass
class Reply:
    content: str
    total_tokens: int
    raw: dict


class ProtocolError(RuntimeError):
    def __init__(self, message, raw=None):
        super().__init__(message)
        self.raw = raw


class APIClient:
    is_real = True

    def __init__(self, max_output_tokens=1200):
        load_env()
        self.key = os.getenv("LLM_API_KEY", "")
        self.base = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        self.thinking = os.getenv("LLM_THINKING", "").strip()
        if self.thinking not in {"", "enabled", "disabled"}:
            raise ValueError("LLM_THINKING must be enabled, disabled, or empty")
        self.max_output_tokens = max_output_tokens
        if not self.key or not self.model:
            raise ValueError("Set LLM_API_KEY and the course-approved LLM_MODEL in .env")
        if not self.base.startswith("https://"):
            raise ValueError("LLM_BASE_URL must use HTTPS")

    def complete(self, messages: list[dict]) -> Reply:
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "max_tokens": self.max_output_tokens,
                   "response_format": {"type": "json_object"}}
        if self.thinking:
            payload["thinking"] = {"type": self.thinking}
        # Do not retry ambiguous failures: an unseen completed request may have incurred tokens.
        request = urllib.request.Request(self.base + "/chat/completions",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + self.key,
                                                  "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").replace(self.key, "[redacted]")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body[:1000]}") from exc
        usage = raw.get("usage", {}).get("total_tokens")
        if type(usage) is not int or usage < 0:
            raise ProtocolError("API omitted usage.total_tokens; cannot produce valid accounting", raw)
        try:
            choice = raw["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProtocolError("Missing assistant message", raw) from exc
        if not isinstance(content, str):
            raise ProtocolError("Assistant content must be text JSON", raw)
        if choice.get("finish_reason") == "length":
            raise ProtocolError("API output hit max_tokens; increase max_output_tokens or set LLM_THINKING=disabled", raw)
        if not content.strip():
            raise ProtocolError("API returned empty assistant content; check thinking mode and output token limit", raw)
        return Reply(content, usage, raw)


def parse_json(content: str):
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value
