from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = ROOT / ".env") -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config(path: str | Path = ROOT / "config.json") -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for name in ("index_dir", "skills_dir"):
        p = Path(config[name])
        config[name] = str(p if p.is_absolute() else ROOT / p)
    if not config["methods"] or not set(config["methods"]) <= {"sparse", "dense", "instruction"}:
        raise ValueError("methods must contain sparse, dense and/or instruction")
    if len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("Duplicate retrieval methods")
    if not 0 <= config["chunk_overlap"] < config["chunk_chars"]:
        raise ValueError("chunk_overlap must be smaller than chunk_chars")
    for key in ("max_rounds", "max_tool_calls", "max_total_tokens", "max_output_tokens",
                "context_chars", "max_evidence", "history_actions", "top_k", "candidate_k",
                "embedding_max_length", "batch_size"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["context_chars"] < 4000:
        raise ValueError("context_chars must be >=4000")
    return config
