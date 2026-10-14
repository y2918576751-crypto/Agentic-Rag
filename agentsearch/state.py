from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SearchState:
    question: str
    subqueries: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    visited_docs: set = field(default_factory=set)
    evidence: list = field(default_factory=list)
    loaded_skills: dict = field(default_factory=dict)
    progress: list = field(default_factory=list)
    candidate_answer: str = ""
    trajectory: list = field(default_factory=list)
    tool_calls: int = 0
    total_tokens: int = 0
    seen_calls: set = field(default_factory=set)
    evidence_keys: set = field(default_factory=set)

    def event(self, kind, **values):
        self.trajectory.append({"step": len(self.trajectory), "type": kind, **values})

    def update(self, update):
        if not isinstance(update, dict):
            return
        for name in ("subqueries", "unresolved"):
            value = update.get(name)
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                setattr(self, name, [v[:500] for v in value[:12]])

    def collect(self, item):
        text = item.get("text", item.get("snippet", ""))
        if not isinstance(text, str) or not text or "doc_id" not in item:
            return
        key = (str(item["doc_id"]), item.get("start", 0), text)
        if key in self.evidence_keys:
            return
        self.evidence_keys.add(key)
        self.visited_docs.add(str(item["doc_id"]))
        self.evidence.append({"evidence_id": f"e{len(self.evidence) + 1}", "doc_id": str(item["doc_id"]),
                              "source_url": item.get("source_url", ""), "start": item.get("start", 0),
                              "end": item.get("end", item.get("start", 0) + len(text)), "text": text})

    def context(self, config):
        if not config["context_management"]:
            # Real append-all ablation, bounded only by the provider's context limit.
            tools = [e for e in self.trajectory if e["type"] == "tool_call"]
            return {"question": self.question, "subqueries": self.subqueries, "unresolved": self.unresolved,
                    "evidence": self.evidence, "loaded_skills": self.loaded_skills, "tool_history": tools}
        evidence_budget = config["context_chars"] // 2
        selected, used = [], 0
        for evidence in reversed(self.evidence):
            copy = dict(evidence)
            copy["text"] = copy["text"][:2400]
            size = len(json.dumps(copy, ensure_ascii=False))
            if used + size > evidence_budget or len(selected) >= config["max_evidence"]:
                continue
            selected.append(copy)
            used += size
        result = {"question": self.question, "subqueries": self.subqueries, "unresolved": self.unresolved,
                  "visited_docs": sorted(self.visited_docs)[:100], "evidence": list(reversed(selected)),
                  "loaded_skills": self.loaded_skills, "candidate_answer": self.candidate_answer,
                  "recent_progress": self.progress[-config["history_actions"]:]}
        # Trimming preserves complete trajectory and evidence in state; only the LLM view is reduced.
        while len(json.dumps(result, ensure_ascii=False)) > config["context_chars"]:
            if result["evidence"]:
                result["evidence"].pop(0)
            elif result["recent_progress"]:
                result["recent_progress"].pop(0)
            else:
                raise ValueError("Question/state exceeds context_chars; increase budget")
        return result
