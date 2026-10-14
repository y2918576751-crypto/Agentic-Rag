from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def documents(path):
    for row in read_jsonl(path):
        doc_id = row.get("doc_id", row.get("id", row.get("DocumentID")))
        text = row.get("text")
        if doc_id is None or not isinstance(text, str) or not text.strip():
            raise ValueError("Document needs doc_id/id/DocumentID and non-empty text")
        yield {"doc_id": str(doc_id), "text": text,
               "source_url": str(row.get("source_url", row.get("sourceURL", row.get("url", ""))))}


def questions(path):
    seen = set()
    for row in read_jsonl(path):
        if "id" not in row or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Question needs id and non-empty question")
        qid = str(row["id"])
        if qid in seen:
            raise ValueError(f"Duplicate question ID: {qid}")
        seen.add(qid)
        # Gold answers are deliberately not passed to the agent.
        yield {"id": qid, "question": row["question"]}


def chunks(text: str, size: int, overlap: int):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size/overlap")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = end - overlap
