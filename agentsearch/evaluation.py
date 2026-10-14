from __future__ import annotations

import json
import re
import string
from pathlib import Path

from .data import read_jsonl, questions

REQUIRED = {"id", "question", "answer", "trajectory", "tool_calls", "total_tokens"}


def normalize_answer(text):
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def validate_record(record, course=False):
    if set(record) != REQUIRED:
        raise ValueError("Prediction must have exactly the six required fields")
    for key in ("id", "question", "answer"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"Invalid {key}")
    for key in ("tool_calls", "total_tokens"):
        if type(record[key]) is not int or record[key] < 0:
            raise ValueError(f"Invalid {key}")
    events = record["trajectory"]
    if not isinstance(events, list) or not events or not all(isinstance(e, dict) for e in events):
        raise ValueError("Missing machine-readable trajectory")
    if any(e.get("step") != i for i, e in enumerate(events)):
        raise ValueError("Trajectory must be chronologically numbered")
    if events[0].get("type") != "run_start" or events[-1].get("type") != "final":
        raise ValueError("Trajectory needs run_start and successful final")
    if events[0].get("id") != record["id"] or events[0].get("question") != record["question"]:
        raise ValueError("Input and trace mismatch")
    if events[-1].get("answer") != record["answer"]:
        raise ValueError("Answer and final trace mismatch")
    if sum(e.get("type") == "tool_call" for e in events) != record["tool_calls"]:
        raise ValueError("Incorrect tool_calls count")
    calls = [e for e in events if e.get("type") == "llm_call"]
    if sum(e["usage"]["total_tokens"] for e in calls) != record["total_tokens"]:
        raise ValueError("Incorrect total_tokens count")
    for call in calls:
        raw_usage = call.get("response", {}).get("usage", {}).get("total_tokens")
        if raw_usage != call["usage"]["total_tokens"]:
            raise ValueError("Tokens must match API response usage")
    if course:
        start = events[0]
        if not start.get("real_api") or start.get("mode") != "agent":
            raise ValueError("Offline fixtures or single-pass baselines cannot be submitted as course agent runs")
        if start.get("corpus", {}).get("document_count") != 100195:
            raise ValueError("Course run must use all 100,195 documents")
        if record["tool_calls"] < 1 or not calls:
            raise ValueError("Course agent must actually use tools and the LLM")
        if any(e.get("type") == "llm_error" for e in events):
            raise ValueError("API failure creates incomplete token accounting")


def validate_file(predictions, input_questions, expected_count=100, course=False):
    expected = {q["id"]: q["question"] for q in questions(input_questions)}
    rows = list(read_jsonl(predictions))
    if len(rows) != expected_count or len(expected) != expected_count:
        raise ValueError(f"Expected {expected_count} questions and records")
    seen = set()
    for row in rows:
        validate_record(row, course)
        if row["id"] in seen or expected.get(row["id"]) != row["question"]:
            raise ValueError("Duplicate ID, unknown ID or question mismatch")
        seen.add(row["id"])
    if seen != set(expected):
        raise ValueError("Missing question IDs")
    return {"records": len(rows), "total_tool_calls": sum(r["tool_calls"] for r in rows),
            "total_tokens": sum(r["total_tokens"] for r in rows), "course_checks": course}


def evaluate(predictions, gold_path):
    pred = list(read_jsonl(predictions))
    gold = {}
    for row in read_jsonl(gold_path):
        qid = str(row["id"])
        if qid in gold:
            raise ValueError("Duplicate gold ID")
        if "answer" not in row and "answers" not in row:
            raise ValueError("Evaluation requires reference answers")
        gold[qid] = row
    if len(pred) != len(gold) or {r["id"] for r in pred} != set(gold):
        raise ValueError("Predictions and gold must cover exactly the same IDs")
    details = []
    for row in pred:
        validate_record(row)
        reference = gold[row["id"]]
        if row["question"] != reference["question"]:
            raise ValueError("Prediction question does not match reference")
        answers = reference.get("answers", reference.get("answer"))
        answers = answers if isinstance(answers, list) else [answers]
        hit = any(normalize_answer(row["answer"]) == normalize_answer(a) for a in answers)
        details.append({"id": row["id"], "em": int(hit), "answer": row["answer"], "references": answers})
    return {"count": len(pred), "normalized_em": sum(d["em"] for d in details) / len(details) if details else 0,
            "total_tool_calls": sum(r["tool_calls"] for r in pred),
            "total_tokens": sum(r["total_tokens"] for r in pred), "details": details}
