from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import Agent, AgentFailure
from .config import ROOT, load_config
from .data import read_jsonl, questions
from .evaluation import evaluate, validate_file, validate_record
from .llm import APIClient
from .retrieval import Retrieval, build_corpus, build_dense


def save_json(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_batch(agent, input_file, output_file, resume=False, expected_count=100):
    items = list(questions(input_file))
    if len(items) != expected_count:
        raise ValueError(f"Expected {expected_count} input questions; found {len(items)}")
    target = Path(output_file)
    checkpoint = Path(str(target) + ".partial")
    error_path = Path(str(target) + ".error.json")
    inflight_path = Path(str(target) + ".inflight.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("Completed output exists; select a new output path")
    if checkpoint.exists() and not resume:
        raise FileExistsError("Checkpoint exists; use --resume after a clean interruption")
    if resume and error_path.exists():
        raise ValueError("Previous API/agent error: use a fresh output path; do not discard failed-attempt accounting")
    done = {}
    expected = {q["id"]: q["question"] for q in items}
    if checkpoint.exists():
        for row in read_jsonl(checkpoint):
            validate_record(row)
            if row["id"] in done or expected.get(row["id"]) != row["question"]:
                raise ValueError("Checkpoint does not match input")
            start = row["trajectory"][0]
            if start["config"] != agent.config or start["corpus"] != agent.retrieval.manifest:
                raise ValueError("Checkpoint config/corpus mismatch")
            if start["llm_model"] != agent.client.model:
                raise ValueError("Checkpoint model mismatch")
            if start.get("llm_base_url") != getattr(agent.client, "base", "offline-test"):
                raise ValueError("Checkpoint API endpoint mismatch")
            done[row["id"]] = row
    if inflight_path.exists():
        inflight = json.loads(inflight_path.read_text(encoding="utf-8"))
        if inflight.get("id") not in done:
            raise ValueError("Interrupted during a question: API token accounting may be incomplete; start a new output path")
        inflight_path.unlink()
    with checkpoint.open("a", encoding="utf-8") as stream:
        for i, item in enumerate(items, 1):
            if item["id"] in done:
                continue
            save_json(inflight_path, {"id": item["id"], "question": item["question"]})
            try:
                row = agent.run(item["question"], item["id"])
                validate_record(row)
            except AgentFailure as exc:
                save_json(error_path, {"id": item["id"], "question": item["question"], "error": str(exc),
                           "trajectory": exc.state.trajectory, "known_total_tokens": exc.state.total_tokens,
                           "tool_calls": exc.state.tool_calls})
                raise
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done[item["id"]] = row
            inflight_path.unlink()
            print(f"{i}/{len(items)} {item['id']}: tokens={row['total_tokens']}, calls={row['tool_calls']}", flush=True)
    # Publish only after all questions have valid completed records.
    validate_file(checkpoint, input_file, expected_count)
    checkpoint.replace(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="COMP5423 skill-augmented fixed-corpus search")
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--documents", required=True)
    build.add_argument("--categories", nargs="+", choices=["sparse", "dense", "instruction"], default=["sparse", "dense", "instruction"])
    build.add_argument("--expected-documents", type=int, default=100195)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--output", default="outputs/interactive.json")
    ask.add_argument("--single", action="store_true")
    batch = sub.add_parser("batch")
    batch.add_argument("--questions", required=True)
    batch.add_argument("--output", default="outputs/test_prediction.jsonl")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--expected-count", type=int, default=100)
    evaluate_p = sub.add_parser("evaluate")
    evaluate_p.add_argument("--predictions", required=True)
    evaluate_p.add_argument("--gold", required=True)
    evaluate_p.add_argument("--output", default="outputs/evaluation.json")
    validate = sub.add_parser("validate")
    validate.add_argument("--predictions", required=True)
    validate.add_argument("--questions", required=True)
    validate.add_argument("--expected-count", type=int, default=100)
    validate.add_argument("--course", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        result = evaluate(args.predictions, args.gold)
        save_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))
        return
    if args.command == "validate":
        print(json.dumps(validate_file(args.predictions, args.questions, args.expected_count, args.course), indent=2))
        return
    config = load_config(args.config)
    if args.command == "build":
        root = Path(config["index_dir"])
        if not (root / "corpus.sqlite").exists():
            manifest = build_corpus(args.documents, config)
        else:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            # Avoid silently reusing an index for a different input corpus.
            import hashlib
            from .data import documents
            digest = hashlib.sha256()
            for doc in documents(args.documents):
                digest.update(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode())
            if digest.hexdigest() != manifest["corpus_sha256"]:
                raise ValueError("Existing index belongs to another corpus")
        if manifest["document_count"] != args.expected_documents:
            raise ValueError(f"Expected {args.expected_documents} documents; indexed {manifest['document_count']}")
        for method in args.categories:
            if method != "sparse":
                build_dense(config, method)
        print("Index build complete")
        return
    retrieval = Retrieval(config)
    try:
        client = APIClient(config["max_output_tokens"])
        agent = Agent(retrieval, client, config)
        if args.command == "ask":
            result = agent.run(args.question, mode="single" if args.single else "agent")
            save_json(args.output, result)
            print(result["answer"])
            print(f"Trace: {args.output}; calls={result['tool_calls']}; tokens={result['total_tokens']}")
        elif args.command == "batch":
            print(run_batch(agent, args.questions, args.output, args.resume, args.expected_count))
    finally:
        retrieval.close()


if __name__ == "__main__":
    main()
