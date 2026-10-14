"""Export course-provided HF dataset/parquet to JSONL; no dataset ID is guessed."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Exact instructor-provided Hugging Face dataset ID")
    parser.add_argument("--subset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--parquet", help="Local parquet file, instead of --dataset")
    parser.add_argument("--kind", choices=["documents", "questions"], required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--url-field", default="url")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.dataset) == bool(args.parquet):
        parser.error("Choose exactly one of --dataset or --parquet")
    from datasets import load_dataset
    dataset = load_dataset("parquet", data_files=args.parquet, split="train", streaming=True) if args.parquet else load_dataset(
        args.dataset, args.subset, split=args.split, streaming=True)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with target.open("w", encoding="utf-8") as stream:
        for record in dataset:
            rid = str(record[args.id_field])
            if rid in seen:
                raise ValueError("Duplicate ID: " + rid)
            seen.add(rid)
            if args.kind == "documents":
                item = {"doc_id": rid, "text": record[args.text_field], "source_url": record.get(args.url_field, "")}
            else:
                item = {"id": rid, "question": record[args.question_field]}
                if args.answer_field in record:
                    item["answer"] = record[args.answer_field]
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Exported", len(seen), "records")


if __name__ == "__main__":
    main()
