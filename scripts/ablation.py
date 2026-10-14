"""Real-API validation-only ablations. Every run is logged; no mock predictions."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentsearch.agent import Agent
from agentsearch.config import ROOT, load_config
from agentsearch.data import questions
from agentsearch.evaluation import evaluate
from agentsearch.llm import APIClient
from agentsearch.retrieval import Retrieval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "config.json"))
    p.add_argument("--questions", required=True, help="Validation JSONL with gold answers")
    p.add_argument("--output-dir", default="outputs/ablations")
    args = p.parse_args()
    base = load_config(args.config)
    variants = {
        "single_rag": {"skills_enabled": False, "mode": "single"},
        "agent_without_skills": {"skills_enabled": False},
        "agent_without_context_management": {"context_management": False},
        "full": {},
        "full_sparse": {"methods": ["sparse"]},
        "full_sparse_dense": {"methods": ["sparse", "dense"]},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, settings in variants.items():
        config = dict(base)
        config.update({k: v for k, v in settings.items() if k != "mode"})
        target = out / f"{name}.jsonl"
        if target.exists():
            raise FileExistsError("Use a new output directory for a new experiment")
        retrieval = Retrieval(config)
        try:
            agent = Agent(retrieval, APIClient(config["max_output_tokens"]), config)
            with target.open("w", encoding="utf-8") as stream:
                for q in questions(args.questions):
                    row = agent.run(q["question"], q["id"], mode=settings.get("mode", "agent"))
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
            stats = evaluate(target, args.questions)
            summary[name] = {k: v for k, v in stats.items() if k != "details"}
            (out / f"{name}.evaluation.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(name, summary[name], flush=True)
        finally:
            retrieval.close()


if __name__ == "__main__":
    main()
