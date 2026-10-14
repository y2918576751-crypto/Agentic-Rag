"""Create source-only ZIP. Never include keys, indexes, outputs or environments."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "indexes", "indexes-demo", "outputs"}
ALLOWED = {".py", ".md", ".json", ".jsonl", ".txt"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(ROOT.parent / "comp5423_search_agent_code.zip"))
    args = p.parse_args()
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob("*")):
            relative = path.relative_to(ROOT)
            if not path.is_file() or set(relative.parts) & EXCLUDED:
                continue
            if path.suffix not in ALLOWED and path.name not in {".env.example", ".gitignore"}:
                continue
            if path.name.startswith(".env") and path.name != ".env.example":
                continue
            archive.write(path, str(Path("code") / relative))
    print(target)


if __name__ == "__main__":
    main()
