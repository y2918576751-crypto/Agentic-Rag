from __future__ import annotations

import re
from pathlib import Path


class Skills:
    """Discover metadata at startup; read full instructions only on load_skill."""

    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.catalog = {}
        for path in sorted(self.root.glob("*/SKILL.md")):
            with path.open(encoding="utf-8") as stream:
                if stream.readline().strip() != "---":
                    raise ValueError(f"Missing skill frontmatter: {path}")
                metadata = {}
                for line in stream:
                    if line.strip() == "---":
                        break
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip("\"'")
                else:
                    raise ValueError(f"Unterminated frontmatter: {path}")
            name, description = metadata.get("name", ""), metadata.get("description", "")
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64 or name != path.parent.name:
                raise ValueError(f"Invalid skill name: {path}")
            if not description or len(description) > 1024:
                raise ValueError(f"Invalid description: {path}")
            self.catalog[name] = {"name": name, "description": description}

    def metadata(self):
        return list(self.catalog.values())

    def load(self, name: str):
        if name not in self.catalog:
            raise ValueError("Unknown skill")
        path = self.root / name / "SKILL.md"
        return {"name": name, "instructions": path.read_text(encoding="utf-8")}
