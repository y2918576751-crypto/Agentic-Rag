from __future__ import annotations

import gc
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from .data import chunks, documents


def connect(path: Path):
    db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    return db


def build_corpus(source: str, config: dict):
    root = Path(config["index_dir"])
    root.mkdir(parents=True, exist_ok=True)
    target = root / "corpus.sqlite"
    if target.exists():
        raise FileExistsError("Index exists: use a new index_dir; rebuilding changes dense IDs")
    temporary = root / "corpus.building.sqlite"
    if temporary.exists():
        raise FileExistsError("Interrupted build exists; inspect corpus.building.sqlite first")
    db = connect(temporary)
    db.executescript("""
        CREATE TABLE documents(doc_id TEXT PRIMARY KEY, text TEXT NOT NULL, source_url TEXT NOT NULL);
        CREATE TABLE chunks(chunk_id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, start INTEGER, end INTEGER, text TEXT);
        CREATE INDEX chunks_by_doc ON chunks(doc_id);
        CREATE VIRTUAL TABLE search USING fts5(text, content='chunks', content_rowid='chunk_id', tokenize='porter unicode61');
    """)
    count = 0
    digest = hashlib.sha256()
    try:
        for doc in documents(source):
            db.execute("INSERT INTO documents VALUES(?,?,?)", tuple(doc[k] for k in ("doc_id", "text", "source_url")))
            digest.update(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode())
            for start, end, text in chunks(doc["text"], config["chunk_chars"], config["chunk_overlap"]):
                cur = db.execute("INSERT INTO chunks(doc_id,start,end,text) VALUES(?,?,?,?)", (doc["doc_id"], start, end, text))
                db.execute("INSERT INTO search(rowid,text) VALUES(?,?)", (cur.lastrowid, text))
            count += 1
            if count % 1000 == 0:
                db.commit()
                print(f"Indexed {count} documents", flush=True)
        if not count:
            raise ValueError("Empty corpus")
        db.execute("INSERT INTO search(search) VALUES('optimize')")
        db.commit()
        chunk_count = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    finally:
        db.close()
    manifest = {"document_count": count, "chunk_count": chunk_count, "corpus_sha256": digest.hexdigest(),
                "chunk_chars": config["chunk_chars"], "chunk_overlap": config["chunk_overlap"], "categories": ["sparse"]}
    temporary.replace(target)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_encoder(config: dict, method: str, model_name: str | None = None):
    from sentence_transformers import SentenceTransformer
    name = model_name or config["dense_model" if method == "dense" else "instruction_model"]
    encoder = SentenceTransformer(name, device=config["device"], trust_remote_code=False)
    encoder.max_seq_length = config["embedding_max_length"]
    if method == "instruction":
        encoder.tokenizer.padding_side = "left"
    return encoder


def encode(encoder, texts, method, is_query):
    options = {"normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False}
    if method == "dense":
        texts = [("query: " if is_query else "passage: ") + t for t in texts]
    elif is_query:
        options["prompt"] = "Instruct: Retrieve passages from the fixed collection that resolve the question's factual clues.\nQuery: "
    return encoder.encode(texts, **options).astype("float32")


def build_dense(config: dict, method: str):
    import faiss
    import numpy as np
    if method not in {"dense", "instruction"}:
        raise ValueError(method)
    root = Path(config["index_dir"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    encoder = load_encoder(config, method)
    db = connect(root / "corpus.sqlite")
    cursor = db.execute("SELECT chunk_id,text FROM chunks ORDER BY chunk_id")
    index = None
    while rows := cursor.fetchmany(config["batch_size"]):
        matrix = encode(encoder, [r["text"] for r in rows], method, False)
        if index is None:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(matrix.shape[1]))
        index.add_with_ids(matrix, np.array([r["chunk_id"] for r in rows], dtype="int64"))
        if index.ntotal % (config["batch_size"] * 100) == 0:
            print(f"{method}: {index.ntotal} chunks", flush=True)
    db.close()
    if index is None:
        raise ValueError("No chunks")
    # FAISS native file I/O may not support Chinese paths; use Python bytes.
    (root / f"{method}.faiss").write_bytes(faiss.serialize_index(index).tobytes())
    info = {"model": config["dense_model" if method == "dense" else "instruction_model"],
            "max_length": config["embedding_max_length"], "dimension": index.d,
            "count": index.ntotal, "corpus_sha256": manifest["corpus_sha256"]}
    (root / f"{method}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if method not in manifest["categories"]:
        manifest["categories"].append(method)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del encoder, index
    gc.collect()
    return info


class Retrieval:
    """One agent-facing retrieval tool, with optional internal RRF fusion."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config["index_dir"])
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.methods = config["methods"]
        self.encoders, self.indexes = {}, {}
        for method in self.methods:
            if method != "sparse":
                for ext in ("faiss", "json"):
                    if not (self.root / f"{method}.{ext}").exists():
                        raise FileNotFoundError(f"Build {method} index or explicitly select sparse-only config")
                info = json.loads((self.root / f"{method}.json").read_text(encoding="utf-8"))
                expected = config["dense_model" if method == "dense" else "instruction_model"]
                if info["corpus_sha256"] != self.manifest["corpus_sha256"] or info["model"] != expected:
                    raise ValueError(f"{method} index/config mismatch")
                if info["max_length"] != config["embedding_max_length"]:
                    raise ValueError(f"{method} max_length mismatch")
        self.db = connect(self.root / "corpus.sqlite")

    def close(self):
        self.db.close()

    def _sparse(self, query, k):
        terms = list(dict.fromkeys(re.findall(r"\w+", query, flags=re.UNICODE)))[:48]
        if not terms:
            return []
        expression = " OR ".join('"' + t + '"' for t in terms)
        rows = self.db.execute("SELECT rowid,bm25(search) AS score FROM search WHERE search MATCH ? ORDER BY score LIMIT ?", (expression, k))
        return [(r["rowid"], -r["score"]) for r in rows]

    def _dense(self, query, k, method):
        import faiss
        import numpy as np
        if method not in self.encoders:
            self.encoders[method] = load_encoder(self.config, method)
            raw = np.frombuffer((self.root / f"{method}.faiss").read_bytes(), dtype="uint8")
            self.indexes[method] = faiss.deserialize_index(raw)
        query_vec = encode(self.encoders[method], [query], method, True)
        scores, ids = self.indexes[method].search(query_vec, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def retrieve(self, query: str, top_k: int = 8, method: str = "hybrid"):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must be 1..2000 characters")
        top_k = max(1, min(int(top_k), 20))
        methods = self.methods if method == "hybrid" else [method]
        if not set(methods) <= set(self.methods):
            raise ValueError(f"Available retrieval methods: {self.methods}")
        candidates = max(top_k, self.config["candidate_k"])
        scores, ranks = {}, {}
        for current in methods:
            hits = self._sparse(query, candidates) if current == "sparse" else self._dense(query, candidates, current)
            for rank, (cid, score) in enumerate(hits, 1):
                scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
                ranks.setdefault(cid, {})[current] = {"rank": rank, "score": score}
        results, seen = [], set()
        for cid in sorted(scores, key=lambda c: (-scores[c], c)):
            row = self.db.execute("SELECT c.*,d.source_url FROM chunks c JOIN documents d USING(doc_id) WHERE chunk_id=?", (cid,)).fetchone()
            if row["doc_id"] in seen:
                continue
            seen.add(row["doc_id"])
            results.append({"doc_id": row["doc_id"], "chunk_id": cid, "score": scores[cid],
                            "snippet": row["text"], "start": row["start"], "end": row["end"],
                            "source_url": row["source_url"], "retrievers": ranks[cid]})
            if len(results) >= top_k:
                break
        return {"query": query, "method": method, "results": results}

    def open_document(self, doc_id: str, start: int = 0, length: int = 3500):
        row = self.db.execute("SELECT * FROM documents WHERE doc_id=?", (str(doc_id),)).fetchone()
        if row is None:
            raise ValueError("Unknown doc_id")
        start, length = max(0, int(start)), max(1, min(int(length), 6000))
        text = row["text"][start:start + length]
        return {"doc_id": row["doc_id"], "source_url": row["source_url"], "start": start,
                "end": start + len(text), "total_chars": len(row["text"]), "text": text}

    def find_in_document(self, doc_id: str, term: str):
        row = self.db.execute("SELECT * FROM documents WHERE doc_id=?", (str(doc_id),)).fetchone()
        if row is None or not isinstance(term, str) or not term.strip():
            raise ValueError("Known doc_id and non-empty term required")
        text = row["text"]
        hits = []
        for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
            start, end = max(0, match.start() - 250), min(len(text), match.end() + 450)
            hits.append({"doc_id": row["doc_id"], "source_url": row["source_url"], "start": start, "end": end, "text": text[start:end]})
            if len(hits) >= 8:
                break
        return {"doc_id": row["doc_id"], "term": term, "matches": hits}
