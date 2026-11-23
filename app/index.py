"""BM25 打分 + 文档库（入库、检索、删除、落盘）。"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .text import chunk_pages, tokenize


class BM25:
    """k1 控制词频饱和，b 控制长度归一化；参数取常用默认值。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = corpus
        self.n = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.n if self.n else 0.0
        self.freq: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for doc in corpus:
            tf: dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.freq.append(tf)
            for term in tf:
                df[term] = df.get(term, 0) + 1
        self.idf = {
            term: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for term, c in df.items()
        }

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, tf in enumerate(self.freq):
            dl = len(self.docs[i]) or 1
            score = 0.0
            for term in query:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * f * (self.k1 + 1) / denom
            out.append(score)
        return out


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    page: int
    text: str
    score: float

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page": self.page,
            "text": self.text,
            "score": round(self.score, 3),
        }


@dataclass
class Store:
    path: Path | None = None
    docs: dict[str, dict] = field(default_factory=dict)
    chunks: list[dict] = field(default_factory=list)
    _index: BM25 | None = None
    _mtime: float = 0.0

    # ---------------------------------------------------------------- 入库

    def add_document(self, doc_id: str, title: str, pages: list, size: int = 320, overlap: int = 60) -> int:
        pieces = chunk_pages(pages, size=size, overlap=overlap)
        if not pieces:
            raise ValueError("文档里没有可入库的正文")
        self.docs[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "pages": len(pages),
            "chunks": len(pieces),
            "added_at": int(time.time()),
        }
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        for i, piece in enumerate(pieces):
            self.chunks.append(
                {
                    "chunk_id": f"{doc_id}#{i}",
                    "doc_id": doc_id,
                    "title": title,
                    "page": piece["page"],
                    "text": piece["text"].strip(),
                }
            )
        self._index = None
        self.save()
        return len(pieces)

    def remove_document(self, doc_id: str) -> bool:
        if doc_id not in self.docs:
            return False
        del self.docs[doc_id]
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        self._index = None
        self.save()
        return True

    # ---------------------------------------------------------------- 检索

    @property
    def index(self) -> BM25:
        if self._index is None:
            corpus = [tokenize(c["title"] + "\n" + c["text"]) for c in self.chunks]
            self._index = BM25(corpus)
        return self._index

    def search(self, query: str, k: int = 4, min_score: float = 0.0) -> list[Hit]:
        if not self.chunks:
            return []
        scores = self.index.scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits = []
        for i in ranked:
            if scores[i] <= 0 or scores[i] < min_score:
                continue
            c = self.chunks[i]
            hits.append(Hit(c["chunk_id"], c["doc_id"], c["title"], c["page"], c["text"], scores[i]))
        return hits

    # ---------------------------------------------------------------- 落盘

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"docs": self.docs, "chunks": self.chunks}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._mtime = self.path.stat().st_mtime

    @classmethod
    def load(cls, path: Path) -> "Store":
        store = cls(path=path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            store.docs = data.get("docs", {})
            store.chunks = data.get("chunks", [])
            store._mtime = path.stat().st_mtime
        return store

    def stats(self) -> dict:
        return {
            "docs": len(self.docs),
            "chunks": len(self.chunks),
            "avg_chunk_chars": round(sum(len(c["text"]) for c in self.chunks) / len(self.chunks), 1)
            if self.chunks
            else 0,
        }
