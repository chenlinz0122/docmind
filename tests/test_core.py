from pathlib import Path

import pytest

from app.index import BM25, Store
from app.parse import ParseError, extract
from app.text import chunk_pages, split_sentences, tokenize


def test_tokenize_keeps_terms_and_drops_punct():
    tokens = tokenize("Redis 的 LRU 淘汰策略，命中率更高。")
    assert "lru" in tokens or "Redis" in tokens
    assert all(t.strip() for t in tokens)


def test_split_sentences():
    sents = split_sentences("第一句。第二句！第三句？")
    assert len(sents) == 3


def test_chunk_pages_keeps_page_number(tmp_path: Path):
    page = type("P", (), {"no": 3, "text": "甲。乙。丙。" * 30})
    chunks = chunk_pages([page], size=320, overlap=60)
    assert chunks and chunks[0]["page"] == 3


def test_bm25_ranks_matching_chunk_first():
    corpus = [tokenize("缓存 LRU 淘汰"), tokenize("日志 归档 压缩")]
    scores = BM25(corpus).scores(tokenize("LRU"))
    assert scores[0] > scores[1]


def test_store_roundtrip(tmp_path: Path):
    path = tmp_path / "index.json"
    store = Store.load(path)
    page = type("P", (), {"no": 1, "text": "Redis 支持 LRU 淘汰策略。"})
    store.add_document("d1", "缓存笔记", [page])
    assert store.search("LRU") and store.stats()["docs"] == 1

    reloaded = Store.load(path)
    assert reloaded.search("LRU")
    assert reloaded.remove_document("d1") is True
    assert reloaded.search("LRU") == []


def test_parse_txt(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("hello 缓存", encoding="utf-8")
    pages = extract(p)
    assert pages[0].no == 1


def test_parse_rejects_unknown_suffix(tmp_path: Path):
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG")
    with pytest.raises(ParseError):
        extract(p)
