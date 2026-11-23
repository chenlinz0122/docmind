from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

DOC = (
    "第二章 缓存策略\n\n"
    "LRU 会淘汰最近最少使用的数据，适合缓存命中率要求高的场景。"
    "Redis 支持 LRU 淘汰策略，常用作会话缓存；内存不足时按策略清理。"
    "LFU 则按访问次数淘汰，适合热点数据变化慢的场景。"
)


def upload(name: str = "demo.txt", text: str = DOC) -> dict:
    resp = client.post(
        "/api/docs",
        files={"file": (name, text.encode("utf-8"), "text/plain")},
        data={"title": "缓存笔记"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health():
    assert client.get("/health").json()["ok"] is True


def test_upload_parses_and_chunks():
    body = upload("upload.txt")
    assert body["pages"] >= 1
    assert body["chunks"] >= 1
    client.delete(f"/api/docs/{body['doc_id']}")


def test_upload_rejects_unsupported_type():
    resp = client.post(
        "/api/docs",
        files={"file": ("a.exe", b"binary", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_ask_returns_answer_with_page_citation():
    body = upload("ask.txt")
    try:
        resp = client.post("/api/ask", json={"question": "Redis 用什么淘汰策略？"})
        data = resp.json()
        assert data["abstained"] is False
        assert data["citations"], "回答必须带引用"
        assert data["citations"][0]["page"] >= 1
        assert data["citations"][0]["chunk_id"].startswith(body["doc_id"])
    finally:
        client.delete(f"/api/docs/{body['doc_id']}")


def test_ask_abstains_on_unrelated_question():
    body = upload("abstain.txt")
    try:
        data = client.post("/api/ask", json={"question": "木星有几颗卫星"}).json()
        assert data["abstained"] is True
        assert data["answer"]
    finally:
        client.delete(f"/api/docs/{body['doc_id']}")


def test_delete_missing_doc_returns_404():
    assert client.delete("/api/docs/not-exist").status_code == 404
