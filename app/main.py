"""FastAPI 服务：上传文档 → 建索引 → 提问 → 带页码引用回答。"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .answer import Answerer
from .index import Store
from .parse import ParseError, extract

BASE = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = BASE / "var" / "uploads"
INDEX_PATH = BASE / "var" / "index.json"
ALLOWED = {".pdf", ".txt", ".md", ".markdown"}
MAX_BYTES = 20 * 1024 * 1024
MIN_SCORE = 1.0

app = FastAPI(title="DocMind", version="1.0.0")
UPLOADS.mkdir(parents=True, exist_ok=True)
store = Store.load(INDEX_PATH)
answerer = Answerer()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)


def _doc_id(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", Path(name).stem).strip("-")[:32] or "doc"
    return f"{stem}-{hashlib.sha1(name.encode('utf-8')).hexdigest()[:6]}"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/docs")
async def upload(file: UploadFile = File(...), title: str | None = Form(default=None)) -> JSONResponse:
    name = file.filename or "upload"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"不支持的类型 {suffix}，支持 PDF / TXT / MD")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="文件超过 20MB")

    doc_id = _doc_id(name)
    saved = UPLOADS / f"{doc_id}{suffix}"
    saved.write_bytes(raw)
    try:
        pages = extract(saved)
        chunks = store.add_document(doc_id, title or Path(name).stem, pages)
    except (ParseError, ValueError) as exc:
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "doc_id": doc_id, "pages": len(pages), "chunks": chunks})


@app.get("/api/docs")
def list_docs() -> JSONResponse:
    return JSONResponse({"docs": list(store.docs.values()), **store.stats()})


@app.delete("/api/docs/{doc_id}")
def delete_doc(doc_id: str) -> JSONResponse:
    if not store.remove_document(doc_id):
        raise HTTPException(status_code=404, detail="没有这份文档")
    for f in UPLOADS.glob(f"{doc_id}.*"):
        f.unlink(missing_ok=True)
    return JSONResponse({"ok": True})


@app.post("/api/ask")
def ask(req: AskRequest) -> JSONResponse:
    started = time.perf_counter()
    hits = store.search(req.question, k=req.top_k, min_score=MIN_SCORE)
    out = answerer.answer(req.question, hits)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return JSONResponse(
        {
            "question": req.question,
            "answer": out["answer"],
            "citations": out["citations"],
            "abstained": out["abstained"],
            "reason": out["reason"],
            "grounded": out["grounded"],
            "hits": [h.as_dict() for h in hits],
            "elapsed_ms": elapsed,
        }
    )


@app.get("/api/stats")
def stats() -> JSONResponse:
    return JSONResponse(store.stats())


@app.get("/health")
def health() -> dict:
    return {"ok": True, "docs": len(store.docs), "chunks": len(store.chunks)}
