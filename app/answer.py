"""回答生成：默认抽取式（无需 Key），配了 Key 走 OpenAI 兼容接口。

不管走哪条路，引用都带「文档名 + 页码 + 片段编号」，方便对照原文核对。
"""

from __future__ import annotations

import os

from .text import split_sentences, tokenize

ABSTAIN_TEXT = "文档里没有找到能回答这个问题的内容。"


class Answerer:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or "deepseek-chat"
        self.timeout = timeout

    def answer(self, question: str, hits: list) -> dict:
        if not hits:
            return self._abstain("no_hits")
        if self.api_key and self.base_url:
            try:
                return self._online(question, hits)
            except Exception:
                # 接口挂了就退回抽取式：宁可答得笨一点，也不能让服务不可用
                return self._offline(question, hits)
        return self._offline(question, hits)

    # ---------------------------------------------------------------- 离线

    def _offline(self, question: str, hits: list) -> dict:
        q = set(tokenize(question))
        candidates = []
        for hit in hits[:3]:
            for sent in split_sentences(hit.text):
                if len(sent) < 6:
                    continue
                overlap = self._overlap(q, set(tokenize(sent)))
                if overlap > 0:
                    candidates.append((overlap, sent, hit))
        if not candidates:
            return self._abstain("no_overlap")
        candidates.sort(key=lambda x: x[0], reverse=True)
        picked = candidates[:3]
        body = "".join(p[1] for p in picked)
        return {
            "answer": body,
            "citations": self._citations(picked),
            "grounded": round(candidates[0][0], 3),
            "abstained": False,
            "reason": "answered_offline",
        }

    # ---------------------------------------------------------------- 在线

    def _online(self, question: str, hits: list) -> dict:
        import httpx

        context = "\n\n".join(
            f"[{h.chunk_id} 第{h.page}页] {h.title}\n{h.text}" for h in hits[:3]
        )
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "你只根据给出的资料回答，答完在句末标注片段编号（如 [a#0]）。资料不够就回答「资料不足」。",
                },
                {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
            ],
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if "资料不足" in text:
            return self._abstain("ungrounded")
        return {
            "answer": text,
            "citations": [
                {
                    "chunk_id": h.chunk_id,
                    "doc_id": h.doc_id,
                    "title": h.title,
                    "page": h.page,
                    "snippet": h.text[:160],
                }
                for h in hits[:3]
            ],
            "grounded": None,
            "abstained": False,
            "reason": "answered_online",
        }

    # ---------------------------------------------------------------- 工具

    @staticmethod
    def _citations(picked: list) -> list[dict]:
        out, seen = [], set()
        for _, _, hit in picked:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            out.append(
                {
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "title": hit.title,
                    "page": hit.page,
                    "snippet": hit.text[:160],
                }
            )
        return out

    @staticmethod
    def _overlap(a: set[str], b: set[str]) -> float:
        if not a:
            return 0.0
        return len(a & b) / len(a)

    def _abstain(self, reason: str) -> dict:
        return {"answer": ABSTAIN_TEXT, "citations": [], "grounded": 0.0, "abstained": True, "reason": reason}
