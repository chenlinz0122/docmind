"""分词与切块：jieba 分词（有就用，没有退化成字二元组），按句聚合 + 滑窗。"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[\s\u3000]+")
_SENT = re.compile(r"(?<=[。！？!?；;\n])")

try:  # jieba 是可选依赖：装了走分词，没装也能跑
    import jieba

    jieba.setLogLevel(60)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    if _HAS_JIEBA:
        words = [w.strip() for w in jieba.lcut(text)]
    else:  # 退化成「英文词 + 中文二元组」，保证零依赖也能用
        latin = re.findall(r"[a-z0-9_]+", text)
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        words = latin + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    out = []
    for w in words:
        w = _PUNCT.sub("", w)
        if len(w) <= 1 and not w.isdigit():
            continue
        if not re.search(r"[a-z0-9\u4e00-\u9fff]", w):
            continue
        out.append(w)
    return out


def split_sentences(text: str) -> list[str]:
    units = [u.strip() for u in _SENT.split(text)]
    return [u for u in units if len(u) >= 2]


def chunk_pages(pages: list, size: int = 320, overlap: int = 60) -> list[dict]:
    """把每页切成若干块，块上带页码——引用能回到「第几页」，这是对照原文的关键。

    切法：按句累积到 size 字左右；换页时把上一页的尾巴（overlap 字）带到下一页开头，
    避免一句话被页边界切断后两边都检索不到。
    """
    chunks: list[dict] = []
    carry = ""
    for page in pages:
        text = re.sub(r"\n{3,}", "\n\n", page.text)
        cur = carry
        for sent in split_sentences(text):
            if cur and len(cur) + len(sent) > size:
                chunks.append({"page": page.no, "text": cur})
                cur = (cur[-overlap:] if overlap else "") + sent
            else:
                cur += sent
        carry = ""
        if cur.strip():
            chunks.append({"page": page.no, "text": cur.strip()})
    return chunks
