"""文档解析：PDF 逐页抽文本，TXT/MD 当作单页。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ParseError(Exception):
    """解析失败时抛这个，接口层统一转成 400。"""


@dataclass
class Page:
    no: int
    text: str


def extract(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".markdown"):
        return [Page(1, path.read_text(encoding="utf-8", errors="ignore"))]
    if suffix != ".pdf":
        raise ParseError(f"不支持的文件类型：{suffix}（目前支持 PDF / TXT / MD）")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ParseError("缺少 pypdf，请先安装依赖") from exc

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, raw in enumerate(reader.pages, start=1):
        text = (raw.extract_text() or "").strip()
        if text:
            pages.append(Page(i, text))
    if not pages:
        raise ParseError("这份 PDF 没有可用文本层，可能是扫描件；扫描件需要先做 OCR")
    return pages
