"""Discovery of MinerU visual artifacts that require image analysis."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.documents import mineru as mineru_client

_HTML_IMAGE = re.compile(r"<img\b", re.IGNORECASE)
_HTML_TABLE = re.compile(r"<table\b", re.IGNORECASE)
_HTML_ROW = re.compile(r"<tr\b", re.IGNORECASE)
_HTML_CELL = re.compile(r"<t[dh]\b", re.IGNORECASE)


@dataclass(frozen=True)
class VisualCandidate:
    """One MinerU image artifact selected for model or OCR analysis."""
    page: int
    block_type: str
    image_path: Path
    relative_path: str
    caption: str = ""
    footnote: str = ""
    html: str = ""
    headings: tuple[str, ...] = ()


def _page_blocks(content_list: list[Any]):
    """兼容两种内容列表结构，逐个产出页码和内容块。"""
    if all(isinstance(page, list) for page in content_list):
        for page_number, blocks in enumerate(content_list):
            for block in blocks:
                if isinstance(block, dict):
                    yield page_number, block
        return
    for block in content_list:
        if not isinstance(block, dict):
            continue
        raw_page = block.get("page_idx", block.get("page", 0))
        try:
            page_number = max(0, int(raw_page))
        except (TypeError, ValueError):
            page_number = 0
        yield page_number, block


def _title_level(content: Any) -> int:
    """从 MinerU 标题块中读取范围受限的 Markdown 标题层级。"""
    raw_level = content.get("level", 2) if isinstance(content, dict) else 2
    try:
        return max(1, min(6, int(raw_level)))
    except (TypeError, ValueError):
        return 2


def _table_needs_visual_analysis(html: str) -> bool:
    """判断表格是否由图片承载，或缺少完整 HTML 结构。"""
    if _HTML_IMAGE.search(html or ""):
        return True
    return not all(
        pattern.search(html or "")
        for pattern in (_HTML_TABLE, _HTML_ROW, _HTML_CELL)
    )


# 图注以“图 1”“Figure 2”开头；表格以“表 1”“Table 2”开头。
_FIGURE_CAPTION = re.compile(
    r"^\s*(?:附?图|Figure|Fig\.?)\s*[A-Za-z]?[\d一二三四五六七八九十]", re.IGNORECASE
)
_TABLE_CAPTION = re.compile(
    r"^\s*(?:附?表|Table)\s*[A-Za-z]?[\d一二三四五六七八九十]", re.IGNORECASE
)
# 以“图 1”开头的正文句子很长且带句号，用长度和句号把它们和真正的图注区分开。
_CAPTION_MAX_CHARS = 120


def _adjacent_caption(
    blocks: list[tuple[int, dict[str, Any]]], index: int, block_type: str
) -> str:
    """MinerU 未把图注挂到图上时，从同页紧邻的段落块中恢复。

    图的图注在图下方，表的图注在表上方；只看紧邻的一个块，宁可漏掉也不张冠李戴。
    """
    page_number = blocks[index][0]
    neighbor_index = index + 1 if block_type == "image" else index - 1
    if not 0 <= neighbor_index < len(blocks):
        return ""
    neighbor_page, neighbor = blocks[neighbor_index]
    if neighbor_page != page_number or neighbor.get("type") != "paragraph":
        return ""
    text = " ".join(mineru_client.content_text(neighbor.get("content")).split())
    pattern = _TABLE_CAPTION if block_type == "table" else _FIGURE_CAPTION
    if not text or len(text) > _CAPTION_MAX_CHARS or "。" in text or not pattern.match(text):
        return ""
    return text


def _content_field(content: Any, key: str) -> str:
    """读取原始 HTML，或把其他结构化内容字段渲染为文本。"""
    if not isinstance(content, dict):
        return ""
    value = content.get(key, "")
    if key == "html" and isinstance(value, str):
        return value
    return mineru_client.content_text(value)


def discover_visual_candidates(file_path: str) -> list[VisualCandidate]:
    """发现需要分析的去重图片和表格，并附带所在标题上下文。"""
    content_list = mineru_client.read_saved_content_list(file_path)
    if not content_list:
        return []
    headings: dict[int, str] = {}
    seen_paths: set[str] = set()
    candidates: list[VisualCandidate] = []
    blocks = list(_page_blocks(content_list))
    for index, (page_number, block) in enumerate(blocks):
        block_type = str(block.get("type", "")).strip().lower()
        content = block.get("content")
        if block_type == "title":
            title = mineru_client.content_text(content)
            if title:
                level = _title_level(content)
                for existing_level in list(headings):
                    if existing_level >= level:
                        headings.pop(existing_level)
                headings[level] = title
            continue
        if block_type not in {"image", "table"} or not isinstance(content, dict):
            continue
        html = _content_field(content, "html")
        if block_type == "table" and not _table_needs_visual_analysis(html):
            continue
        image_source = content.get("image_source")
        relative_path = (
            image_source.get("path", "") if isinstance(image_source, dict) else ""
        )
        if not isinstance(relative_path, str) or relative_path in seen_paths:
            continue
        image_path = mineru_client.resolve_saved_artifact(file_path, relative_path)
        if image_path is None:
            continue
        seen_paths.add(relative_path)
        caption_key = (
            "table_caption" if block_type == "table" else "image_caption"
        )
        footnote_key = (
            "table_footnote" if block_type == "table" else "image_footnote"
        )
        candidates.append(
            VisualCandidate(
                page=page_number,
                block_type=block_type,
                image_path=image_path,
                relative_path=relative_path.replace("\\", "/"),
                caption=_content_field(content, caption_key)
                or _adjacent_caption(blocks, index, block_type),
                footnote=_content_field(content, footnote_key),
                html=html,
                headings=tuple(headings[level] for level in sorted(headings)),
            )
        )
    return candidates
