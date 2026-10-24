"""Conversion of MinerU structured content into indexable page text."""

import html
import re
from typing import Any

from .errors import MinerUError

_REFERENCE_LIST_ITEM = re.compile(r"^\s*(?:[-*]\s*)?\[\d+(?:\s*[-,，]\s*\d+)*\]")
_LABELED_LIST_ITEM = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\[\d+(?:\s*[-,，]\s*\d+)*\]|\d+\s*[.、）)]|[一二三四五六七八九十]+\s*[、.）)])"
)
_BIBLIOGRAPHY_HEADING = re.compile(
    r"^(?:第[一二三四五六七八九十百千万零〇0-9]+[章节篇部分]\s*)?"
    r"(?:参考文献|参考资料|参考书目|文献目录|references?|bibliography)$",
    re.IGNORECASE,
)
# 只匹配真正的标签与注释：“<”后必须紧跟字母或“/字母”，且标签内不含“<”“>”。
# 这样“x < 5”“温度<10℃”“>80%”这类正文比较符不会被当成标签整段删掉。
_HTML_TAG_OR_COMMENT = re.compile(r"<!--.*?-->|</?[A-Za-z][^<>]*>", re.DOTALL)


def _clean_text(value: str) -> str:
    """将简单 HTML 和转义内容转换为紧凑纯文本。"""
    value = html.unescape(value or "")
    if "<" in value and ">" in value:
        value = re.sub(
            r"<(?:br|/p|/div|/tr|/li|/td|/th)[^>]*>", "\n", value, flags=re.I
        )
        value = _HTML_TAG_OR_COMMENT.sub(" ", value)
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _text_from_value(value: Any) -> list[str]:
    """递归提取 MinerU 已知内容字段中的可读文本。"""
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _text_from_value(item)]
    if not isinstance(value, dict):
        return []
    texts: list[str] = []
    for key in (
        "content", "text", "text_content", "title_content", "paragraph_content",
        "list_content", "list_items", "item_content", "algorithm_content",
        "algorithm_caption", "equation_content", "math_content", "latex", "html",
        "table_caption", "table_body", "table_footnote", "image_caption",
        "image_footnote", "caption", "footnote",
    ):
        if key in value:
            texts.extend(_text_from_value(value[key]))
    return texts


def _unique_text(values: list[str]) -> list[str]:
    """规范化并去重文本片段，同时保持原始顺序。"""
    unique: list[str] = []
    for value in values:
        normalized = _clean_text(value)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def content_text(value: Any) -> str:
    """将任意 MinerU 内容值转换为可读文本。"""
    return "\n".join(_unique_text(_text_from_value(value)))


def _is_reference_list_item(text: str) -> bool:
    """判断列表项是否类似带编号的参考文献条目。"""
    return bool(_REFERENCE_LIST_ITEM.match(text or ""))


def _starts_labeled_list_item(text: str) -> bool:
    """判断文本是否以常见列表标记开头。"""
    return bool(_LABELED_LIST_ITEM.match(text or ""))


def is_bibliography_heading(text: str) -> bool:
    """识别独立参考文献标题，同时避免误匹配正文内引用。"""
    normalized = re.sub(r"^\s*#{1,6}\s*", "", text or "")
    normalized = " ".join(normalized.split()).strip("：:。.")
    normalized = re.sub(
        r"^(?:第[一二三四五六七八九十百千万零〇0-9]+[章节篇部分]\s*|"
        r"[一二三四五六七八九十百千万零〇]+[、.]\s*|\d+(?:\.\d+)*[.、]?\s*)",
        "",
        normalized,
    )
    return bool(_BIBLIOGRAPHY_HEADING.fullmatch(normalized))


def _render_block(block: Any) -> str:
    """渲染一个结构化内容块，并丢弃页眉页脚等版面噪声。"""
    if not isinstance(block, dict):
        return ""
    block_type = str(block.get("type", "")).lower()
    if block_type in {"page_header", "page_footer", "page_number"}:
        return ""
    content = block.get("content")
    values = _unique_text(_text_from_value(content))
    if not values:
        return ""
    text = "\n".join(values)
    if block_type == "title":
        level = content.get("level", 2) if isinstance(content, dict) else 2
        try:
            level = max(1, min(6, int(level)))
        except (TypeError, ValueError):
            level = 2
        return f"{'#' * level} {text}"
    if block_type == "list":
        return "\n".join(f"- {item}" for item in values)
    return text


def content_list_to_page_texts(content_list: Any) -> list[tuple[int, str]]:
    """将 MinerU 结构化内容块转换为非空的零起始页文本列表。"""
    if not isinstance(content_list, list):
        raise MinerUError("MinerU 返回的结构化结果格式无效")
    if all(isinstance(page, list) for page in content_list):
        pages = [(index, page) for index, page in enumerate(content_list)]
    else:
        grouped: dict[int, list[Any]] = {}
        for block in content_list:
            if not isinstance(block, dict):
                continue
            raw_page = block.get("page_idx", block.get("page", 0))
            try:
                page_number = int(raw_page)
            except (TypeError, ValueError):
                page_number = 0
            grouped.setdefault(max(0, page_number), []).append(block)
        pages = sorted(grouped.items())

    rendered_pages: list[tuple[int, str]] = []
    in_reference_list = False
    in_bibliography = False
    for page_number, blocks in pages:
        fragments = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).lower()
            if block_type in {"page_header", "page_footer", "page_number", "page_footnote"}:
                continue
            fragment = _render_block(block)
            if not fragment:
                continue
            if block_type == "title":
                if is_bibliography_heading(fragment):
                    in_bibliography = True
                    in_reference_list = False
                    continue
                if in_bibliography:
                    in_bibliography = False
            if in_bibliography:
                continue
            if block_type == "list":
                if _is_reference_list_item(fragment):
                    in_reference_list = True
                    continue
                if in_reference_list and not _starts_labeled_list_item(fragment):
                    continue
                in_reference_list = False
            else:
                in_reference_list = False
            fragments.append(fragment)
        text = "\n\n".join(fragment for fragment in fragments if fragment.strip()).strip()
        if text:
            rendered_pages.append((page_number, text))
    return rendered_pages
