"""Section hierarchy parsing and public structured-document chunking."""

import os

from langchain_core.documents import Document as LCDocument

from services.core.config import DEFAULT_CHUNK_SIZE
from services.documents import mineru as mineru_client
from services.documents.heading_patterns import (
    APPENDIX_CHILD_HEADING,
    APPENDIX_HEADING,
    ARABIC_ITEM_HEADING,
    CHAPTER_HEADING,
    CHINESE_MAJOR_HEADING,
    CHINESE_SUB_HEADING,
    CIRCLED_NUMBER_HEADING,
    PAREN_NUMBER_HEADING,
    PREFACE_HEADING,
    STANDARD_HEADING,
    STANDARD_TOP_HEADING,
)

from .common import (
    _MARKDOWN_HEADING,
    CHUNK_ROUTE_STRUCTURED,
    STRUCTURED_CHUNK_MAX_CHARS,
    STRUCTURED_CHUNK_MIN_CHARS,
    _clean_chunk_text,
    _StructuredSection,
    assign_chunk_indexes,
)
from .structured_tables import normalize_mineru_markdown, split_structured_content


def _clean_structure_heading(value: str) -> str:
    """移除标题中的 Markdown 标记，并合并多余空白。"""
    return " ".join(str(value or "").strip().strip("#").split())


def _heading_style_and_level(
    heading: str,
    markdown_level: int,
    has_chapters: bool,
    stack: list[_StructuredSection],
    *,
    infer_numbered_hierarchy: bool = True,
) -> tuple[str, int]:
    """根据 Markdown 标记和章节编号推断稳定的标题样式与层级。"""
    if not infer_numbered_hierarchy:
        return "markdown", max(1, min(6, markdown_level))
    if CHAPTER_HEADING.match(heading):
        return "chapter", 1
    if APPENDIX_CHILD_HEADING.match(heading):
        return "appendix_child", 2
    if APPENDIX_HEADING.match(heading):
        return "appendix", 1
    if PREFACE_HEADING.fullmatch(heading):
        return "preface", 1
    standard = STANDARD_HEADING.match(heading)
    if standard:
        return "standard", min(6, len(standard.group(1).split(".")))
    if STANDARD_TOP_HEADING.match(heading):
        return "standard", 1
    if CHINESE_MAJOR_HEADING.match(heading):
        return "chinese_major", 2 if has_chapters else 1
    if CHINESE_SUB_HEADING.match(heading):
        return "chinese_sub", 3 if has_chapters else 2
    for style, pattern in (
        ("arabic_item", ARABIC_ITEM_HEADING),
        ("paren_number", PAREN_NUMBER_HEADING),
        ("circled_number", CIRCLED_NUMBER_HEADING),
    ):
        if not pattern.match(heading):
            continue
        if stack and stack[-1].style == style:
            return style, stack[-1].level
        parent_level = stack[-1].level if stack else 0
        return style, min(6, max(1, parent_level + 1))
    return "markdown", max(1, min(6, markdown_level))


def _parse_markdown_sections(
    text: str, *, infer_numbered_hierarchy: bool = True
) -> _StructuredSection:
    """即使 MinerU 将标题层级扁平化，也尽量恢复章节树。"""
    matches = list(_MARKDOWN_HEADING.finditer(text))
    root = _StructuredSection(title="", level=0, style="root", body=[], children=[])
    if not matches:
        root.body.append(text.strip())
        return root
    has_chapters = any(
        CHAPTER_HEADING.match(_clean_structure_heading(match.group(2)))
        for match in matches
    )
    stack = [root]
    current = root
    cursor = 0
    for match in matches:
        before = text[cursor : match.start()].strip()
        if before:
            current.body.append(before)
        title = _clean_structure_heading(match.group(2))
        if not title:
            cursor = match.end()
            continue
        style, level = _heading_style_and_level(
            title,
            len(match.group(1)),
            has_chapters,
            stack[1:],
            infer_numbered_hierarchy=infer_numbered_hierarchy,
        )
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        current = _StructuredSection(
            title=title, level=level, style=style, body=[], children=[]
        )
        stack[-1].children.append(current)
        stack.append(current)
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        current.body.append(tail)
    return root


def _remove_bibliography_sections(section: _StructuredSection) -> None:
    """从解析后的章节树中递归移除参考文献部分。"""
    kept_children = []
    for child in section.children:
        if mineru_client.is_bibliography_heading(child.title):
            continue
        _remove_bibliography_sections(child)
        kept_children.append(child)
    section.children = kept_children


def _remove_leading_noncontent_sections(root: _StructuredSection) -> None:
    """丢弃首个正文标题之前的封面和目录章节。"""
    content_styles = {"chapter", "standard", "chinese_major", "appendix"}
    first_content = next(
        (
            index
            for index, child in enumerate(root.children)
            if child.style in content_styles
        ),
        None,
    )
    if first_content is None:
        return
    root.children = [
        child
        for index, child in enumerate(root.children)
        if index >= first_content or child.style == "preface"
    ]


def _render_section_unit(
    section: _StructuredSection, path: tuple[str, ...]
) -> tuple[tuple[str, ...], str] | None:
    """将章节正文和完整标题路径一起渲染为文本单元。"""
    body = "\n\n".join(
        block.strip() for block in section.body if block.strip()
    ).strip()
    if not body:
        return None
    return path, f"{section.title}\n{body}".strip()


def _common_heading_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    """返回多个标题路径共有的最长前缀。"""
    if not paths:
        return ()
    common = list(paths[0])
    for path in paths[1:]:
        while common and tuple(common) != path[: len(common)]:
            common.pop()
    return tuple(common)


def _group_sibling_units(
    units: list[tuple[tuple[str, ...], str]], target_chars: int
) -> list[tuple[tuple[str, ...], str]]:
    """合并相邻叶子章节，直到达到适合作答的文本大小。"""
    grouped: list[tuple[tuple[str, ...], str]] = []
    pending: list[tuple[tuple[str, ...], str]] = []
    pending_size = 0

    def flush() -> None:
        """输出当前缓存的同级章节组，并重置累计状态。"""
        nonlocal pending, pending_size
        if pending:
            paths = [path for path, _ in pending]
            content = "\n\n".join(content for _, content in pending)
            grouped.append((_common_heading_path(paths), content))
        pending = []
        pending_size = 0

    for unit in units:
        unit_size = len(unit[1])
        if (
            pending
            and pending_size + 2 + unit_size > target_chars
            and pending_size >= STRUCTURED_CHUNK_MIN_CHARS
        ):
            flush()
        pending.append(unit)
        pending_size += unit_size + (2 if pending_size else 0)
        if pending_size >= target_chars:
            flush()
    flush()
    return grouped


def _collect_structured_units(
    section: _StructuredSection,
    path: tuple[str, ...],
    target_chars: int,
) -> list[tuple[tuple[str, ...], str]]:
    """收集适合作答的文本单元，同时避免合并无关根章节。"""
    current_path = path + ((section.title,) if section.title else ())
    collected: list[tuple[tuple[str, ...], str]] = []
    own = _render_section_unit(section, current_path)
    if not section.title and section.children:
        own = None
    pending_leaves: list[tuple[tuple[str, ...], str]] = []
    if own and not section.children:
        collected.append(own)
    elif own:
        pending_leaves.append(own)
    for child in section.children:
        child_path = current_path + (child.title,)
        if child.children:
            collected.extend(_group_sibling_units(pending_leaves, target_chars))
            pending_leaves = []
            collected.extend(
                _collect_structured_units(child, current_path, target_chars)
            )
            continue
        unit = _render_section_unit(child, child_path)
        if unit:
            if section.title:
                pending_leaves.append(unit)
            else:
                collected.append(unit)
    collected.extend(_group_sibling_units(pending_leaves, target_chars))
    return collected


def _contextual_title(document: LCDocument, path: tuple[str, ...]) -> str:
    """根据文档元数据和章节标题生成展示及检索标题。"""
    source = str(document.metadata.get("source", "")).strip()
    parts = [os.path.splitext(os.path.basename(source))[0]] if source else []
    for heading in path:
        if heading and heading not in parts:
            parts.append(heading)
    return " > ".join(parts)[:300]


def split_structured_markdown_documents(
    docs: list[LCDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    default_profile: str = "native",
) -> list[LCDocument]:
    """在保留章节和表格边界的前提下切分 Markdown 文档。"""
    target_chars = max(STRUCTURED_CHUNK_MIN_CHARS, min(1000, int(chunk_size)))
    maximum_chars = max(target_chars + 200, STRUCTURED_CHUNK_MAX_CHARS)
    splits: list[LCDocument] = []
    for document in docs:
        profile = str(
            document.metadata.get("structured_profile", default_profile)
        ).strip().lower()
        is_mineru = profile == "mineru"
        normalized = normalize_mineru_markdown(document.page_content)
        if not normalized:
            continue
        root = _parse_markdown_sections(
            normalized, infer_numbered_hierarchy=is_mineru
        )
        if is_mineru:
            _remove_bibliography_sections(root)
            _remove_leading_noncontent_sections(root)
        for path, content in _collect_structured_units(root, (), target_chars):
            for part in split_structured_content(
                content,
                maximum_chars,
                table_fallback_title=path[-1] if path else "",
            ):
                part = _clean_chunk_text(part)
                if not part:
                    continue
                metadata = document.metadata.copy()
                metadata.pop("page", None)
                metadata.pop("structured_profile", None)
                metadata["chunk_route"] = CHUNK_ROUTE_STRUCTURED
                contextual_title = _contextual_title(document, path)
                if contextual_title:
                    metadata["contextual_title"] = contextual_title
                splits.append(LCDocument(page_content=part, metadata=metadata))
    return assign_chunk_indexes(splits)
