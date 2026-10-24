"""HTML table normalization and boundary-preserving structured text splitting.

内部实现分两个逻辑层：
  1. HTML 规范化（normalize_mineru_markdown 及其辅助函数）
  2. 表格切分（split_structured_content 及其辅助函数）
"""

import html
import re
from typing import Any

from services.documents.heading_patterns import APPENDIX_HEADING

from .common import (
    _HTML_TABLE,
    _HTML_TABLE_CELL,
    _HTML_TABLE_ROW,
    _HTML_TAG,
    _PAGE_CONTINUATION_MARKER,
    _TABLE_CAPTION_LINE,
    _TABLE_END,
    _TABLE_HEADER_END,
    _TABLE_HEADER_PREFIX,
    _TABLE_NOTE_BLOCK,
    _TABLE_START,
    _TABLE_TITLE_PREFIX,
    _normalize_index_text,
    _ParsedHTMLTable,
)

# ---------------------------------------------------------------------------
# 1. HTML 表格规范化
# ---------------------------------------------------------------------------


def _positive_span(value: str | None) -> int:
    """解析 HTML 行列跨度，缺失或异常时回退为 1。"""
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def _html_table_cell_text(cell) -> str:
    """提取单元格文本，避免将编码后的比较符误认为 HTML 标记。"""
    from lxml import html as lxml_html

    markup = lxml_html.tostring(cell, encoding="unicode", method="html")
    markup = re.sub(r"<(?:br|/p|/div|/li)[^>]*>", "\n", markup, flags=re.IGNORECASE)
    text = html.unescape(_HTML_TAG.sub(" ", markup))
    return " ".join(text.split())


def _fallback_html_table(table_html: str) -> _ParsedHTMLTable:
    """当 lxml 拒绝格式异常的 MinerU 片段时，仍保留可用的表格。"""
    rows: list[list[str]] = []
    for raw_row in _HTML_TABLE_ROW.findall(table_html):
        cells = []
        for raw_cell in _HTML_TABLE_CELL.findall(raw_row):
            text = html.unescape(_HTML_TAG.sub(" ", raw_cell))
            cells.append(" ".join(text.split()))
        if cells:
            rows.append(cells)
    width = max((len(row) for row in rows), default=0)
    return _ParsedHTMLTable(
        rows=[row + [""] * (width - len(row)) for row in rows],
        header_rows=1 if rows else 0,
        notes={},
    )


def _parse_html_table(table_html: str) -> _ParsedHTMLTable:
    """展开 rowspan/colspan 为矩形网格，并识别表头行。"""
    try:
        from lxml import html as lxml_html

        root = lxml_html.fromstring(table_html)
        table = root if str(root.tag).lower() == "table" else root.xpath(".//table")[0]
        row_elements = table.xpath(".//tr[count(ancestor::table) = 1]")
    except (ImportError, IndexError, TypeError, ValueError):
        return _fallback_html_table(table_html)

    occupied: dict[tuple[int, int], str] = {}
    row_details: list[dict[str, Any]] = []
    for row_index, row in enumerate(row_elements):
        cells = row.xpath("./th | ./td")
        column_index = 0
        maximum_rowspan = 1
        has_colspan = False
        cell_details: list[tuple[str, int]] = []
        for cell in cells:
            while (row_index, column_index) in occupied:
                column_index += 1
            text = _html_table_cell_text(cell)
            rowspan = _positive_span(cell.get("rowspan"))
            colspan = _positive_span(cell.get("colspan"))
            maximum_rowspan = max(maximum_rowspan, rowspan)
            has_colspan = has_colspan or colspan > 1
            cell_details.append((text, colspan))
            for row_offset in range(rowspan):
                for column_offset in range(colspan):
                    occupied.setdefault(
                        (row_index + row_offset, column_index + column_offset), text
                    )
            column_index += colspan

        ancestors = {str(node.tag).lower() for node in row.iterancestors()}
        row_details.append(
            {
                "has_th": any(str(cell.tag).lower() == "th" for cell in cells),
                "is_thead": "thead" in ancestors,
                "has_colspan": has_colspan,
                "maximum_rowspan": maximum_rowspan,
                "cells": cell_details,
            }
        )

    if not row_elements:
        return _fallback_html_table(table_html)
    width = max((column + 1 for row, column in occupied if row < len(row_elements)), default=0)
    if not width:
        return _fallback_html_table(table_html)
    rows = [
        [occupied.get((row_index, column_index), "") for column_index in range(width)]
        for row_index in range(len(row_elements))
    ]

    header_rows = 1
    explicit_header_rows = 0
    for details in row_details:
        if details["is_thead"] or details["has_th"]:
            explicit_header_rows += 1
        else:
            break
    header_rows = max(header_rows, explicit_header_rows)
    first = row_details[0]
    if first["has_colspan"] or first["maximum_rowspan"] > 1:
        inferred_depth = max(2, int(first["maximum_rowspan"]))
        header_rows = max(header_rows, inferred_depth)
        while header_rows < len(rows) and row_details[header_rows - 1]["has_colspan"]:
            header_rows += 1
    if len(rows) > 1:
        header_rows = min(header_rows, len(rows) - 1)
    else:
        header_rows = 1

    notes: dict[int, str] = {}
    for row_index, details in enumerate(row_details[header_rows:], start=header_rows):
        cells = details["cells"]
        if len(cells) == 1 and cells[0][0] and cells[0][1] >= width:
            notes[row_index] = cells[0][0]
    return _ParsedHTMLTable(rows=rows, header_rows=header_rows, notes=notes)


def _table_column_headers(table: _ParsedHTMLTable) -> list[str]:
    """合并多级表头，生成每列完整的展示和检索名称。"""
    headers: list[str] = []
    width = len(table.rows[0]) if table.rows else 0
    for column_index in range(width):
        path: list[str] = []
        for row_index in range(table.header_rows):
            value = table.rows[row_index][column_index].strip()
            if value and (not path or path[-1] != value):
                path.append(value)
        if not path:
            path.append("项目" if column_index == 0 else f"第{column_index + 1}列")
        headers.append(" / ".join(path))
    return headers


def _serialized_table_text(table: _ParsedHTMLTable) -> str:
    """将已解析表格转为带表头边界哨兵的可切分文本。"""
    if not table.rows:
        return ""
    header = f"{_TABLE_HEADER_PREFIX}" + " | ".join(_table_column_headers(table))
    body: list[str] = []
    for row_index, row in enumerate(table.rows[table.header_rows:], start=table.header_rows):
        if row_index in table.notes:
            body.append(table.notes[row_index])
            continue
        if any(cell for cell in row):
            body.append(" | ".join(row).rstrip())
    lines = [_TABLE_START, header, _TABLE_HEADER_END, *body, _TABLE_END]
    return "\n\n" + "\n".join(lines) + "\n\n"


def _table_to_text(match: re.Match) -> str:
    """将一个 HTML 表格转为带表头边界哨兵的可切分文本。"""
    return _serialized_table_text(_parse_html_table(match.group(0)))


def _split_markdown_pipe_row(line: str) -> list[str] | None:
    """解析一行 Markdown pipe 表格，保留转义竖线和行内代码中的竖线。"""
    stripped = line.strip()
    if "|" not in stripped or stripped.startswith(("```", "~~~")):
        return None
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    code_ticks = 0
    index = 0
    while index < len(stripped):
        character = stripped[index]
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if character == "\\":
            if index + 1 < len(stripped) and stripped[index + 1] in {"|", "\\"}:
                escaped = True
            else:
                current.append(character)
            index += 1
            continue
        if character == "`":
            run_end = index
            while run_end < len(stripped) and stripped[run_end] == "`":
                run_end += 1
            ticks = run_end - index
            code_ticks = 0 if code_ticks == ticks else ticks
            current.extend(stripped[index:run_end])
            index = run_end
            continue
        if character == "|" and code_ticks == 0:
            cells.append(" ".join("".join(current).split()))
            current = []
        else:
            current.append(character)
        index += 1
    if escaped:
        current.append("\\")
    cells.append(" ".join("".join(current).split()))
    return cells if len(cells) >= 2 else None


def _is_markdown_pipe_separator(cells: list[str] | None) -> bool:
    """判断单元格序列是否为 Markdown 表头分隔行。"""
    return bool(cells) and all(re.fullmatch(r":?-+:?", cell) for cell in cells)


def _normalize_markdown_pipe_tables(text: str) -> str:
    """将代码围栏之外的 Markdown pipe 表格转换为统一表格哨兵。"""
    lines = (text or "").splitlines()
    normalized: list[str] = []
    index = 0
    fence_marker = ""
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        fence = re.match(r"(`{3,}|~{3,})", stripped)
        if fence:
            marker = fence.group(1)[0]
            fence_marker = "" if fence_marker == marker else marker
            normalized.append(line)
            index += 1
            continue
        if fence_marker or line.startswith("    ") or index + 1 >= len(lines):
            normalized.append(line)
            index += 1
            continue

        header = _split_markdown_pipe_row(line)
        separator = _split_markdown_pipe_row(lines[index + 1])
        if header is None or not _is_markdown_pipe_separator(separator):
            normalized.append(line)
            index += 1
            continue

        rows = [header]
        index += 2
        while index < len(lines):
            row = _split_markdown_pipe_row(lines[index])
            if row is None:
                break
            rows.append(row)
            index += 1
        width = max(len(row) for row in rows)
        rectangular_rows = [row + [""] * (width - len(row)) for row in rows]
        normalized.append(
            _serialized_table_text(
                _ParsedHTMLTable(rows=rectangular_rows, header_rows=1, notes={})
            ).strip()
        )
    return "\n".join(normalized)


def normalize_mineru_markdown(text: str) -> str:
    """保留 Markdown 结构，并统一 HTML 与 pipe 表格的安全切分表示。"""
    normalized = _normalize_markdown_pipe_tables(text or "")
    normalized = _HTML_TABLE.sub(_table_to_text, normalized)
    return _normalize_index_text(normalized)


# ---------------------------------------------------------------------------
# 2. 表格切分
# ---------------------------------------------------------------------------


def _is_table_block(block: str) -> bool:
    """判断规范化结构块是否被表格哨兵完整包裹。"""
    return block.startswith(_TABLE_START) and block.endswith(_TABLE_END)


def _table_caption_from_block(block: str) -> str:
    """从相邻源文本块中恢复表格图注或附录标题。"""
    for raw_line in reversed((block or "").splitlines()):
        line = " ".join(raw_line.strip().strip("*_# ").split())
        if _TABLE_CAPTION_LINE.fullmatch(line) or APPENDIX_HEADING.match(line):
            return line[:400]
    return ""


def _is_table_continuation_block(block: str) -> bool:
    """识别应附着到前一张表格的跨页续表标记。"""
    lines = [line.strip() for line in (block or "").splitlines() if line.strip()]
    return len(lines) == 1 and bool(_PAGE_CONTINUATION_MARKER.fullmatch(lines[0]))


def _attach_table_title(block: str, title: str) -> str:
    """在序列化表格块中插入规范化的继承章节标题。"""
    title = " ".join((title or "").split())
    if not title:
        return block
    lines = block.splitlines()
    if not lines or lines[0].strip() != _TABLE_START:
        return block
    return "\n".join([lines[0], f"{_TABLE_TITLE_PREFIX}{title[:400]}", *lines[1:]])


def _is_redundant_table_context(block: str, caption: str, fallback_title: str) -> bool:
    """识别信息已在表格中重复出现的标题或图注片段。"""
    caption_is_plain = bool(caption) and len(caption) <= 160 and not re.search(
        r"[。！？]", caption
    )
    repeated = {fallback_title.strip()}
    if caption_is_plain:
        repeated.add(caption.strip())
    for raw_line in (block or "").splitlines():
        line = " ".join(raw_line.strip().strip("*_# ").split())
        if not line:
            continue
        if line in repeated or _PAGE_CONTINUATION_MARKER.fullmatch(line):
            continue
        return False
    return bool(block.strip())


def _is_table_note_block(block: str) -> bool:
    """识别属于前一张表格而非普通正文的注释块。"""
    first_line = next((line.strip() for line in (block or "").splitlines() if line.strip()), "")
    return bool(_TABLE_NOTE_BLOCK.match(first_line))


def _append_table_note(block: str, note: str) -> str:
    """将表注插入结束哨兵之前，不破坏表格结构。"""
    lines = block.splitlines()
    try:
        end_index = next(
            index for index in range(len(lines) - 1, -1, -1) if lines[index].strip() == _TABLE_END
        )
    except StopIteration:
        return block
    return "\n".join([*lines[:end_index], note.strip(), *lines[end_index:]])


def _split_table_block(block: str, maximum_chars: int) -> list[str]:
    """按完整行切分过大的表格，并在每一部分重复表头。"""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(block) <= maximum_chars or len(lines) <= 4:
        return [block]
    try:
        header_end = lines.index(_TABLE_HEADER_END)
    except ValueError:
        header_end = 1
    repeated_prefix = lines[: header_end + 1]
    rows = lines[header_end + 1 : -1]
    chunks: list[str] = []
    current = repeated_prefix.copy()
    for row in rows:
        candidate = "\n".join(current + [row, _TABLE_END])
        if len(candidate) > maximum_chars and len(current) > len(repeated_prefix):
            chunks.append("\n".join(current + [_TABLE_END]))
            current = repeated_prefix.copy()
        current.append(row)
    if len(current) > len(repeated_prefix):
        chunks.append("\n".join(current + [_TABLE_END]))
    return chunks or [block]


def _split_long_text_block(block: str, maximum_chars: int) -> list[str]:
    """使用感知句子的分隔符切分过长非表格正文，不引入重叠。"""
    if len(block) <= maximum_chars:
        return [block]
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=maximum_chars,
        chunk_overlap=0,
        separators=["\n", "。", "！", "？", "；", " ", ""],
    ).split_text(block)


def split_structured_content(
    content: str,
    maximum_chars: int,
    *,
    table_fallback_title: str = "",
) -> list[str]:
    """切分混合正文和表格的章节内容，同时保持表格自身完整。"""
    isolated = (content or "").replace(_TABLE_START, f"\n\n{_TABLE_START}").replace(
        _TABLE_END, f"{_TABLE_END}\n\n"
    )
    blocks = [block.strip() for block in re.split(r"\n\s*\n", isolated) if block.strip()]
    chunks: list[str] = []
    pending: list[str] = []
    pending_size = 0
    last_table_title = ""

    def flush() -> None:
        """在独立表格或大小边界之前输出已缓存文本。"""
        nonlocal pending, pending_size
        if pending:
            chunks.append("\n\n".join(pending))
        pending = []
        pending_size = 0

    block_index = 0
    while block_index < len(blocks):
        block = blocks[block_index]
        if _is_table_block(block):
            caption = (
                _table_caption_from_block(blocks[block_index - 1]) if block_index > 0 else ""
            )
            continuation = block_index > 0 and _is_table_continuation_block(
                blocks[block_index - 1]
            )
            table_title = caption or (last_table_title if continuation else "") or table_fallback_title
            block = _attach_table_title(block, table_title)
            if table_title:
                last_table_title = table_title
            if pending and _is_redundant_table_context(
                pending[-1], caption, table_fallback_title
            ):
                pending.pop()
                pending_size = sum(len(piece) for piece in pending) + max(0, len(pending) - 1) * 2
            if block_index + 1 < len(blocks) and _is_table_note_block(blocks[block_index + 1]):
                block = _append_table_note(block, blocks[block_index + 1])
                block_index += 1
            flush()
            chunks.extend(_split_table_block(block, maximum_chars))
            block_index += 1
            continue
        pieces = _split_long_text_block(block, maximum_chars)
        for piece in pieces:
            if pending and pending_size + 2 + len(piece) > maximum_chars:
                flush()
            pending.append(piece)
            pending_size += len(piece) + (2 if pending_size else 0)
        block_index += 1
    flush()
    return chunks
