"""共享切片元数据，以及固定路线的文本规范化辅助函数。"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

CHUNK_ROUTE_FIXED = "fixed"
CHUNK_ROUTE_STRUCTURED = "structured"
CHUNK_ROUTE_VISUAL = "visual"
FIXED_CHILD_MIN_CHARS = 60
STRUCTURED_CHUNK_MIN_CHARS = 450
STRUCTURED_CHUNK_MAX_CHARS = 1200


@dataclass
class _StructuredSection:
    """从 MinerU 扁平标题层级重建出的一个 Markdown 标题树节点。"""

    title: str
    level: int
    style: str
    body: list[str]
    children: list["_StructuredSection"]


@dataclass
class _ChildChunkSpan:
    """用于从原文精确重建合并后固定子切片的临时范围。"""

    document: LCDocument
    start: int | None
    end: int | None


@dataclass
class _ParsedHTMLTable:
    """由 HTML 行列合并信息还原得到的矩形表格。"""

    rows: list[list[str]]
    header_rows: int
    notes: dict[int, str]


def _split_source(split) -> str:
    """读取切片所属的源文件名。"""
    return str(split.metadata.get("source", ""))


def chunk_route(document: LCDocument) -> str:
    """读取切片路线；缺失或空值时回退为固定路线。"""
    route = str(document.metadata.get("chunk_route", CHUNK_ROUTE_FIXED)).strip()
    return route or CHUNK_ROUTE_FIXED


def assign_chunk_indexes(splits: list) -> list:
    """在每个源文件和切片路线内分配稳定的顺序编号。"""
    positions: dict[tuple[str, str], int] = {}
    for split in splits:
        source = _split_source(split)
        route = chunk_route(split)
        split.metadata["chunk_route"] = route
        position_key = (source, route)
        position = positions.get(position_key, 0)
        split.metadata["chunk_index"] = position
        positions[position_key] = position + 1
    return splits


def contextualized_chunk_text(document: LCDocument) -> str:
    """返回供检索使用的文本，保留原始正文供后续展示。"""
    title = str(document.metadata.get("contextual_title", "")).strip()
    content = document.page_content.strip()
    return f"{title}\n{content}" if title and content else title or content



_MARKDOWN_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_HTML_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_HTML_TABLE_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_TABLE_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_TABLE_START = "[[TABLE]]"
_TABLE_END = "[[/TABLE]]"
_TABLE_HEADER_END = "[[/TABLE_HEADER]]"
_TABLE_TITLE_PREFIX = "表名："
_TABLE_HEADER_PREFIX = "表头："
_TABLE_CAPTION_LINE = re.compile(
    r"^(?:附\s*)?(?:表|table)\s*"
    r"[A-Za-zＡ-Ｚａ-ｚ0-9０-９一二三四五六七八九十]+"
    r"(?:[.．\-—－][A-Za-zＡ-Ｚａ-ｚ0-9０-９一二三四五六七八九十]+)*\s*.+$",
    re.IGNORECASE,
)
_TABLE_NOTE_BLOCK = re.compile(r"^(?:注(?:释)?|备注)\s*[:：]", re.IGNORECASE)
_REFERENCE_MARKER = re.compile(r"\^\{\[[^\]\n]{1,80}\]\}")
_REFERENCE_TRAILING_PUNCTUATION = re.compile(
    r"(\^\{\[[^\]\n]{1,80}\]\})[ \t]*(?:\n[ \t]*)+([。！？；，、,.!?])"
)
_REFERENCE_LEADING_LINE = re.compile(
    r"[ \t]*\n[ \t]*(\^\{\[[^\]\n]{1,80}\]\})"
)
_NON_SEMANTIC_CHARS = re.compile(
    r"[\s#*_~`。；，、,.!?！？:：()（）\[\]{}\\^%\-]+"
)
_PAGE_CONTINUATION_MARKER = re.compile(
    r"^续(?:附录|表|图)\s*[0-9A-Za-z一二三四五六七八九十.-]*$",
    re.IGNORECASE,
)


def _normalize_index_text(value: str) -> str:
    """移除仅由解析器产生的断行，同时保留原文措辞和标记。"""
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    # MinerU 常把引用标记及其标点拆成两行，保留拆分会产生没有实际含义的纯引用向量。
    text = _REFERENCE_TRAILING_PUNCTUATION.sub(r"\1\2", text)
    text = _REFERENCE_LEADING_LINE.sub(r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _normalize_document_for_splitting(document: LCDocument) -> LCDocument:
    """返回适合切分和索引的副本，不修改解析器的原始文档对象。"""
    return LCDocument(
        page_content=_normalize_index_text(document.page_content),
        metadata=document.metadata.copy(),
    )


def _clean_chunk_text(value: str) -> str:
    """结构切分完成后，移除内部使用的表格哨兵标记。"""
    return _normalize_index_text(
        (value or "")
        .replace(_TABLE_START, "")
        .replace(_TABLE_END, "")
        .replace(_TABLE_HEADER_END, "")
    )


def _semantic_chunk_length(value: str) -> int:
    """计算去除标点、空白和引用标记后的有效语义长度。"""
    text = _REFERENCE_MARKER.sub("", value or "")
    return len(_NON_SEMANTIC_CHARS.sub("", text))


def _is_trailing_fragment(value: str) -> bool:
    """判断短子切片是否只包含标点或来源引用标记。"""
    return _semantic_chunk_length(value) == 0


def _suffix_prefix_overlap(left: str, right: str) -> int:
    """计算有意义的精确首尾重叠，单个相同字符不算重叠。"""
    maximum = min(len(left), len(right))
    for length in range(maximum, 7, -1):
        if left.endswith(right[:length]):
            return length
    return 0


def _join_chunk_text(left: str, right: str) -> str:
    """拼接两个切片，去除首尾重复并保持中文标点连接自然。"""
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    overlap = _suffix_prefix_overlap(left, right)
    if overlap:
        return left + right[overlap:]
    if right[0] in "。；，、,.!?！？" or right.startswith("^{["):
        return left.rstrip() + right.lstrip()
    return f"{left.rstrip()}\n\n{right.lstrip()}"


def _source_span_text(
    left: _ChildChunkSpan,
    right: _ChildChunkSpan,
    *,
    parent_content: str,
    parent_start: int,
) -> str | None:
    """从未改变的父段落原文中重建两个重叠子切片的并集。"""
    if (
        not parent_content
        or left.start is None
        or left.end is None
        or right.start is None
        or right.end is None
    ):
        return None
    local_start = min(left.start, right.start) - parent_start
    local_end = max(left.end, right.end) - parent_start
    if local_start < 0 or local_end > len(parent_content) or local_end <= local_start:
        return None
    return _clean_chunk_text(parent_content[local_start:local_end])


def _merge_child_spans(
    left: _ChildChunkSpan,
    right: _ChildChunkSpan,
    *,
    parent_content: str,
    parent_start: int,
) -> tuple[str, int | None, int | None]:
    """合并相邻小切片，并返回合并后的文本及其原文范围。"""
    content = _source_span_text(
        left,
        right,
        parent_content=parent_content,
        parent_start=parent_start,
    )
    if content is None:
        content = _join_chunk_text(left.document.page_content, right.document.page_content)
    starts = [value for value in (left.start, right.start) if value is not None]
    ends = [value for value in (left.end, right.end) if value is not None]
    return content, min(starts) if starts else None, max(ends) if ends else None


def _merge_small_child_chunks(
    children: list[LCDocument],
    *,
    minimum_chars: int = FIXED_CHILD_MIN_CHARS,
    parent_content: str = "",
    parent_start: int = 0,
) -> list[LCDocument]:
    """将信息量过低的子切片合并到同一父段落中的相邻切片。"""
    cleaned: list[_ChildChunkSpan] = []
    for child in children:
        raw_content = child.page_content
        raw_start = child.metadata.get("start_index")
        start = raw_start if isinstance(raw_start, int) else None
        end = start + len(raw_content) if start is not None else None
        child.page_content = _clean_chunk_text(child.page_content)
        if not child.page_content or _PAGE_CONTINUATION_MARKER.fullmatch(child.page_content):
            continue
        cleaned.append(_ChildChunkSpan(document=child, start=start, end=end))

    merged: list[_ChildChunkSpan] = []
    index = 0
    while index < len(cleaned):
        child = cleaned[index]
        if _semantic_chunk_length(child.document.page_content) >= minimum_chars:
            merged.append(child)
            index += 1
            continue

        if _is_trailing_fragment(child.document.page_content) and merged:
            content, start, end = _merge_child_spans(
                merged[-1],
                child,
                parent_content=parent_content,
                parent_start=parent_start,
            )
            merged[-1].document.page_content = content
            merged[-1].start = start
            merged[-1].end = end
        elif index + 1 < len(cleaned):
            following = cleaned[index + 1]
            content, start, end = _merge_child_spans(
                child,
                following,
                parent_content=parent_content,
                parent_start=parent_start,
            )
            following.document.page_content = content
            following.start = start
            following.end = end
        elif merged:
            content, start, end = _merge_child_spans(
                merged[-1],
                child,
                parent_content=parent_content,
                parent_start=parent_start,
            )
            merged[-1].document.page_content = content
            merged[-1].start = start
            merged[-1].end = end
        else:
            # 父段落可能本就只有一个短章节，不能仅为了满足理想长度而丢弃来源内容。
            merged.append(child)
        index += 1
    for child in merged:
        if child.start is not None:
            child.document.metadata["start_index"] = child.start
    return [child.document for child in merged]


def _adjacent_split_key(document: LCDocument) -> tuple[str, str, str, str]:
    """生成用于比较相邻切片是否重复的键。"""
    return (
        _split_source(document),
        str(document.metadata.get("page", document.metadata.get("row", ""))),
        chunk_route(document),
        re.sub(r"\s+", "", document.page_content),
    )


def _deduplicate_adjacent_splits(splits: list[LCDocument]) -> list[LCDocument]:
    """删除相邻的完全重复切片，但保留相距较远的合法重复内容。"""
    retained: list[LCDocument] = []
    for split in splits:
        if retained and _adjacent_split_key(retained[-1]) == _adjacent_split_key(split):
            continue
        retained.append(split)
    return retained
