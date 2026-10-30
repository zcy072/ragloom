"""视觉路线与固定父子切片路线。"""

import os
import uuid

from langchain_core.documents import Document as LCDocument

from services.core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SEPARATORS,
    DEFAULT_CHUNK_SIZE,
)

from .common import (
    _MARKDOWN_HEADING,
    CHUNK_ROUTE_FIXED,
    CHUNK_ROUTE_VISUAL,
    _deduplicate_adjacent_splits,
    _merge_small_child_chunks,
    _normalize_document_for_splitting,
    _normalize_index_text,
    assign_chunk_indexes,
)


def split_mineru_visual_documents(docs: list[LCDocument]) -> list[LCDocument]:
    """将每个分析后的 MinerU 视觉内容保留为独立且完整的检索段落。"""
    splits: list[LCDocument] = []
    for document in docs:
        content = _normalize_index_text(document.page_content)
        if not content:
            continue
        metadata = document.metadata.copy()
        metadata["chunk_route"] = CHUNK_ROUTE_VISUAL

        source = str(metadata.get("source", "")).strip()
        source_title = os.path.splitext(os.path.basename(source))[0] if source else ""
        existing_title = str(metadata.get("contextual_title", "")).strip()
        title_parts = [source_title] if source_title else []
        for part in existing_title.split(" > "):
            part = part.strip()
            if part and part != source_title and part not in title_parts:
                title_parts.append(part)
        if title_parts:
            metadata["contextual_title"] = " > ".join(title_parts)[:300]
        splits.append(LCDocument(page_content=content, metadata=metadata))
    return assign_chunk_indexes(splits)


def _heading_path(text: str, offset: int | None) -> list[str]:
    """根据文本偏移量回溯该位置所在的 Markdown 标题路径。"""
    if offset is None:
        return []
    hierarchy: dict[int, str] = {}
    for match in _MARKDOWN_HEADING.finditer(text):
        if match.start() > offset:
            break
        level = len(match.group(1))
        heading = " ".join(match.group(2).strip().strip("#").split())
        if not heading:
            continue
        for existing_level in list(hierarchy):
            if existing_level >= level:
                hierarchy.pop(existing_level)
        hierarchy[level] = heading
    return [hierarchy[level] for level in sorted(hierarchy)]


def _contextual_title(document: LCDocument, source_text: str = "", offset: int | None = None) -> str:
    """组合文件名和章节路径，生成切片的上下文标题。"""
    metadata = document.metadata
    parts: list[str] = []
    source = str(metadata.get("source", "")).strip()
    if source:
        parts.append(os.path.splitext(os.path.basename(source))[0])

    headings = [
        str(metadata[key]).strip()
        for key in ("Header 1", "Header 2", "Header 3", "Header 4", "Header 5", "Header 6")
        if metadata.get(key)
    ]
    if not headings:
        headings = _heading_path(source_text, offset)
    for heading in headings:
        if heading and heading not in parts:
            parts.append(heading)
    return " > ".join(parts)[:300]


def split_documents(
    docs,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """构建所有知识库共用的固定父子切片路线。

    小子切片用于精确召回，大父段落仅在 SQLite parent_chunks 保存一次并在命中后恢复；这样回答
    上下文完整，又不必把完整父段落重复写入每个 Milvus 向量。
    """
    normalized_documents: list[LCDocument] = []
    for document in docs:
        if not document.page_content:
            continue
        normalized = _normalize_document_for_splitting(document)
        normalized.metadata.setdefault("chunk_route", CHUNK_ROUTE_FIXED)
        normalized_documents.append(normalized)

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    separators: list[str] = list(DEFAULT_CHUNK_SEPARATORS)
    child_chunk_size = max(100, int(chunk_size / 4))
    # 父子切分器共用边界并按比例重叠，保证子切片偏移相对未改动父段落仍然有效。
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        add_start_index=True,
        keep_separator="end",
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=int(chunk_overlap / 4),
        separators=separators,
        add_start_index=True,
        keep_separator="end",
    )
    splits: list[LCDocument] = []
    parent_contents: dict[str, str] = {}
    for document in normalized_documents:
        for parent in parent_splitter.split_documents([document]):
            parent_start = parent.metadata.get("start_index", 0)
            if not isinstance(parent_start, int):
                parent_start = 0
            parent_title = _contextual_title(
                parent,
                document.page_content,
                parent_start,
            )
            if parent_title:
                parent.metadata["contextual_title"] = parent_title

            children = child_splitter.split_documents([parent])
            for child in children:
                # LangChain 子偏移相对父段落；合并短片段前需换算回全文坐标。
                local_start = child.metadata.get("start_index")
                metadata = parent.metadata.copy()
                if isinstance(parent_start, int) and isinstance(local_start, int):
                    metadata["start_index"] = parent_start + local_start
                child.metadata = metadata
            children = _merge_small_child_chunks(
                children,
                parent_content=parent.page_content,
                parent_start=parent_start,
            )
            if not children:
                continue

            parent_id = str(uuid.uuid4())
            parent_contents[parent_id] = parent.page_content
            for child in children:
                child.metadata["parent_id"] = parent_id
                splits.append(child)

    splits = _deduplicate_adjacent_splits(splits)
    # 仅一个临时子切片携带 parent_content 给父块表写入；Milvus 记录只保留 parent_id。
    seen_parents: set[str] = set()
    for split in splits:
        parent_id = str(split.metadata["parent_id"])
        split.metadata.pop("parent_content", None)
        if parent_id not in seen_parents:
            split.metadata["parent_content"] = parent_contents[parent_id]
            seen_parents.add(parent_id)
    return assign_chunk_indexes(splits)
