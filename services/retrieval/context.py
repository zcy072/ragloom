"""父段落恢复、相邻片段扩展，以及聊天提示上下文格式化。"""

import collections
import functools
import hashlib
import logging
import os
import sqlite3
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

from services.core.config import (
    RETRIEVAL_CONTEXT_MAX_CHARS,
    RETRIEVAL_CONTEXT_MAX_SPANS,
    RETRIEVAL_CONTEXT_NEIGHBOR_CHUNKS,
)
from services.core.container import get_application_container
from services.indexing.chunking import (
    CHUNK_ROUTE_STRUCTURED,
    CHUNK_ROUTE_VISUAL,
    STRUCTURED_CHUNK_MIN_CHARS,
    chunk_route,
)
from services.retrieval import parent_store
from services.retrieval.chunk_index import get_chunk_index, run_searches

logger = logging.getLogger(__name__)
PARENT_CONTEXT_RESOLVED = "_parent_context_resolved"


@dataclass(frozen=True)
class ContextSpan:
    """用于支撑一次聊天回答的、按来源聚合并带排序的上下文段。"""

    source: str
    content: str
    pages: tuple[str, ...]
    page_unit: str
    rank: int
    chunk_start: int | None = None
    chunk_end: int | None = None
    contextual_titles: tuple[str, ...] = ()
    kb_name: str = ""
    chunk_routes: tuple[str, ...] = ()
    # 网页搜索结果的地址；知识库片段为空。
    url: str = ""


def _chunk_index(document: LCDocument) -> int | None:
    """读取文档的稳定切片序号；缺失或不合法时返回 None。"""
    value = document.metadata.get("chunk_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def source_name(document: LCDocument) -> str:
    """读取文档来源文件名；缺失时提供展示用默认值。"""
    return str(document.metadata.get("source", "未知文档"))


def document_kb_name(document: LCDocument, fallback_kb_name: str) -> str:
    """读取文档所属知识库；缺失时使用调用方提供的默认知识库。"""
    value = str(document.metadata.get("kb_name") or fallback_kb_name).strip()
    return value or fallback_kb_name


def _kb_id(kb_name: str) -> str | None:
    """把知识库名称解析为稳定的 kb_id；知识库不存在时返回 None。"""
    return get_application_container().knowledge_bases.kb_id_for(kb_name)


def _load_parent_chunks(
    kb_name: str,
    parent_ids: set[str],
) -> dict[str, dict[str, str]]:
    """从父段落表批量加载指定 parent_id 对应的完整段落。"""
    if not parent_ids:
        return {}
    kb_id = _kb_id(kb_name)
    if not kb_id:
        return {}
    try:
        rows = parent_store.load(kb_id, parent_ids)
    except sqlite3.Error as exc:
        logger.warning("Parent passage lookup failed: %s", exc)
        return {}
    return {
        parent_id: {
            "source": source,
            "page": page,
            "content": content,
            "contextual_title": contextual_title,
        }
        for parent_id, (_, source, page, content, contextual_title) in rows.items()
    }


def restore_parent_documents(
    kb_name: str,
    documents: list[LCDocument],
) -> list[LCDocument]:
    """每个命中的父段落只恢复一次，无需把全文重复存入每个子切片向量。"""
    parent_ids_by_kb: dict[str, set[str]] = collections.defaultdict(set)
    for document in documents:
        parent_id = document.metadata.get("parent_id")
        if parent_id:
            parent_ids_by_kb[document_kb_name(document, kb_name)].add(str(parent_id))

    parent_chunks_by_kb = {
        parent_kb_name: _load_parent_chunks(parent_kb_name, parent_ids)
        for parent_kb_name, parent_ids in parent_ids_by_kb.items()
    }
    restored: list[LCDocument] = []
    seen_parents: set[tuple[str, str]] = set()
    for document in documents:
        parent_id = document.metadata.get("parent_id")
        if not parent_id:
            restored.append(document)
            continue

        parent_kb_name = document_kb_name(document, kb_name)
        parent_key = (parent_kb_name, str(parent_id))
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)

        parent = parent_chunks_by_kb.get(parent_kb_name, {}).get(str(parent_id))
        if parent:
            document.page_content = parent["content"]
            if parent["page"]:
                document.metadata["page"] = parent["page"]
            if parent["contextual_title"]:
                document.metadata["contextual_title"] = parent["contextual_title"]
        document.metadata["kb_name"] = parent_kb_name
        document.metadata[PARENT_CONTEXT_RESOLVED] = True
        restored.append(document)
    return restored



def _document_key(document: LCDocument) -> tuple[str, str, str, int | None, str]:
    """为来自任意索引后端的检索文档生成稳定去重键。"""
    return (
        document_kb_name(document, ""),
        source_name(document),
        chunk_route(document),
        _chunk_index(document),
        hashlib.sha256(document.page_content.encode("utf-8")).hexdigest(),
    )


NeighborKey = tuple[str, str, str]


def _neighbor_window(index: int, neighbor_chunks: int) -> tuple[int, int]:
    """返回一个命中需要的相邻子块闭区间。"""
    return max(0, index - neighbor_chunks), index + neighbor_chunks


def _needs_neighbors(document: LCDocument) -> bool:
    """判断命中是否要补相邻子块：父段落、视觉和足够长的结构化段落保持自身边界。"""
    if document.metadata.get("parent_id") or _chunk_index(document) is None:
        return False
    route = chunk_route(document)
    return not (
        route == CHUNK_ROUTE_VISUAL
        or (
            route == CHUNK_ROUTE_STRUCTURED
            and len(document.page_content.strip()) >= STRUCTURED_CHUNK_MIN_CHARS
        )
    )


def _fetch_context_neighbors(
    documents: list[LCDocument],
    fallback_kb_name: str,
    neighbor_chunks: int,
) -> dict[NeighborKey, list[LCDocument]]:
    """批量取所有命中的相邻子块，按 (知识库, 文件, 路线) 返回已取到的子块。

    同一文件同一路线的窗口先合并成不重叠的区间，每个区间查一次索引，所有区间并发执行；
    相距很远的命中不合并成一个大区间，避免取回中间无关的子块。
    """
    windows: dict[NeighborKey, list[tuple[int, int]]] = collections.defaultdict(list)
    for document in documents:
        if _needs_neighbors(document):
            key = (
                document_kb_name(document, fallback_kb_name),
                source_name(document),
                chunk_route(document),
            )
            windows[key].append(_neighbor_window(_chunk_index(document) or 0, neighbor_chunks))

    kb_ids = {kb_name: _kb_id(kb_name) for kb_name, _, _ in windows}
    queries: list[tuple[NeighborKey, str, int, int]] = []
    for key, ranges in windows.items():
        kb_id = kb_ids[key[0]]
        if not kb_id:
            continue
        merged: list[list[int]] = []
        for lower, upper in sorted(ranges):
            if merged and lower <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], upper)
            else:
                merged.append([lower, upper])
        queries.extend((key, kb_id, lower, upper) for lower, upper in merged)

    index = get_chunk_index()
    results = run_searches(
        [
            functools.partial(index.neighbors, kb_id, source, route, lower, upper)
            for (_, source, route), kb_id, lower, upper in queries
        ]
    )
    fetched: dict[NeighborKey, list[LCDocument]] = collections.defaultdict(list)
    for (key, _, _, _), neighbors in zip(queries, results):
        for document in neighbors:
            document.metadata["kb_name"] = key[0]
        fetched[key].extend(neighbors)
    return fetched


def _span_from_documents(
    source: str,
    documents: list[LCDocument],
    rank: int,
    *,
    kb_name: str = "",
    chunk_start: int | None = None,
    chunk_end: int | None = None,
) -> ContextSpan:
    """将同一来源的一组文档聚合成带页码、标题和排序信息的上下文段。"""
    unique_contents: list[str] = []
    seen_contents: set[str] = set()
    pages: list[str] = []
    contextual_titles: list[str] = []
    resolved_kb_name = kb_name
    for document in documents:
        if not resolved_kb_name:
            resolved_kb_name = document_kb_name(document, "")
        content = document.page_content.strip()
        if content and content not in seen_contents:
            unique_contents.append(content)
            seen_contents.add(content)
        raw_page = document.metadata.get("page", document.metadata.get("row"))
        if raw_page not in (None, ""):
            page = str(raw_page)
            if page not in pages:
                pages.append(page)
        contextual_title = str(document.metadata.get("contextual_title", "")).strip()
        if contextual_title and contextual_title not in contextual_titles:
            contextual_titles.append(contextual_title)
    page_unit = "页"
    return ContextSpan(
        source=source,
        content="\n\n".join(unique_contents),
        pages=tuple(pages),
        page_unit=page_unit,
        rank=rank,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        contextual_titles=tuple(contextual_titles),
        kb_name=resolved_kb_name,
        chunk_routes=tuple(dict.fromkeys(chunk_route(document) for document in documents)),
    )


def _classify_context_hit(
    rank: int,
    document: LCDocument,
    fallback_kb_name: str,
    neighbor_chunks: int,
    fetched_neighbors: dict[NeighborKey, list[LCDocument]],
) -> tuple[str, tuple[str, ...], dict]:
    """将单个排序命中分类为独立证据段或带位置区间。"""
    source = source_name(document)
    kb_name = document_kb_name(document, fallback_kb_name)
    route = chunk_route(document)
    parent_id = document.metadata.get("parent_id")
    index = _chunk_index(document)

    if parent_id:
        # Restored parent passages are already complete; neighbor expansion would duplicate them.
        parent_document = LCDocument(
            page_content=document.page_content,
            metadata={
                "source": source,
                "page": document.metadata.get("page", ""),
                "kb_name": kb_name,
                **(
                    {"contextual_title": document.metadata["contextual_title"]}
                    if document.metadata.get("contextual_title")
                    else {}
                ),
            },
        )
        return (
            "standalone",
            (kb_name, source, f"parent:{parent_id}"),
            {
                "rank": rank,
                "kb_name": kb_name,
                "source": source,
                "documents": [parent_document],
            },
        )

    if index is not None:
        self_contained = not _needs_neighbors(document)
        if self_contained:
            neighbors = [document]
        else:
            lower, upper = _neighbor_window(index, neighbor_chunks)
            neighbors = [
                item
                for item in fetched_neighbors.get((kb_name, source, route), [])
                if lower <= (_chunk_index(item) or 0) <= upper
            ] or [document]
        indexes = [
            value
            for value in (_chunk_index(item) for item in neighbors)
            if value is not None
        ]
        return (
            "positioned",
            (kb_name, source, route),
            {
                "rank": rank,
                "start": min(indexes) if indexes else index,
                "end": max(indexes) if indexes else index,
                "documents": neighbors,
                "mergeable": not self_contained,
            },
        )

    return (
        "standalone",
        (kb_name, source, route, _document_key(document)[-1]),
        {
            "rank": rank,
            "kb_name": kb_name,
            "source": source,
            "documents": [document],
        },
    )


def _merge_context_intervals(
    positioned: dict[tuple[str, str, str], list[dict]],
    standalone: list[dict],
) -> list[dict]:
    """合并重叠的位置窗口，并与独立证据段组合。"""
    intervals: list[dict] = []
    for (kb_name, source, _chunk_route_name), source_intervals in positioned.items():
        source_intervals.sort(key=lambda item: (item["start"], item["end"], item["rank"]))
        merged: list[dict] = []
        for interval in source_intervals:
            if (
                merged
                and interval.get("mergeable", True)
                and merged[-1].get("mergeable", True)
                and interval["start"] <= merged[-1]["end"] + 1
            ):
                current = merged[-1]
                current["rank"] = min(current["rank"], interval["rank"])
                current["end"] = max(current["end"], interval["end"])
                by_key = {_document_key(item): item for item in current["documents"]}
                by_key.update({_document_key(item): item for item in interval["documents"]})
                current["documents"] = sorted(
                    by_key.values(),
                    key=lambda item: (_chunk_index(item) or 0, item.page_content),
                )
            else:
                merged.append(interval)
        intervals.extend(
            {"kb_name": kb_name, "source": source, **interval}
            for interval in merged
        )
    intervals.extend(standalone)
    intervals.sort(key=lambda item: item["rank"])
    return intervals


CONTEXT_TRUNCATION_MARKER = "\n[片段因上下文长度限制而截断]"


def _limit_context_spans(
    intervals: list[dict],
    fallback_kb_name: str,
    max_spans: int,
    max_chars: int,
) -> list[ContextSpan]:
    """将排序区间转换为上下文段，并应用全局字符预算。"""
    spans: list[ContextSpan] = []
    used_chars = 0
    for interval in intervals:
        if len(spans) >= max_spans or used_chars >= max_chars:
            break
        span = _span_from_documents(
            interval["source"],
            interval["documents"],
            interval["rank"],
            kb_name=interval.get("kb_name", fallback_kb_name),
            chunk_start=interval.get("start"),
            chunk_end=interval.get("end"),
        )
        # 空段不占名额，否则后面有内容的区间会被段数上限截掉。
        if not span.content:
            continue
        remaining_chars = max_chars - used_chars
        if len(span.content) > remaining_chars:
            truncation_marker = CONTEXT_TRUNCATION_MARKER
            if remaining_chars <= len(truncation_marker):
                truncated_content = truncation_marker[:remaining_chars]
            else:
                truncated_content = (
                    span.content[: remaining_chars - len(truncation_marker)].rstrip()
                    + truncation_marker
                )
            span = ContextSpan(
                source=span.source,
                content=truncated_content,
                pages=span.pages,
                page_unit=span.page_unit,
                rank=span.rank,
                chunk_start=span.chunk_start,
                chunk_end=span.chunk_end,
                contextual_titles=span.contextual_titles,
                kb_name=span.kb_name,
                chunk_routes=span.chunk_routes,
            )
        spans.append(span)
        used_chars += len(span.content)
    return spans


def build_context_spans(
    kb_name: str,
    documents: list[LCDocument],
    *,
    neighbor_chunks: int = RETRIEVAL_CONTEXT_NEIGHBOR_CHUNKS,
    max_spans: int = RETRIEVAL_CONTEXT_MAX_SPANS,
    max_chars: int = RETRIEVAL_CONTEXT_MAX_CHARS,
) -> list[ContextSpan]:
    """把排序命中扩展为有长度上限、按来源局部聚合的聊天提示段落。

    当前索引（Milvus）为每个子块保存文件内位置。完整结构化段落和图片描述保持
    自身边界，不拉入无关的相邻章节或图片。
    """
    if any(
        document.metadata.get("parent_id")
        and not document.metadata.get(PARENT_CONTEXT_RESOLVED)
        for document in documents
    ):
        # 固定路线子切片先恢复为完整父段落，再考虑相邻扩展。
        documents = restore_parent_documents(kb_name, documents)

    # 有位置的命中可扩展并合并区间；父段落、视觉和无位置证据保持独立，防止拼接无关内容。
    positioned: dict[tuple[str, str, str], list[dict]] = collections.defaultdict(list)
    standalone: list[dict] = []
    seen_standalone: set[tuple[str, ...]] = set()

    fetched_neighbors = _fetch_context_neighbors(documents, kb_name, neighbor_chunks)
    for rank, document in enumerate(documents):
        kind, key, interval = _classify_context_hit(
            rank, document, kb_name, neighbor_chunks, fetched_neighbors
        )
        if kind == "positioned":
            positioned[(key[0], key[1], key[2])].append(interval)
        elif key not in seen_standalone:
            standalone.append(interval)
            seen_standalone.add(key)

    # Merge source-local windows before applying the global prompt budget.
    intervals = _merge_context_intervals(positioned, standalone)
    return _limit_context_spans(intervals, kb_name, max_spans, max_chars)


def context_span_page_label(span: ContextSpan) -> str:
    """格式化来源页码或表格行号，供提示词和界面显示。"""
    if not span.pages:
        return ""
    display_pages = []
    for raw_page in span.pages[:4]:
        try:
            display_pages.append(str(int(raw_page) + 1))
        except (TypeError, ValueError):
            display_pages.append(str(raw_page))
    suffix = " 等" if len(span.pages) > 4 else ""
    return f"第 {', '.join(display_pages)}{suffix} {span.page_unit}"


def format_context_spans(spans: list[ContextSpan]) -> str:
    """将限长上下文段渲染为带稳定来源编号的聊天模型输入。"""
    blocks = []
    for index, span in enumerate(spans, start=1):
        if span.url:
            details = "，".join([span.source, *span.contextual_titles, span.url])
            blocks.append(f"【来源 {index}：网页，{details}】\n{span.content}")
            continue
        source = os.path.basename(span.source) or span.source
        page = context_span_page_label(span)
        location = f"，{page}" if page else ""
        titles = " / ".join(span.contextual_titles[:3])
        context_title = f"，上下文：{titles}" if titles else ""
        blocks.append(f"【来源 {index}：{source}{location}{context_title}】\n{span.content}")
    return "\n\n".join(blocks)
