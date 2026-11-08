"""Prompt source catalogues and answer-citation projection."""

import re

import jieba  # pyright: ignore[reportMissingImports]

from services.chat.persistence_contracts import ChatSource
from services.retrieval import context as context_builder

CITATION_PATTERN = re.compile(r"[\[【]\s*来源\s*(\d+)\s*[\]】]")
SOURCE_PREVIEW_MAX_CHARS = 180


def _query_phrases(question: str) -> list[str]:
    """根据结巴分词结果生成少量按长度优先的查询短语。"""
    query = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", question or "")[:48]
    if not query:
        return []
    tokens = [
        re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", token)
        for token in jieba.lcut(question or "")
    ]
    tokens = [token for token in tokens if token][:16]
    phrases = [query]
    for length in range(len(tokens), 0, -1):
        for start in range(len(tokens) - length + 1):
            phrase = "".join(tokens[start : start + length])
            if len(phrase) >= 3 and phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _source_preview(content: str, question: str, limit: int = SOURCE_PREVIEW_MAX_CHARS) -> str:
    """截取最接近问题关键词的原文片段，用于前端来源卡片预览。"""
    normalized = " ".join(str(content or "").split())
    if len(normalized) <= limit:
        return normalized

    fragment = ""
    position = -1
    for candidate in _query_phrases(question):
        found_at = normalized.find(candidate)
        if found_at >= 0:
            fragment = candidate
            position = found_at
            break

    if position < 0:
        return normalized[:limit].rstrip() + "..."

    before = min(48, max(0, (limit - len(fragment)) // 3))
    start = max(0, position - before)
    end = min(len(normalized), start + limit)
    if end == len(normalized):
        start = max(0, end - limit)
    prefix = "..." if start else ""
    suffix = "..." if end < len(normalized) else ""
    return prefix + normalized[start:end].strip() + suffix


def _web_source(index: int, span: context_builder.ContextSpan, question: str) -> ChatSource:
    """网页来源：标题作来源名，站点和日期作位置，带原始地址供前端打开。"""
    label = " · ".join(span.contextual_titles)
    content = _source_preview(span.content, question)
    return {
        "id": index,
        "citation_ids": [index],
        "source": span.source,
        "page": label,
        "context_title": "",
        "content": content,
        "url": span.url,
        "evidence": [{
            "id": index,
            "source": span.source,
            "kb_name": "",
            "page": label,
            "page_numbers": [],
            "context_title": "",
            "content": content,
            "url": span.url,
        }],
    }


def build_source_catalog(
    spans: list[context_builder.ContextSpan], question: str = ""
) -> list[ChatSource]:
    """将上下文片段编号并序列化为 Prompt 和前端共用的来源目录。"""
    sources: list[ChatSource] = []
    for index, span in enumerate(spans, start=1):
        if span.url:
            sources.append(_web_source(index, span, question))
            continue
        source = span.source.replace("\\", "/").split("/")[-1] or "未知文档"
        item: ChatSource = {
            "id": index,
            "citation_ids": [index],
            "source": source,
            "page": context_builder.context_span_page_label(span),
            "context_title": " / ".join(span.contextual_titles[:3]),
            "content": _source_preview(span.content, question),
        }
        if span.kb_name:
            item["kb_name"] = span.kb_name
        item["evidence"] = [{
            "id": index,
            "source": source,
            "kb_name": span.kb_name,
            "page": item["page"],
            "page_numbers": sorted({
                int(page) + 1 for page in span.pages
                if source.lower().endswith(".pdf") and str(page).isdigit()
            }),
            "context_title": item["context_title"],
            "content": item["content"],
        }]
        sources.append(item)
    return sources


def _cited_source_ids(answer: str, source_catalog: list[ChatSource]) -> list[int]:
    """按答案出现顺序提取合法引用编号，并去除重复编号。"""
    available_ids = {source["id"] for source in source_catalog}
    cited_ids: list[int] = []
    for match in CITATION_PATTERN.finditer(answer or ""):
        source_id = int(match.group(1))
        if source_id in available_ids and source_id not in cited_ids:
            cited_ids.append(source_id)
    return cited_ids


def select_cited_sources(
    answer: str, source_catalog: list[ChatSource]
) -> list[ChatSource]:
    """仅返回答案实际引用的来源，并按知识库和文件名合并展示。"""
    # The prompt catalogue can contain useful but uncited context. Exposing only IDs that the
    # answer actually referenced keeps the UI from implying unsupported citations.
    source_by_id = {source["id"]: source for source in source_catalog}
    selected = [source_by_id[source_id] for source_id in _cited_source_ids(answer, source_catalog)]
    # Group display cards by both knowledge base and filename; common and selected knowledge
    # bases may legally contain files with the same name.
    grouped: dict[tuple[str, str, str], list[ChatSource]] = {}
    for source in selected:
        source_name = str(source.get("source", "未知文档"))
        source_kb_name = str(source.get("kb_name", ""))
        # 不同网页可能同名，网页来源按地址分组。
        grouped.setdefault(
            (source_kb_name, source_name, source.get("url", "")), []
        ).append(source)

    display_sources: list[ChatSource] = []
    for (source_kb_name, source_name, url), entries in grouped.items():
        pages = list(dict.fromkeys(entry.get("page", "") for entry in entries if entry.get("page")))
        titles = list(
            dict.fromkeys(entry.get("context_title", "") for entry in entries if entry.get("context_title"))
        )
        first = entries[0]
        item: ChatSource = {
            "id": first["id"],
            "citation_ids": [entry["id"] for entry in entries],
            "source": source_name,
            "page": "、".join(pages),
            "context_title": " / ".join(titles[:2]),
            "content": first.get("content", ""),
        }
        if source_kb_name:
            item["kb_name"] = source_kb_name
        if url:
            item["url"] = url
        evidence = [entry for source in entries for entry in source.get("evidence", [])]
        if evidence:
            item["evidence"] = evidence
        display_sources.append(item)
    return display_sources
