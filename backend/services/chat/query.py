"""Retrieval candidate orchestration for chat requests."""

import asyncio
import logging
from typing import Any

from services.chat.diagnostics import RetrievalDiagnostics
from services.chat.ranking import select_rerank_candidates
from services.core.container import ApplicationContainer, get_application_container
from services.core.contracts import RetrievalFilters
from services.core.errors import InvalidRequestError, InvalidStateError
from services.retrieval import retriever as retrieval_service

logger = logging.getLogger(__name__)


def _collect_candidates(
    kb_name: str,
    question: str,
    has_image: bool,
    filters: RetrievalFilters | None,
    include_common: bool,
    container: ApplicationContainer,
) -> tuple[list, list]:
    """同步完成筛选校验、检索器构建以及语义和 BM25 召回，返回 (语义候选, 词法候选)。

    这里全是同步的 SQLite、Milvus 和检索库调用，必须整体在线程里执行，不能占用事件循环。
    请求级文件快照用 ContextVar 保存，设置和使用都留在同一个线程里。
    """
    try:
        filtered = container.documents.has_filters(filters)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_retrieval_filters") from exc

    # The cached graph stays valid across requests; its source filters are refreshed once per
    # request so dense, BM25, and lexical supplementation use the same durable state.
    with retrieval_service.request_source_snapshot(
        kb_name, container, include_common=include_common
    ):
        # Filtered retrievers are request-specific and must not enter the knowledge-base cache.
        cached = (
            retrieval_service.get_cached_retriever(kb_name, include_common=include_common)
            if not filtered
            else None
        )
        if cached is None:
            try:
                retriever = retrieval_service.build_retriever(
                    kb_name, filters=filters, include_common=include_common, container=container
                )
            except ValueError as exc:
                raise InvalidRequestError(
                    str(exc), code="invalid_retrieval_request"
                ) from exc
            if retriever:
                if not filtered:
                    retrieval_service.cache_retriever(
                        kb_name, retriever, include_common=include_common
                    )
                cached = retriever
            elif not has_image:
                detail = "筛选范围内没有可用文档，请调整检索筛选条件。" if filtered else "当前知识库未上传文档，请先上传文件后再提问。"
                raise InvalidStateError(
                    detail,
                    code=(
                        "empty_filtered_retrieval_scope"
                        if filtered
                        else "knowledge_base_has_no_documents"
                    ),
                )
        semantic_docs = cached.invoke(question) if cached else []
        lexical_docs = retrieval_service.retrieve_lexical_candidates(
            kb_name, question, filters, include_common=include_common, container=container
        )
    return semantic_docs, lexical_docs


async def retrieve_rerank_candidates(
    kb_name: str,
    question: str,
    has_image: bool,
    filters: RetrievalFilters | None = None,
    *,
    include_common: bool = False,
    container: ApplicationContainer | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
):
    """执行语义检索，并额外补充一小批 BM25 精确关键词候选。

    ``include_common`` 表示提问者能读取通用知识库，此时补充检索通用知识库。
    """
    container = container or get_application_container()
    # 检索库和 SQLite 都是同步的，整段放进线程，避免阻塞 FastAPI 事件循环。
    semantic_docs, lexical_docs = await asyncio.to_thread(
        _collect_candidates, kb_name, question, has_image, filters, include_common, container
    )
    # Alternate exact-term and semantic candidates so the best of each survives the 30-candidate
    # cap; concatenating them let a large lexical pool push semantic hits out of the reranker.
    docs = _interleave_candidates(lexical_docs, semantic_docs)
    docs = await asyncio.to_thread(retrieval_service.deduplicate_chunk_routes, docs)
    if diagnostics is not None:
        diagnostics.counts["deduplicated_candidates"] = len(docs)
    return select_rerank_candidates(docs)


def _interleave_candidates(primary: list, variant: list) -> list:
    """交替排列两路候选，让两路结果共同竞争有限的重排名额，先 primary 后 variant。"""
    merged: list = []
    for index in range(max(len(primary), len(variant))):
        if index < len(primary):
            merged.append(primary[index])
        if index < len(variant):
            merged.append(variant[index])
    return merged


async def _retrieve_variant_candidates(
    kb_name: str,
    variant_query: str,
    has_image: bool,
    filters: RetrievalFilters | None,
    container_kwargs: dict,
) -> list:
    """补充查询只是增强，任何失败都按没有补充候选处理，不能影响主查询。"""
    try:
        return await retrieve_rerank_candidates(
            kb_name, variant_query, has_image, filters, **container_kwargs
        )
    except Exception as exc:
        logger.debug("Query variant retrieval skipped: %s", exc)
        return []


async def _merge_variant_candidates(
    primary: list, variant: list, diagnostics: RetrievalDiagnostics | None
) -> list:
    """合并主查询与补充查询候选，复用跨路线去重和重排候选限额。"""
    variant_ids = {id(document) for document in variant}
    merged = await asyncio.to_thread(
        retrieval_service.deduplicate_chunk_routes,
        _interleave_candidates(primary, variant),
    )
    selected = select_rerank_candidates(merged)
    if diagnostics is not None:
        diagnostics.counts["variant_candidates"] = sum(
            id(document) in variant_ids for document in selected
        )
    return selected


async def retrieve_candidates_with_query_fallback(
    kb_name: str,
    question: str,
    search_query: str,
    has_image: bool,
    filters: RetrievalFilters | None = None,
    *,
    variant_query: str = "",
    include_common: bool = False,
    container: ApplicationContainer | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
) -> tuple[list, str]:
    """优先使用改写问题检索；只有完全无命中时才回退用户原问题。

    ``variant_query`` 是可选的书面化补充查询：它与主查询并发召回，候选交替合并后
    一起进入重排。重排和返回的实际查询仍然是主查询，补充查询不会取代它。
    """
    # A rewrite is preferred when it succeeds, but it can occasionally over-specify the
    # user's intent.  Retry the original wording only when the rewrite found no evidence.
    container_kwargs: dict[str, Any] = {"include_common": include_common}
    if container is not None:
        container_kwargs["container"] = container
    primary_kwargs: dict[str, Any] = dict(container_kwargs)
    if diagnostics is not None:
        primary_kwargs["diagnostics"] = diagnostics
    if variant_query and variant_query != search_query:
        documents, variant_documents = await asyncio.gather(
            retrieve_rerank_candidates(
                kb_name, search_query, has_image, filters, **primary_kwargs
            ),
            _retrieve_variant_candidates(
                kb_name, variant_query, has_image, filters, container_kwargs
            ),
        )
        if variant_documents:
            documents = await _merge_variant_candidates(
                documents, variant_documents, diagnostics
            )
    else:
        documents = await retrieve_rerank_candidates(
            kb_name, search_query, has_image, filters, **primary_kwargs
        )
    if documents or search_query == question:
        return documents, search_query

    if diagnostics is not None:
        diagnostics.fallback_attempted = True
    try:
        fallback_documents = await retrieve_rerank_candidates(
            kb_name, question, has_image, filters, **primary_kwargs
        )
    except Exception as exc:
        logger.debug("Original-query retrieval fallback failed: %s", exc)
        return documents, search_query
    if fallback_documents:
        logger.debug("Original-query retrieval fallback recovered documents")
        return fallback_documents, question
    return documents, search_query
