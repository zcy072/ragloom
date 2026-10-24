"""SiliconFlow reranking for bounded retrieval candidates."""

import collections
import logging
import math
import re

import httpx

from services.chat.diagnostics import RetrievalDiagnostics
from services.core import llm_config as model_config
from services.core.config import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_SILICONFLOW_BASE_URL,
    RETRIEVAL_RERANK_MAX_CANDIDATES,
)
from services.core.container import ApplicationContainer, get_application_container
from services.indexing import chunking

logger = logging.getLogger(__name__)
RERANK_TOP_N = 10
RERANK_MAX_PER_SOURCE_ROUTE = 6
RERANK_MAX_PER_PARENT = 2
_rerank_client: httpx.AsyncClient | None = None


def start_rerank_client() -> None:
    """应用启动时创建重排序接口复用的 HTTP 连接池。"""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = httpx.AsyncClient(
            timeout=httpx.Timeout(15.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )


async def close_rerank_client() -> None:
    """应用关闭时释放重排序连接池，避免遗留网络连接。"""
    global _rerank_client
    client, _rerank_client = _rerank_client, None
    if client is not None:
        await client.aclose()


async def _get_rerank_client() -> httpx.AsyncClient:
    """返回进程级重排客户端，首次使用时才创建。"""
    start_rerank_client()
    return _rerank_client


def _parent_candidate_key(document) -> tuple[str, str, str] | None:
    """生成用于候选多样性限制的稳定知识库、来源和父块联合键。"""
    metadata = document.metadata or {}
    parent_id = str(metadata.get("parent_id", "")).strip()
    if not parent_id:
        return None
    return (
        str(metadata.get("kb_name", "")),
        str(metadata.get("source", "")),
        parent_id,
    )


def select_rerank_candidates(documents: list) -> list:
    """限制远程重排输入数量，同时保留检索路线和父块多样性。"""
    selected: list = []
    pending = list(documents)
    route_counts: collections.defaultdict = collections.defaultdict(int)
    parent_counts: collections.defaultdict = collections.defaultdict(int)
    for parent_limit in (1, RERANK_MAX_PER_PARENT):
        deferred: list = []
        for document in pending:
            if len(selected) >= RETRIEVAL_RERANK_MAX_CANDIDATES:
                return selected
            metadata = document.metadata or {}
            route_key = (
                str(metadata.get("kb_name", "")),
                str(metadata.get("source", "")),
                str(metadata.get("chunk_route", chunking.CHUNK_ROUTE_FIXED)),
            )
            parent_key = _parent_candidate_key(document)
            if (
                route_counts[route_key] >= RERANK_MAX_PER_SOURCE_ROUTE
                or parent_key is not None
                and parent_counts[parent_key] >= parent_limit
            ):
                deferred.append(document)
                continue
            selected.append(document)
            route_counts[route_key] += 1
            if parent_key is not None:
                parent_counts[parent_key] += 1
        pending = deferred
    selected.extend(
        pending[: RETRIEVAL_RERANK_MAX_CANDIDATES - len(selected)]
    )
    return selected


async def rerank(
    question: str,
    docs: list,
    *,
    container: ApplicationContainer | None = None,
    diagnostics: RetrievalDiagnostics | None = None,
) -> list:
    """对数量受限且来源多样的候选重排序；失败时保留原候选顺序并截到与成功时相同的数量。"""
    if not docs:
        return docs
    # Candidate bounding is repeated here because callers may also use this helper
    # directly; the remote reranker must never receive an unbounded document list.
    candidates = select_rerank_candidates(docs)
    if diagnostics is not None:
        diagnostics.counts["rerank_input"] = len(candidates)
        diagnostics.rerank_status = "degraded"
    try:
        client = await _get_rerank_client()
        dependencies = container or get_application_container()
        response = await client.post(
            f"{DEFAULT_SILICONFLOW_BASE_URL}/rerank",
            json={
                "model": DEFAULT_RERANK_MODEL,
                "query": question,
                "documents": [chunking.contextualized_chunk_text(doc) for doc in candidates],
                "top_n": RERANK_TOP_N,
            },
            headers={
                "Authorization": f"Bearer {model_config.get_siliconflow_api_key(dependencies.model_providers)}",
                "Content-Type": "application/json",
            },
        )
        if response.status_code != 200:
            logger.warning("Reranking unavailable: HTTP %s", response.status_code)
            return candidates[:RERANK_TOP_N]
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list) or not results:
            logger.warning("Reranking returned no usable results")
            return candidates[:RERANK_TOP_N]
        # Preserve exact identifiers such as model numbers that a semantic reranker may blur.
        exact_phrases = [
            value for value in re.findall(r"[A-Za-z0-9-]+", question)
            if len(value) >= 4 and re.search(r"[A-Za-z]", value) and re.search(r"[0-9]", value)
        ]
        ranked = []
        seen = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            index = result.get("index")
            if type(index) is not int or not 0 <= index < len(candidates) or index in seen:
                continue
            score = result.get("relevance_score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                continue
            seen.add(index)
            document = candidates[index].model_copy(deep=True)
            matches = [phrase for phrase in exact_phrases if re.search(
                r"(?<![A-Za-z0-9-])" + re.escape(phrase) + r"(?![A-Za-z0-9-])",
                document.page_content, re.IGNORECASE,
            )]
            document.metadata["relevance_score"] = score
            document.metadata["exact_identifiers"] = matches
            document.metadata["ranking_rule"] = "model_score_then_identifier_tie_break"
            ranked.append(document)
        ranked = sorted(ranked, key=lambda doc: (
            doc.metadata["relevance_score"], bool(doc.metadata["exact_identifiers"])
        ), reverse=True)[:RERANK_TOP_N]
        if diagnostics is not None and ranked:
            diagnostics.rerank_status = "success"
            diagnostics.counts["rerank_returned"] = min(len(ranked), RERANK_TOP_N)
            diagnostics.ranking = [
                {"position": index, "model_score": doc.metadata["relevance_score"],
                 "exact_identifiers": doc.metadata["exact_identifiers"],
                 "rule": doc.metadata["ranking_rule"]}
                for index, doc in enumerate(ranked, 1)
            ]
        if not ranked:
            logger.warning("Reranking results were all invalid")
            return candidates[:RERANK_TOP_N]
        return ranked
    except Exception as exc:
        logger.warning("Reranking failed: %s", exc)
        return candidates[:RERANK_TOP_N]
