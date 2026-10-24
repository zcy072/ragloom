"""检索器缓存、多来源融合和候选证据编排。"""

import collections
import hashlib
import logging
import re
import threading
from concurrent.futures import Future
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

from services.core import config as application_config
from services.core import sqlite_connections as application_db
from services.core.config import (
    RETRIEVAL_CHROMA_K,
    RETRIEVAL_LEXICAL_K,
    RETRIEVAL_ROUTE_WEIGHTS,
)
from services.core.container import ApplicationContainer, get_application_container
from services.core.contracts import RetrievalFilters
from services.core.settings import get_settings
from services.indexing.chunking import (
    CHUNK_ROUTE_FIXED,
    CHUNK_ROUTE_STRUCTURED,
    CHUNK_ROUTE_VISUAL,
    chunk_route,
)
from services.retrieval import chroma as chroma_index
from services.retrieval import context as context_builder
from services.retrieval import fts as fts_index
from services.retrieval import source_state
from services.retrieval.cache import RetrieverCache

logger = logging.getLogger(__name__)

_CACHE_MAX_SIZE = get_settings().retriever_cache_max_size
_QUERY_EMBEDDING_CACHE_SIZE = get_settings().query_embedding_cache_size

_retriever_cache = RetrieverCache(_CACHE_MAX_SIZE)

# 短文本需要完全相同，避免少量探针把不同的标题或摘要误判为重叠内容。
_OVERLAP_EXACT_MATCH_THRESHOLD = 160
# 长文本通过等距采样探针比较，平衡跨路线重复内容的漏判和误判。
_OVERLAP_PROBE_LENGTH = 24
_OVERLAP_PROBE_STRIDE = 48
_OVERLAP_MATCH_RATIO = 0.8


class CachedQueryEmbeddings:
    """在多条检索路线间共享查询向量，同时不在内存保留原始问题文本。

    一次问题会分发到多个切片路线和知识库；``_inflight`` 实现单飞机制，让并发
    路线等待同一次远程嵌入调用。缓存键是哈希，避免在内存保存潜在敏感问题。
    """

    def __init__(self, embeddings, max_size: int = _QUERY_EMBEDDING_CACHE_SIZE):
        """初始化 LRU 向量缓存、在途请求表和并发锁。"""
        self._embeddings = embeddings
        self._max_size = max(1, max_size)
        self._cache: collections.OrderedDict[bytes, tuple[float, ...]] = collections.OrderedDict()
        self._inflight: dict[bytes, Future] = {}
        self._lock = threading.Lock()

    def embed_query(self, query: str) -> list[float]:
        """返回查询向量；并发相同查询只会触发一次底层嵌入调用。"""
        key = hashlib.sha256(query.encode("utf-8")).digest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return list(cached)
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._inflight[key] = future

        if not owner:
            # 另一路已在计算同一问题的向量，直接复用其最终结果。
            return list(future.result())

        try:
            vector = tuple(self._embeddings.embed_query(query))
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
                future.set_exception(exc)
            raise

        with self._lock:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            self._inflight.pop(key, None)
            future.set_result(vector)
        return list(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档批量嵌入无需查询缓存，直接委托给底层模型。"""
        return self._embeddings.embed_documents(texts)


def get_cached_retriever(kb_name: str):
    """获取无过滤条件的 LRU 检索图；命中后提升其最近使用顺序。"""
    return _retriever_cache.get(kb_name)


def cache_retriever(kb_name: str, retriever) -> None:
    """只缓存可复用的全知识库检索图，超限时淘汰最久未使用项。"""
    _retriever_cache.put(kb_name, retriever)


def clear_retriever_cache() -> None:
    """清空所有知识库的检索器缓存。"""
    _retriever_cache.clear()


def warm_retriever_cache(container: ApplicationContainer | None = None) -> int:
    """为文档均已就绪的知识库预热检索图，不会调用嵌入 API。"""
    dependencies = container or get_application_container()
    with application_db.connect(dependencies.database_path()) as conn:
        knowledge_bases = [
            row[0]
            for row in conn.execute(
                """
                SELECT kb_name FROM documents
                GROUP BY kb_name
                HAVING COUNT(*) > 0
                   AND SUM(CASE WHEN status = 'ready' THEN 0 ELSE 1 END) = 0
                ORDER BY kb_name
                """
            ).fetchall()
        ]

    warmed = 0
    for kb_name in knowledge_bases:
        if get_cached_retriever(kb_name) is not None:
            continue
        sources_before = dependencies.documents.ready_sources(kb_name)
        try:
            retriever = build_retriever(kb_name, container=dependencies)
        except Exception as exc:
            logger.debug("Retriever warmup skipped for %s: %s", kb_name, exc)
            continue
        # 组装检索图期间若文档发生变化，就不缓存这个已经过时的图。
        if retriever and sources_before == dependencies.documents.ready_sources(kb_name):
            cache_retriever(kb_name, retriever)
            warmed += 1
    logger.info("Retriever cache warmup completed: %s knowledge bases", warmed)
    return warmed


def start_retriever_cache_warmup(
    container: ApplicationContainer | None = None,
) -> threading.Thread:
    """在后台线程中启动检索器缓存预热，并返回该线程。"""
    thread = threading.Thread(
        target=warm_retriever_cache,
        args=(container,),
        name="retriever-cache-warmup",
        daemon=True,
    )
    thread.start()
    return thread


def drop_cached_retriever(kb_name: str) -> None:
    """移除一个知识库的缓存检索图，使下次查询按最新索引重建。"""
    _retriever_cache.drop(kb_name)


def rename_cached_retriever(old_name: str, new_name: str) -> None:
    """知识库改名后失效新旧名称的缓存检索图。"""
    # 检索器内嵌部门过滤条件，直接迁移缓存实例会使其失效。
    drop_cached_retriever(old_name)
    drop_cached_retriever(new_name)



def _normalized_overlap_text(value: str) -> str:
    """移除空白字符，供跨路线文本重叠比较使用。"""
    return re.sub(r"\s+", "", value or "")


def _substantially_contains(left: str, right: str) -> bool:
    """判断一段文本是否基本包含另一段，不进行模糊的主题相似度猜测。"""
    shorter, longer = sorted((_normalized_overlap_text(left), _normalized_overlap_text(right)), key=len)
    if len(shorter) < _OVERLAP_EXACT_MATCH_THRESHOLD:
        return shorter == longer
    if shorter in longer:
        return True
    probes = [
        shorter[index : index + _OVERLAP_PROBE_LENGTH]
        for index in range(
            0,
            len(shorter) - _OVERLAP_PROBE_LENGTH + 1,
            _OVERLAP_PROBE_STRIDE,
        )
    ]
    return (
        bool(probes)
        and sum(probe in longer for probe in probes) / len(probes)
        >= _OVERLAP_MATCH_RATIO
    )


def deduplicate_chunk_routes(documents: list[LCDocument]) -> list[LCDocument]:
    """去掉完全重复和跨路线重叠的证据，同时保留原始排序。

    同一段源文可同时出现在固定和结构化路线，只应占一个重排名额；真正不同的
    路线证据，例如图片描述，仍应保留。
    """
    retained: list[LCDocument] = []
    exact_keys: set[tuple[str, str, str]] = set()
    for document in documents:
        source = context_builder.source_name(document)
        kb_name = context_builder.document_kb_name(document, "")
        route = chunk_route(document)
        exact_key = (kb_name, source, _normalized_overlap_text(document.page_content))
        if exact_key in exact_keys:
            continue
        duplicate = any(
            context_builder.document_kb_name(existing, "") == kb_name
            and context_builder.source_name(existing) == source
            and chunk_route(existing) != route
            and _substantially_contains(existing.page_content, document.page_content)
            for existing in retained
        )
        if not duplicate:
            retained.append(document)
            exact_keys.add(exact_key)
    return retained


def _retrieval_sources(
    kb_name: str,
    filters: RetrievalFilters | None,
    container: ApplicationContainer | None = None,
) -> list[str]:
    """在 SQLite 权威源文件允许列表内解析用户传入的过滤条件。"""
    dependencies = container or get_application_container()
    current_sources = source_state.ready_sources(kb_name, dependencies)
    filtered_sources = dependencies.documents.filtered_sources(kb_name, filters)
    if filtered_sources is None:
        return current_sources
    current_set = set(current_sources)
    return [source for source in filtered_sources if source in current_set]


request_source_snapshot = source_state.request_source_snapshot


@dataclass(frozen=True)
class _BaseCapabilities:
    """Available sources and routes discovered once for one knowledge base."""
    current_sources: tuple[str, ...]
    chroma_sources: tuple[str, ...]
    fts_routes: frozenset[str]
    chroma_routes: frozenset[str]


def _probe_base_capabilities(
    kb_name: str, allowed_sources: list[str] | None
) -> _BaseCapabilities:
    """为指定知识库一次性探测来源白名单和可用检索路线。"""
    current_sources = tuple(
        source_state.current_source_allowlist(kb_name, allowed_sources)
    )
    if not current_sources:
        return _BaseCapabilities((), (), frozenset(), frozenset())

    db_path = fts_index.fts_db_path(kb_name)
    fts_routes = fts_index.fts5_available_routes(db_path, list(current_sources))
    if fts_routes is None:
        raise ValueError(
            "知识库索引缺失或版本过旧，请在知识库管理中完整重建索引后再提问。"
        )

    chroma_sources = tuple(
        chroma_index.current_chroma_source_allowlist(kb_name, current_sources)
    )
    chroma_routes: set[str] = set()
    try:
        collection = chroma_index.get_chroma_collection(create=False)
        if collection is not None and chroma_sources:
            for route in (
                CHUNK_ROUTE_FIXED,
                CHUNK_ROUTE_STRUCTURED,
                CHUNK_ROUTE_VISUAL,
            ):
                records = collection.get(
                    where=chroma_index.chroma_retrieval_filter(
                        kb_name, chroma_sources, route
                    ),
                    include=[],
                    limit=1,
                )
                if records.get("ids"):
                    chroma_routes.add(route)
    except Exception as exc:
        logger.warning("Chroma retrieval unavailable: %s", exc)

    return _BaseCapabilities(
        current_sources,
        chroma_sources,
        frozenset(fts_routes or ()),
        frozenset(chroma_routes),
    )


def _get_base_retrievers(
    kb_name: str,
    embeddings,
    capabilities: _BaseCapabilities,
    route: str,
):
    """为指定知识库和切片路线创建 Chroma、FTS5 两个基础检索器。"""
    if not capabilities.current_sources:
        return None, None
    db_path = fts_index.fts_db_path(kb_name)
    chroma_retriever = (
        chroma_index.ChromaRetriever(
            kb_name=kb_name,
            embeddings=embeddings,
            k=RETRIEVAL_CHROMA_K,
            allowed_sources=capabilities.chroma_sources,
            chunk_route=route,
            enforce_current_sources=True,
        )
        if route in capabilities.chroma_routes
        else None
    )
    fts5_retriever = (
        fts_index.FTS5Retriever(
            db_path=db_path,
            # FTS5 participates in the main ensemble, so it needs the same candidate
            # budget as Chroma before fusion. The smaller lexical-only pass uses K=6.
            k=RETRIEVAL_CHROMA_K,
            allowed_sources=capabilities.current_sources,
            kb_name=kb_name,
            chunk_route=route,
            enforce_current_sources=True,
        )
        if route in capabilities.fts_routes
        else None
    )
    return chroma_retriever, fts5_retriever


def _append_backend_pairs(pairs: list, retrievers: tuple, total_weight: float) -> None:
    """在可用的关键词和语义后端之间分配一条路线的权重。

    两者都可用时，语义检索略高权重，FTS5 保留编号、文档号和罕见词等精确命中；
    任一后端不可用时，剩余后端获得该路线全部权重。
    """
    chroma_retriever, fts5_retriever = retrievers
    if chroma_retriever and fts5_retriever:
        pairs.extend([(fts5_retriever, total_weight * 0.4), (chroma_retriever, total_weight * 0.6)])
    elif chroma_retriever:
        pairs.append((chroma_retriever, total_weight))
    elif fts5_retriever:
        pairs.append((fts5_retriever, total_weight))


def _append_chunk_route_pairs(
    pairs: list,
    kb_name: str,
    embeddings,
    allowed_sources: list[str] | None,
    total_weight: float,
) -> None:
    """融合固定、结构化和视觉路线，只给实际含有证据的路线分配权重。"""
    capabilities = _probe_base_capabilities(kb_name, allowed_sources)
    route_weights = (
        (CHUNK_ROUTE_FIXED, RETRIEVAL_ROUTE_WEIGHTS["fixed"]),
        (CHUNK_ROUTE_STRUCTURED, RETRIEVAL_ROUTE_WEIGHTS["structured"]),
        (CHUNK_ROUTE_VISUAL, RETRIEVAL_ROUTE_WEIGHTS["visual"]),
    )
    available = []
    for route, weight in route_weights:
        retrievers = _get_base_retrievers(kb_name, embeddings, capabilities, route)
        if any(retrievers):
            available.append((retrievers, weight))
    available_weight = sum(weight for _, weight in available)
    # 只在实际有文档的路线间归一化权重，避免空视觉路线无声压缩固定路线的召回预算。
    for retrievers, weight in available:
        _append_backend_pairs(pairs, retrievers, total_weight * weight / available_weight)


def _retrieve_lexical_candidates_for_base(
    kb_name: str,
    query: str,
    allowed_sources: list[str] | None,
    per_route: int,
) -> list[LCDocument]:
    """在语义融合和重排前收集精确匹配补充证据。

    该受限路径保护包含编号、日期或产品名的问题，避免其精确性被向量相似度稀释；
    结果稍后与主候选集合合并，不作为第二套最终排序。
    """
    if allowed_sources == []:
        return []
    db_path = fts_index.fts_db_path(kb_name)
    if not fts_index.fts5_index_is_current(db_path):
        return []
    return fts_index.retrieve_all_routes(
        db_path,
        query,
        kb_name,
        tuple(allowed_sources) if allowed_sources is not None else None,
        per_route,
        enforce_current_sources=True,
    )


def retrieve_lexical_candidates(
    kb_name: str,
    query: str,
    filters: RetrievalFilters | None = None,
    *,
    per_route: int = RETRIEVAL_LEXICAL_K,
    container: ApplicationContainer | None = None,
) -> list[LCDocument]:
    """保留小规模 FTS5 证据池，让精确术语穿过向量融合噪声。"""
    try:
        dependencies = container or get_application_container()
        kb_name = application_config.validate_kb_name(kb_name)
        if not dependencies.documents.knowledge_base_exists(kb_name):
            return []
        candidates = _retrieve_lexical_candidates_for_base(
            kb_name,
            query,
            _retrieval_sources(kb_name, filters, dependencies),
            per_route,
        )
        # 部门知识库可补充通用政策材料；查询通用知识库本身时不重复加入。
        if (
            kb_name != source_state.COMMON_KNOWLEDGE_BASE
            and dependencies.documents.knowledge_base_exists(
                source_state.COMMON_KNOWLEDGE_BASE
            )
        ):
            candidates.extend(
                _retrieve_lexical_candidates_for_base(
                    source_state.COMMON_KNOWLEDGE_BASE,
                    query,
                    _retrieval_sources(
                        source_state.COMMON_KNOWLEDGE_BASE, filters, dependencies
                    ),
                    per_route,
                )
            )
        return candidates
    except Exception as exc:
        logger.debug("Lexical candidate supplement failed: %s", exc)
        return []


def build_retriever(
    kb_name: str,
    filters: RetrievalFilters | None = None,
    *,
    container: ApplicationContainer | None = None,
):
    """为一次请求或缓存项构建加权检索图。

    当前选择的知识库占主要召回预算；通用知识库作为补充，让全组织政策文档可回答
    部门问题，无需把同一文件复制到每个部门。
    """
    from langchain_classic.retrievers import EnsembleRetriever

    dependencies = container or get_application_container()
    kb_name = application_config.validate_kb_name(kb_name)
    if not dependencies.documents.knowledge_base_exists(kb_name):
        raise ValueError("知识库不存在")

    embeddings = CachedQueryEmbeddings(dependencies.create_query_embeddings())
    pairs = []
    sources = _retrieval_sources(kb_name, filters, dependencies)
    # 当前知识库主导召回，通用知识库负责补充；路线/后端辅助函数会为缺失索引重新归一化权重。
    _append_chunk_route_pairs(pairs, kb_name, embeddings, sources, 0.70)

    if (
        kb_name != source_state.COMMON_KNOWLEDGE_BASE
        and dependencies.documents.knowledge_base_exists(
            source_state.COMMON_KNOWLEDGE_BASE
        )
    ):
        common_sources = _retrieval_sources(
            source_state.COMMON_KNOWLEDGE_BASE, filters, dependencies
        )
        _append_chunk_route_pairs(
            pairs,
            source_state.COMMON_KNOWLEDGE_BASE,
            embeddings,
            common_sources,
            0.30,
        )

    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    return EnsembleRetriever(
        retrievers=[retriever for retriever, _ in pairs],
        weights=[round(weight / total_weight, 6) for _, weight in pairs],
    )
