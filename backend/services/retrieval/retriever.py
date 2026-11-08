"""检索器缓存、多来源融合和候选证据编排。"""

import collections
import functools
import hashlib
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument

from services.core import config as application_config
from services.core import sqlite_connections as application_db
from services.core.config import (
    RETRIEVAL_DENSE_K,
    RETRIEVAL_KB_WEIGHT_COMMON,
    RETRIEVAL_KB_WEIGHT_CURRENT,
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
from services.retrieval import context as context_builder
from services.retrieval import source_state
from services.retrieval.cache import RetrieverCache
from services.retrieval.chunk_index import ChunkIndex, get_chunk_index
from services.retrieval.chunk_index import run_searches as _run_searches

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


class QueryVectorCache:
    """进程级查询向量 LRU，以模型名和问题文本的哈希为键，不在内存保留原始问题文本。

    ``_inflight`` 实现单飞机制：并发路线或并发请求遇到同一问题时，只发一次远程嵌入调用。
    """

    def __init__(self, max_size: int = _QUERY_EMBEDDING_CACHE_SIZE):
        """初始化 LRU 向量缓存、在途请求表和并发锁。"""
        self._max_size = max(1, max_size)
        self._cache: collections.OrderedDict[bytes, tuple[float, ...]] = collections.OrderedDict()
        self._inflight: dict[bytes, Future] = {}
        self._lock = threading.Lock()

    def get_or_compute(
        self, model: str, query: str, compute: Callable[[], list[float]]
    ) -> list[float]:
        """返回缓存向量；未命中时由第一个调用方计算，其他并发调用方等待其结果。"""
        key = hashlib.sha256(f"{model}\x1f{query}".encode()).digest()
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
            vector = tuple(compute())
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

    def clear(self) -> None:
        """清空缓存的向量；在途计算不受影响。"""
        with self._lock:
            self._cache.clear()


_query_vector_cache = QueryVectorCache()


class CachedQueryEmbeddings:
    """给嵌入客户端套上进程级查询向量缓存。

    缓存不随检索器实例走，带筛选条件的请求每次新建检索器，也能复用之前算过的向量。
    """

    def __init__(self, embeddings, cache: QueryVectorCache | None = None):
        """保存底层嵌入客户端和共享缓存；默认使用进程级缓存。"""
        self._embeddings = embeddings
        self._cache = cache or _query_vector_cache
        self._model = str(getattr(embeddings, "model", "") or type(embeddings).__name__)

    def embed_query(self, query: str) -> list[float]:
        """返回查询向量；并发相同查询只会触发一次底层嵌入调用。"""
        return self._cache.get_or_compute(
            self._model, query, lambda: self._embeddings.embed_query(query)
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档批量嵌入无需查询缓存，直接委托给底层模型。"""
        return self._embeddings.embed_documents(texts)


class _RouteCache:
    """缓存“知识库 + 文件集合 → 已有路线”，省去每次构建检索器时的路线探测查询。

    键用知识库名，与检索器缓存同步失效：发布、删除和重建都会调用 drop_cached_retriever。
    每个知识库有一个代号，失效时加一；探测期间代号变了，结果就不写入缓存。
    """

    _MAX_SIZE = 256

    def __init__(self) -> None:
        """创建空缓存。"""
        self._items: collections.OrderedDict[tuple[str, tuple[str, ...]], frozenset[str]] = (
            collections.OrderedDict()
        )
        self._generations: dict[str, int] = collections.defaultdict(int)
        self._lock = threading.Lock()

    def routes(
        self, kb_name: str, sources: tuple[str, ...], probe: Callable[[], set[str]]
    ) -> set[str]:
        """返回缓存的路线；未命中时调用 probe，非空结果在代号未变时写入缓存。"""
        key = (kb_name, tuple(sorted(sources)))
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self._items.move_to_end(key)
                return set(cached)
            generation = self._generations[kb_name]
        routes = probe()
        with self._lock:
            if routes and self._generations[kb_name] == generation:
                self._items[key] = frozenset(routes)
                while len(self._items) > self._MAX_SIZE:
                    self._items.popitem(last=False)
        return routes

    def drop(self, kb_name: str) -> None:
        """失效一个知识库的全部路线缓存。"""
        with self._lock:
            self._generations[kb_name] += 1
            for key in [key for key in self._items if key[0] == kb_name]:
                del self._items[key]

    def clear(self) -> None:
        """失效所有知识库的路线缓存。"""
        with self._lock:
            for kb_name in {key[0] for key in self._items} | set(self._generations):
                self._generations[kb_name] += 1
            self._items.clear()


_route_cache = _RouteCache()


def get_cached_retriever(kb_name: str, *, include_common: bool = False):
    """获取无过滤条件的 LRU 检索图；命中后提升其最近使用顺序。

    是否补充通用知识库取决于提问者的权限，两种检索图分开缓存。
    """
    return _retriever_cache.get((kb_name, include_common))


def cache_retriever(kb_name: str, retriever, *, include_common: bool = False) -> None:
    """只缓存可复用的全知识库检索图，超限时淘汰最久未使用项。"""
    _retriever_cache.put((kb_name, include_common), retriever)


def clear_retriever_cache() -> None:
    """清空所有知识库的检索器缓存和路线缓存。"""
    _retriever_cache.clear()
    _route_cache.clear()


def warm_retriever_cache(container: ApplicationContainer | None = None) -> int:
    """为文档均已就绪的知识库预热检索图，不会调用嵌入 API。"""
    dependencies = container or get_application_container()
    with application_db.connect(str(dependencies.database_path())) as conn:
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

    # 大多数角色都能读取通用知识库，只预热补充通用知识库的检索图。
    warmed = 0
    for kb_name in knowledge_bases:
        if get_cached_retriever(kb_name, include_common=True) is not None:
            continue
        sources_before = dependencies.documents.ready_sources(kb_name)
        try:
            retriever = build_retriever(kb_name, include_common=True, container=dependencies)
        except Exception as exc:
            logger.debug("Retriever warmup skipped for %s: %s", kb_name, exc)
            continue
        # 组装检索图期间若文档发生变化，就不缓存这个已经过时的图。
        if retriever and sources_before == dependencies.documents.ready_sources(kb_name):
            cache_retriever(kb_name, retriever, include_common=True)
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
    """移除一个知识库的缓存检索图，使下次查询按最新索引重建。

    部门库的检索图在构建时固定了通用知识库的路线计划，所以通用知识库变化时要清空全部缓存。
    """
    _route_cache.drop(kb_name)
    if kb_name == source_state.COMMON_KNOWLEDGE_BASE:
        _retriever_cache.clear()
    else:
        _retriever_cache.drop((kb_name, False))
        _retriever_cache.drop((kb_name, True))


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
    shorter, longer = sorted(
        (_normalized_overlap_text(left), _normalized_overlap_text(right)), key=len
    )
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
        return list(current_sources)
    current_set = set(current_sources)
    return [source for source in filtered_sources if source in current_set]


request_source_snapshot = source_state.request_source_snapshot


_ROUTE_ORDER = (CHUNK_ROUTE_FIXED, CHUNK_ROUTE_STRUCTURED, CHUNK_ROUTE_VISUAL)
_FUSION_RRF_K = 60


def _kb_id(dependencies: ApplicationContainer, kb_name: str) -> str:
    """读取知识库的稳定 kb_id，知识库不存在时视为请求无效。"""
    kb_id = dependencies.knowledge_bases.kb_id_for(kb_name)
    if not kb_id:
        raise ValueError("知识库不存在")
    return kb_id


@dataclass(frozen=True)
class _RoutePlan:
    """一个 (知识库, 路线) 的检索单元及其在融合中的权重。"""
    kb_name: str
    kb_id: str
    route: str
    weight: float
    requested_sources: tuple[str, ...] | None


class WeightedRouteRetriever:
    """按知识库和路线分别做稠密加 BM25 混合检索，再按权重做跨路线 RRF 融合。

    路线内的稠密与 BM25 融合由索引完成；这里只负责知识库权重（当前库主导、通用库
    补充）和路线权重（固定、结构化、视觉），与原先 EnsembleRetriever 的分层一致。
    """

    def __init__(self, index: ChunkIndex, embeddings, plans: list[_RoutePlan]) -> None:
        """保存索引、共享的查询向量缓存和已按存在路线归一化的检索计划。"""
        self._index = index
        self._embeddings = embeddings
        self._plans = plans

    def invoke(self, query: str) -> list[LCDocument]:
        """返回融合后的候选文档；索引不可用时直接抛出，不静默返回空。"""
        query_vector = self._embeddings.embed_query(query)
        # 文件白名单读取请求级快照，必须在当前线程内解析。
        allowlists = {
            plan.kb_name: source_state.current_source_allowlist(
                plan.kb_name, plan.requested_sources
            )
            for plan in self._plans
        }
        active = [plan for plan in self._plans if allowlists[plan.kb_name]]
        results = _run_searches(
            [
                functools.partial(
                    self._index.search_hybrid,
                    plan.kb_id,
                    query,
                    query_vector,
                    plan.route,
                    list(allowlists[plan.kb_name]),
                    RETRIEVAL_DENSE_K,
                )
                for plan in active
            ]
        )
        scores: dict[tuple, float] = {}
        documents: dict[tuple, LCDocument] = {}
        # 融合按计划顺序累加，保证并发不改变同分时的排序。
        for plan, hits in zip(active, results):
            for rank, document in enumerate(hits):
                document.metadata["kb_name"] = plan.kb_name
                key = (
                    plan.kb_name,
                    document.metadata.get("source"),
                    plan.route,
                    document.metadata.get("chunk_index"),
                )
                scores[key] = scores.get(key, 0.0) + plan.weight / (_FUSION_RRF_K + rank + 1)
                documents.setdefault(key, document)
        return [documents[key] for key in sorted(scores, key=lambda key: -scores[key])]


def _route_plans(
    dependencies: ApplicationContainer,
    index: ChunkIndex,
    kb_name: str,
    requested_sources: list[str] | None,
    total_weight: float,
) -> list[_RoutePlan]:
    """探测一个知识库实际存在的路线，并把总权重按路线权重重新归一分配。"""
    current_sources = source_state.current_source_allowlist(kb_name, requested_sources)
    if not current_sources:
        return []
    kb_id = _kb_id(dependencies, kb_name)
    routes = _route_cache.routes(
        kb_name,
        tuple(current_sources),
        functools.partial(index.available_routes, kb_id, list(current_sources)),
    )
    if not routes:
        raise ValueError("知识库索引缺失或版本过旧，请在知识库管理中完整重建索引后再提问。")
    weights = [
        (route, RETRIEVAL_ROUTE_WEIGHTS[route]) for route in _ROUTE_ORDER if route in routes
    ]
    available_weight = sum(weight for _, weight in weights)
    requested = tuple(requested_sources) if requested_sources is not None else None
    # 只在实际有文档的路线间归一化权重，避免空视觉路线无声压缩固定路线的召回预算。
    return [
        _RoutePlan(kb_name, kb_id, route, total_weight * weight / available_weight, requested)
        for route, weight in weights
    ]


def _retrieve_lexical_candidates_for_base(
    dependencies: ApplicationContainer,
    kb_name: str,
    query: str,
    allowed_sources: list[str] | None,
    per_route: int,
) -> list[LCDocument]:
    """在语义融合和重排前收集 BM25 精确匹配补充证据。

    该受限路径保护包含编号、日期或产品名的问题，避免其精确性被向量相似度稀释；
    结果稍后与主候选集合合并，不作为第二套最终排序。
    """
    sources = source_state.current_source_allowlist(kb_name, allowed_sources)
    if not sources:
        return []
    kb_id = _kb_id(dependencies, kb_name)
    index = get_chunk_index()
    results = _run_searches(
        [
            functools.partial(
                index.search_lexical, kb_id, query, route, list(sources), per_route
            )
            for route in _ROUTE_ORDER
        ]
    )
    candidates: list[LCDocument] = []
    for hits in results:
        for document in hits:
            document.metadata["kb_name"] = kb_name
            candidates.append(document)
    return candidates


def retrieve_lexical_candidates(
    kb_name: str,
    query: str,
    filters: RetrievalFilters | None = None,
    *,
    per_route: int = RETRIEVAL_LEXICAL_K,
    include_common: bool = False,
    container: ApplicationContainer | None = None,
) -> list[LCDocument]:
    """保留小规模 BM25 证据池，让精确术语穿过向量融合噪声；失败时只记录日志。"""
    try:
        dependencies = container or get_application_container()
        kb_name = application_config.validate_kb_name(kb_name)
        if not dependencies.documents.knowledge_base_exists(kb_name):
            return []
        candidates = _retrieve_lexical_candidates_for_base(
            dependencies,
            kb_name,
            query,
            _retrieval_sources(kb_name, filters, dependencies),
            per_route,
        )
        # 能读取通用知识库时补充通用政策材料；查询通用知识库本身时不重复加入。
        if source_state.supplements_common(kb_name, include_common, dependencies):
            candidates.extend(
                _retrieve_lexical_candidates_for_base(
                    dependencies,
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
        logger.warning("Lexical candidate supplement failed: %s", exc)
        return []


def build_retriever(
    kb_name: str,
    filters: RetrievalFilters | None = None,
    *,
    include_common: bool = False,
    container: ApplicationContainer | None = None,
):
    """为一次请求或缓存项构建加权检索器；没有可检索文档时返回 None。

    当前选择的知识库占主要召回预算；``include_common`` 为真时补充通用知识库，让全组织
    政策文档可回答部门问题，无需把同一文件复制到每个部门。调用方负责确认提问者能读取
    通用知识库。
    """
    dependencies = container or get_application_container()
    kb_name = application_config.validate_kb_name(kb_name)
    if not dependencies.documents.knowledge_base_exists(kb_name):
        raise ValueError("知识库不存在")

    index = get_chunk_index()
    embeddings = CachedQueryEmbeddings(dependencies.create_query_embeddings())
    sources = _retrieval_sources(kb_name, filters, dependencies)
    # 当前知识库主导召回，通用知识库负责补充；缺失路线会重新归一化权重。
    plans = _route_plans(
        dependencies, index, kb_name, sources, RETRIEVAL_KB_WEIGHT_CURRENT
    )

    if source_state.supplements_common(kb_name, include_common, dependencies):
        common_sources = _retrieval_sources(
            source_state.COMMON_KNOWLEDGE_BASE, filters, dependencies
        )
        plans.extend(
            _route_plans(
                dependencies,
                index,
                source_state.COMMON_KNOWLEDGE_BASE,
                common_sources,
                RETRIEVAL_KB_WEIGHT_COMMON,
            )
        )

    if not plans:
        return None
    total_weight = sum(plan.weight for plan in plans)
    normalized = [
        _RoutePlan(
            plan.kb_name, plan.kb_id, plan.route, plan.weight / total_weight,
            plan.requested_sources,
        )
        for plan in plans
    ]
    return WeightedRouteRetriever(index, embeddings, normalized)
