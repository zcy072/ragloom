"""子块索引接口：向量与全文检索的统一入口，生产实现见 milvus_index。"""

import hashlib
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from langchain_core.documents import Document as LCDocument

from services.core.concurrency import resource_slot
from services.core.request_context import carry_request_id
from services.core.settings import get_settings
from services.indexing.chunking import chunk_route, contextualized_chunk_text


class ChunkIndex(Protocol):
    """子块的写入、删除与检索。所有方法按 kb_id 隔离，sources 为 None 表示不限文件。"""

    def publish_source(
        self,
        kb_id: str,
        source: str,
        splits: list,
        vectors: list[list[float]],
    ) -> None:
        """先写入一个文件的新版本，成功后再删除旧版本；失败时旧版本保持可用。"""
        ...

    def delete_source(self, kb_id: str, source: str) -> None:
        """删除一个文件的全部子块；幂等。"""
        ...

    def delete_kb(self, kb_id: str) -> None:
        """删除一个知识库的全部子块；幂等。"""
        ...

    def delete_stale_sources(self, kb_id: str, active_sources: set[str]) -> int:
        """删除不属于 active_sources 的文件的子块，返回删除数量。"""
        ...

    def ensure_collection(self) -> None:
        """启动时确认索引存储可用（Milvus 需等待就绪）；幂等。"""
        ...

    def reset_if_empty(self, has_documents: Callable[[], bool]) -> bool:
        """系统中不再有任何文档时清空索引存储，返回是否执行了清空。"""
        ...

    def check_ready(self, timeout: float) -> str:
        """就绪探测：索引可查询时返回简短说明，否则抛出异常；不创建或加载任何东西。"""
        ...

    def search_hybrid(
        self,
        kb_id: str,
        query: str,
        query_vector: list[float],
        route: str | None,
        sources: Sequence[str] | None,
        k: int,
    ) -> list[LCDocument]:
        """稠密向量与 BM25 融合检索，由索引内部完成融合排序。"""
        ...

    def search_dense(
        self,
        kb_id: str,
        query_vector: list[float],
        route: str | None,
        sources: Sequence[str] | None,
        k: int,
    ) -> list[LCDocument]:
        """只做稠密向量检索，供评测对照。"""
        ...

    def search_lexical(
        self,
        kb_id: str,
        query: str,
        route: str | None,
        sources: Sequence[str] | None,
        k: int,
    ) -> list[LCDocument]:
        """只做 BM25 全文检索。"""
        ...

    def available_routes(self, kb_id: str, sources: Sequence[str] | None) -> set[str]:
        """返回这些文件下实际存在的分块路线；索引不存在时返回空集合。"""
        ...

    def neighbors(
        self, kb_id: str, source: str, route: str, lo: int, hi: int
    ) -> list[LCDocument]:
        """按 chunk_index 闭区间 [lo, hi] 取同一文件同一路线的相邻子块，按序返回。"""
        ...


def escape_filter_value(value: str) -> str:
    """转义过滤表达式中的字符串值：先转义反斜杠，再转义双引号。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def build_filter(
    kb_id: str,
    *,
    source: str | None = None,
    sources: Sequence[str] | None = None,
    route: str | None = None,
    chunk_index_range: tuple[int, int] | None = None,
) -> str:
    """生成 Milvus 过滤表达式。文件名是用户输入，所有字符串值都必须经过转义。"""
    conditions = [f'kb_id == "{escape_filter_value(kb_id)}"']
    if source is not None:
        conditions.append(f'source == "{escape_filter_value(source)}"')
    if sources is not None:
        quoted = ", ".join(f'"{escape_filter_value(item)}"' for item in sources)
        conditions.append(f"source in [{quoted}]")
    if route:
        conditions.append(f'chunk_route == "{escape_filter_value(route)}"')
    if chunk_index_range is not None:
        lo, hi = int(chunk_index_range[0]), int(chunk_index_range[1])
        conditions.append(f"chunk_index >= {lo} and chunk_index <= {hi}")
    return " and ".join(conditions)


def record_id(kb_id: str, source: str, position: int, split, generation: str) -> str:
    """生成包含发布代际的稳定记录 ID。"""
    value = "\x1f".join(
        (kb_id, source, chunk_route(split), str(position), split.page_content, generation)
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def embed_splits(splits: list, embeddings) -> list[list[float]]:
    """为分块计算向量；不接触索引，可在不持有知识库锁时调用。

    向量只取决于分块文本，与 chunk_index 无关，因此可以先于最终写入计算。
    所有索引任务合计最多同时发出 EMBEDDING_MAX_CONCURRENCY 个嵌入请求。
    """
    if not splits:
        return []
    texts = [contextualized_chunk_text(split) for split in splits]
    with resource_slot("embedding", get_settings().embedding_max_concurrency):
        return embeddings.embed_documents(texts)


# 一次问题最多分发到 (知识库 × 路线) 次索引查询，组装上下文时还要取相邻块；
# 这些查询互不依赖，并发执行以压缩延迟。
# 线程池进程级共享，用并发上限保护 Milvus，调用方线程不在池内，不会互相死锁。
_SEARCH_WORKERS = 8
_search_executor: ThreadPoolExecutor | None = None
_search_executor_lock = threading.Lock()


def _get_search_executor() -> ThreadPoolExecutor:
    """惰性创建检索线程池。"""
    global _search_executor
    with _search_executor_lock:
        if _search_executor is None:
            _search_executor = ThreadPoolExecutor(
                max_workers=_SEARCH_WORKERS, thread_name_prefix="route-search"
            )
        return _search_executor


def run_searches(calls: list[Callable[[], list[LCDocument]]]) -> list[list[LCDocument]]:
    """并发执行一组检索调用，按输入顺序返回结果；任一调用失败则抛出其异常。"""
    if len(calls) <= 1:
        return [call() for call in calls]
    executor = _get_search_executor()
    futures = [executor.submit(carry_request_id(call)) for call in calls]
    try:
        return [future.result() for future in futures]
    except BaseException:
        for future in futures:
            future.cancel()
        raise


_chunk_index: ChunkIndex | None = None
_chunk_index_lock = threading.Lock()


def get_chunk_index() -> ChunkIndex:
    """返回进程级索引实例；首次调用时按配置创建 Milvus 实现。"""
    global _chunk_index
    if _chunk_index is None:
        with _chunk_index_lock:
            if _chunk_index is None:
                from services.retrieval.milvus_index import MilvusChunkIndex

                _chunk_index = MilvusChunkIndex.from_settings(get_settings())
    return _chunk_index


def set_chunk_index(index: ChunkIndex | None) -> None:
    """替换进程级索引实例；测试注入假实现，传 None 恢复默认。"""
    global _chunk_index
    with _chunk_index_lock:
        _chunk_index = index
