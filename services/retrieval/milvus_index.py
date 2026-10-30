"""Milvus 子块索引：稠密向量加内置 BM25 全文检索，是唯一直接依赖 pymilvus 的模块。"""

import functools
import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence

from langchain_core.documents import Document as LCDocument

from services.core.errors import ServiceUnavailableError
from services.indexing import schema as index_schema
from services.indexing.chunking import (
    assign_chunk_indexes,
    chunk_route,
    contextualized_chunk_text,
)
from services.retrieval.chunk_index import build_filter, record_id

logger = logging.getLogger(__name__)

_INSERT_BATCH = 500
_TEXT_MAX = 65535
_RRF_K = 60
_OUTPUT_FIELDS = [
    "content", "source", "page", "chunk_index", "chunk_route", "parent_id",
    "contextual_title", "visual_kind", "visual_image", "visual_analysis",
]
_METADATA_FIELDS = [name for name in _OUTPUT_FIELDS if name != "content"]
_STARTUP_WAIT_SECONDS = 60.0


def _unavailable_on_milvus_error(method):
    """把读路径上的 Milvus 异常转换为 503，调用方得到明确提示而不是裸 500。"""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            from pymilvus.exceptions import MilvusException

            if not isinstance(exc, MilvusException):
                raise
            logger.warning("Milvus 检索失败：%s", exc)
            raise ServiceUnavailableError(
                "检索服务暂时不可用，请稍后重试", code="retrieval_index_unavailable"
            ) from exc

    return wrapper


def _entity_to_document(entity: dict) -> LCDocument:
    """把 Milvus 返回的实体转换为带检索元数据的 Document，丢弃空字段。"""
    metadata = {
        name: entity[name]
        for name in _METADATA_FIELDS
        if entity.get(name) not in (None, "")
    }
    return LCDocument(page_content=entity.get("content", ""), metadata=metadata)


def _clip(value: object, max_bytes: int) -> str:
    """按 UTF-8 字节数截断；Milvus 的 VARCHAR max_length 以字节计，中文一字占 3 字节。"""
    text = str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class MilvusChunkIndex:
    """基于 Milvus Standalone 的 ChunkIndex 实现，一个 collection 容纳所有知识库。"""

    def __init__(
        self,
        uri: str,
        token: str,
        collection_prefix: str,
        dimension: int,
        *,
        ranker: str = "rrf",
    ) -> None:
        """保存连接参数；客户端和 collection 在首次使用时才创建。"""
        self._uri = uri
        self._token = token
        self._collection = f"{collection_prefix}chunks_v{index_schema.CURRENT_INDEX_SCHEMA_VERSION}"
        self._dimension = dimension
        self._ranker = ranker
        self._client = None
        self._ready = False
        self._lock = threading.RLock()

    @classmethod
    def from_settings(cls, settings) -> "MilvusChunkIndex":
        """按应用配置创建实例。"""
        return cls(
            settings.milvus_uri,
            settings.milvus_token,
            settings.milvus_collection_prefix,
            settings.embedding_dimension,
        )

    def _get_client(self):
        """惰性创建 MilvusClient，所有线程共用一个实例。"""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from pymilvus import MilvusClient

                    self._client = MilvusClient(uri=self._uri, token=self._token or "")
        return self._client

    def _build_schema(self, client):
        """构造 collection 的字段、BM25 函数和索引参数。"""
        from pymilvus import DataType, Function, FunctionType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self._dimension)
        schema.add_field(
            "search_text", DataType.VARCHAR, max_length=_TEXT_MAX,
            enable_analyzer=True, analyzer_params={"type": "chinese"},
        )
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("content", DataType.VARCHAR, max_length=_TEXT_MAX)
        schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_route", DataType.VARCHAR, max_length=32)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("page", DataType.VARCHAR, max_length=64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=128)
        schema.add_field("contextual_title", DataType.VARCHAR, max_length=2048)
        schema.add_field("visual_kind", DataType.VARCHAR, max_length=64)
        schema.add_field("visual_image", DataType.VARCHAR, max_length=1024)
        schema.add_field("visual_analysis", DataType.VARCHAR, max_length=_TEXT_MAX)
        schema.add_field("generation", DataType.VARCHAR, max_length=64)
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["search_text"],
                output_field_names=["sparse"],
            )
        )
        index_params = client.prepare_index_params()
        index_params.add_index(
            "dense", index_type="HNSW", metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            "sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        return schema, index_params

    def ensure_collection(self, wait_seconds: float = _STARTUP_WAIT_SECONDS) -> None:
        """等待 Milvus 就绪后创建并加载 collection；幂等，可在启动时调用。"""
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                self._ensure_collection_once()
                return
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise
                logger.info("等待 Milvus 就绪：%s", exc)
                time.sleep(2.0)

    def _ensure_collection_once(self) -> None:
        """单次尝试：不存在则创建，随后确保已加载。"""
        with self._lock:
            client = self._get_client()
            if not client.has_collection(self._collection):
                schema, index_params = self._build_schema(client)
                client.create_collection(
                    self._collection, schema=schema, index_params=index_params,
                    consistency_level="Strong",
                )
            client.load_collection(self._collection)
            self._ready = True

    def check_ready(self, timeout: float) -> str:
        """只读探测 collection 加载状态；Milvus 不可达或超时时由 pymilvus 抛出异常。"""
        from pymilvus.client.types import LoadState

        state = self._get_client().get_load_state(self._collection, timeout=timeout)["state"]
        if state != LoadState.Loaded:
            raise RuntimeError(f"collection {self._collection} 状态为 {state.name}")
        return self._collection

    def _ready_client(self):
        """返回已确保 collection 存在的客户端；Milvus 不可用时抛出异常，不静默返回空。"""
        if not self._ready:
            self._ensure_collection_once()
        return self._get_client()

    @staticmethod
    def _row(kb_id: str, source: str, position: int, split, vector, generation: str) -> dict:
        """把一个检索切片转换为 Milvus 行；sparse 字段由 BM25 函数生成，不传。"""
        meta = split.metadata
        return {
            "id": record_id(kb_id, source, position, split, generation),
            "dense": [float(value) for value in vector],
            "search_text": _clip(contextualized_chunk_text(split), _TEXT_MAX),
            "content": _clip(split.page_content, _TEXT_MAX),
            "kb_id": kb_id,
            "source": source,
            "chunk_route": chunk_route(split),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "page": _clip(meta.get("page", meta.get("row", "")), 64),
            "parent_id": _clip(meta.get("parent_id", ""), 128),
            "contextual_title": _clip(meta.get("contextual_title", ""), 2048),
            "visual_kind": _clip(meta.get("visual_kind", ""), 64),
            "visual_image": _clip(meta.get("visual_image", ""), 1024),
            "visual_analysis": _clip(meta.get("visual_analysis", ""), _TEXT_MAX),
            "generation": generation,
        }

    def publish_source(
        self, kb_id: str, source: str, splits: list, vectors: list[list[float]]
    ) -> None:
        """先写入新代，成功后删除同文件的其他代；写入失败则撤销新代并抛出。

        旧代主键在写入前查出，写入后按主键删除。collection 是强一致性，紧接在写入后
        按条件查询或删除要等这批写入可见，600 条约 1.7 秒；按主键删除不需要等。
        调用方须持有该文件的锁，否则写入前后可能有其他发布插进来，留下查不到的旧代。
        """
        splits = assign_chunk_indexes(splits)
        if len(vectors) != len(splits):
            raise ValueError(f"向量数量与分块数量不一致：{len(vectors)} != {len(splits)}")
        client = self._ready_client()
        if not splits:
            self.delete_source(kb_id, source)
            return
        generation = uuid.uuid4().hex
        rows = [
            self._row(kb_id, source, position, split, vector, generation)
            for position, (split, vector) in enumerate(zip(splits, vectors))
        ]
        previous_ids = self._ids(build_filter(kb_id, source=source))
        try:
            for offset in range(0, len(rows), _INSERT_BATCH):
                client.insert(self._collection, rows[offset : offset + _INSERT_BATCH])
        except Exception:
            try:
                self._delete_ids([row["id"] for row in rows])
            except Exception as cleanup_exc:
                logger.warning("Milvus 新代清理失败：%s", cleanup_exc)
            raise
        self._delete_ids(previous_ids)
        logger.info("Milvus 发布 [%s/%s] %d 条", kb_id, source, len(rows))

    def _ids(self, filter_expression: str, keep: Callable[[dict], bool] | None = None) -> list[str]:
        """分批读取满足条件的主键；条数不受单次查询上限限制。"""
        ids: list[str] = []
        iterator = self._get_client().query_iterator(
            self._collection, batch_size=1000, filter=filter_expression,
            output_fields=["id", "source"],
        )
        try:
            while batch := iterator.next():
                ids.extend(row["id"] for row in batch if keep is None or keep(row))
        finally:
            iterator.close()
        return ids

    def _delete_ids(self, ids: list[str]) -> None:
        """按主键分批删除。"""
        client = self._get_client()
        for offset in range(0, len(ids), _INSERT_BATCH):
            client.delete(self._collection, ids=ids[offset : offset + _INSERT_BATCH])

    def delete_source(self, kb_id: str, source: str) -> None:
        """删除一个文件的全部子块；collection 不存在时视为已删除。"""
        client = self._get_client()
        if not client.has_collection(self._collection):
            return
        client.delete(self._collection, filter=build_filter(kb_id, source=source))

    def delete_kb(self, kb_id: str) -> None:
        """删除一个知识库的全部子块；Milvus 不可用时抛出，由调用方中止操作。"""
        client = self._get_client()
        if not client.has_collection(self._collection):
            return
        client.delete(self._collection, filter=build_filter(kb_id))

    def delete_stale_sources(self, kb_id: str, active_sources: set[str]) -> int:
        """删除知识库内不属于 active_sources 的文件的子块，返回删除的条数。"""
        client = self._get_client()
        if not client.has_collection(self._collection):
            return 0
        stale_ids = self._ids(
            build_filter(kb_id), lambda row: row["source"] not in active_sources
        )
        self._delete_ids(stale_ids)
        return len(stale_ids)

    def reset_if_empty(self, has_documents: Callable[[], bool]) -> bool:
        """系统里没有任何文档时删除整个 collection，下次写入再重建。"""
        if has_documents():
            return False
        with self._lock:
            # 第一次判断到拿到锁之间可能有新文档登记；drop 前在锁内再确认一次。
            if has_documents():
                return False
            client = self._get_client()
            if client.has_collection(self._collection):
                client.drop_collection(self._collection)
            self._ready = False
        return True

    def _hits_to_documents(self, hits) -> list[LCDocument]:
        """把一次搜索的命中列表转换为 Document 列表。"""
        return [_entity_to_document(hit["entity"]) for hit in hits]

    @_unavailable_on_milvus_error
    def search_dense(
        self,
        kb_id: str,
        query_vector: list[float],
        route: str | None,
        sources: Sequence[str] | None,
        k: int,
    ) -> list[LCDocument]:
        """只做稠密向量检索。"""
        if sources is not None and not sources:
            return []
        result = self._ready_client().search(
            self._collection,
            data=[query_vector],
            anns_field="dense",
            search_params={"metric_type": "COSINE"},
            limit=k,
            filter=build_filter(kb_id, sources=sources, route=route),
            output_fields=_OUTPUT_FIELDS,
        )
        return self._hits_to_documents(result[0])

    @_unavailable_on_milvus_error
    def search_lexical(
        self,
        kb_id: str,
        query: str,
        route: str | None,
        sources: Sequence[str] | None,
        k: int,
    ) -> list[LCDocument]:
        """只做 BM25 全文检索。"""
        if (sources is not None and not sources) or not query.strip():
            return []
        result = self._ready_client().search(
            self._collection,
            data=[query],
            anns_field="sparse",
            search_params={"metric_type": "BM25"},
            limit=k,
            filter=build_filter(kb_id, sources=sources, route=route),
            output_fields=_OUTPUT_FIELDS,
        )
        return self._hits_to_documents(result[0])

    @_unavailable_on_milvus_error
    def search_hybrid(
        self,
        kb_id: str,
        query: str,
        query_vector: list[float],
        route: str | None,
        sources: Sequence[str] | None,
        k: int,
    ) -> list[LCDocument]:
        """一次 hybrid_search 完成稠密与 BM25 融合；两路请求各自带过滤条件。"""
        if sources is not None and not sources:
            return []
        if not query.strip():
            return self.search_dense(kb_id, query_vector, route, sources, k)
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        expr = build_filter(kb_id, sources=sources, route=route)
        requests = [
            AnnSearchRequest(
                data=[query_vector], anns_field="dense",
                param={"metric_type": "COSINE"}, limit=k, expr=expr,
            ),
            AnnSearchRequest(
                data=[query], anns_field="sparse",
                param={"metric_type": "BM25"}, limit=k, expr=expr,
            ),
        ]
        ranker = WeightedRanker(0.6, 0.4) if self._ranker == "weighted" else RRFRanker(_RRF_K)
        result = self._ready_client().hybrid_search(
            self._collection, reqs=requests, ranker=ranker, limit=k,
            output_fields=_OUTPUT_FIELDS,
        )
        return self._hits_to_documents(result[0])

    @_unavailable_on_milvus_error
    def available_routes(self, kb_id: str, sources: Sequence[str] | None) -> set[str]:
        """逐路线探测是否至少有一条记录；Milvus 没有 DISTINCT，路线只有三种。"""
        from services.indexing.chunking import (
            CHUNK_ROUTE_FIXED,
            CHUNK_ROUTE_STRUCTURED,
            CHUNK_ROUTE_VISUAL,
        )

        if sources is not None and not sources:
            return set()
        client = self._get_client()
        if not client.has_collection(self._collection):
            return set()
        client = self._ready_client()
        found = set()
        for route in (CHUNK_ROUTE_FIXED, CHUNK_ROUTE_STRUCTURED, CHUNK_ROUTE_VISUAL):
            rows = client.query(
                self._collection,
                filter=build_filter(kb_id, sources=sources, route=route),
                output_fields=["id"],
                limit=1,
            )
            if rows:
                found.add(route)
        return found

    @_unavailable_on_milvus_error
    def neighbors(
        self, kb_id: str, source: str, route: str, lo: int, hi: int
    ) -> list[LCDocument]:
        """按 chunk_index 闭区间取同一文件同一路线的相邻子块，按序返回。"""
        rows = self._ready_client().query(
            self._collection,
            filter=build_filter(
                kb_id, source=source, route=route, chunk_index_range=(lo, hi)
            ),
            output_fields=_OUTPUT_FIELDS,
            limit=max(1, hi - lo + 1),
        )
        rows.sort(key=lambda row: row.get("chunk_index", 0))
        return [_entity_to_document(row) for row in rows]
