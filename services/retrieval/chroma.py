"""Chroma 向量索引、存储生命周期和语义检索。"""

import hashlib
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable
from typing import Any, List

from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from services.core.concurrency import resource_slot
from services.core.settings import get_settings
from services.indexing import schema as index_schema
from services.indexing.chunking import (
    assign_chunk_indexes,
    chunk_route,
    contextualized_chunk_text,
)
from services.retrieval import source_state

logger = logging.getLogger(__name__)

CHROMA_PATH = get_settings().chroma_path
CHROMA_COLLECTION = "pymodel_knowledge"
CHROMA_COLLECTION_CONFIG = {"hnsw": {"space": "cosine"}}
_chroma_client = None
_chroma_storage_lock = threading.RLock()


def current_chroma_source_allowlist(
    kb_name: str, requested_sources: tuple[str, ...] | list[str] | None
) -> tuple[str, ...]:
    """排除当前版本向量写入失败、仅允许 FTS 检索的源文件。"""
    current_sources = source_state.current_source_allowlist(kb_name, requested_sources)
    disabled_sources = source_state.chroma_disabled_sources(kb_name)
    return tuple(source for source in current_sources if source not in disabled_sources)


class ChromaRetriever(BaseRetriever):
    """使用配置的嵌入模型，并按元数据过滤的 Chroma 检索器。"""

    kb_name: str = Field(...)
    embeddings: Any = Field(exclude=True)
    k: int = Field(default=20)
    allowed_sources: tuple[str, ...] | None = Field(default=None)
    chunk_route: str | None = Field(default=None)
    enforce_current_sources: bool = Field(default=False)

    def _get_collection(self):
        """获取现有 Chroma 集合，不因查询创建新的空集合。"""
        return get_chroma_collection(create=False)

    def _get_relevant_documents(self, query: str) -> List[LCDocument]:
        """计算查询向量后执行带知识库、源文件和路线过滤的相似度检索。"""
        try:
            collection = self._get_collection()
            if collection is None:
                return []
            allowed_sources = self.allowed_sources
            if self.enforce_current_sources:
                allowed_sources = current_chroma_source_allowlist(
                    self.kb_name, allowed_sources
                )
            if allowed_sources is not None and not allowed_sources:
                return []
            where = chroma_retrieval_filter(
                self.kb_name, allowed_sources, self.chunk_route
            )
            results = collection.query(
                query_embeddings=[self.embeddings.embed_query(query)],
                n_results=self.k,
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.warning("Chroma search failed: %s", exc)
            return []

        documents = (results.get("documents") or [[]])[0] or []
        metadatas = (results.get("metadatas") or [[]])[0] or []
        returned = []
        for document, metadata in zip(documents, metadatas):
            normalized_metadata = dict(metadata or {})
            normalized_metadata.setdefault("kb_name", self.kb_name)
            returned.append(LCDocument(page_content=document or "", metadata=normalized_metadata))
        return returned

    async def _aget_relevant_documents(self, query: str) -> List[LCDocument]:
        """异步接口复用同步 Chroma 查询实现。"""
        return self._get_relevant_documents(query)



def get_chroma_client():
    """为每个进程创建一个嵌入式、磁盘持久化的 Chroma 客户端。"""
    global _chroma_client
    if _chroma_client is None:
        # 不同知识库的索引任务会并行发布，首次创建客户端必须只发生一次。
        with _chroma_storage_lock:
            if _chroma_client is None:
                import chromadb

                os.makedirs(CHROMA_PATH, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def get_chroma_collection(create: bool = True):
    """获取主集合；create 为真时在缺失时创建并配置余弦距离。"""
    from chromadb.errors import NotFoundError

    client = get_chroma_client()
    try:
        return client.get_collection(name=CHROMA_COLLECTION, embedding_function=None)
    except (ValueError, NotFoundError):
        if not create:
            return None
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=None,
            configuration=CHROMA_COLLECTION_CONFIG,
        )


def chroma_metadata_for_split(split, kb_name: str, source_filename: str) -> dict:
    """把检索切片的来源、路线和展示元数据转换为 Chroma 元数据。"""
    metadata = {
        "kb_name": kb_name,
        "source": source_filename,
        index_schema.CHROMA_SCHEMA_VERSION_FIELD: index_schema.CURRENT_INDEX_SCHEMA_VERSION,
        "page": str(split.metadata.get("page", split.metadata.get("row", ""))),
        "chunk_index": int(split.metadata.get("chunk_index", 0)),
        "chunk_route": chunk_route(split),
    }
    for source_key, metadata_key in (
        ("contextual_title", "contextual_title"),
        ("visual_kind", "visual_kind"),
        ("visual_image", "visual_image"),
        ("visual_analysis", "visual_analysis"),
    ):
        value = split.metadata.get(source_key)
        if value:
            metadata[metadata_key] = str(value)
    if split.metadata.get("parent_id"):
        metadata["parent_id"] = split.metadata["parent_id"]
    return metadata


def _source_filter(kb_name: str, source_filename: str) -> dict:
    """构造精确限制到一个知识库和一个源文件的 Chroma 过滤条件。"""
    return {"$and": [{"kb_name": kb_name}, {"source": source_filename}]}


def chroma_retrieval_filter(
    kb_name: str,
    allowed_sources: tuple[str, ...] | list[str] | None,
    chunk_route: str | None = None,
) -> dict:
    """构造查询过滤条件，强制当前索引架构并可附加源文件和路线限制。"""
    conditions = [
        {"kb_name": kb_name},
        {
            index_schema.CHROMA_SCHEMA_VERSION_FIELD: index_schema.CURRENT_INDEX_SCHEMA_VERSION
        },
    ]
    if allowed_sources is not None:
        conditions.append({"source": {"$in": list(allowed_sources)}})
    if chunk_route:
        conditions.append({"chunk_route": chunk_route})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _chroma_record_id(
    kb_name: str,
    source_filename: str,
    position: int,
    split,
    *,
    generation: str = "",
) -> str:
    """生成包含写入代际的稳定记录 ID，使新版本可先写后删旧版本。"""
    value = "\x1f".join(
        (
            kb_name,
            source_filename,
            chunk_route(split),
            str(position),
            split.page_content,
            generation,
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _chroma_batch_size() -> int:
    """读取 Chroma 客户端可接受的最大批量大小，并设置保守上限。"""
    try:
        return min(500, max(1, int(get_chroma_client().get_max_batch_size())))
    except (AttributeError, TypeError, ValueError):
        return 500


def _iter_chroma_batches(collection, where: dict, include: list):
    """按批分页返回满足过滤条件的 Chroma 记录，每批为 (ids, records) 元组。"""
    batch_size = _chroma_batch_size()
    offset = 0
    while True:
        records = collection.get(where=where, include=include, limit=batch_size, offset=offset)
        ids = list(records.get("ids") or [])
        if not ids:
            break
        yield ids, records
        if len(ids) < batch_size:
            break
        offset += len(ids)


def _chroma_ids_for_filter(collection, where: dict) -> list[str]:
    """分页返回满足 Chroma 过滤条件的全部 ID，不加载文档正文。"""
    return [id_ for ids, _ in _iter_chroma_batches(collection, where, []) for id_ in ids]


def _delete_chroma_ids(collection, ids: list[str]) -> None:
    """按客户端批量限制删除一组 Chroma 记录 ID。"""
    if not ids:
        return
    batch_size = _chroma_batch_size()
    for offset in range(0, len(ids), batch_size):
        collection.delete(ids=ids[offset : offset + batch_size])


def embed_splits(splits: list, embeddings) -> list[list[float]]:
    """为分块计算向量；不接触 Chroma，可在不持有知识库锁时调用。

    向量只取决于分块文本，与 chunk_index 无关，因此可以先于最终写入计算。
    所有索引任务合计最多同时发出 EMBEDDING_MAX_CONCURRENCY 个嵌入请求。
    """
    if not splits:
        return []
    texts = [contextualized_chunk_text(split) for split in splits]
    with resource_slot("embedding", get_settings().embedding_max_concurrency):
        return embeddings.embed_documents(texts)


def chroma_upsert_splits(
    splits: list,
    kb_name: str,
    source_filename: str,
    embeddings,
    vectors: list[list[float]] | None = None,
) -> None:
    """先暂存一个源文件的新向量，再替换其旧 Chroma 版本。

    传入 ``vectors`` 时直接使用，不再请求嵌入模型；其长度必须与分块一致。
    """
    splits = assign_chunk_indexes(splits)
    if vectors is not None and len(vectors) != len(splits):
        raise ValueError(f"向量数量与分块数量不一致：{len(vectors)} != {len(splits)}")
    collection = get_chroma_collection()
    source_filter = _source_filter(kb_name, source_filename)
    existing_ids = _chroma_ids_for_filter(collection, source_filter)
    if not splits:
        _delete_chroma_ids(collection, existing_ids)
        return

    if vectors is None:
        vectors = embed_splits(splits, embeddings)
    generation = uuid.uuid4().hex
    ids = [
        _chroma_record_id(
            kb_name,
            source_filename,
            position,
            split,
            generation=generation,
        )
        for position, split in enumerate(splits)
    ]
    metadatas = [chroma_metadata_for_split(split, kb_name, source_filename) for split in splits]
    batch_size = _chroma_batch_size()
    staged_ids: list[str] = []
    try:
        for offset in range(0, len(splits), batch_size):
            current_ids = ids[offset : offset + batch_size]
            staged_ids.extend(current_ids)
            collection.upsert(
                ids=current_ids,
                embeddings=vectors[offset : offset + batch_size],
                documents=[split.page_content for split in splits[offset : offset + batch_size]],
                metadatas=metadatas[offset : offset + batch_size],
            )
    except Exception:
        # 暂存 ID 带有新代际标识，清理失败写入不会误删此前可用的源文件版本。
        try:
            _delete_chroma_ids(collection, staged_ids)
        except Exception as cleanup_exc:
            logger.warning("Chroma staged-vector cleanup failed: %s", cleanup_exc)
        raise

    _delete_chroma_ids(collection, existing_ids)
    logger.info(
        "Chroma upsert [%s/%s] %d vectors", kb_name, source_filename, len(splits)
    )


def chroma_delete_by_source(kb_name: str, source_filename: str) -> None:
    """删除一个源文件的向量记录；失败由持久化清理任务继续处理。"""
    collection = get_chroma_collection(create=False)
    if collection is not None:
        collection.delete(where=_source_filter(kb_name, source_filename))


def chroma_delete_by_knowledge_base(kb_name: str) -> None:
    """删除一个知识库的全部向量记录；Chroma 不可用时让调用方中止操作。"""
    collection = get_chroma_collection(create=False)
    if collection is not None:
        collection.delete(where={"kb_name": kb_name})


def chroma_delete_stale_sources(kb_name: str, active_sources: set[str]) -> int:
    """全量重建成功暂存后，删除不再属于活跃源文件的旧向量。"""
    collection = get_chroma_collection(create=False)
    if collection is None:
        return 0
    stale_ids: list[str] = []
    for ids, records in _iter_chroma_batches(collection, {"kb_name": kb_name}, ["metadatas"]):
        metadatas = list(records.get("metadatas") or [])
        stale_ids.extend(
            record_id
            for record_id, metadata in zip(ids, metadatas)
            if str((metadata or {}).get("source", "")) not in active_sources
        )
    _delete_chroma_ids(collection, stale_ids)
    return len(stale_ids)


def reset_empty_chroma_storage(
    has_documents: Callable[[], bool], clear_cache: Callable[[], None] | None = None
) -> bool:
    """仅当系统中不再有任何源文件时，物理重置 Chroma 存储。"""
    global _chroma_client
    if has_documents():
        return False

    with _chroma_storage_lock:
        client = _chroma_client
        try:
            if client is not None:
                client.close()
                client.clear_system_cache()
            _chroma_client = None
            if clear_cache is not None:
                clear_cache()
            if os.path.isdir(CHROMA_PATH):
                shutil.rmtree(CHROMA_PATH)
            return True
        except Exception as exc:
            logger.warning("Chroma storage reset failed: %s", exc)
            return False


def rename_chroma_knowledge_base(old_name: str, new_name: str) -> None:
    """知识库改名时仅更新 Chroma 元数据，无需重新计算向量。"""
    collection = get_chroma_collection(create=False)
    if collection is None:
        return
    batch_size = _chroma_batch_size()
    while True:
        records = collection.get(
            where={"kb_name": old_name}, include=["metadatas"], limit=batch_size
        )
        ids = records.get("ids") or []
        if not ids:
            break
        updated_metadatas = []
        for metadata in records.get("metadatas") or []:
            updated = dict(metadata or {})
            updated["kb_name"] = new_name
            updated_metadatas.append(updated)
        collection.update(ids=ids, metadatas=updated_metadatas)
