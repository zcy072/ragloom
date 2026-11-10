"""Helpers for tests that need a knowledge base, ready documents and indexed chunks."""

import sqlite3

import app_storage
from services.retrieval.chunk_index import get_chunk_index
from tests.support import add_knowledge_bases, kb_id_by_name

DEFAULT_VECTOR = [1.0, 0.0]


def create_knowledge_base(name: str) -> str:
    """初始化数据库（幂等）并创建知识库，返回其稳定 kb_id。名称重复时返回已有的那个。"""
    app_storage.initialize_database()
    try:
        return kb_id_by_name(name)
    except AssertionError:
        return add_knowledge_bases(name)[0]


def common_kb_id() -> str:
    """返回通用知识库的 kb_id；它由 system_key 标识，写入默认数据时创建。"""
    app_storage.initialize_database()
    app_storage.bootstrap_defaults()
    kb_id = app_storage.knowledge_bases.common_kb_id()
    assert kb_id is not None
    return kb_id


def _ensure_document(kb_id: str, document_id: str, filename: str = "fixture.pdf") -> None:
    if not app_storage.documents.document_exists(kb_id, document_id):
        app_storage.documents.create_document_with_job(
            kb_id,
            document_id,
            filename,
            size=1,
            parser_mode="native_text",
            metadata={},
        )


def mark_ready(kb_id: str, document_id: str) -> None:
    """将指定夹具 ID 标为 ready；其他夹具需显式持有不同 ID。"""
    _ensure_document(kb_id, document_id)
    with sqlite3.connect(app_storage.DB_FILE) as conn:
        conn.execute(
            "UPDATE documents SET status = 'ready' WHERE kb_id = ? AND document_id = ?",
            (kb_id, document_id),
        )


def publish(kb_id: str, document_id: str, splits: list, vectors: list | None = None) -> str:
    """显式传入文档 ID，文件名仅来自分块展示元数据。返回知识库 ID。"""
    filename = str(splits[0].metadata.get("source", "fixture.pdf")) if splits else "fixture.pdf"
    _ensure_document(kb_id, document_id, filename)
    for split in splits:
        split.metadata.update(kb_id=kb_id, document_id=document_id)
    get_chunk_index().publish_document(
        kb_id, document_id, splits, vectors or [list(DEFAULT_VECTOR) for _ in splits]
    )
    return kb_id
