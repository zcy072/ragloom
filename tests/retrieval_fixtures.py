"""Helpers for tests that need a knowledge base, ready documents and indexed chunks."""

import sqlite3

import app_storage
from services.core import config as application_config
from services.retrieval.chunk_index import get_chunk_index

DEFAULT_VECTOR = [1.0, 0.0]


def create_knowledge_base(kb_name: str) -> str:
    """初始化数据库（幂等）并创建知识库，返回其稳定 kb_id。"""
    app_storage.initialize_database()
    if not app_storage.knowledge_bases.exists(kb_name):
        app_storage.knowledge_bases.create(kb_name, 800, 100)
    kb_id = app_storage.knowledge_bases.kb_id_for(kb_name)
    assert kb_id
    return kb_id


def mark_ready(kb_name: str, source: str) -> None:
    """为文件写入一条状态为 ready 的文档记录，使其进入检索白名单。"""
    with sqlite3.connect(app_storage.DB_FILE) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents "
            "(kb_name, filename, path, size, uploaded_at, status) "
            "VALUES (?, ?, ?, ?, ?, 'ready')",
            (kb_name, source, f"/tmp/{source}", 1, application_config.db_now()),
        )


def publish(kb_name: str, source: str, splits: list, vectors: list | None = None) -> str:
    """把分块写入当前注入的 ChunkIndex，返回 kb_id；默认所有分块使用同一向量。"""
    kb_id = create_knowledge_base(kb_name)
    get_chunk_index().publish_source(
        kb_id, source, splits, vectors or [list(DEFAULT_VECTOR) for _ in splits]
    )
    return kb_id
