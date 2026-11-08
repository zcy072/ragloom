"""Knowledge-base level settings stored in the knowledge_bases table."""

from collections.abc import Callable
from typing import cast

from services.core import sqlite_connections as application_db
from services.core.contracts import KnowledgeBaseConfig


class KnowledgeBaseSettings:
    """Thin accessor for per-KB configuration flags stored in the knowledge_bases table.

    Kept separate from DocumentStore so that the documents layer doesn't have
    to know about chat or retrieval configuration concerns.
    """

    def __init__(
        self,
        db_file: Callable[[], str],
        validate_kb_name: Callable[[str], str],
    ) -> None:
        """保存延迟解析的数据库路径和名称校验依赖。"""
        self._db_file = db_file
        self._validate_kb_name = validate_kb_name

    def query_rewrite_enabled(self, kb_name: str) -> bool:
        """返回指定知识库是否启用追问查询改写。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT query_rewrite_enabled FROM knowledge_bases WHERE name = ?",
                (kb_name,),
            ).fetchone()
        if not row:
            raise ValueError("知识库不存在")
        return bool(row[0])

    def set_query_rewrite_enabled(self, kb_name: str, enabled: bool) -> bool:
        """持久化指定知识库的查询改写开关。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                "UPDATE knowledge_bases SET query_rewrite_enabled = ? WHERE name = ?",
                (1 if enabled else 0, kb_name),
            )
        if cursor.rowcount != 1:
            raise ValueError("知识库不存在")
        return bool(enabled)

    def chunk_config(
        self, kb_name: str, *, default_size: int, default_overlap: int
    ) -> KnowledgeBaseConfig:
        """返回指定知识库当前生效的文档分块配置。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT chunk_size, chunk_overlap FROM knowledge_bases WHERE name = ?",
                (kb_name,),
            ).fetchone()
        if not row:
            raise ValueError("知识库不存在")
        return KnowledgeBaseConfig(
            # 重叠长度 0 是合法配置（不重叠），只有 NULL 才表示尚未设置。
            cast(int, row[0]) if row[0] is not None else default_size,
            cast(int, row[1]) if row[1] is not None else default_overlap,
        )

    def set_chunk_config(self, kb_name: str, chunk_size: int, chunk_overlap: int) -> None:
        """保存指定知识库的文档分块设置。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                "UPDATE knowledge_bases SET chunk_size = ?, chunk_overlap = ? WHERE name = ?",
                (chunk_size, chunk_overlap, kb_name),
            )
        if cursor.rowcount != 1:
            raise ValueError("知识库不存在")
