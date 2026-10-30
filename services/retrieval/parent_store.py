"""父段落存储：命中子块后用于恢复上下文，位于主 SQLite 数据库。"""

import sqlite3
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager

from services.core import sqlite_connections as application_db
from services.core.settings import get_settings

# (parent_id, source, page, content, contextual_title)
ParentRow = tuple[str, str, str, str, str]

_SQLITE_PARAM_BATCH = 500


def _db_file() -> str:
    """返回当前配置的主数据库路径。"""
    return get_settings().db_file


def parent_chunk_rows(splits: Iterable) -> list[ParentRow]:
    """从内存中的父子切分结果收集去重后的父段落记录。"""
    parents: dict[str, ParentRow] = {}
    for document in splits:
        parent_id = document.metadata.get("parent_id")
        parent_content = document.metadata.get("parent_content")
        if not parent_id or not parent_content:
            continue
        parent_id = str(parent_id)
        parents[parent_id] = (
            parent_id,
            str(document.metadata.get("source", "")),
            str(document.metadata.get("page", document.metadata.get("row", ""))),
            str(parent_content),
            str(document.metadata.get("contextual_title", "")),
        )
    return list(parents.values())


def replace_source(
    kb_id: str,
    source: str,
    rows: list[ParentRow],
    *,
    connect: Callable[[str], AbstractContextManager[sqlite3.Connection]] = application_db.connect,
) -> None:
    """在同一事务中先删除一个文件的旧父段落，再写入新记录。"""
    with connect(_db_file()) as conn:
        conn.execute(
            "DELETE FROM parent_chunks WHERE kb_id = ? AND source = ?", (kb_id, source)
        )
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO parent_chunks "
                "(kb_id, parent_id, source, page, content, contextual_title) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(kb_id, *row) for row in rows],
            )


def delete_source(kb_id: str, source: str) -> None:
    """删除一个文件的全部父段落；记录不存在时不报错。"""
    with application_db.connect(_db_file()) as conn:
        conn.execute(
            "DELETE FROM parent_chunks WHERE kb_id = ? AND source = ?", (kb_id, source)
        )


def delete_stale_sources(kb_id: str, active_sources: set[str]) -> None:
    """删除知识库内不属于 active_sources 的文件的父段落。"""
    with application_db.connect(_db_file()) as conn:
        sources = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT source FROM parent_chunks WHERE kb_id = ?", (kb_id,)
            ).fetchall()
            if row[0] not in active_sources
        ]
        conn.executemany(
            "DELETE FROM parent_chunks WHERE kb_id = ? AND source = ?",
            [(kb_id, source) for source in sources],
        )


def delete_kb(kb_id: str) -> None:
    """删除一个知识库的全部父段落；记录不存在时不报错。"""
    with application_db.connect(_db_file()) as conn:
        conn.execute("DELETE FROM parent_chunks WHERE kb_id = ?", (kb_id,))


def load(kb_id: str, parent_ids: Iterable[str]) -> dict[str, ParentRow]:
    """按 parent_id 批量读取父段落，返回 parent_id 到记录的映射。"""
    ids = list(dict.fromkeys(str(parent_id) for parent_id in parent_ids if parent_id))
    found: dict[str, ParentRow] = {}
    if not ids:
        return found
    with application_db.connect(_db_file()) as conn:
        for offset in range(0, len(ids), _SQLITE_PARAM_BATCH):
            batch = ids[offset : offset + _SQLITE_PARAM_BATCH]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                "SELECT parent_id, source, page, content, contextual_title "
                f"FROM parent_chunks WHERE kb_id = ? AND parent_id IN ({placeholders})",
                (kb_id, *batch),
            ).fetchall()
            for row in rows:
                found[str(row[0])] = (
                    str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])
                )
    return found
