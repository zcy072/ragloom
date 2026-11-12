"""父段落存储：命中子块后用于恢复上下文，位于主 SQLite 数据库。"""

from collections.abc import Iterable

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
            str(document.metadata.get("page", "")),
            str(parent_content),
            str(document.metadata.get("contextual_title", "")),
        )
    return list(parents.values())


def add_document_version(kb_id: str, document_id: str, rows: list[ParentRow]) -> set[str]:
    """写入文件新版本的父段落并保留旧版本，返回写入前已有的 parent_id。

    新旧版本的子块在 Milvus 中切换期间都可能被检索到，两版父段落须同时存在；
    调用方在新子块生效后再删除返回的旧 parent_id。新版本的 parent_id 须与旧版本
    不同（切分时以 uuid4 生成），重复时整个写入失败，旧版本不受影响。
    """
    with application_db.connect(_db_file()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = {
            str(row[0])
            for row in conn.execute(
                "SELECT parent_id FROM parent_chunks WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            ).fetchall()
        }
        conn.executemany(
            "INSERT INTO parent_chunks "
            "(kb_id, document_id, parent_id, source, page, content, contextual_title) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(kb_id, document_id, *row) for row in rows],
        )
    return previous


def delete_parents(kb_id: str, document_id: str, parent_ids: Iterable[str]) -> None:
    """删除一个文件中指定 parent_id 的父段落；记录不存在时不报错。"""
    ids = list(dict.fromkeys(str(parent_id) for parent_id in parent_ids))
    if not ids:
        return
    with application_db.connect(_db_file()) as conn:
        for offset in range(0, len(ids), _SQLITE_PARAM_BATCH):
            batch = ids[offset : offset + _SQLITE_PARAM_BATCH]
            placeholders = ",".join("?" for _ in batch)
            conn.execute(
                "DELETE FROM parent_chunks WHERE kb_id = ? AND document_id = ? "
                f"AND parent_id IN ({placeholders})",
                (kb_id, document_id, *batch),
            )


def delete_document(kb_id: str, document_id: str) -> None:
    """删除一个文件的全部父段落；记录不存在时不报错。"""
    with application_db.connect(_db_file()) as conn:
        conn.execute(
            "DELETE FROM parent_chunks WHERE kb_id = ? AND document_id = ?", (kb_id, document_id)
        )


def delete_stale_documents(kb_id: str, active_document_ids: set[str]) -> None:
    """删除知识库内不属于 active_document_ids 的文件的父段落。"""
    with application_db.connect(_db_file()) as conn:
        sources = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT document_id FROM parent_chunks WHERE kb_id = ?", (kb_id,)
            ).fetchall()
            if row[0] not in active_document_ids
        ]
        conn.executemany(
            "DELETE FROM parent_chunks WHERE kb_id = ? AND document_id = ?",
            [(kb_id, document_id) for document_id in sources],
        )


def delete_kb(kb_id: str) -> None:
    """删除一个知识库的全部父段落；记录不存在时不报错。"""
    with application_db.connect(_db_file()) as conn:
        conn.execute("DELETE FROM parent_chunks WHERE kb_id = ?", (kb_id,))


def load(kb_id: str, document_id: str, parent_ids: Iterable[str]) -> dict[str, ParentRow]:
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
                f"FROM parent_chunks WHERE kb_id = ? AND document_id = ? AND parent_id IN ({placeholders})",
                (kb_id, document_id, *batch),
            ).fetchall()
            for row in rows:
                found[str(row[0])] = (
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                )
    return found
