"""聊天短写事务内的知识库与会话生命周期检查。"""

import sqlite3

from services.chat import source_access
from services.core.errors import ResourceNotFoundError


def active_kb(conn: sqlite3.Connection, kb_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM knowledge_bases WHERE kb_id = ? AND state = 'active'",
            (kb_id,),
        ).fetchone()
        is not None
    )


def require_active_kb(conn: sqlite3.Connection, kb_id: str) -> None:
    if not active_kb(conn, kb_id):
        raise ResourceNotFoundError(
            "知识库已删除或正在删除", code="chat_knowledge_base_unavailable"
        )


def require_active_session(conn: sqlite3.Connection, session_id: str) -> str:
    row = conn.execute(
        "SELECT s.kb_id FROM chat_sessions s JOIN knowledge_bases k ON k.kb_id = s.kb_id "
        "WHERE s.id = ? AND k.state = 'active'",
        (session_id,),
    ).fetchone()
    if row is None:
        raise ResourceNotFoundError("会话或知识库已删除", code="chat_session_unavailable")
    return str(row[0])


def active_source_reader(conn: sqlite3.Connection) -> source_access.KbReadable:
    """使用调用方已有写事务，避免清理与来源检查之间穿插删除。"""
    return source_access.kb_reader(lambda _kb_id: True, lambda kb_id: active_kb(conn, kb_id))
