"""管理审计：记录管理员查看用户对话正文等敏感操作。

审计记录与被查看的账户分开保存，删除账户时不删除，以便事后追查。
审计写入与读取在同一事务中完成：写不进审计时不返回对话内容。
"""

import sqlite3
from typing import Literal

from services.analytics.contracts import AuditPage, AuditRecord
from services.core import config as application_config
from services.core import sqlite_connections as application_db
from services.identity.pagination import resolve_page_window

AuditAction = Literal["view_conversation"]
DETAIL_MAX_CHARS = 300


def record(
    conn: sqlite3.Connection,
    *,
    actor_account_id: int,
    actor_username: str,
    action: AuditAction,
    target_account_id: int,
    target_username: str,
    detail: str,
) -> None:
    """在调用方的事务中写入一条审计记录。"""
    conn.execute(
        "INSERT INTO admin_audit_log (created_at, actor_account_id, actor_username, action, "
        "target_account_id, target_username, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            application_config.db_now(),
            actor_account_id,
            actor_username,
            action,
            target_account_id,
            target_username,
            detail[:DETAIL_MAX_CHARS],
        ),
    )


def list_records(db_file: str, *, page: int, page_size: int) -> AuditPage:
    """按时间倒序分页返回审计记录。"""
    with application_db.connect(db_file) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM admin_audit_log").fetchone()[0])
        window = resolve_page_window(total, page, page_size)
        rows = conn.execute(
            "SELECT id, created_at, actor_username, action, target_username, detail "
            "FROM admin_audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
            (window.page_size, window.offset),
        ).fetchall()
    items: list[AuditRecord] = [
        {
            "id": int(row[0]),
            "created_at": str(row[1]),
            "actor_username": str(row[2]),
            "action": str(row[3]),
            "target_username": str(row[4]),
            "detail": str(row[5]),
        }
        for row in rows
    ]
    return {
        "items": items,
        "total": window.total,
        "page": window.page,
        "page_size": window.page_size,
        "total_pages": window.total_pages,
    }
