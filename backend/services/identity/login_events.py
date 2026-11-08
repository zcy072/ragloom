"""登录记录：每次登录尝试保存结果和来源地址，供用户分析页面统计活跃度和失败次数。

写入失败只记录警告，不影响登录本身。用户名按输入原样截断保存，未知用户名的尝试
`account_id` 为空，方便识别对不存在账号的试探。
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Literal

from services.core import config as application_config
from services.core import sqlite_connections as application_db
from services.core.settings import get_settings

logger = logging.getLogger(__name__)

LoginFailureReason = Literal["", "invalid_credentials", "account_not_active", "rate_limited"]
USERNAME_MAX_CHARS = 80
CLIENT_MAX_CHARS = 64


def record(
    db_file: str,
    *,
    username: str,
    success: bool,
    reason: LoginFailureReason = "",
    client: str = "",
) -> None:
    """写入一条登录记录；账号存在时关联其 ID，任何异常都只记警告。"""
    name = " ".join(str(username).split())[:USERNAME_MAX_CHARS]
    try:
        with application_db.connect(db_file) as conn:
            row = conn.execute("SELECT id FROM accounts WHERE username = ?", (name,)).fetchone()
            conn.execute(
                "INSERT INTO login_events (created_at, account_id, username, success, reason, client) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    application_config.db_now(),
                    int(row[0]) if row else None,
                    name,
                    int(success),
                    reason,
                    client[:CLIENT_MAX_CHARS],
                ),
            )
    except Exception as exc:
        logger.warning("Failed to record login event: %s", exc, exc_info=True)


def prune(db_file: str, retention_days: int | None = None) -> int:
    """删除超过保留期的登录记录，返回删除条数。"""
    days = max(1, retention_days or get_settings().login_event_retention_days)
    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    with application_db.connect(db_file) as conn:
        cursor = conn.execute("DELETE FROM login_events WHERE created_at < ?", (cutoff,))
    return cursor.rowcount
