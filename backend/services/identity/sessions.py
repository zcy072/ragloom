"""服务端登录会话：随机令牌只交给浏览器 Cookie，SQLite 只保存其 SHA-256 摘要。

令牌是 256 bit 随机值，摘要足以作为查找键，不需要加盐或慢哈希；数据库里的摘要
被当作 Cookie 提交时会再被摘要一次，因此无法直接重放。

组合用的 SQL 辅助函数接收调用方的连接、不自行提交，让“改密/停用与撤销会话”
处于同一个事务；只有 `create`、`delete`、`prune_expired` 自己打开连接。
"""

import hashlib
import logging
import re
import secrets
import sqlite3
import time

from services.core import sqlite_connections as application_db

logger = logging.getLogger(__name__)

COOKIE_NAME = "ragloom_session"
COOKIE_PATH = "/api"
# token_urlsafe(32) 恰好生成 43 个 URL 安全字符。
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
ACTIVE_ACCOUNT_STATUS = "正常"
# 空闲过期时间最多每 5 分钟写一次，避免每个请求都抢 SQLite 写锁。
RENEW_INTERVAL_SECONDS = 300


class CredentialsChangedError(Exception):
    """登录校验后账号密码或状态已变化，本次不应签发会话。"""


def now() -> int:
    """当前 UTC Unix 秒；会话时间统一经过这里，测试可替换为可控时钟。"""
    return int(time.time())


def new_token() -> str:
    """生成一个新的随机会话令牌；从不接受客户端指定的值，以防会话固定。"""
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str | None:
    """返回格式正确的令牌摘要；缺失、超长或格式不符时返回 None。"""
    if not token or len(token) != 43 or not _TOKEN_PATTERN.fullmatch(token):
        return None
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def create(
    db_file: str,
    *,
    account_id: int,
    verified_password_hash: str,
    ttl_seconds: int,
    max_age_seconds: int,
    max_sessions: int,
    replaced_token_hash: str | None = None,
) -> str:
    """在账号仍有效且密码未变时插入新会话，返回原始令牌。

    bcrypt 校验在事务外完成；这里在写事务内重读账号，若期间改密、停用或删除，
    抛出 CredentialsChangedError 而不是用旧密码签发会话。同一浏览器带来的旧会话一并删除。
    ttl_seconds 是空闲过期时间，max_age_seconds 是从登录起的绝对上限；每个账号最多保留
    max_sessions 个会话，超出时删除最早登录的，防止循环登录无限插入会话。
    """
    absolute_ttl = max(ttl_seconds, max_age_seconds)
    token = new_token()
    created_at = now()
    with application_db.connect(db_file) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT password, status FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()
        if (
            not row
            or row[1] != ACTIVE_ACCOUNT_STATUS
            or not secrets.compare_digest(str(row[0]), verified_password_hash)
        ):
            raise CredentialsChangedError
        conn.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (created_at,))
        if replaced_token_hash:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (replaced_token_hash,))
        conn.execute(
            "INSERT INTO auth_sessions "
            "(token_hash, account_id, created_at, expires_at, absolute_expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                token_hash(token),
                account_id,
                created_at,
                created_at + ttl_seconds,
                created_at + absolute_ttl,
            ),
        )
        conn.execute(
            """
            DELETE FROM auth_sessions
            WHERE account_id = ? AND rowid NOT IN (
                SELECT rowid FROM auth_sessions WHERE account_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?
            )
            """,
            (account_id, account_id, max(1, max_sessions)),
        )
    return token


def renew(db_file: str, digest: str, expires_at: int, ttl_seconds: int) -> None:
    """把空闲过期时间推到现在加 ttl_seconds，不超过绝对上限；距上次续期不足间隔时不写。

    只等锁 1 秒：续期失败只记日志，会话仍按原过期时间有效，不应拖慢或打断本次请求。
    """
    current = now()
    if current + ttl_seconds - expires_at < RENEW_INTERVAL_SECONDS:
        return
    try:
        with application_db.connect(db_file, timeout=1.0) as conn:
            conn.execute(
                "UPDATE auth_sessions SET expires_at = MIN(?, absolute_expires_at) "
                "WHERE token_hash = ? AND expires_at > ?",
                (current + ttl_seconds, digest, current),
            )
    except sqlite3.OperationalError as exc:
        logger.warning("Session renewal skipped: %s", exc)


def find_account_id(conn: sqlite3.Connection, digest: str) -> int | None:
    """返回未过期会话所属的账号 ID；过期判断不依赖清理任务是否运行过。"""
    row = conn.execute(
        "SELECT account_id FROM auth_sessions WHERE token_hash = ? AND expires_at > ?",
        (digest, now()),
    ).fetchone()
    return int(row[0]) if row else None


def delete(db_file: str, digest: str) -> None:
    """删除一个会话；不存在时静默成功，使重复退出保持幂等。"""
    with application_db.connect(db_file) as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (digest,))


def delete_for_account(conn: sqlite3.Connection, account_id: int) -> None:
    """在调用方事务内删除账号的全部会话；不另开连接，也不提交。"""
    conn.execute("DELETE FROM auth_sessions WHERE account_id = ?", (account_id,))


def prune_expired(db_file: str) -> int:
    """物理删除已过期的会话，返回删除条数。"""
    with application_db.connect(db_file) as conn:
        cursor = conn.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ?", (now(),)
        )
    return cursor.rowcount
