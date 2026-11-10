"""SQLite persistence for the single account identity model."""

from collections.abc import Callable
from typing import cast

from services.core import sqlite_connections as application_db
from services.identity import roles, sessions
from services.identity.contracts import AuthenticatedAccount, UserPage, UserSummary
from services.identity.pagination import like_pattern, page_of, resolve_page_window
from services.identity.permissions import decode_permission_list


def _authenticated_account_from_row(row: tuple[object, ...]) -> AuthenticatedAccount:
    """Convert one account/role join into the authentication contract."""
    return {
        "id": cast(int, row[0]),
        "name": str(row[1]),
        "phone": str(row[2]),
        "status": str(row[3]),
        "role_id": cast(int, row[4]) if row[4] is not None else None,
        "password": str(row[5]),
        "username": str(row[6]),
        "role": str(row[7]) if row[7] else "未分配",
        "all_knowledge_bases": bool(row[8]),
        "knowledge_base_ids": [],
        "menu_permissions": decode_permission_list(row[9]),
    }


def _user_summary_from_row(row: tuple[object, ...]) -> UserSummary:
    """Convert one account row into the administration list contract."""
    created_at = str(row[4]) if row[4] else ""
    return {
        "id": cast(int, row[0]),
        "name": str(row[1]),
        "phone": str(row[2]),
        "status": str(row[3]),
        "role": str(row[6]) if row[6] else "未分配",
        "time": created_at[:19],
        "username": str(row[7]),
    }


def find_account_by_username(db_file: str, username: str) -> AuthenticatedAccount | None:
    """Load one account and its current role permissions."""
    with application_db.connect(db_file) as conn:
        row = conn.execute(
            """
            SELECT a.id, a.name, a.phone, a.status, a.role_id, a.password, a.username,
                   r.name, r.all_knowledge_bases, r.menu_permissions
            FROM accounts a LEFT JOIN roles r ON a.role_id = r.id
            WHERE a.username = ?
            """,
            (username,),
        ).fetchone()
        if not row:
            return None
        account = _authenticated_account_from_row(row)
        if account["role_id"] is not None:
            account["knowledge_base_ids"] = roles.role_knowledge_base_ids(
                conn, [account["role_id"]], include_deleting=True
            )[account["role_id"]]
    return account


def register_user(
    db_file: str,
    now: Callable[[], str],
    *,
    username: str,
    name: str,
    phone: str,
    password: str,
) -> bool:
    """Create a pending account when the username is available."""
    with application_db.connect(db_file) as conn:
        if conn.execute("SELECT 1 FROM accounts WHERE username = ?", (username,)).fetchone():
            return False
        conn.execute(
            "INSERT INTO accounts "
            "(username, name, phone, password, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username, name, phone, password, "审核中", now()),
        )
    return True


def find_password_hash(db_file: str, account_id: int) -> str | None:
    """读取账号当前的密码哈希，供写事务外的 bcrypt 校验使用。"""
    with application_db.connect(db_file) as conn:
        row = conn.execute("SELECT password FROM accounts WHERE id = ?", (account_id,)).fetchone()
    return str(row[0]) if row else None


def replace_password(
    db_file: str,
    *,
    account_id: int,
    session_digest: str,
    expected_hash: str,
    new_hash: str,
) -> bool:
    """在一个写事务内条件更新密码并撤销该账号全部会话；凭证已变化时不写入并返回 False。

    bcrypt 校验和新哈希在事务外算好；这里只复查当前会话仍有效、账号仍正常，
    再以旧哈希为条件更新，避免并发改密时后写覆盖先写。任一步失败整体回滚。
    """
    with application_db.connect(db_file) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if sessions.find_account_id(conn, session_digest) != account_id:
            return False
        cursor = conn.execute(
            "UPDATE accounts SET password = ? WHERE id = ? AND password = ? AND status = ?",
            (new_hash, account_id, expected_hash, sessions.ACTIVE_ACCOUNT_STATUS),
        )
        if cursor.rowcount != 1:
            return False
        sessions.delete_for_account(conn, account_id)
    return True


def list_users_page(db_file: str, *, page: int, page_size: int, query: str = "") -> UserPage:
    """Return one user page, filtered by account username or display name."""
    pattern = like_pattern(query)
    where_clause = ""
    parameters: list[object] = []
    if pattern:
        where_clause = "WHERE a.username LIKE ? ESCAPE '!' OR a.name LIKE ? ESCAPE '!'"
        parameters = [pattern, pattern]

    with application_db.connect(db_file) as conn:
        total = int(
            conn.execute(f"SELECT COUNT(*) FROM accounts a {where_clause}", parameters).fetchone()[
                0
            ]
        )
        window = resolve_page_window(total, page, page_size)
        rows = conn.execute(
            f"""
            SELECT a.id, a.name, a.phone, a.status, a.created_at,
                   a.role_id, r.name, a.username
            FROM accounts a
            LEFT JOIN roles r ON a.role_id = r.id
            {where_clause}
            ORDER BY a.created_at DESC, a.id DESC
            LIMIT ? OFFSET ?
            """,
            [*parameters, window.page_size, window.offset],
        ).fetchall()

    return page_of([_user_summary_from_row(row) for row in rows], window)


def find_account_permissions(
    db_file: str, user_id: int
) -> tuple[list[str], bool, list[str]] | None:
    """Return one account's current (menu, all-KBs flag, KB ids) permissions, or None."""
    with application_db.connect(db_file) as conn:
        row = conn.execute(
            """
            SELECT a.role_id, r.menu_permissions, r.all_knowledge_bases
            FROM accounts a LEFT JOIN roles r ON a.role_id = r.id
            WHERE a.id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        kb_ids = roles.role_knowledge_base_ids(conn, [row[0]])[row[0]] if row[0] is not None else []
    return decode_permission_list(row[1]), bool(row[2]), kb_ids


def delete_user(db_file: str, user_id: int) -> None:
    """Delete an account, its chat data, chat run records, and login events.

    Admin audit rows stay: they record who viewed the account and keep the username.
    登录会话由 auth_sessions 外键级联删除。
    """
    with application_db.connect(db_file) as conn:
        if not conn.execute("SELECT 1 FROM accounts WHERE id = ?", (user_id,)).fetchone():
            raise ValueError("用户不存在")
        conn.execute(
            "DELETE FROM chat_session_memory WHERE session_id IN "
            "(SELECT id FROM chat_sessions WHERE account_id = ?)",
            (user_id,),
        )
        conn.execute(
            "DELETE FROM chat_messages WHERE session_id IN "
            "(SELECT id FROM chat_sessions WHERE account_id = ?)",
            (user_id,),
        )
        conn.execute("DELETE FROM chat_sessions WHERE account_id = ?", (user_id,))
        conn.execute("DELETE FROM chat_runs WHERE account_id = ?", (user_id,))
        conn.execute("DELETE FROM chat_traces WHERE account_id = ?", (user_id,))
        conn.execute("DELETE FROM login_events WHERE account_id = ?", (user_id,))
        conn.execute("DELETE FROM accounts WHERE id = ?", (user_id,))


def update_user_role(db_file: str, user_id: int, role_id: int) -> None:
    """Assign an existing role to an existing account."""
    with application_db.connect(db_file) as conn:
        if not conn.execute("SELECT 1 FROM roles WHERE id = ?", (role_id,)).fetchone():
            raise ValueError("角色不存在")
        cursor = conn.execute("UPDATE accounts SET role_id = ? WHERE id = ?", (role_id, user_id))
        if cursor.rowcount != 1:
            raise ValueError("用户不存在")


def update_user_status(db_file: str, user_id: int, status: str) -> None:
    """Update an existing account's authentication status.

    改为非“正常”状态时在同一事务内删除该账号全部会话；恢复“正常”不会复活旧会话。
    """
    with application_db.connect(db_file) as conn:
        cursor = conn.execute("UPDATE accounts SET status = ? WHERE id = ?", (status, user_id))
        if cursor.rowcount != 1:
            raise ValueError("用户不存在")
        if status != sessions.ACTIVE_ACCOUNT_STATUS:
            sessions.delete_for_account(conn, user_id)
