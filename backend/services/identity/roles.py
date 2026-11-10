"""SQLite persistence for roles and their permissions."""

import json
import sqlite3
from collections.abc import Callable, Sequence
from typing import cast

from services.core import sqlite_connections as application_db
from services.identity.contracts import RolePage, RoleRecord
from services.identity.pagination import like_pattern, page_of, resolve_page_window
from services.identity.permissions import MENU_PERMISSIONS, decode_permission_list

SUPER_ADMIN_ROLE_NAME = "超级管理员"


def role_knowledge_base_ids(
    conn: sqlite3.Connection, role_ids: Sequence[int], *, include_deleting: bool = False
) -> dict[int, list[str]]:
    """批量读取角色的显式知识库授权；正在删除的库不计入。"""
    if not role_ids:
        return {}
    placeholders = ",".join("?" for _ in role_ids)
    state_clause = "" if include_deleting else "AND k.state = 'active' "
    rows = conn.execute(
        f"SELECT rk.role_id, rk.kb_id FROM role_knowledge_bases rk "
        f"JOIN knowledge_bases k ON k.kb_id = rk.kb_id "
        f"{state_clause}"
        f"WHERE rk.role_id IN ({placeholders}) ORDER BY rk.kb_id",
        list(role_ids),
    ).fetchall()
    result: dict[int, list[str]] = {role_id: [] for role_id in role_ids}
    for role_id, kb_id in rows:
        result[int(role_id)].append(str(kb_id))
    return result


def _roles_from_rows(
    conn: sqlite3.Connection, rows: Sequence[tuple[object, ...]]
) -> list[RoleRecord]:
    """把角色记录转换为对象，并附上显式授权的知识库 ID。"""
    kb_ids = role_knowledge_base_ids(conn, [cast(int, row[0]) for row in rows])
    return [
        {
            "id": cast(int, row[0]),
            "name": str(row[1]),
            "description": str(row[2]),
            "menu_permissions": decode_permission_list(row[3]),
            "all_knowledge_bases": bool(row[4]),
            "knowledge_base_ids": kb_ids[cast(int, row[0])],
        }
        for row in rows
    ]


def list_roles_page(db_file: str, *, page: int, page_size: int, query: str = "") -> RolePage:
    """Return one role page, filtered by role name or description."""
    pattern = like_pattern(query)
    where_clause = ""
    parameters: list[object] = []
    if pattern:
        where_clause = "WHERE name LIKE ? ESCAPE '!' OR description LIKE ? ESCAPE '!'"
        parameters = [pattern, pattern]

    with application_db.connect(db_file) as conn:
        total = int(
            conn.execute(f"SELECT COUNT(*) FROM roles {where_clause}", parameters).fetchone()[0]
        )
        window = resolve_page_window(total, page, page_size)
        rows = conn.execute(
            f"""
            SELECT id, name, description, menu_permissions, all_knowledge_bases
            FROM roles
            {where_clause}
            ORDER BY id ASC
            LIMIT ? OFFSET ?
            """,
            [*parameters, window.page_size, window.offset],
        ).fetchall()
        records = _roles_from_rows(conn, rows)

    return page_of(records, window)


def get_role(db_file: str, role_id: int) -> RoleRecord | None:
    """Return one role with decoded permissions."""
    with application_db.connect(db_file) as conn:
        row = conn.execute(
            "SELECT id, name, description, menu_permissions, all_knowledge_bases "
            "FROM roles WHERE id = ?",
            (role_id,),
        ).fetchone()
        return _roles_from_rows(conn, [row])[0] if row else None


def create_role(db_file: str, now: Callable[[], str], name: str, description: str = "") -> None:
    """创建一个空权限角色，供管理员后续配置。"""
    with application_db.connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO roles
            (name, description, menu_permissions, all_knowledge_bases, created_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (name, description, "[]", now()),
        )


def update_role_permissions(
    db_file: str,
    role_id: int,
    menu_permissions: Sequence[str],
    all_knowledge_bases: bool,
    knowledge_base_ids: Sequence[str],
) -> None:
    """整体替换指定角色的菜单权限和知识库授权；授权只接受存在且未在删除的知识库。"""
    invalid_menu_permissions = set(menu_permissions) - set(MENU_PERMISSIONS)
    if invalid_menu_permissions:
        raise ValueError(f"未知菜单权限：{', '.join(sorted(invalid_menu_permissions))}")
    requested = list(dict.fromkeys(knowledge_base_ids))
    with application_db.connect(db_file) as conn:
        active_ids = {
            str(row[0])
            for row in conn.execute("SELECT kb_id FROM knowledge_bases WHERE state = 'active'")
        }
        invalid_knowledge_bases = set(requested) - active_ids
        if invalid_knowledge_bases:
            raise ValueError("包含不存在的知识库")
        cursor = conn.execute(
            "UPDATE roles SET menu_permissions = ?, all_knowledge_bases = ? WHERE id = ?",
            (
                json.dumps(list(menu_permissions), ensure_ascii=False),
                1 if all_knowledge_bases else 0,
                role_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("角色不存在")
        conn.execute("DELETE FROM role_knowledge_bases WHERE role_id = ?", (role_id,))
        conn.executemany(
            "INSERT INTO role_knowledge_bases (role_id, kb_id) VALUES (?, ?)",
            [(role_id, kb_id) for kb_id in requested],
        )


def delete_role(db_file: str, role_id: int) -> None:
    """删除未被使用的角色，同时保护系统内置管理员角色。"""
    with application_db.connect(db_file) as conn:
        row = conn.execute("SELECT name FROM roles WHERE id = ?", (role_id,)).fetchone()
        if not row:
            raise ValueError("角色不存在")
        if row[0] == SUPER_ADMIN_ROLE_NAME:
            raise ValueError("超级管理员角色不能删除")
        if conn.execute("SELECT 1 FROM accounts WHERE role_id = ? LIMIT 1", (role_id,)).fetchone():
            raise ValueError("角色仍有用户使用，不能删除")
        conn.execute("DELETE FROM roles WHERE id = ?", (role_id,))
