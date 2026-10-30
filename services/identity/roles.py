"""SQLite persistence for roles and their permissions."""

import json
from collections.abc import Callable, Sequence
from typing import cast

from services.core import sqlite_connections as application_db
from services.identity.contracts import RolePage, RoleRecord
from services.identity.pagination import like_pattern, resolve_page_window
from services.identity.permissions import MENU_PERMISSIONS, decode_permission_list

SUPER_ADMIN_ROLE_NAME = "超级管理员"


def _role_from_row(row: tuple[object, ...]) -> RoleRecord:
    """把角色记录转换为对象，并反序列化两类权限列表。"""
    return {
        "id": cast(int, row[0]),
        "name": str(row[1]),
        "description": str(row[2]),
        "menu_permissions": decode_permission_list(row[3]),
        "knowledge_base_permissions": decode_permission_list(row[4]),
    }


def list_roles_page(
    db_file: str, *, page: int, page_size: int, query: str = ""
) -> RolePage:
    """Return one role page, filtered by role name or description."""
    pattern = like_pattern(query)
    where_clause = ""
    parameters: list[object] = []
    if pattern:
        where_clause = "WHERE name LIKE ? ESCAPE '!' OR description LIKE ? ESCAPE '!'"
        parameters = [pattern, pattern]

    with application_db.connect(db_file) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM roles {where_clause}", parameters
            ).fetchone()[0]
        )
        window = resolve_page_window(total, page, page_size)
        rows = conn.execute(
            f"""
            SELECT id, name, description, menu_permissions, knowledge_base_permissions
            FROM roles
            {where_clause}
            ORDER BY id ASC
            LIMIT ? OFFSET ?
            """,
            [*parameters, window.page_size, window.offset],
        ).fetchall()

    return {
        "items": [_role_from_row(row) for row in rows],
        "total": window.total,
        "page": window.page,
        "page_size": window.page_size,
        "total_pages": window.total_pages,
    }


def get_role(db_file: str, role_id: int) -> RoleRecord | None:
    """Return one role with decoded permissions."""
    with application_db.connect(db_file) as conn:
        row = conn.execute(
            "SELECT id, name, description, menu_permissions, knowledge_base_permissions "
            "FROM roles WHERE id = ?",
            (role_id,),
        ).fetchone()
    return _role_from_row(row) if row else None


def create_role(
    db_file: str, now: Callable[[], str], name: str, description: str = ""
) -> None:
    """创建一个空权限角色，供管理员后续配置。"""
    with application_db.connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO roles
            (name, description, menu_permissions, knowledge_base_permissions, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, description, "[]", "[]", now()),
        )


def update_role_permissions(
    db_file: str,
    role_id: int,
    menu_permissions: Sequence[str],
    knowledge_base_permissions: Sequence[str],
) -> None:
    """整体替换指定角色的两类权限集合。"""
    invalid_menu_permissions = set(menu_permissions) - set(MENU_PERMISSIONS)
    if invalid_menu_permissions:
        raise ValueError(
            f"未知菜单权限：{', '.join(sorted(invalid_menu_permissions))}"
        )
    with application_db.connect(db_file) as conn:
        allowed_knowledge_bases = {
            str(row[0]) for row in conn.execute("SELECT name FROM knowledge_bases")
        } | {"全公司"}
        invalid_knowledge_bases = set(knowledge_base_permissions) - allowed_knowledge_bases
        if invalid_knowledge_bases:
            raise ValueError(
                f"未知知识库权限：{', '.join(sorted(invalid_knowledge_bases))}"
            )
        cursor = conn.execute(
            "UPDATE roles SET menu_permissions = ?, knowledge_base_permissions = ? WHERE id = ?",
            (
                json.dumps(list(menu_permissions), ensure_ascii=False),
                json.dumps(list(knowledge_base_permissions), ensure_ascii=False),
                role_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("角色不存在")


def delete_role(db_file: str, role_id: int) -> None:
    """删除未被使用的角色，同时保护系统内置管理员角色。"""
    with application_db.connect(db_file) as conn:
        row = conn.execute(
            "SELECT name FROM roles WHERE id = ?", (role_id,)
        ).fetchone()
        if not row:
            raise ValueError("角色不存在")
        if row[0] == SUPER_ADMIN_ROLE_NAME:
            raise ValueError("超级管理员角色不能删除")
        if conn.execute(
            "SELECT 1 FROM accounts WHERE role_id = ? LIMIT 1", (role_id,)
        ).fetchone():
            raise ValueError("角色仍有用户使用，不能删除")
        conn.execute("DELETE FROM roles WHERE id = ?", (role_id,))
