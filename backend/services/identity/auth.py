"""Cookie 会话鉴权、写请求来源校验，以及 FastAPI 路由权限依赖。"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from services.core import sqlite_connections as application_db
from services.core.config import validate_kb_id
from services.core.container import database_file, get_application_container
from services.core.errors import AccessDeniedError
from services.core.settings import get_settings
from services.identity import roles, sessions
from services.identity.permissions import decode_permission_list

# 认证相关响应不允许被浏览器或代理缓存。
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
WRITE_HEADER_NAME = "X-RAGLoom-Request"
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


@dataclass(frozen=True)
class Principal:
    """Authenticated identity with resolved menu and knowledge-base access."""

    username: str
    role: str
    account_id: int = 0
    role_id: int | None = None
    menu_permissions: tuple[str, ...] = ()
    all_knowledge_bases: bool = False
    knowledge_base_ids: tuple[str, ...] = ()
    name: str = ""

    def can_read_knowledge_base(self, kb_id: str) -> bool:
        """判断主体是否拥有指定知识库的读取权限。"""
        return self.all_knowledge_bases or kb_id in self.knowledge_base_ids

    def can_manage_knowledge_bases(self) -> bool:
        """判断主体是否拥有知识库管理菜单权限。"""
        return "知识库管理" in self.menu_permissions


def require_api_write_header(request: Request) -> None:
    """全局 CSRF 防护：/api 下的非安全方法必须带固定请求头 X-RAGLoom-Request: 1。

    跨站表单无法设置自定义请求头，跨源脚本设置它必须先预检，而服务端不批准跨源预检。
    该头不是秘密；直接读取原始请求头，避免缺头先被声明式校验转成 422。
    登录、注册、退出也受此约束，防止登录 CSRF。
    """
    path = request.url.path
    if request.method in _SAFE_METHODS or not (path == "/api" or path.startswith("/api/")):
        return
    if request.headers.get(WRITE_HEADER_NAME) != "1":
        raise AccessDeniedError("请求缺少来源校验头，已拒绝", code="csrf_rejected")


def session_digest(request: Request) -> str | None:
    """返回请求 Cookie 中会话令牌的摘要；缺失或格式不符时返回 None。"""
    return sessions.token_hash(request.cookies.get(sessions.COOKIE_NAME, ""))


def _unauthenticated(message: str) -> HTTPException:
    """未登录或会话失效的 401；不附带删除 Cookie，避免晚到的旧请求清掉新登录。"""
    return HTTPException(401, message, headers=NO_STORE_HEADERS)


def _load_principal(digest: str) -> Principal | None:
    """在一个连接内按会话摘要读取账号、角色和知识库权限；会话无效时返回 None。

    权限每次请求实时读取，角色修改无需撤销会话即可在下一请求生效。
    有效会话顺带滑动续期，持续使用的用户不会在空闲时限到点时被登出。
    数据库异常照常抛出，不能被当作未登录。
    """
    with application_db.connect(database_file()) as conn:
        row = conn.execute(
            """
            SELECT a.id, a.username, a.name, a.role_id, r.name,
                   r.menu_permissions, r.all_knowledge_bases, s.expires_at
            FROM auth_sessions s
            JOIN accounts a ON a.id = s.account_id
            LEFT JOIN roles r ON a.role_id = r.id
            WHERE s.token_hash = ? AND s.expires_at > ? AND a.status = ?
            """,
            (digest, sessions.now(), sessions.ACTIVE_ACCOUNT_STATUS),
        ).fetchone()
        if not row:
            return None
        kb_ids = (
            roles.role_knowledge_base_ids(conn, [row[3]], include_deleting=True)[row[3]]
            if row[3] is not None
            else []
        )
    sessions.renew(database_file(), digest, int(row[7]), get_settings().auth_session_ttl_seconds)
    return Principal(
        username=str(row[1]),
        role=str(row[4] or "未分配"),
        account_id=int(row[0]),
        role_id=int(row[3]) if row[3] is not None else None,
        menu_permissions=tuple(decode_permission_list(row[5])),
        all_knowledge_bases=bool(row[6]),
        knowledge_base_ids=tuple(kb_ids),
        name=str(row[2] or ""),
    )


def current_principal(request: Request) -> Principal:
    """只从 ragloom_session Cookie 读取会话，并加载当前仍有效的访问主体。"""
    if sessions.COOKIE_NAME not in request.cookies:
        raise _unauthenticated("请先登录")
    digest = session_digest(request)
    principal = _load_principal(digest) if digest else None
    if principal is None:
        raise _unauthenticated("登录状态已失效，请重新登录")
    return principal


def require_menu_permission(permission: str):
    """生成一个 FastAPI 依赖，用于校验某项菜单权限。"""

    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        """校验当前主体是否拥有闭包中指定的菜单权限。"""
        if permission not in principal.menu_permissions:
            raise HTTPException(403, f"缺少“{permission}”权限")
        return principal

    dependency.__name__ = f"require_{permission}_permission"
    return dependency


def _path_knowledge_base_id(request: Request) -> str:
    """所有知识库路径参数统一叫 kb_id；规范化失败的值一律视为无权访问。"""
    try:
        return validate_kb_id(str(request.path_params.get("kb_id") or ""))
    except ValueError:
        return ""


def require_knowledge_base_reader(
    request: Request, principal: Principal = Depends(current_principal)
) -> Principal:
    """校验当前主体是否有读取请求路径中知识库的权限。"""
    kb_id = _path_knowledge_base_id(request)
    if not kb_id or not principal.can_read_knowledge_base(kb_id):
        raise HTTPException(403, "无权访问该知识库")
    if not get_application_container().knowledge_bases.exists(kb_id):
        raise HTTPException(404, "知识库不存在")
    return principal


def require_knowledge_base_manager(
    request: Request, principal: Principal = Depends(current_principal)
) -> Principal:
    """校验当前主体是否有管理权限，并可访问请求路径中的目标知识库。"""
    if not principal.can_manage_knowledge_bases():
        raise HTTPException(403, "无权管理知识库")
    kb_id = _path_knowledge_base_id(request)
    if kb_id and not principal.can_read_knowledge_base(kb_id):
        raise HTTPException(403, "无权管理该知识库")
    if (
        kb_id
        and request.method != "DELETE"
        and not get_application_container().knowledge_bases.exists(kb_id)
    ):
        raise HTTPException(404, "知识库不存在")
    return principal
