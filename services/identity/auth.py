"""登录令牌签发、验证，以及 FastAPI 路由权限依赖。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import cast

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.core import sqlite_connections as application_db
from services.core.container import database_file
from services.core.settings import get_settings
from services.identity.contracts import AccessTokenPayload
from services.identity.permissions import decode_permission_list

_TOKEN_VERSION = 4
_bearer = HTTPBearer(auto_error=False)

# 签名密钥在进程生命周期内固定不变；按数据库路径缓存，确保测试隔离。
_signing_key_cache: dict[str, bytes] = {}


@dataclass(frozen=True)
class Principal:
    """Authenticated identity with resolved menu and knowledge-base access."""
    username: str
    role: str
    account_id: int = 0
    role_id: int | None = None
    menu_permissions: tuple[str, ...] = ()
    knowledge_base_permissions: tuple[str, ...] = ()

    def can_read_knowledge_base(self, kb_name: str) -> bool:
        """判断主体是否拥有指定知识库的读取权限。"""
        return "全公司" in self.knowledge_base_permissions or (
            kb_name in self.knowledge_base_permissions
        )

    def can_manage_knowledge_bases(self) -> bool:
        """判断主体是否拥有知识库管理菜单权限。"""
        return "知识库管理" in self.menu_permissions


def _urlsafe_encode(value: bytes) -> str:
    """将字节编码为令牌中使用的无填充 URL 安全 Base64 文本。"""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _urlsafe_decode(value: str) -> bytes:
    """把无填充 URL 安全 Base64 文本还原为字节。"""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _signing_key() -> bytes:
    """读取稳定的令牌签名密钥；未配置时生成并持久化到应用设置。"""
    configured = get_settings().auth_secret_key
    if configured:
        return configured.encode("utf-8")
    db_file = database_file()
    cached = _signing_key_cache.get(db_file)
    if cached is not None:
        return cached
    with application_db.connect(db_file) as conn:
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = 'auth_signing_key'"
        ).fetchone()
        if row:
            _signing_key_cache[db_file] = row[0].encode("utf-8")
            return _signing_key_cache[db_file]
        # 持久化随机密钥，避免普通服务重启导致所有用户令牌全部失效。
        # 并发的首个请求可能同时走到这里：OR IGNORE 让后到者不报错，再读回胜出的那一个。
        conn.execute(
            "INSERT OR IGNORE INTO app_settings (key, value) VALUES ('auth_signing_key', ?)",
            (secrets.token_urlsafe(48),),
        )
        stored = conn.execute(
            "SELECT value FROM app_settings WHERE key = 'auth_signing_key'"
        ).fetchone()[0]
    _signing_key_cache[db_file] = stored.encode("utf-8")
    return _signing_key_cache[db_file]


def _credential_fingerprint(value: str) -> str:
    """生成不可逆的凭证指纹，用于判断密码修改后令牌是否仍有效。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def issue_access_token(
    account_id: int,
    username: str,
    credential: str,
) -> str:
    """签发与当前密码凭证绑定的紧凑 HMAC 登录令牌。"""
    now = int(time.time())
    # 令牌中的角色仅作身份提示；每次请求仍从 SQLite 读取权限，使角色修改立即生效。
    payload = {
        "v": _TOKEN_VERSION,
        "aid": account_id,
        "sub": username,
        "iat": now,
        "exp": now + max(60, get_settings().auth_token_ttl_seconds),
        "cred": _credential_fingerprint(credential),
    }
    body = _urlsafe_encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    signature = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_urlsafe_encode(signature)}"


def _decode_access_token(token: str) -> AccessTokenPayload:
    """验证令牌格式、规范 Base64 编码、HMAC 签名和过期时间。"""
    try:
        body, encoded_signature = token.split(".", 1)
        signature = _urlsafe_decode(encoded_signature)
        if _urlsafe_encode(signature) != encoded_signature:
            raise ValueError("non-canonical signature")
        if _urlsafe_encode(_urlsafe_decode(body)) != body:
            raise ValueError("non-canonical payload")
        expected = hmac.new(_signing_key(), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid signature")
        payload = json.loads(_urlsafe_decode(body))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(401, "登录凭证无效") from exc
    if not isinstance(payload, dict):
        raise HTTPException(401, "登录凭证无效")
    if (
        payload.get("v") != _TOKEN_VERSION
        or not isinstance(payload.get("exp"), int)
        or not isinstance(payload.get("sub"), str)
        or not payload["sub"]
        or not isinstance(payload.get("aid"), int)
        or payload["aid"] <= 0
        or not isinstance(payload.get("iat"), int)
        or not isinstance(payload.get("cred"), str)
    ):
        raise HTTPException(401, "登录凭证无效")
    if payload["exp"] <= int(time.time()):
        raise HTTPException(401, "登录凭证已过期")
    return cast(AccessTokenPayload, payload)


def _load_principal(payload: AccessTokenPayload) -> Principal:
    """重新读取账号和角色状态，而不是直接信任令牌中携带的旧权限。"""
    username = payload["sub"]
    account_id = payload["aid"]
    db_file = database_file()
    with application_db.connect(db_file) as conn:
        row = conn.execute(
            """
            SELECT a.password, a.status, a.role_id, r.name,
                   r.menu_permissions, r.knowledge_base_permissions
            FROM accounts a LEFT JOIN roles r ON a.role_id = r.id
            WHERE a.id = ? AND a.username = ?
            """,
            (account_id, username),
        ).fetchone()
    # 对比凭证指纹即可在不维护令牌黑名单表的情况下，让改密后的旧令牌失效。
    if not row or row[1] != "正常" or not hmac.compare_digest(
        _credential_fingerprint(row[0]), str(payload.get("cred", ""))
    ):
        raise HTTPException(401, "登录状态已失效，请重新登录")
    return Principal(
        username=username,
        role=str(row[3] or "未分配"),
        account_id=account_id,
        role_id=int(row[2]) if row[2] is not None else None,
        menu_permissions=tuple(decode_permission_list(row[4])),
        knowledge_base_permissions=tuple(decode_permission_list(row[5])),
    )


def current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> Principal:
    """验证 Bearer 凭证并从数据库加载当前仍有效的访问主体。"""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "请先登录")
    # 签名只能证明令牌来自本服务；后续数据库查询还要确认账号仍启用且登录后没有改密。
    return _load_principal(_decode_access_token(credentials.credentials))


def require_menu_permission(permission: str):
    """生成一个 FastAPI 依赖，用于校验某项菜单权限。"""

    def dependency(principal: Principal = Depends(current_principal)) -> Principal:
        """校验当前主体是否拥有闭包中指定的菜单权限。"""
        if permission not in principal.menu_permissions:
            raise HTTPException(403, f"缺少“{permission}”权限")
        return principal

    dependency.__name__ = f"require_{permission}_permission"
    return dependency


def _path_knowledge_base_name(request: Request) -> str:
    """路由使用了不同的路径参数名；在这里统一解析，保证所有接口都做同一套后端鉴权。"""
    params = request.path_params
    return str(params.get("kb_name") or params.get("name") or params.get("old_name") or "")


def require_knowledge_base_reader(
    request: Request, principal: Principal = Depends(current_principal)
) -> Principal:
    """校验当前主体是否有读取请求路径中知识库的权限。"""
    kb_name = _path_knowledge_base_name(request)
    if not kb_name or not principal.can_read_knowledge_base(kb_name):
        raise HTTPException(403, "无权访问该知识库")
    return principal


def require_knowledge_base_manager(
    request: Request, principal: Principal = Depends(current_principal)
) -> Principal:
    """校验当前主体是否有管理权限，并可访问请求路径中的目标知识库。"""
    if not principal.can_manage_knowledge_bases():
        raise HTTPException(403, "无权管理知识库")
    kb_name = _path_knowledge_base_name(request)
    if kb_name and not principal.can_read_knowledge_base(kb_name):
        raise HTTPException(403, "无权管理该知识库")
    return principal
