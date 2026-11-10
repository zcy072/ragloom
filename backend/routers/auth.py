"""Unified account authentication, registration, and password endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response

from routers import schemas
from services.core import config as application_config
from services.core.container import database_file
from services.core.errors import (
    AccessDeniedError,
    AuthenticationError,
    InvalidStateError,
    RateLimitedError,
)
from services.core.settings import get_settings
from services.identity import auth as auth_service
from services.identity import login_events, rate_limit, sessions, use_cases

router = APIRouter(prefix="/api", tags=["auth"])


def client_address(request: Request) -> str:
    """返回限流使用的来源地址；只有显式信任反向代理时才采用 X-Real-IP。

    后端端口若直接对外开放，这个请求头可以被伪造，因此默认不信任；
    经 Compose 内的 Nginx 访问时，Nginx 会用真实连接地址覆盖该请求头。
    """
    if get_settings().auth_trust_x_real_ip:
        forwarded = request.headers.get("x-real-ip", "").strip()
        if forwarded:
            return forwarded[:64]
    return request.client.host if request.client else ""


ClientAddress = Annotated[str, Depends(client_address)]


def public_user(
    *,
    account_id: int,
    username: str,
    name: str,
    role: str,
    menu_permissions: list[str] | tuple[str, ...],
    all_knowledge_bases: bool,
    knowledge_base_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """登录响应与 /me 共用的公开用户字段；显式列出，不序列化账号行或整个 Principal。"""
    return {
        "id": account_id,
        "username": username,
        "name": name,
        "role": role,
        "menuPermissions": list(menu_permissions),
        "allKnowledgeBases": all_knowledge_bases,
        "knowledgeBaseIds": list(knowledge_base_ids),
    }


def _set_session_cookie(response: Response, token: str, max_age_seconds: int) -> None:
    """写入会话 Cookie：HttpOnly、SameSite=Lax、仅 /api 路径、host-only。"""
    response.set_cookie(
        sessions.COOKIE_NAME,
        token,
        max_age=max_age_seconds,
        path=sessions.COOKIE_PATH,
        secure=get_settings().auth_cookie_secure,
        httponly=True,
        samesite="lax",
    )


def _clear_session_cookie(response: Response) -> None:
    """删除会话 Cookie，名称、路径和安全属性与写入时保持一致。"""
    response.delete_cookie(
        sessions.COOKIE_NAME,
        path=sessions.COOKIE_PATH,
        secure=get_settings().auth_cookie_secure,
        httponly=True,
        samesite="lax",
    )


@router.post("/login")
def login(
    data: schemas.LoginRequest,
    request: Request,
    response: Response,
    client: ClientAddress = "",
) -> dict[str, Any]:
    """Authenticate any active account and start a cookie session."""
    response.headers.update(auth_service.NO_STORE_HEADERS)
    db_file = database_file()
    limiter = rate_limit.get_auth_rate_limiter()
    try:
        attempt = limiter.reserve_login(data.username, client)
    except RateLimitedError:
        login_events.record(
            db_file, username=data.username, success=False, reason="rate_limited", client=client
        )
        raise
    try:
        account = use_cases.authenticate_account(
            db_file=db_file,
            username=data.username,
            password=data.password,
        )
    except AuthenticationError:
        # 只统计凭证错误：占用的名额保留，即记为一次失败。
        login_events.record(
            db_file, username=data.username, success=False,
            reason="invalid_credentials", client=client,
        )
        raise
    except AccessDeniedError:
        # 账号待审核等状态说明密码已经正确，不计入失败。
        attempt.succeeded(clear_username=False)
        login_events.record(
            db_file, username=data.username, success=False,
            reason="account_not_active", client=client,
        )
        raise
    except Exception:
        # 数据库等内部错误不是凭证错误，退回名额。
        attempt.succeeded(clear_username=False)
        raise
    settings = get_settings()
    # Cookie 活到绝对上限，空闲过期由服务端判断，续期时不用重写 Cookie。
    cookie_max_age = max(settings.auth_session_ttl_seconds, settings.auth_session_max_age_seconds)
    try:
        # 同一浏览器带来的旧会话在同一事务里删除，避免切换账号后遗留。
        token = sessions.create(
            db_file,
            account_id=account["id"],
            verified_password_hash=account["password"],
            ttl_seconds=settings.auth_session_ttl_seconds,
            max_age_seconds=settings.auth_session_max_age_seconds,
            max_sessions=settings.auth_max_sessions_per_account,
            replaced_token_hash=auth_service.session_digest(request),
        )
    except sessions.CredentialsChangedError as exc:
        # 校验后密码被改、账号被停用或删除：拒绝本次登录，不拿旧密码签发会话。
        attempt.succeeded(clear_username=False)
        login_events.record(
            db_file, username=data.username, success=False,
            reason="invalid_credentials", client=client,
        )
        raise AuthenticationError("用户名或密码错误", code="invalid_credentials") from exc
    except Exception:
        attempt.succeeded(clear_username=False)
        raise
    attempt.succeeded(clear_username=True)
    login_events.record(db_file, username=account["username"], success=True, client=client)
    # 会话已提交才下发 Cookie。
    _set_session_cookie(response, token, cookie_max_age)
    return {
        "status": "success",
        "user": public_user(
            account_id=account["id"],
            username=account["username"],
            name=account["name"],
            role=account["role"],
            menu_permissions=account["menu_permissions"],
            all_knowledge_bases=account["all_knowledge_bases"],
            knowledge_base_ids=account["knowledge_base_ids"],
        ),
    }


@router.post("/logout", status_code=204)
def logout(request: Request) -> Response:
    """删除当前会话并清除 Cookie；无需有效会话，缺失、过期或重复退出都返回 204。"""
    digest = auth_service.session_digest(request)
    if digest:
        sessions.delete(database_file(), digest)
    response = Response(status_code=204, headers=auth_service.NO_STORE_HEADERS)
    _clear_session_cookie(response)
    return response


@router.get("/me")
def current_user(
    response: Response,
    principal: auth_service.Principal = Depends(auth_service.current_principal),
) -> dict[str, Any]:
    """返回当前账号实时的角色与权限，供前端在管理员改角色后刷新菜单。"""
    response.headers.update(auth_service.NO_STORE_HEADERS)
    return public_user(
        account_id=principal.account_id,
        username=principal.username,
        name=principal.name,
        role=principal.role,
        menu_permissions=principal.menu_permissions,
        all_knowledge_bases=principal.all_knowledge_bases,
        knowledge_base_ids=principal.knowledge_base_ids,
    )


@router.post("/register")
def register_user(
    data: schemas.RegistrationRequest, response: Response, client: ClientAddress = ""
) -> dict[str, str]:
    """Submit a new account registration for review."""
    response.headers.update(auth_service.NO_STORE_HEADERS)
    rate_limit.get_auth_rate_limiter().check_registration(client)
    use_cases.register_account(
        db_file=database_file(),
        now=application_config.db_now,
        username=data.username,
        name=data.name,
        phone=data.phone,
        password=data.password,
    )
    return {"status": "success", "message": "注册申请已提交，等待管理员审核"}


@router.post("/change_password")
def change_password(
    data: schemas.PasswordChangeRequest,
    request: Request,
    response: Response,
    principal: auth_service.Principal = Depends(auth_service.current_principal),
) -> dict[str, str]:
    """修改当前账号密码并撤销其全部会话；成功后清除 Cookie，需要重新登录。"""
    response.headers.update(auth_service.NO_STORE_HEADERS)
    limiter = rate_limit.get_auth_rate_limiter()
    release = limiter.reserve_password_change(principal.account_id)
    try:
        use_cases.change_account_password(
            db_file=database_file(),
            account_id=principal.account_id,
            # 鉴权依赖已验证过 Cookie，这里摘要必然有效；事务内还会复查会话是否仍存在。
            session_digest=auth_service.session_digest(request) or "",
            old_password=data.old_password,
            new_password=data.new_password,
        )
    except Exception as exc:
        # 只有原密码错误保留名额（记为一次失败）；新密码不合规等其他错误退回名额。
        if not (isinstance(exc, InvalidStateError) and exc.code == "invalid_current_password"):
            release()
        raise
    limiter.record_password_change_success(principal.account_id)
    _clear_session_cookie(response)
    return {"status": "success", "message": "密码修改成功"}
