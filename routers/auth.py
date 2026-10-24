"""Unified account authentication, registration, and password endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from routers import schemas
from services.core import config as application_config
from services.core.container import get_application_container
from services.core.errors import AuthenticationError
from services.core.settings import get_settings
from services.identity import auth as auth_service
from services.identity import rate_limit, use_cases

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


@router.post("/login")
def login(data: schemas.LoginRequest, client: ClientAddress = "") -> dict[str, Any]:
    """Authenticate any active account through the same credential path."""
    limiter = rate_limit.get_auth_rate_limiter()
    limiter.check_login(data.username, client)
    try:
        account = use_cases.authenticate_account(
            db_file=str(get_application_container().database_path()),
            username=data.username,
            password=data.password,
        )
    except AuthenticationError:
        # 只统计凭证错误；账号待审核等状态说明密码已经正确，不应计入失败。
        limiter.record_login_failure(data.username, client)
        raise
    limiter.record_login_success(data.username)
    return {
        "status": "success",
        "access_token": auth_service.issue_access_token(
            account["id"], account["username"], account["password"]
        ),
        "token_type": "bearer",
        "user": {
            "role": account["role"],
            "username": account["username"],
            "name": account["name"],
            "menuPermissions": account["menu_permissions"],
            "knowledgeBasePermissions": account["knowledge_base_permissions"],
        },
    }


@router.get("/me")
def current_user(
    principal: auth_service.Principal = Depends(auth_service.current_principal),
) -> dict[str, Any]:
    """返回当前账号实时的角色与权限，供前端在管理员改角色后刷新菜单。"""
    return {
        "role": principal.role,
        "username": principal.username,
        "menuPermissions": list(principal.menu_permissions),
        "knowledgeBasePermissions": list(principal.knowledge_base_permissions),
    }


@router.post("/register")
def register_user(
    data: schemas.RegistrationRequest, client: ClientAddress = ""
) -> dict[str, str]:
    """Submit a new account registration for review."""
    rate_limit.get_auth_rate_limiter().check_registration(client)
    use_cases.register_account(
        db_file=str(get_application_container().database_path()),
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
    principal: auth_service.Principal = Depends(auth_service.current_principal),
) -> dict[str, str]:
    """Change the authenticated account's password and invalidate its old token."""
    use_cases.change_account_password(
        db_file=str(get_application_container().database_path()),
        account_id=principal.account_id,
        old_password=data.old_password,
        new_password=data.new_password,
    )
    return {"status": "success", "message": "密码修改成功"}
