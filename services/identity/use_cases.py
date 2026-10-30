"""Framework-independent account and role administration use cases."""

import logging
import sqlite3
from collections.abc import Callable, Sequence

from services.core import config as application_config
from services.core.errors import (
    AccessDeniedError,
    AuthenticationError,
    ConflictError,
    InvalidRequestError,
    InvalidStateError,
    OperationFailedError,
    ResourceNotFoundError,
)
from services.identity import accounts, roles
from services.identity.contracts import RolePage, UserPage

logger = logging.getLogger(__name__)


def authenticate_account(*, db_file: str, username: str, password: str):
    """Load and validate an active account for the supplied credentials."""
    account = accounts.find_account_by_username(db_file, username)
    if account is None:
        application_config.burn_password_check(password)
        raise AuthenticationError("用户名或密码错误", code="invalid_credentials")
    if not application_config.verify_password(password, account["password"]):
        raise AuthenticationError("用户名或密码错误", code="invalid_credentials")
    if account["status"] != "正常":
        raise AccessDeniedError(
            f"账号目前状态: {account['status']} (请联系管理员审核)",
            code="account_not_active",
        )
    return account


def register_account(
    *,
    db_file: str,
    now: Callable[[], str],
    username: str,
    name: str,
    phone: str,
    password: str,
) -> None:
    """Create a pending account while preserving registration error semantics."""
    try:
        application_config.validate_registration_fields(username, phone)
        application_config.validate_new_password(password)
        created = accounts.register_user(
            db_file,
            now,
            username=username,
            name=name,
            phone=phone,
            password=application_config.hash_password(password),
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_registration") from exc
    except sqlite3.IntegrityError:
        # 并发注册同名账号时，查重通过但插入撞上唯一约束。
        created = False
    except Exception as exc:
        logger.exception("Account registration failed")
        raise OperationFailedError("注册失败，请稍后重试", code="registration_failed") from exc
    if not created:
        raise InvalidStateError("该账号名称已被注册", code="username_already_registered")


def change_account_password(
    *, db_file: str, account_id: int, old_password: str, new_password: str
) -> None:
    """Replace an account password only when its current password matches."""
    try:
        changed = accounts.change_password(db_file, account_id, old_password, new_password)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_new_password") from exc
    if not changed:
        raise InvalidStateError("原密码不正确", code="invalid_current_password")


def _can_grant_permissions(
    *,
    actor_menu_permissions: Sequence[str],
    actor_knowledge_base_permissions: Sequence[str],
    menu_permissions: Sequence[str],
    knowledge_base_permissions: Sequence[str],
) -> bool:
    """Return whether an actor may delegate the requested permission sets."""
    exceeds_menu_permissions = set(menu_permissions) - set(actor_menu_permissions)
    exceeds_knowledge_permissions = (
        application_config.ALL_KNOWLEDGE_BASES_PERMISSION not in actor_knowledge_base_permissions
    ) and bool(
        set(knowledge_base_permissions) - set(actor_knowledge_base_permissions)
    )
    return not exceeds_menu_permissions and not exceeds_knowledge_permissions


def list_users(db_file: str, *, page: int, page_size: int, query: str) -> UserPage:
    """Return a paginated account administration view."""
    return accounts.list_users_page(db_file, page=page, page_size=page_size, query=query)


def require_manageable_target(
    *,
    db_file: str,
    actor_menu_permissions: Sequence[str],
    actor_knowledge_base_permissions: Sequence[str],
    user_id: int,
) -> None:
    """Reject managing an account whose permissions exceed the actor's own."""
    target = accounts.find_account_permissions(db_file, user_id)
    if target is None:
        raise ResourceNotFoundError("用户不存在", code="user_not_found")
    menu_permissions, knowledge_base_permissions = target
    if not _can_grant_permissions(
        actor_menu_permissions=actor_menu_permissions,
        actor_knowledge_base_permissions=actor_knowledge_base_permissions,
        menu_permissions=menu_permissions,
        knowledge_base_permissions=knowledge_base_permissions,
    ):
        raise AccessDeniedError(
            "不能管理权限超出当前账号的用户", code="target_permissions_exceed_actor"
        )


def delete_user(
    *,
    db_file: str,
    actor_account_id: int,
    actor_menu_permissions: Sequence[str],
    actor_knowledge_base_permissions: Sequence[str],
    user_id: int,
) -> None:
    """Delete another account and its chat history."""
    if user_id == actor_account_id:
        raise AccessDeniedError("不能删除当前登录账号", code="cannot_delete_self")
    require_manageable_target(
        db_file=db_file,
        actor_menu_permissions=actor_menu_permissions,
        actor_knowledge_base_permissions=actor_knowledge_base_permissions,
        user_id=user_id,
    )
    try:
        accounts.delete_user(db_file, user_id)
    except ValueError as exc:
        raise ResourceNotFoundError(str(exc), code="user_not_found") from exc


def assign_user_role(
    *,
    db_file: str,
    actor_account_id: int,
    actor_menu_permissions: Sequence[str],
    actor_knowledge_base_permissions: Sequence[str],
    user_id: int,
    role_id: int,
) -> None:
    """Assign a role when it is within the actor's permission scope."""
    if user_id == actor_account_id:
        raise AccessDeniedError("不能修改当前账号的角色", code="cannot_change_own_role")
    require_manageable_target(
        db_file=db_file,
        actor_menu_permissions=actor_menu_permissions,
        actor_knowledge_base_permissions=actor_knowledge_base_permissions,
        user_id=user_id,
    )
    target_role = roles.get_role(db_file, role_id)
    if target_role is None:
        raise ResourceNotFoundError("角色不存在", code="role_not_found")
    if not _can_grant_permissions(
        actor_menu_permissions=actor_menu_permissions,
        actor_knowledge_base_permissions=actor_knowledge_base_permissions,
        menu_permissions=target_role["menu_permissions"],
        knowledge_base_permissions=target_role["knowledge_base_permissions"],
    ):
        raise AccessDeniedError("不能授予当前账号不具备的权限", code="permission_delegation_denied")
    try:
        accounts.update_user_role(db_file, user_id, role_id)
    except ValueError as exc:
        raise ResourceNotFoundError(str(exc), code="user_not_found") from exc


def update_user_status(
    *,
    db_file: str,
    actor_account_id: int,
    actor_menu_permissions: Sequence[str],
    actor_knowledge_base_permissions: Sequence[str],
    user_id: int,
    status: str,
) -> None:
    """Update another account's authentication status."""
    if user_id == actor_account_id:
        raise AccessDeniedError("不能修改当前账号的状态", code="cannot_change_own_status")
    require_manageable_target(
        db_file=db_file,
        actor_menu_permissions=actor_menu_permissions,
        actor_knowledge_base_permissions=actor_knowledge_base_permissions,
        user_id=user_id,
    )
    try:
        accounts.update_user_status(db_file, user_id, status)
    except ValueError as exc:
        raise ResourceNotFoundError(str(exc), code="user_not_found") from exc


def list_roles(db_file: str, *, page: int, page_size: int, query: str) -> RolePage:
    """Return a paginated role administration view."""
    return roles.list_roles_page(db_file, page=page, page_size=page_size, query=query)


def create_role(
    *,
    db_file: str,
    now: Callable[[], str],
    name: str,
    description: str,
) -> None:
    """Create an initially unprivileged role."""
    try:
        roles.create_role(db_file, now, name, description)
    except sqlite3.IntegrityError as exc:
        raise InvalidStateError("角色名称已存在", code="role_already_exists") from exc


def update_role_permissions(
    *,
    db_file: str,
    actor_role_id: int | None,
    actor_menu_permissions: Sequence[str],
    actor_knowledge_base_permissions: Sequence[str],
    role_id: int,
    menu_permissions: Sequence[str],
    knowledge_base_permissions: Sequence[str],
) -> None:
    """Replace a role's permissions when the actor may safely delegate them."""
    if role_id == actor_role_id:
        raise AccessDeniedError("不能修改当前账号正在使用的角色", code="cannot_edit_own_role")
    if not _can_grant_permissions(
        actor_menu_permissions=actor_menu_permissions,
        actor_knowledge_base_permissions=actor_knowledge_base_permissions,
        menu_permissions=menu_permissions,
        knowledge_base_permissions=knowledge_base_permissions,
    ):
        raise AccessDeniedError("不能授予当前账号不具备的权限", code="permission_delegation_denied")
    target = roles.get_role(db_file, role_id)
    if target is None:
        raise ResourceNotFoundError("角色不存在", code="role_not_found")
    if target["name"] == roles.SUPER_ADMIN_ROLE_NAME:
        raise AccessDeniedError("超级管理员角色的权限不能修改", code="protected_role")
    if not _can_grant_permissions(
        actor_menu_permissions=actor_menu_permissions,
        actor_knowledge_base_permissions=actor_knowledge_base_permissions,
        menu_permissions=target["menu_permissions"],
        knowledge_base_permissions=target["knowledge_base_permissions"],
    ):
        raise AccessDeniedError(
            "不能修改权限超出当前账号的角色", code="target_permissions_exceed_actor"
        )
    try:
        roles.update_role_permissions(
            db_file, role_id, menu_permissions, knowledge_base_permissions
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_role_permissions") from exc


def delete_role(*, db_file: str, role_id: int) -> None:
    """Delete a role only when no protected state conflicts with the request."""
    try:
        roles.delete_role(db_file, role_id)
    except ValueError as exc:
        raise ConflictError(str(exc), code="role_deletion_conflict") from exc
