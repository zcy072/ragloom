"""User-account management endpoints."""

from fastapi import APIRouter, Depends, Query

from routers import schemas
from services.core.container import get_application_container
from services.identity import auth as auth_service
from services.identity import use_cases

router = APIRouter(prefix="/api", tags=["users"])


@router.delete("/users/{user_id}", response_model=schemas.SuccessResponse)
def delete_user(
    user_id: int,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("用户管理")
    ),
) -> dict[str, str]:
    """删除用户账号及其拥有的聊天历史。"""
    use_cases.delete_user(
        db_file=str(get_application_container().database_path()),
        actor_account_id=principal.account_id,
        user_id=user_id,
    )
    return {"status": "success"}


@router.put("/users/{user_id}/role", response_model=schemas.SuccessResponse)
def update_user_role(
    user_id: int,
    data: schemas.UserRoleRequest,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("用户管理")
    ),
) -> dict[str, str]:
    """为用户账号分配一个已有角色。"""
    use_cases.assign_user_role(
        db_file=str(get_application_container().database_path()),
        actor_account_id=principal.account_id,
        actor_menu_permissions=principal.menu_permissions,
        actor_knowledge_base_permissions=principal.knowledge_base_permissions,
        user_id=user_id,
        role_id=data.role_id,
    )
    return {"status": "success"}


@router.put("/users/{user_id}/status", response_model=schemas.SuccessResponse)
def update_user_status(
    user_id: int,
    data: schemas.UserStatusRequest,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("用户管理")
    ),
) -> dict[str, str]:
    """修改用户账号的审核或启用状态。"""
    use_cases.update_user_status(
        db_file=str(get_application_container().database_path()),
        actor_account_id=principal.account_id,
        user_id=user_id,
        status=data.status,
    )
    return {"status": "success"}


@router.get("/users", response_model=schemas.UserPageResponse)
def list_users(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=10, le=100),
    query: str = Query(default="", max_length=80),
    _principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("用户管理")
    ),
) -> dict:
    """分页返回用户账号列表，并附带当前角色名称。"""
    return use_cases.list_users(
        str(get_application_container().database_path()),
        page=page,
        page_size=page_size,
        query=query,
    )
