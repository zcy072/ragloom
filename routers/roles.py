"""Role and permission management endpoints."""

from fastapi import APIRouter, Depends, Query

from routers import schemas
from services.core import config as application_config
from services.core.container import database_file
from services.identity import auth as auth_service
from services.identity import use_cases
from services.identity.contracts import RolePage

router = APIRouter(prefix="/api", tags=["roles"])


@router.get("/roles", response_model=schemas.RolePageResponse)
def list_roles(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=10, le=100),
    query: str = Query(default="", max_length=80),
    _principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("角色管理")
    ),
) -> RolePage:
    """分页返回角色及其权限集合。"""
    return use_cases.list_roles(
        database_file(),
        page=page,
        page_size=page_size,
        query=query,
    )


@router.post("/roles", response_model=schemas.SuccessResponse)
def create_role(
    data: schemas.RoleCreateRequest,
    _principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("角色管理")
    ),
) -> dict[str, str]:
    """创建一个初始不带任何权限的角色。"""
    use_cases.create_role(
        db_file=database_file(),
        now=application_config.db_now,
        name=data.name,
        description=data.description,
    )
    return {"status": "success"}


@router.put("/roles/{role_id}", response_model=schemas.SuccessResponse)
def update_role(
    role_id: int,
    data: schemas.RolePermissionsRequest,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("角色管理")
    ),
) -> dict[str, str]:
    """整体替换角色的菜单权限和知识库权限。"""
    use_cases.update_role_permissions(
        db_file=database_file(),
        actor_role_id=principal.role_id,
        actor_menu_permissions=principal.menu_permissions,
        actor_knowledge_base_permissions=principal.knowledge_base_permissions,
        role_id=role_id,
        menu_permissions=data.menu_permissions,
        knowledge_base_permissions=data.knowledge_base_permissions,
    )
    return {"status": "success"}


@router.delete("/roles/{role_id}", response_model=schemas.SuccessResponse)
def delete_role(
    role_id: int,
    _principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("角色管理")
    ),
) -> dict[str, str]:
    """删除未被用户使用且不是系统内置的角色。"""
    use_cases.delete_role(
        db_file=database_file(), role_id=role_id
    )
    return {"status": "success"}
