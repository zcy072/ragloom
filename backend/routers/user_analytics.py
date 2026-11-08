"""User analytics endpoints: activity overview, per-user statistics, audited conversation view."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from routers import schemas
from services.analytics import audit, user_activity
from services.analytics import use_cases as analytics_use_cases
from services.analytics.contracts import (
    ActivityOverview,
    AuditPage,
    UserActivityDetail,
    UserActivityPage,
    UserConversation,
)
from services.core.container import database_file
from services.identity import auth as auth_service

router = APIRouter(prefix="/api/user-analytics", tags=["user-analytics"])

_ANALYTICS = auth_service.require_menu_permission(analytics_use_cases.ANALYTICS_PERMISSION)
_DAYS = Query(default=30, ge=1, le=user_activity.DAYS_MAX)


def _visible(principal: auth_service.Principal) -> tuple[str, ...] | None:
    return analytics_use_cases.visible_knowledge_bases(principal.knowledge_base_permissions)


@router.get("/overview", response_model=schemas.AnalyticsOverviewResponse)
def analytics_overview(
    days: int = _DAYS,
    principal: auth_service.Principal = Depends(_ANALYTICS),
) -> ActivityOverview:
    """返回窗口内的用户规模、活跃度、按天趋势、按小时分布和各知识库使用情况。"""
    return user_activity.overview(database_file(), days=days, visible_kbs=_visible(principal))


@router.get("/users", response_model=schemas.AnalyticsUserPageResponse)
def analytics_users(
    days: int = _DAYS,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    query: str = Query(default="", max_length=80),
    sort: Literal[
        "questions", "last_question_at", "last_login_at", "created_at", "downvotes"
    ] = "questions",
    order: Literal["asc", "desc"] = "desc",
    principal: auth_service.Principal = Depends(_ANALYTICS),
) -> UserActivityPage:
    """分页返回账户及其窗口内的提问、会话、评价和登录统计；提问统计只含可读知识库。"""
    return user_activity.list_users(
        database_file(),
        days=days,
        visible_kbs=_visible(principal),
        page=page,
        page_size=page_size,
        query=query,
        sort=sort,
        descending=order == "desc",
    )


@router.get("/users/{user_id}", response_model=schemas.AnalyticsUserDetailResponse)
def analytics_user_detail(
    user_id: int,
    days: int = _DAYS,
    principal: auth_service.Principal = Depends(_ANALYTICS),
) -> UserActivityDetail:
    """返回单个用户的统计、趋势、最近登录、会话和提问预览。"""
    return user_activity.user_detail(
        database_file(), user_id, days=days, visible_kbs=_visible(principal)
    )


@router.get(
    "/users/{user_id}/sessions/{session_id}",
    response_model=schemas.AnalyticsConversationResponse,
)
def analytics_user_conversation(
    user_id: int,
    session_id: str,
    principal: auth_service.Principal = Depends(_ANALYTICS),
) -> UserConversation:
    """返回用户的完整会话；需要“查看用户对话”权限，每次读取都写入审计记录。"""
    if analytics_use_cases.VIEW_CONVERSATION_PERMISSION not in principal.menu_permissions:
        raise HTTPException(403, "缺少“查看用户对话”权限")
    return analytics_use_cases.view_conversation(
        database_file(),
        actor_account_id=principal.account_id,
        actor_username=principal.username,
        actor_menu_permissions=principal.menu_permissions,
        actor_knowledge_base_permissions=principal.knowledge_base_permissions,
        user_id=user_id,
        session_id=session_id,
    )


@router.get("/audit-log", response_model=schemas.AnalyticsAuditPageResponse)
def analytics_audit_log(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    _principal: auth_service.Principal = Depends(_ANALYTICS),
) -> AuditPage:
    """按时间倒序返回对话查看的审计记录。"""
    return audit.list_records(database_file(), page=page, page_size=page_size)
