"""用户分析用例：把调用者的知识库权限转换为统计范围，并协调对话查看与审计。"""

from collections.abc import Sequence

from services.analytics import audit, user_activity
from services.analytics.contracts import UserConversation
from services.chat import lifecycle, source_access
from services.core import sqlite_connections as application_db
from services.core.errors import ResourceNotFoundError
from services.identity import use_cases as identity_use_cases

ANALYTICS_PERMISSION = "用户分析"
VIEW_CONVERSATION_PERMISSION = "查看用户对话"


def visible_knowledge_bases(
    all_knowledge_bases: bool, knowledge_base_ids: Sequence[str]
) -> tuple[str, ...] | None:
    """返回统计时可见的知识库 ID；可读全部知识库时为 None，表示不限。"""
    if all_knowledge_bases:
        return None
    return tuple(dict.fromkeys(knowledge_base_ids))


def view_conversation(
    db_file: str,
    *,
    actor_account_id: int,
    actor_username: str,
    actor_menu_permissions: Sequence[str],
    actor_all_knowledge_bases: bool,
    actor_knowledge_base_ids: Sequence[str],
    user_id: int,
    session_id: str,
) -> UserConversation:
    """读取某个用户的完整会话并写入审计记录。

    与用户管理相同，不能查看权限超出自己的账号。会话不存在、不属于该用户、或调用者
    不能读其知识库时统一返回 404，不暴露会话是否存在。
    审计与读取在同一事务中：审计写入失败时整个请求失败，不会出现“看了但没记录”。
    """
    identity_use_cases.require_manageable_target(
        db_file=db_file,
        actor_menu_permissions=actor_menu_permissions,
        actor_all_knowledge_bases=actor_all_knowledge_bases,
        actor_knowledge_base_ids=actor_knowledge_base_ids,
        user_id=user_id,
    )
    visible = visible_knowledge_bases(actor_all_knowledge_bases, actor_knowledge_base_ids)
    with application_db.connect(db_file) as conn:
        conn.execute("BEGIN IMMEDIATE")
        owner = user_activity.find_session_owner(conn, user_id, session_id)
        if owner is None or (visible is not None and owner[0] not in visible):
            raise ResourceNotFoundError("会话不存在或无权访问", code="chat_session_not_found")
        kb_id, kb_name, title, username = owner
        audit.record(
            conn,
            actor_account_id=actor_account_id,
            actor_username=actor_username,
            action="view_conversation",
            target_account_id=user_id,
            target_username=username,
            detail=f"{kb_name} / {title}",
        )
        return user_activity.load_conversation(
            conn,
            session_id,
            kb_id=kb_id,
            kb_name=kb_name,
            title=title,
            username=username,
            readable=source_access.kb_reader(
                lambda source_id: visible is None or source_id in visible,
                lambda source_id: lifecycle.active_kb(conn, source_id),
            ),
        )
