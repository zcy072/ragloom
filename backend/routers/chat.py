"""Chat sessions and streaming-answer HTTP endpoints."""

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from routers import schemas
from services.chat import service as chat_service
from services.chat import sessions as chat_sessions
from services.chat import source_access, web_search
from services.chat import trace as chat_trace
from services.chat.contracts import ChatRequest
from services.chat.persistence_contracts import ChatHistoryMessage, ChatSessionSummary
from services.chat.sessions import SessionOwner
from services.chat.trace import ChatTracePayload
from services.core import config as application_config
from services.core.container import ApplicationContainer, get_application_container
from services.identity import auth as auth_service
from services.identity import rate_limit

router = APIRouter(prefix="/api", tags=["chat"])


def _session_owner(principal: auth_service.Principal) -> SessionOwner:
    """把认证主体转换为聊天持久化层使用的稳定账号标识。"""
    return SessionOwner(principal.account_id, principal.username)


def _can_read_common(principal: auth_service.Principal, container: ApplicationContainer) -> bool:
    """判断提问者能否读取通用知识库；通用库由 system_key 识别。"""
    common_id = container.knowledge_bases.common_kb_id()
    return bool(common_id) and principal.can_read_knowledge_base(common_id)


def _source_reader(
    principal: auth_service.Principal, container: ApplicationContainer
) -> source_access.KbReadable:
    """来源库必须仍为 active 且主体有读取权限，响应内的结果会被缓存。"""
    return source_access.kb_reader(
        principal.can_read_knowledge_base, container.knowledge_bases.exists
    )


@router.get("/chat/sessions")
def get_chat_sessions(
    principal: auth_service.Principal = Depends(auth_service.require_menu_permission("问答对话")),
    container: ApplicationContainer = Depends(get_application_container),
) -> list[ChatSessionSummary]:
    """返回当前用户拥有且仍有知识库访问权限的聊天会话。"""
    sessions = chat_sessions.list_owned_sessions(_session_owner(principal), container=container)
    return [session for session in sessions if principal.can_read_knowledge_base(session["kb_id"])]


@router.get("/chat/sessions/{session_id}")
def get_chat_session_messages(
    session_id: str,
    principal: auth_service.Principal = Depends(auth_service.require_menu_permission("问答对话")),
    container: ApplicationContainer = Depends(get_application_container),
) -> list[ChatHistoryMessage]:
    """返回当前用户拥有的指定会话消息记录。"""
    kb_id, messages = chat_sessions.load_owned_session_messages(
        session_id, _session_owner(principal), container=container
    )
    if not principal.can_read_knowledge_base(kb_id):
        raise HTTPException(403, "无权访问该知识库")
    readable = _source_reader(principal, container)
    return [
        {**message, "sources": source_access.filter_message_sources(message["sources"], readable)}
        for message in messages
    ]


@router.delete("/chat/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    principal: auth_service.Principal = Depends(auth_service.require_menu_permission("问答对话")),
    container: ApplicationContainer = Depends(get_application_container),
) -> dict[str, str]:
    """删除当前用户的指定聊天会话及其全部消息。"""
    owner = _session_owner(principal)
    kb_id = chat_sessions.get_owned_session_kb(session_id, owner, container=container)
    if not principal.can_read_knowledge_base(kb_id):
        raise HTTPException(403, "无权访问该知识库")
    chat_sessions.delete_owned_session(session_id, owner, container=container)
    return {"status": "success"}


@router.put("/chat/sessions/{session_id}/title", response_model=schemas.SuccessResponse)
def rename_chat_session(
    session_id: str,
    data: schemas.ChatSessionTitleRequest,
    principal: auth_service.Principal = Depends(auth_service.require_menu_permission("问答对话")),
    container: ApplicationContainer = Depends(get_application_container),
) -> dict[str, str]:
    """修改当前用户拥有的指定会话的标题。"""
    owner = _session_owner(principal)
    kb_id = chat_sessions.get_owned_session_kb(session_id, owner, container=container)
    if not principal.can_read_knowledge_base(kb_id):
        raise HTTPException(403, "无权访问该知识库")
    chat_sessions.rename_owned_session(session_id, owner, data.title, container=container)
    return {"status": "success"}


@router.post("/chat")
async def stream_chat_answer(
    req: ChatRequest,
    principal: auth_service.Principal = Depends(auth_service.require_menu_permission("问答对话")),
    container: ApplicationContainer = Depends(get_application_container),
) -> StreamingResponse:
    """针对指定问题和知识库，以 SSE 方式流式返回 RAG 回答。"""
    try:
        req.kb_id = application_config.validate_kb_id(req.kb_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not principal.can_read_knowledge_base(req.kb_id):
        raise HTTPException(403, "无权访问该知识库")
    if req.web_search and web_search.WEB_SEARCH_PERMISSION not in principal.menu_permissions:
        raise HTTPException(403, "当前角色没有联网搜索权限")
    release = rate_limit.get_chat_request_limiter().acquire(principal.account_id)
    try:
        stream = await chat_service.chat(
            req,
            _session_owner(principal),
            container,
            include_common=_can_read_common(principal, container),
        )
    except BaseException:
        release()
        raise

    async def events():
        # 回答流结束、出错或客户端断开时都释放名额。
        try:
            async for event in chat_service.with_heartbeat(
                stream.events, application_config.CHAT_SSE_HEARTBEAT_SECONDS
            ):
                yield event
        finally:
            release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        background=BackgroundTask(stream.refresh_summary),
    )


@router.get("/chat/traces/{trace_id}")
def get_chat_trace(
    trace_id: str,
    principal: auth_service.Principal = Depends(auth_service.current_principal),
    container: ApplicationContainer = Depends(get_application_container),
) -> ChatTracePayload:
    """返回一次问答的追踪记录：提问者本人或有运行监控权限的用户可看，且须能读该知识库。"""
    stored = chat_trace.load_trace(trace_id, container=container)
    is_owner = (
        stored.account_id == principal.account_id and "问答对话" in principal.menu_permissions
    )
    # 无权查看时与不存在返回同样的结果，不暴露他人的追踪 ID 是否有效。
    if not is_owner and "运行监控" not in principal.menu_permissions:
        raise HTTPException(404, "追踪记录不存在或已过期")
    if not principal.can_read_knowledge_base(stored.kb_id):
        raise HTTPException(403, "无权访问该知识库")
    return cast(
        ChatTracePayload,
        source_access.filter_trace_payload(
            cast(dict[str, Any], stored.payload), _source_reader(principal, container)
        ),
    )


@router.put("/chat/traces/{trace_id}/feedback", response_model=schemas.SuccessResponse)
def set_chat_trace_feedback(
    trace_id: str,
    data: schemas.ChatFeedbackRequest,
    principal: auth_service.Principal = Depends(auth_service.require_menu_permission("问答对话")),
    container: ApplicationContainer = Depends(get_application_container),
) -> dict[str, str]:
    """记录提问者对回答的评价：1 有用，-1 没用，0 取消评价。

    知识库读取权限被收回后不能再评价，否则评价仍会计入差评统计和评测导出。
    """
    stored = chat_trace.load_trace(trace_id, container=container)
    if stored.account_id != principal.account_id:
        raise HTTPException(404, "追踪记录不存在或已过期")
    if not principal.can_read_knowledge_base(stored.kb_id):
        raise HTTPException(403, "无权访问该知识库")
    chat_trace.set_feedback(trace_id, principal.account_id, data.feedback, container=container)
    return {"status": "success"}
