"""Chat sessions and streaming-answer HTTP endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from routers import schemas
from services.chat import service as chat_service
from services.chat import sessions as chat_sessions
from services.chat.contracts import ChatRequest
from services.chat.persistence_contracts import ChatHistoryMessage, ChatSessionSummary
from services.chat.sessions import SessionOwner
from services.core import config as application_config
from services.core.container import ApplicationContainer, get_application_container
from services.identity import auth as auth_service

router = APIRouter(prefix="/api", tags=["chat"])


def _session_owner(principal: auth_service.Principal) -> SessionOwner:
    """把认证主体转换为聊天持久化层使用的稳定账号标识。"""
    return SessionOwner(principal.account_id, principal.username)


@router.get("/chat/sessions")
def get_chat_sessions(
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("问答对话")
    ),
    container: ApplicationContainer = Depends(get_application_container),
) -> list[ChatSessionSummary]:
    """返回当前用户拥有且仍有知识库访问权限的聊天会话。"""
    sessions = chat_sessions.list_owned_sessions(
        _session_owner(principal), container=container
    )
    return [
        session
        for session in sessions
        if principal.can_read_knowledge_base(session["kb_name"])
    ]


@router.get("/chat/sessions/{session_id}")
def get_chat_session_messages(
    session_id: str,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("问答对话")
    ),
    container: ApplicationContainer = Depends(get_application_container),
) -> list[ChatHistoryMessage]:
    """返回当前用户拥有的指定会话消息记录。"""
    kb_name, messages = chat_sessions.load_owned_session_messages(
        session_id, _session_owner(principal), container=container
    )
    if not principal.can_read_knowledge_base(kb_name):
        raise HTTPException(403, "无权访问该知识库")
    return messages


@router.delete("/chat/sessions/{session_id}")
def delete_chat_session(
    session_id: str,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("问答对话")
    ),
    container: ApplicationContainer = Depends(get_application_container),
) -> dict[str, str]:
    """删除当前用户的指定聊天会话及其全部消息。"""
    owner = _session_owner(principal)
    kb_name = chat_sessions.get_owned_session_kb(
        session_id, owner, container=container
    )
    if not principal.can_read_knowledge_base(kb_name):
        raise HTTPException(403, "无权访问该知识库")
    chat_sessions.delete_owned_session(session_id, owner, container=container)
    return {"status": "success"}


@router.put("/chat/sessions/{session_id}/title", response_model=schemas.SuccessResponse)
def rename_chat_session(
    session_id: str,
    data: schemas.ChatSessionTitleRequest,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("问答对话")
    ),
    container: ApplicationContainer = Depends(get_application_container),
) -> dict[str, str]:
    """修改当前用户拥有的指定会话的标题。"""
    owner = _session_owner(principal)
    kb_name = chat_sessions.get_owned_session_kb(
        session_id, owner, container=container
    )
    if not principal.can_read_knowledge_base(kb_name):
        raise HTTPException(403, "无权访问该知识库")
    chat_sessions.rename_owned_session(
        session_id, owner, data.title, container=container
    )
    return {"status": "success"}


@router.post("/chat")
async def stream_chat_answer(
    req: ChatRequest,
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("问答对话")
    ),
    container: ApplicationContainer = Depends(get_application_container),
) -> StreamingResponse:
    """针对指定问题和知识库，以 SSE 方式流式返回 RAG 回答。"""
    if not principal.can_read_knowledge_base(req.kb_name):
        raise HTTPException(403, "无权访问该知识库")
    stream = await chat_service.chat(
        req,
        _session_owner(principal),
        container,
    )
    return StreamingResponse(
        chat_service.with_heartbeat(stream.events, application_config.CHAT_SSE_HEARTBEAT_SECONDS),
        media_type="text/event-stream",
        background=BackgroundTask(stream.refresh_summary),
    )
