"""聊天会话和消息的持久化，以及会话归属校验。"""

import json
import uuid
from dataclasses import dataclass
from typing import cast

from services.chat.persistence_contracts import (
    ChatHistoryMessage,
    ChatSessionSummary,
    ChatSource,
)
from services.core import config as application_config
from services.core import sqlite_connections as application_db
from services.core.container import ApplicationContainer, get_application_container
from services.core.errors import AccessDeniedError, ResourceNotFoundError


@dataclass(frozen=True)
class SessionOwner:
    """用于校验持久化聊天会话归属的稳定账号身份。"""
    account_id: int
    username: str


@dataclass(frozen=True)
class ConversationMemory:
    """回答当前问题时使用的早期摘要和近期原文消息。"""

    summary: str
    recent_history: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SummaryBatch:
    """一次增量摘要需要吸收的消息范围和旧摘要状态。"""

    previous_summary: str
    previous_message_id: int
    messages: tuple[tuple[int, str, str], ...]


@dataclass(frozen=True)
class SessionResolution:
    """会话解析结果，并标记本次请求是否新建了持久化会话。"""

    session_id: str
    created: bool


SESSION_TITLE_MAX_CHARS = 30
# 手动重命名允许比自动标题更长，前端输入框用同一个上限。
SESSION_TITLE_RENAME_MAX_CHARS = 50


def session_title_from_question(question: str) -> str:
    """用首个问题的前若干字作为会话标题，让侧边栏里的会话可以互相区分。"""
    title = " ".join(str(question or "").split())
    if len(title) > SESSION_TITLE_MAX_CHARS:
        title = title[:SESSION_TITLE_MAX_CHARS].rstrip() + "…"
    return title or "新会话"


def _session_summary_from_row(row: tuple[object, ...]) -> ChatSessionSummary:
    """把数据库中的会话记录转换为列表接口使用的摘要结构。"""
    created_at = str(row[4]) if row[4] else ""
    return {
        "id": str(row[0]),
        "username": str(row[1]),
        "kb_name": str(row[2]),
        "title": str(row[3]),
        "date": created_at[:10],
    }


def _history_message_from_row(row: tuple[object, ...]) -> ChatHistoryMessage:
    """把消息记录转换为聊天历史，并还原其中保存的来源列表。"""
    sources = cast(list[ChatSource], json.loads(str(row[2]))) if row[2] else []
    return {
        "role": str(row[0]),
        "content": str(row[1]),
        "sources": sources,
    }


def _database_path(container: ApplicationContainer | None) -> str:
    """从显式容器或默认容器中取得当前业务数据库路径。"""
    return str((container or get_application_container()).database_path())


def list_owned_sessions(
    owner: SessionOwner, *, container: ApplicationContainer | None = None
) -> list[ChatSessionSummary]:
    """按最近更新时间返回一个账号拥有的全部会话。"""
    with application_db.connect(_database_path(container)) as conn:
        rows = conn.execute(
            "SELECT id, username, kb_name, title, created_at FROM chat_sessions "
            "WHERE account_id = ? ORDER BY updated_at DESC",
            (owner.account_id,),
        ).fetchall()
    return [_session_summary_from_row(row) for row in rows]


def get_owned_session_kb(
    session_id: str,
    owner: SessionOwner,
    *,
    container: ApplicationContainer | None = None,
) -> str:
    """返回本人会话绑定的知识库；不存在或不属于本人时统一返回 404。"""
    with application_db.connect(_database_path(container)) as conn:
        row = conn.execute(
            "SELECT kb_name FROM chat_sessions "
            "WHERE id = ? AND account_id = ?",
            (session_id, owner.account_id),
        ).fetchone()
    if not row:
        raise ResourceNotFoundError(
            "会话不存在或无权访问", code="chat_session_not_found"
        )
    return str(row[0])


def load_owned_session_messages(
    session_id: str,
    owner: SessionOwner,
    *,
    container: ApplicationContainer | None = None,
) -> tuple[str, list[ChatHistoryMessage]]:
    """校验会话归属，并返回其知识库和按时间排序的消息。"""
    with application_db.connect(_database_path(container)) as conn:
        row = conn.execute(
            "SELECT kb_name FROM chat_sessions "
            "WHERE id = ? AND account_id = ?",
            (session_id, owner.account_id),
        ).fetchone()
        if not row:
            raise ResourceNotFoundError(
                "会话不存在或无权访问", code="chat_session_not_found"
            )
        message_rows = conn.execute(
            "SELECT role, content, sources FROM chat_messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return str(row[0]), [_history_message_from_row(message) for message in message_rows]


def delete_owned_session(
    session_id: str,
    owner: SessionOwner,
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """再次校验归属后，原子删除会话及其全部消息。"""
    with application_db.connect(_database_path(container)) as conn:
        if not conn.execute(
            "SELECT 1 FROM chat_sessions "
            "WHERE id = ? AND account_id = ?",
            (session_id, owner.account_id),
        ).fetchone():
            raise ResourceNotFoundError(
                "会话不存在或无权访问", code="chat_session_not_found"
            )
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_session_memory WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


def rename_owned_session(
    session_id: str,
    owner: SessionOwner,
    title: str,
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """校验归属后更新会话标题；标题的空值和长度由接口层的请求模型保证。"""
    with application_db.connect(_database_path(container)) as conn:
        cursor = conn.execute(
            "UPDATE chat_sessions SET title = ? WHERE id = ? AND account_id = ?",
            (title, session_id, owner.account_id),
        )
        if cursor.rowcount == 0:
            raise ResourceNotFoundError(
                "会话不存在或无权访问", code="chat_session_not_found"
            )


def save_user_message(
    session_id: str,
    content: str,
    *,
    container: ApplicationContainer | None = None,
) -> int:
    """在回答生成前保存用户消息，返回消息 ID；会话的第一条消息同时决定会话标题。"""
    with application_db.connect(_database_path(container)) as conn:
        is_first_message = conn.execute(
            "SELECT 1 FROM chat_messages WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone() is None
        if is_first_message:
            # 只覆盖创建时的占位标题，已被手动重命名的会话保留用户设定的标题。
            conn.execute(
                "UPDATE chat_sessions SET title = ? WHERE id = ? AND title = kb_name || ' 会话'",
                (session_title_from_question(content), session_id),
            )
        cursor = conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, "user", content, "[]", application_config.db_now()),
        )
    return int(cursor.lastrowid or 0)


def discard_user_message(
    session_id: str,
    message_id: int,
    *,
    container: ApplicationContainer | None = None,
) -> bool:
    """撤回没有得到任何回答的用户消息；会话因此变空时一并删除，返回会话是否已删除。"""
    with application_db.connect(_database_path(container)) as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE id = ? AND session_id = ? AND role = 'user'",
            (message_id, session_id),
        )
        if conn.execute(
            "SELECT 1 FROM chat_messages WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone():
            return False
        conn.execute("DELETE FROM chat_session_memory WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    return True


def save_assistant_message(
    session_id: str,
    content: str,
    sources: list[ChatSource],
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """保存助手回答和引用，并同步会话的最近更新时间。"""
    now = application_config.db_now()
    with application_db.connect(_database_path(container)) as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                "assistant",
                content,
                json.dumps(sources, ensure_ascii=False),
                now,
            ),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )


def resolve_session(
    session_id: str | None,
    owner: SessionOwner,
    kb_name: str,
    *,
    force_new: bool = False,
    container: ApplicationContainer | None = None,
) -> SessionResolution:
    """解析或创建会话，并告知调用方是否需要在前置失败时清理新会话。"""
    created = False
    with application_db.connect(_database_path(container)) as conn:
        if session_id:
            row = conn.execute(
                "SELECT account_id, kb_name FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                raise ResourceNotFoundError(
                    "会话不存在", code="chat_session_not_found"
                )
            # 会话绑定知识库，也避免合法用户把一段历史对话悄悄带到另一套检索范围中。
            if row != (owner.account_id, kb_name):
                raise AccessDeniedError(
                    "无权访问该会话", code="chat_session_access_denied"
                )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (application_config.db_now(), session_id),
            )
            return SessionResolution(session_id=session_id, created=False)

        existing = None
        if not force_new:
            existing = conn.execute(
                "SELECT id FROM chat_sessions "
                "WHERE account_id = ? AND kb_name = ? "
                "ORDER BY updated_at DESC LIMIT 1",
                (owner.account_id, kb_name),
            ).fetchone()
        if existing:
            session_id = str(existing[0])
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (application_config.db_now(), session_id),
            )
        else:
            session_id = str(uuid.uuid4())
            created = True
            conn.execute(
                "INSERT INTO chat_sessions "
                "(id, account_id, username, kb_name, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    owner.account_id,
                    owner.username,
                    kb_name,
                    f"{kb_name} 会话",
                    application_config.db_now(),
                    application_config.db_now(),
                ),
            )
    return SessionResolution(session_id=session_id, created=created)


def _group_recent_turns(
    rows: list[tuple[int, str, str]],
) -> list[list[tuple[int, str, str]]]:
    """把倒序消息整理成倒序问答轮，避免只留下回答而丢掉对应问题。"""
    turns: list[list[tuple[int, str, str]]] = []
    pending_assistants: list[tuple[int, str, str]] = []
    for message_id, role, content in rows:
        message = (int(message_id), str(role), str(content or ""))
        if role == "assistant":
            pending_assistants.append(message)
            continue
        if role == "user":
            turns.append(pending_assistants + [message])
            pending_assistants = []
            continue
        turns.append([message])
    if pending_assistants:
        # 异常历史中找不到对应问题时仍保留回答，但它作为独立的一轮参与预算判断。
        turns.append(pending_assistants)
    return turns


def _pack_recent_turns(
    rows: list[tuple[int, str, str]], recent_max_chars: int
) -> tuple[list[tuple[str, str]], int | None]:
    """按字符预算从最新一轮开始装入近期原文。

    `rows` 是 (消息 ID, 角色, 内容)，从新到旧。返回 (近期原文, 其中最早一条的消息 ID)；
    摘要据此知道哪些较早消息没有进入提示词。
    """
    history: list[tuple[str, str]] = []
    oldest_kept_id: int | None = None
    used_chars = 0
    for turn in _group_recent_turns(rows):
        bounded_turn = [
            (message_id, role, text[: application_config.CHAT_MEMORY_RECENT_MESSAGE_MAX_CHARS])
            for message_id, role, text in turn
        ]
        turn_chars = sum(len(text) for _, _, text in bounded_turn)
        if used_chars + turn_chars > recent_max_chars:
            # 最新一轮本身超过预算时优先保住用户问题，再用剩余空间保留回答；
            # 更早轮次若无法完整放入，则整轮舍弃。
            if history:
                break
            remaining = recent_max_chars
            clipped_by_index: dict[int, tuple[int, str, str]] = {}
            for index in range(len(bounded_turn) - 1, -1, -1):
                if remaining <= 0:
                    break
                message_id, role, text = bounded_turn[index]
                clipped = text[:remaining]
                clipped_by_index[index] = (message_id, role, clipped)
                remaining -= len(clipped)
            kept = [clipped_by_index[index] for index in sorted(clipped_by_index)]
            history.extend((role, text) for _, role, text in kept)
            if kept:
                oldest_kept_id = min(message_id for message_id, _, _ in kept)
            break
        history.extend((role, text) for _, role, text in bounded_turn)
        used_chars += turn_chars
        oldest_kept_id = min(
            [message_id for message_id, _, _ in bounded_turn]
            + ([oldest_kept_id] if oldest_kept_id is not None else [])
        )
    return history, oldest_kept_id


def _effective_recent_limit(unsummarized_count: int, recent_limit: int) -> int:
    """摘要积压时临时放宽近期原文的读取条数，字符预算仍会限制实际长度。"""
    if unsummarized_count > application_config.CHAT_MEMORY_SUMMARY_TRIGGER_MESSAGE_COUNT:
        return max(recent_limit, application_config.CHAT_MEMORY_UNSUMMARIZED_MESSAGE_LIMIT)
    return recent_limit


def load_conversation_memory(
    session_id: str,
    recent_limit: int,
    recent_max_chars: int,
    *,
    container: ApplicationContainer | None = None,
) -> ConversationMemory:
    """读取会话摘要，并按消息数和字符数限制近期原文。"""
    with application_db.connect(_database_path(container)) as conn:
        memory_row = conn.execute(
            "SELECT summary, summary_message_id FROM chat_session_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        summary = str(memory_row[0] or "") if memory_row else ""
        summary_message_id = int(memory_row[1] or 0) if memory_row else 0
        unsummarized_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = ? AND id > ?",
                (session_id, summary_message_id),
            ).fetchone()[0]
        )
        rows = conn.execute(
            "SELECT id, role, content FROM chat_messages "
            "WHERE session_id = ? AND id > ? ORDER BY id DESC LIMIT ?",
            (
                session_id,
                summary_message_id,
                _effective_recent_limit(unsummarized_count, recent_limit),
            ),
        ).fetchall()
    history, _ = _pack_recent_turns(rows, recent_max_chars)
    return ConversationMemory(summary=summary, recent_history=tuple(history))


def load_summary_batch(
    session_id: str,
    trigger_message_count: int,
    keep_recent: int,
    *,
    recent_limit: int = application_config.CHAT_MEMORY_RECENT_MESSAGE_LIMIT,
    recent_max_chars: int = application_config.CHAT_MEMORY_RECENT_MAX_CHARS,
    container: ApplicationContainer | None = None,
) -> SummaryBatch | None:
    """读取需要并入摘要的较早消息；没有需要并入的消息时返回 None。

    两种情况会产生待摘要消息：未摘要消息超过触发阈值，此时保留最近 `keep_recent` 条原文；
    或者近期原文窗口因字符预算放不下较早的消息。后者既不在摘要里也不在提示词里，
    必须尽早并入摘要，否则这些轮次会被模型彻底遗忘。
    """
    with application_db.connect(_database_path(container)) as conn:
        memory_row = conn.execute(
            "SELECT summary, summary_message_id FROM chat_session_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        previous_summary = str(memory_row[0] or "") if memory_row else ""
        previous_message_id = int(memory_row[1] or 0) if memory_row else 0
        rows = conn.execute(
            "SELECT id, role, content FROM chat_messages "
            "WHERE session_id = ? AND id > ? ORDER BY id ASC",
            (session_id, previous_message_id),
        ).fetchall()

    cut = 0
    if len(rows) > trigger_message_count:
        cut = max(0, len(rows) - keep_recent) if keep_recent else len(rows)
    # 与 load_conversation_memory 使用同一套窗口规则，找出被挤出窗口的较早消息。
    window = rows[-_effective_recent_limit(len(rows), recent_limit):]
    _, oldest_kept_id = _pack_recent_turns(list(reversed(window)), recent_max_chars)
    if oldest_kept_id is not None:
        cut = max(cut, sum(1 for row in rows if row[0] < oldest_kept_id))
    if cut == 0:
        return None
    return SummaryBatch(
        previous_summary=previous_summary,
        previous_message_id=previous_message_id,
        messages=tuple(rows[:cut]),
    )


def save_conversation_summary(
    session_id: str,
    summary: str,
    *,
    expected_message_id: int,
    through_message_id: int,
    container: ApplicationContainer | None = None,
) -> bool:
    """以乐观并发方式保存摘要，旧状态已变化时放弃覆盖。"""
    now = application_config.db_now()
    with application_db.connect(_database_path(container)) as conn:
        cursor = conn.execute(
            "INSERT INTO chat_session_memory "
            "(session_id, summary, summary_message_id, updated_at) "
            "SELECT ?, ?, ?, ? WHERE EXISTS "
            "(SELECT 1 FROM chat_sessions WHERE id = ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "summary = excluded.summary, "
            "summary_message_id = excluded.summary_message_id, "
            "updated_at = excluded.updated_at "
            "WHERE chat_session_memory.summary_message_id = ?",
            (
                session_id,
                summary,
                through_message_id,
                now,
                session_id,
                expected_message_id,
            ),
        )
    return cursor.rowcount > 0
