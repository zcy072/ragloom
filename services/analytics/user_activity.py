"""用户分析：按账户聚合提问、会话、评价和登录记录，全部从 SQLite 实时计算。

数据量是单机部署的量级（运行记录默认保留 90 天），聚合查询走 `account_id` 和
`created_at` 上的索引即可，不维护预聚合表，因此也不存在统计与明细不一致的问题。

时间字段以 UTC 保存；按天和按小时分组用 SQLite 的 `localtime`，与进程时区一致
（Compose 中由 `TZ` 设置）。“最近 N 天”指今天和之前 N-1 个自然日。

`visible_kbs` 为 None 表示调用者能读全部知识库；否则只统计这些知识库内的提问和评价。
登录记录与知识库无关，不受该参数影响。
"""

import json
import sqlite3
from datetime import UTC, datetime, time, timedelta
from typing import Literal

from services.analytics.contracts import (
    ActivityOverview,
    ConversationMessage,
    DailyActivityPoint,
    DailyQuestionPoint,
    KnowledgeBaseUsage,
    LoginEventRecord,
    UserActivityDetail,
    UserActivityPage,
    UserActivityRow,
    UserConversation,
    UserQuestionRecord,
    UserSessionRecord,
)
from services.core import sqlite_connections as application_db
from services.core.errors import ResourceNotFoundError
from services.identity.pagination import like_pattern, resolve_page_window

DAYS_MAX = 90
RECENT_LIMIT = 20
SortKey = Literal["questions", "last_question_at", "last_login_at", "created_at", "downvotes"]
# 排序键到 SQL 表达式的白名单；NULL（从未提问或登录）总是排在最后。
_SORT_EXPRESSIONS: dict[str, str] = {
    "questions": "questions",
    "last_question_at": "last_question_at",
    "last_login_at": "last_login_at",
    "created_at": "a.created_at",
    "downvotes": "downvotes",
}


def _window_start(days: int) -> tuple[str, list[str]]:
    """返回窗口起点（UTC，格式与 `db_now` 一致）和窗口内的本地日期列表。"""
    today = datetime.now().astimezone().date()
    first_day = today - timedelta(days=days - 1)
    start = datetime.combine(first_day, time.min).astimezone()
    dates = [(first_day + timedelta(days=offset)).isoformat() for offset in range(days)]
    return start.astimezone(UTC).isoformat(timespec="seconds"), dates


def _kb_filter(column: str, visible_kbs: tuple[str, ...] | None) -> tuple[str, list[str]]:
    """生成知识库过滤条件；无权读任何知识库时条件恒假。"""
    if visible_kbs is None:
        return "", []
    if not visible_kbs:
        return " AND 0", []
    placeholders = ",".join("?" for _ in visible_kbs)
    return f" AND {column} IN ({placeholders})", list(visible_kbs)


def _clamp_days(days: int) -> int:
    return max(1, min(days, DAYS_MAX))


def overview(
    db_file: str, *, days: int, visible_kbs: tuple[str, ...] | None
) -> ActivityOverview:
    """统计窗口内的用户规模、活跃度、按天趋势、按小时分布和各知识库使用情况。"""
    days = _clamp_days(days)
    start, dates = _window_start(days)
    run_kb, run_kb_params = _kb_filter("kb_name", visible_kbs)
    trace_kb, trace_kb_params = _kb_filter("kb_name", visible_kbs)
    with application_db.connect(db_file) as conn:
        total_users, pending_users, new_users = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(status = '审核中'), 0), "
            "COALESCE(SUM(created_at >= ?), 0) FROM accounts",
            (start,),
        ).fetchone()
        questions, active_users = conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT account_id) FROM chat_runs "
            f"WHERE created_at >= ?{run_kb}",
            (start, *run_kb_params),
        ).fetchone()
        upvotes, downvotes = conn.execute(
            f"SELECT COALESCE(SUM(feedback = 1), 0), COALESCE(SUM(feedback = -1), 0) "
            f"FROM chat_traces WHERE created_at >= ?{trace_kb}",
            (start, *trace_kb_params),
        ).fetchone()
        login_users, login_failures = conn.execute(
            "SELECT COUNT(DISTINCT CASE WHEN success = 1 THEN account_id END), "
            "COALESCE(SUM(success = 0), 0) FROM login_events WHERE created_at >= ?",
            (start,),
        ).fetchone()
        daily_runs = {
            str(day): (int(count), int(users))
            for day, count, users in conn.execute(
                f"SELECT DATE(created_at, 'localtime'), COUNT(*), COUNT(DISTINCT account_id) "
                f"FROM chat_runs WHERE created_at >= ?{run_kb} GROUP BY 1",
                (start, *run_kb_params),
            )
        }
        daily_logins = {
            str(day): int(users)
            for day, users in conn.execute(
                "SELECT DATE(created_at, 'localtime'), COUNT(DISTINCT account_id) "
                "FROM login_events WHERE created_at >= ? AND success = 1 GROUP BY 1",
                (start,),
            )
        }
        hourly = [0] * 24
        for hour, count in conn.execute(
            f"SELECT CAST(strftime('%H', created_at, 'localtime') AS INTEGER), COUNT(*) "
            f"FROM chat_runs WHERE created_at >= ?{run_kb} GROUP BY 1",
            (start, *run_kb_params),
        ):
            hourly[int(hour)] = int(count)
        knowledge_bases = _knowledge_base_usage(
            conn, start=start, visible_kbs=visible_kbs, account_id=None
        )
    daily: list[DailyActivityPoint] = [
        {
            "date": day,
            "questions": daily_runs.get(day, (0, 0))[0],
            "active_users": daily_runs.get(day, (0, 0))[1],
            "login_users": daily_logins.get(day, 0),
        }
        for day in dates
    ]
    return {
        "days": days,
        "total_users": int(total_users),
        "pending_users": int(pending_users),
        "new_users": int(new_users),
        "active_users": int(active_users),
        "login_users": int(login_users),
        "login_failures": int(login_failures),
        "questions": int(questions),
        "upvotes": int(upvotes),
        "downvotes": int(downvotes),
        "daily": daily,
        "hourly": hourly,
        "knowledge_bases": knowledge_bases,
    }


def _knowledge_base_usage(
    conn: sqlite3.Connection,
    *,
    start: str,
    visible_kbs: tuple[str, ...] | None,
    account_id: int | None,
) -> list[KnowledgeBaseUsage]:
    """按知识库汇总提问人数、提问数和差评数，按提问数降序。"""
    run_kb, run_kb_params = _kb_filter("kb_name", visible_kbs)
    account_clause = " AND account_id = ?" if account_id is not None else ""
    account_params = [account_id] if account_id is not None else []
    usage: dict[str, KnowledgeBaseUsage] = {}
    for kb_name, users, count in conn.execute(
        f"SELECT kb_name, COUNT(DISTINCT account_id), COUNT(*) FROM chat_runs "
        f"WHERE created_at >= ?{run_kb}{account_clause} GROUP BY kb_name",
        (start, *run_kb_params, *account_params),
    ):
        usage[str(kb_name)] = {
            "kb_name": str(kb_name), "users": int(users), "questions": int(count), "downvotes": 0,
        }
    for kb_name, downvotes in conn.execute(
        f"SELECT kb_name, COUNT(*) FROM chat_traces "
        f"WHERE created_at >= ? AND feedback = -1{run_kb}{account_clause} GROUP BY kb_name",
        (start, *run_kb_params, *account_params),
    ):
        if str(kb_name) in usage:
            usage[str(kb_name)]["downvotes"] = int(downvotes)
    return sorted(usage.values(), key=lambda item: (-item["questions"], item["kb_name"]))


def _kb_named_filter(column: str, visible_kbs: tuple[str, ...] | None) -> tuple[str, dict]:
    """与 `_kb_filter` 相同，但使用命名参数 :kb0、:kb1……，供同一语句中多次引用。"""
    if visible_kbs is None:
        return "", {}
    if not visible_kbs:
        return " AND 0", {}
    names = {f"kb{index}": name for index, name in enumerate(visible_kbs)}
    return f" AND {column} IN ({','.join(':' + key for key in names)})", names


def _user_rows_statement(
    *, visible_kbs: tuple[str, ...] | None, where: str, order_by: str
) -> tuple[str, dict]:
    """构造用户统计查询；返回 SQL 和知识库过滤用到的命名参数。"""
    kb_clause, kb_params = _kb_named_filter("kb_name", visible_kbs)
    statement = f"""
        WITH q AS (
            SELECT account_id, COUNT(*) AS questions,
                   COUNT(DISTINCT session_id) AS sessions,
                   SUM(status = 'error') AS errors
            FROM chat_runs WHERE created_at >= :start{kb_clause}
            GROUP BY account_id
        ),
        f AS (
            SELECT account_id, SUM(feedback = 1) AS upvotes, SUM(feedback = -1) AS downvotes
            FROM chat_traces WHERE created_at >= :start{kb_clause}
            GROUP BY account_id
        ),
        l AS (
            SELECT account_id, SUM(success) AS logins, SUM(1 - success) AS login_failures
            FROM login_events WHERE created_at >= :start AND account_id IS NOT NULL
            GROUP BY account_id
        ),
        last_question AS (
            SELECT account_id, MAX(created_at) AS last_question_at
            FROM chat_runs WHERE 1{kb_clause}
            GROUP BY account_id
        ),
        last_login AS (
            SELECT account_id, MAX(created_at) AS last_login_at
            FROM login_events WHERE success = 1 AND account_id IS NOT NULL
            GROUP BY account_id
        )
        SELECT a.id, a.username, a.name, a.status, COALESCE(r.name, '未分配') AS role,
               a.created_at,
               COALESCE(q.questions, 0) AS questions, COALESCE(q.sessions, 0) AS sessions,
               COALESCE(q.errors, 0) AS errors,
               COALESCE(f.upvotes, 0) AS upvotes, COALESCE(f.downvotes, 0) AS downvotes,
               COALESCE(l.logins, 0) AS logins, COALESCE(l.login_failures, 0) AS login_failures,
               last_question.last_question_at, last_login.last_login_at
        FROM accounts a
        LEFT JOIN roles r ON r.id = a.role_id
        LEFT JOIN q ON q.account_id = a.id
        LEFT JOIN f ON f.account_id = a.id
        LEFT JOIN l ON l.account_id = a.id
        LEFT JOIN last_question ON last_question.account_id = a.id
        LEFT JOIN last_login ON last_login.account_id = a.id
        {where}
        {order_by}
    """
    return statement, kb_params


def _user_row(row: sqlite3.Row) -> UserActivityRow:
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "name": str(row["name"]),
        "status": str(row["status"]),
        "role": str(row["role"]),
        "created_at": str(row["created_at"]),
        "questions": int(row["questions"]),
        "sessions": int(row["sessions"]),
        "errors": int(row["errors"]),
        "upvotes": int(row["upvotes"]),
        "downvotes": int(row["downvotes"]),
        "logins": int(row["logins"]),
        "login_failures": int(row["login_failures"]),
        "last_question_at": str(row["last_question_at"] or ""),
        "last_login_at": str(row["last_login_at"] or ""),
    }


def _query_user_rows(
    conn: sqlite3.Connection,
    *,
    start: str,
    visible_kbs: tuple[str, ...] | None,
    where: str,
    order_by: str,
    extra: dict[str, object],
) -> list[sqlite3.Row]:
    statement, kb_params = _user_rows_statement(
        visible_kbs=visible_kbs, where=where, order_by=order_by
    )
    conn.row_factory = sqlite3.Row
    return conn.execute(statement, {"start": start, **kb_params, **extra}).fetchall()


def list_users(
    db_file: str,
    *,
    days: int,
    visible_kbs: tuple[str, ...] | None,
    page: int,
    page_size: int,
    query: str = "",
    sort: SortKey = "questions",
    descending: bool = True,
) -> UserActivityPage:
    """分页返回全部账户及其窗口内的使用统计，可按用户名或姓名搜索、按白名单字段排序。"""
    start, _ = _window_start(_clamp_days(days))
    pattern = like_pattern(query)
    where = ""
    extra: dict[str, object] = {}
    if pattern:
        where = "WHERE a.username LIKE :pattern ESCAPE '!' OR a.name LIKE :pattern ESCAPE '!'"
        extra["pattern"] = pattern
    direction = "DESC" if descending else "ASC"
    expression = _SORT_EXPRESSIONS[sort]
    order_by = f"ORDER BY {expression} IS NULL, {expression} {direction}, a.id ASC"
    with application_db.connect(db_file) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM accounts a {where.replace(':pattern', '?')}",
                [pattern, pattern] if pattern else [],
            ).fetchone()[0]
        )
        window = resolve_page_window(total, page, page_size)
        rows = _query_user_rows(
            conn,
            start=start,
            visible_kbs=visible_kbs,
            where=where,
            order_by=f"{order_by} LIMIT :limit OFFSET :offset",
            extra={**extra, "limit": window.page_size, "offset": window.offset},
        )
    return {
        "items": [_user_row(row) for row in rows],
        "total": window.total,
        "page": window.page,
        "page_size": window.page_size,
        "total_pages": window.total_pages,
    }


def user_detail(
    db_file: str, user_id: int, *, days: int, visible_kbs: tuple[str, ...] | None
) -> UserActivityDetail:
    """返回单个用户的统计、按天提问数、知识库分布、最近登录、会话和提问。"""
    days = _clamp_days(days)
    start, dates = _window_start(days)
    run_kb, run_kb_params = _kb_filter("kb_name", visible_kbs)
    session_kb, session_kb_params = _kb_filter("s.kb_name", visible_kbs)
    question_kb, question_kb_params = _kb_filter("r.kb_name", visible_kbs)
    with application_db.connect(db_file) as conn:
        rows = _query_user_rows(
            conn, start=start, visible_kbs=visible_kbs,
            where="WHERE a.id = :user_id", order_by="", extra={"user_id": user_id},
        )
        if not rows:
            raise ResourceNotFoundError("用户不存在", code="user_not_found")
        user = _user_row(rows[0])
        conn.row_factory = None
        no_evidence = int(
            conn.execute(
                f"SELECT COUNT(*) FROM chat_runs WHERE account_id = ? AND created_at >= ? "
                f"AND status = 'no_evidence'{run_kb}",
                (user_id, start, *run_kb_params),
            ).fetchone()[0]
        )
        daily_counts = {
            str(day): int(count)
            for day, count in conn.execute(
                f"SELECT DATE(created_at, 'localtime'), COUNT(*) FROM chat_runs "
                f"WHERE account_id = ? AND created_at >= ?{run_kb} GROUP BY 1",
                (user_id, start, *run_kb_params),
            )
        }
        knowledge_bases = _knowledge_base_usage(
            conn, start=start, visible_kbs=visible_kbs, account_id=user_id
        )
        recent_logins: list[LoginEventRecord] = [
            {
                "created_at": str(created_at),
                "success": bool(success),
                "reason": str(reason),
                "client": str(client),
            }
            for created_at, success, reason, client in conn.execute(
                "SELECT created_at, success, reason, client FROM login_events "
                "WHERE account_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (user_id, RECENT_LIMIT),
            )
        ]
        recent_sessions: list[UserSessionRecord] = [
            {
                "id": str(session_id),
                "title": str(title),
                "kb_name": str(kb_name),
                "created_at": str(created_at),
                "updated_at": str(updated_at),
                "questions": int(questions),
            }
            for session_id, title, kb_name, created_at, updated_at, questions in conn.execute(
                f"""
                SELECT s.id, s.title, s.kb_name, s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM chat_messages m
                         WHERE m.session_id = s.id AND m.role = 'user')
                FROM chat_sessions s
                WHERE s.account_id = ?{session_kb}
                ORDER BY s.updated_at DESC LIMIT ?
                """,
                (user_id, *session_kb_params, RECENT_LIMIT),
            )
        ]
        recent_questions: list[UserQuestionRecord] = [
            {
                "created_at": str(created_at),
                "kb_name": str(kb_name),
                "question_preview": str(preview),
                "status": str(status),
                "total_ms": float(total_ms),
                "feedback": int(feedback or 0),
            }
            for created_at, kb_name, preview, status, total_ms, feedback in conn.execute(
                f"""
                SELECT r.created_at, r.kb_name, r.question_preview, r.status, r.total_ms,
                       t.feedback
                FROM chat_runs r LEFT JOIN chat_traces t ON t.run_id = r.id
                WHERE r.account_id = ?{question_kb}
                ORDER BY r.id DESC LIMIT ?
                """,
                (user_id, *question_kb_params, RECENT_LIMIT),
            )
        ]
    daily: list[DailyQuestionPoint] = [
        {"date": day, "questions": daily_counts.get(day, 0)} for day in dates
    ]
    return {
        "days": days,
        "user": user,
        "no_evidence": no_evidence,
        "daily": daily,
        "knowledge_bases": knowledge_bases,
        "recent_logins": recent_logins,
        "recent_sessions": recent_sessions,
        "recent_questions": recent_questions,
    }


def _source_names(raw_sources: object) -> list[str]:
    """从消息的来源快照中取出去重后的文件名；损坏的快照按没有来源处理。"""
    try:
        sources = json.loads(str(raw_sources or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(sources, list):
        return []
    names = [
        str(item.get("source") or "") for item in sources if isinstance(item, dict)
    ]
    return list(dict.fromkeys(name for name in names if name))


def find_session_owner(
    conn: sqlite3.Connection, user_id: int, session_id: str
) -> tuple[str, str, str] | None:
    """返回属于该用户的会话的 (知识库, 标题, 用户名)；不存在或不属于该用户时返回 None。"""
    row = conn.execute(
        "SELECT kb_name, title, username FROM chat_sessions WHERE id = ? AND account_id = ?",
        (session_id, user_id),
    ).fetchone()
    return (str(row[0]), str(row[1]), str(row[2])) if row else None


def load_conversation(
    conn: sqlite3.Connection, session_id: str, *, kb_name: str, title: str, username: str
) -> UserConversation:
    """读取会话的全部消息，包括回答引用的文件名和提问者给出的评价。"""
    messages: list[ConversationMessage] = [
        {
            "role": str(role),
            "content": str(content),
            "created_at": str(created_at),
            "source_names": _source_names(sources),
            "feedback": int(feedback or 0),
        }
        for role, content, created_at, sources, feedback in conn.execute(
            "SELECT m.role, m.content, m.created_at, m.sources, t.feedback FROM chat_messages m "
            "LEFT JOIN chat_traces t ON t.assistant_message_id = m.id "
            "WHERE m.session_id = ? ORDER BY m.id ASC",
            (session_id,),
        )
    ]
    return {
        "session_id": session_id,
        "title": title,
        "kb_name": kb_name,
        "username": username,
        "messages": messages,
    }
