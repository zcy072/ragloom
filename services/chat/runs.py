"""问答运行记录：每次提问保存耗时与结果状态，供运行监控页面查询。

记录只含统计事实和问题前若干字，不含回答正文与检索文本，这些仍只存在于
`chat_messages`。写入失败只记录警告，绝不影响回答本身。
"""

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, TypedDict

from services.chat import trace as chat_trace
from services.core import config as application_config
from services.core import sqlite_connections as application_db
from services.core.container import ApplicationContainer, database_file
from services.core.settings import get_settings

logger = logging.getLogger(__name__)

RunStatus = Literal[
    "answered", "no_evidence", "direct", "interrupted", "cancelled", "error"
]
RUN_STATUSES: tuple[RunStatus, ...] = (
    "answered", "no_evidence", "direct", "interrupted", "cancelled", "error",
)
QUESTION_PREVIEW_CHARS = 200
ERROR_SUMMARY_CHARS = 200
# 概览统计的行数上限：分位数要在 Python 中计算，窗口过大时只取最近的记录。
OVERVIEW_ROW_LIMIT = 20000
LIST_LIMIT_MAX = 100
# 阶段耗时中有独立列的键，其余键进入 elapsed_extra。
_STAGE_COLUMNS = {"retrieval": "retrieval_ms", "rerank": "rerank_ms", "context": "context_ms"}


class ChatRunRecord(TypedDict):
    """列表接口返回的一条问答运行记录。"""

    id: int
    created_at: str
    username: str
    kb_name: str
    session_id: str
    question_preview: str
    model_provider: str
    status: str
    rerank_status: str
    rewritten: bool
    retrieval_skipped: bool
    context_spans: int
    cited_sources: int
    answer_chars: int
    retrieval_ms: float | None
    rerank_ms: float | None
    context_ms: float | None
    first_token_ms: float | None
    generation_ms: float | None
    total_ms: float
    elapsed_extra: dict[str, float]
    error_summary: str
    trace_id: str
    feedback: int


class LatencySummary(TypedDict):
    """某项耗时的分位数，单位毫秒；没有样本时为 None。"""

    p50: float | None
    p95: float | None


class DailyRunPoint(TypedDict):
    """按天汇总的问答数、失败数和总耗时 P95。"""

    date: str
    runs: int
    errors: int
    p95_total_ms: float | None


class RunsOverview(TypedDict):
    """概览接口返回的统计结果。"""

    days: int
    total: int
    truncated: bool
    status_counts: dict[str, int]
    error_rate: float
    latency: dict[str, LatencySummary]
    daily: list[DailyRunPoint]


@dataclass
class ChatRunRecorder:
    """在一次问答过程中累积记录所需的事实，结束时由 `finish` 一次性写库。"""

    account_id: int
    username: str
    kb_name: str
    session_id: str = ""
    user_message_id: int | None = None
    question: str = ""
    model_provider: str = ""
    started: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    generation_started: float | None = None
    elapsed_ms: dict[str, float] = field(default_factory=dict)
    rerank_status: str = ""
    rewritten: bool = False
    retrieval_skipped: bool = False
    context_spans: int = 0
    cited_sources: int = 0
    answer_chars: int = 0
    status: RunStatus | None = None
    error: BaseException | str | None = None
    _written: bool = False

    def mark_generation_start(self) -> None:
        """记录开始调用回答模型的时刻，用来计算首字耗时和生成耗时。"""
        if self.generation_started is None:
            self.generation_started = time.perf_counter()

    def mark_token(self) -> None:
        """记录第一个非空输出片段的到达时刻。"""
        if self.first_token_at is None:
            self.first_token_at = time.perf_counter()

    def first_token_ms(self) -> float | None:
        """从开始调用回答模型到第一个非空片段的毫秒数；尚无输出时为 None。"""
        return _milliseconds(self.generation_started, self.first_token_at)

    def fail(self, error: BaseException | str) -> None:
        """标记本次问答失败；只保留异常类型和截断后的消息。"""
        self.status = "error"
        self.error = error

    def finish(
        self,
        container: ApplicationContainer | None = None,
        *,
        answer_chars: int | None = None,
        cited_sources: int | None = None,
    ) -> int | None:
        """写入记录并返回记录 ID；每个实例只写一次，任何异常都只记警告并返回 None。"""
        if self._written:
            return None
        self._written = True
        if answer_chars is not None:
            self.answer_chars = answer_chars
        if cited_sources is not None:
            self.cited_sources = cited_sources
        try:
            return record_run(self, container=container)
        except Exception as exc:
            logger.warning("Failed to record chat run: %s", exc, exc_info=True)
            return None


def _milliseconds(start: float | None, end: float | None) -> float | None:
    """两个单调时钟读数之间的毫秒数；任一缺失时返回 None。"""
    if start is None or end is None:
        return None
    return round(max(0.0, end - start) * 1000, 2)


def _error_summary(error: BaseException | str | None) -> str:
    """压缩为异常类型加截断消息，避免把上游响应原文写进记录。"""
    if error is None:
        return ""
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    return " ".join(text.split())[:ERROR_SUMMARY_CHARS]


def _resolved_status(recorder: ChatRunRecorder) -> RunStatus:
    """流程在得出结果前被中断（客户端断开、任务取消）时不会设置状态，按取消记录。"""
    return recorder.status or "cancelled"


def record_run(
    recorder: ChatRunRecorder, *, container: ApplicationContainer | None = None
) -> int:
    """把一次问答的累积事实写成一行记录，返回记录 ID。"""
    now = time.perf_counter()
    extra = {
        key: value
        for key, value in recorder.elapsed_ms.items()
        if key not in _STAGE_COLUMNS
    }
    stages = {
        column: recorder.elapsed_ms.get(key) for key, column in _STAGE_COLUMNS.items()
    }
    with application_db.connect(database_file(container)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_runs (
                created_at, account_id, username, kb_name, session_id, user_message_id,
                question_preview, model_provider, status, rerank_status, rewritten,
                retrieval_skipped, context_spans, cited_sources, answer_chars,
                retrieval_ms, rerank_ms, context_ms, first_token_ms, generation_ms,
                total_ms, elapsed_extra, error_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                application_config.db_now(),
                recorder.account_id,
                recorder.username,
                recorder.kb_name,
                recorder.session_id,
                recorder.user_message_id,
                " ".join(recorder.question.split())[:QUESTION_PREVIEW_CHARS],
                recorder.model_provider,
                _resolved_status(recorder),
                recorder.rerank_status,
                int(recorder.rewritten),
                int(recorder.retrieval_skipped),
                recorder.context_spans,
                recorder.cited_sources,
                recorder.answer_chars,
                stages["retrieval_ms"],
                stages["rerank_ms"],
                stages["context_ms"],
                _milliseconds(recorder.generation_started, recorder.first_token_at),
                _milliseconds(recorder.generation_started, now),
                _milliseconds(recorder.started, now) or 0.0,
                json.dumps(extra, ensure_ascii=False),
                _error_summary(recorder.error),
            ),
        )
    return int(cursor.lastrowid or 0)


def _percentile(values: list[float], fraction: float) -> float | None:
    """最近秩法分位数；空样本返回 None。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 2)


def _summary(values: list[float]) -> LatencySummary:
    """计算一组耗时的 P50 与 P95。"""
    return {"p50": _percentile(values, 0.50), "p95": _percentile(values, 0.95)}


def _cutoff(days: int) -> str:
    """窗口起点，格式与 `db_now` 一致，可直接做字符串比较。"""
    start = datetime.now(UTC) - timedelta(days=days)
    return start.isoformat(timespec="seconds")


def overview(
    days: int, *, container: ApplicationContainer | None = None
) -> RunsOverview:
    """统计最近 `days` 天的问答状态分布、耗时分位数和按天趋势。"""
    days = max(1, min(days, 90))
    with application_db.connect(database_file(container)) as conn:
        total = int(
            conn.execute(
                "SELECT COUNT(*) FROM chat_runs WHERE created_at >= ?", (_cutoff(days),)
            ).fetchone()[0]
        )
        rows = conn.execute(
            """
            SELECT created_at, status, total_ms, retrieval_ms, rerank_ms, first_token_ms
            FROM chat_runs WHERE created_at >= ?
            ORDER BY id DESC LIMIT ?
            """,
            (_cutoff(days), OVERVIEW_ROW_LIMIT),
        ).fetchall()
    status_counts = {status: 0 for status in RUN_STATUSES}
    samples: dict[str, list[float]] = {
        "total": [], "retrieval": [], "rerank": [], "first_token": [],
    }
    by_day: dict[str, list[tuple[str, float]]] = {}
    for created_at, status, total_ms, retrieval_ms, rerank_ms, first_token_ms in rows:
        status_counts[status] = status_counts.get(status, 0) + 1
        by_day.setdefault(str(created_at)[:10], []).append((status, float(total_ms)))
        # 出错的请求耗时不代表正常链路，不计入分位数。
        if status == "error":
            continue
        samples["total"].append(float(total_ms))
        for name, value in (
            ("retrieval", retrieval_ms), ("rerank", rerank_ms), ("first_token", first_token_ms),
        ):
            if value is not None:
                samples[name].append(float(value))
    counted = len(rows)
    daily: list[DailyRunPoint] = [
        {
            "date": day,
            "runs": len(items),
            "errors": sum(1 for status, _ in items if status == "error"),
            "p95_total_ms": _percentile(
                [ms for status, ms in items if status != "error"], 0.95
            ),
        }
        for day, items in sorted(by_day.items())
    ]
    return {
        "days": days,
        "total": total,
        "truncated": total > counted,
        "status_counts": status_counts,
        "error_rate": round(status_counts["error"] / counted, 4) if counted else 0.0,
        "latency": {name: _summary(values) for name, values in samples.items()},
        "daily": daily,
    }


def _record_from_row(row: sqlite3.Row) -> ChatRunRecord:
    """把数据库行转换为列表接口使用的记录结构。"""
    try:
        extra = json.loads(row["elapsed_extra"] or "{}")
    except json.JSONDecodeError:
        extra = {}
    return {
        "id": int(row["id"]),
        "created_at": str(row["created_at"]),
        "username": str(row["username"]),
        "kb_name": str(row["kb_name"]),
        "session_id": str(row["session_id"]),
        "question_preview": str(row["question_preview"]),
        "model_provider": str(row["model_provider"]),
        "status": str(row["status"]),
        "rerank_status": str(row["rerank_status"]),
        "rewritten": bool(row["rewritten"]),
        "retrieval_skipped": bool(row["retrieval_skipped"]),
        "context_spans": int(row["context_spans"]),
        "cited_sources": int(row["cited_sources"]),
        "answer_chars": int(row["answer_chars"]),
        "retrieval_ms": row["retrieval_ms"],
        "rerank_ms": row["rerank_ms"],
        "context_ms": row["context_ms"],
        "first_token_ms": row["first_token_ms"],
        "generation_ms": row["generation_ms"],
        "total_ms": float(row["total_ms"]),
        "elapsed_extra": extra if isinstance(extra, dict) else {},
        "error_summary": str(row["error_summary"]),
        "trace_id": str(row["trace_id"] or ""),
        "feedback": int(row["feedback"] or 0),
    }


_RUNS_WITH_TRACE = "FROM chat_runs r LEFT JOIN chat_traces t ON t.run_id = r.id"


def list_runs(
    *,
    status: str | None = None,
    kb_name: str | None = None,
    min_total_ms: float | None = None,
    feedback: int | None = None,
    limit: int = 20,
    offset: int = 0,
    container: ApplicationContainer | None = None,
) -> tuple[int, list[ChatRunRecord]]:
    """按时间倒序分页返回记录，并给出过滤后的总数；每条记录带上对应的追踪 ID 和用户评价。"""
    clauses: list[str] = []
    params: list[object] = []
    if status:
        clauses.append("r.status = ?")
        params.append(status)
    if kb_name:
        clauses.append("r.kb_name = ?")
        params.append(kb_name)
    if min_total_ms is not None:
        clauses.append("r.total_ms >= ?")
        params.append(min_total_ms)
    if feedback is not None:
        clauses.append("COALESCE(t.feedback, 0) = ?")
        params.append(feedback)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit = max(1, min(limit, LIST_LIMIT_MAX))
    with application_db.connect(database_file(container)) as conn:
        conn.row_factory = sqlite3.Row
        total = int(
            conn.execute(f"SELECT COUNT(*) {_RUNS_WITH_TRACE} {where}", params).fetchone()[0]
        )
        rows = conn.execute(
            f"SELECT r.*, t.id AS trace_id, t.feedback AS feedback {_RUNS_WITH_TRACE} {where} "
            "ORDER BY r.id DESC LIMIT ? OFFSET ?",
            (*params, limit, max(0, offset)),
        ).fetchall()
    return total, [_record_from_row(row) for row in rows]


def prune(
    retention_days: int | None = None,
    *,
    container: ApplicationContainer | None = None,
    db_file: str | None = None,
) -> int:
    """删除超过保留期的运行记录和追踪记录，返回删除的运行记录条数；启动维护时直接传入数据库路径。"""
    days = max(1, retention_days or get_settings().chat_run_retention_days)
    with application_db.connect(db_file or database_file(container)) as conn:
        cursor = conn.execute(
            "DELETE FROM chat_runs WHERE created_at < ?", (_cutoff(days),)
        )
        chat_trace.prune(conn, _cutoff(days))
    return cursor.rowcount
