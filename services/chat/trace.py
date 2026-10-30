"""问答过程追踪：按阶段记录耗时和关键事实，实时推送给前端，并落库供事后排查。

与 `chat_runs` 的统计记录不同，追踪记录包含候选片段前若干字、重排分数和上下文来源，
读取时必须校验调用者对该知识库的访问权限。
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from services.core import config as application_config
from services.core import sqlite_connections as application_db
from services.core.container import ApplicationContainer, database_file
from services.core.errors import ResourceNotFoundError
from services.core.request_context import current_request_id

if TYPE_CHECKING:
    from services.chat.diagnostics import RetrievalDiagnostics

logger = logging.getLogger(__name__)

TRACE_SNIPPET_CHARS = 200
TRACE_QUESTION_CHARS = 4000
TRACE_MAX_BYTES = 128 * 1024
ERROR_SUMMARY_CHARS = 200
# 重排最高分低于它时提示"召回不足或库中没有答案"；库内问题实测前几名都在 0.8 以上。
LOW_TOP_SCORE = 0.3

StageStatus = Literal["ok", "skipped", "degraded", "error", "cancelled"]


class TraceStage(TypedDict):
    """一个处理阶段的结果。"""

    stage: str
    status: StageStatus
    elapsed_ms: float
    detail: dict[str, object]


class TraceDocument(TypedDict):
    """一个检索候选：位置、来源和原文前若干字；重排后带分数和是否进入后续环节。"""

    position: int
    kb_name: str
    source: str
    page: str
    route: str
    snippet: str
    score: NotRequired[float]
    kept: NotRequired[bool]
    url: NotRequired[str]


class TraceContextSpan(TypedDict):
    """送给模型的一段上下文。"""

    id: int
    kb_name: str
    source: str
    page: str
    routes: list[str]
    chars: int
    truncated: bool
    url: NotRequired[str]


class TraceQuery(TypedDict):
    """补充检索的一轮查询及其新增结果数。"""

    query: str
    results: int


class ChatTracePayload(TypedDict):
    """一次问答的完整追踪记录。"""

    id: str
    created_at: str
    kb_name: str
    status: str
    question: str
    actual_query: str
    variant_query: str
    rewritten: bool
    skipped: bool
    fallback_attempted: bool
    rerank_status: str
    stages: list[TraceStage]
    candidates: list[TraceDocument]
    reranked: list[TraceDocument]
    web_results: list[TraceDocument]
    agentic_queries: list[TraceQuery]
    context: list[TraceContextSpan]
    prompt_chars: int
    cited_ids: list[int]
    hints: list[str]


def document_summary(document: object, position: int) -> TraceDocument:
    """把检索文档压缩为追踪记录使用的来源描述和原文开头。"""
    metadata = getattr(document, "metadata", None) or {}
    page = metadata.get("page")
    url = str(metadata.get("url") or "")
    source = str(metadata.get("source", ""))
    summary: TraceDocument = {
        "position": position,
        "kb_name": str(metadata.get("kb_name", "")),
        # 网页标题里可能有斜杠，不能按路径取文件名。
        "source": source if url else os.path.basename(source),
        "page": "" if page is None else str(page),
        "route": str(metadata.get("chunk_route", "")),
        "snippet": " ".join(str(getattr(document, "page_content", "")).split())[
            :TRACE_SNIPPET_CHARS
        ],
    }
    if url:
        summary["url"] = url
    return summary


def error_summary(error: BaseException | str) -> str:
    """压缩为异常类型加截断消息，避免把上游响应原文写进记录。"""
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    return " ".join(text.split())[:ERROR_SUMMARY_CHARS]


def stage_event(record: TraceStage) -> str:
    """把阶段结果编码为一条 SSE 事件。"""
    return f"data: {json.dumps(record)}\n\n"


@dataclass
class ChatTrace:
    """在一次问答过程中累积追踪事实，结束时由 `save_trace` 一次性写库。"""

    account_id: int
    kb_name: str = ""
    session_id: str = ""
    # 在 HTTP 请求中取请求 ID，使日志、响应头 X-Request-ID 和追踪记录用同一个值。
    trace_id: str = field(default_factory=lambda: current_request_id() or uuid.uuid4().hex)
    stages: list[TraceStage] = field(default_factory=list)
    candidates: list[TraceDocument] = field(default_factory=list)
    web_results: list[TraceDocument] = field(default_factory=list)
    context: list[TraceContextSpan] = field(default_factory=list)
    prompt_chars: int = 0
    cited_ids: list[int] = field(default_factory=list)
    _current: tuple[str, float] | None = None

    def start(self, name: str) -> None:
        """开始计时一个阶段；出错时 `fail_current` 据此记录是哪个阶段失败。"""
        self._current = (name, time.perf_counter())

    def finish(self, status: StageStatus = "ok", **detail: object) -> TraceStage:
        """结束当前阶段并记录结果。"""
        name, started = self._current or ("unknown", time.perf_counter())
        self._current = None
        record: TraceStage = {
            "stage": name,
            "status": status,
            "elapsed_ms": round(max(0.0, time.perf_counter() - started) * 1000, 2),
            "detail": detail,
        }
        self.stages.append(record)
        return record

    def record(
        self, name: str, status: StageStatus, elapsed_ms: float, **detail: object
    ) -> TraceStage:
        """记录一个与其他阶段并行执行、自行计时的阶段。"""
        record: TraceStage = {
            "stage": name,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "detail": detail,
        }
        self.stages.append(record)
        return record

    def discard(self) -> None:
        """放弃当前阶段，用于事后发现该阶段并未实际执行的情况。"""
        self._current = None

    def fail_current(self, error: BaseException | str) -> None:
        """把正在进行的阶段记为失败；没有进行中的阶段时不做任何事。"""
        if self._current is not None:
            self.finish("error", error=error_summary(error))

    def cancel_current(self) -> None:
        """把正在进行的阶段记为被取消；没有进行中的阶段时不做任何事。"""
        if self._current is not None:
            self.finish("cancelled")

    def payload(self, diagnostics: "RetrievalDiagnostics", status: str) -> ChatTracePayload:
        """合并本对象与检索诊断中的事实，生成追踪记录和排查提示。"""
        result: ChatTracePayload = {
            "id": self.trace_id,
            "created_at": application_config.db_now(),
            "kb_name": self.kb_name,
            "status": status,
            "question": diagnostics.original_question[:TRACE_QUESTION_CHARS],
            "actual_query": diagnostics.actual_query[:TRACE_QUESTION_CHARS],
            "variant_query": diagnostics.variant_query[:TRACE_QUESTION_CHARS],
            "rewritten": diagnostics.rewritten,
            "skipped": diagnostics.skipped,
            "fallback_attempted": diagnostics.fallback_attempted,
            "rerank_status": diagnostics.rerank_status,
            "stages": list(self.stages),
            "candidates": list(self.candidates),
            "reranked": list(diagnostics.rerank_results),
            "web_results": list(self.web_results),
            "agentic_queries": list(diagnostics.agentic_queries),
            "context": list(self.context),
            "prompt_chars": self.prompt_chars,
            "cited_ids": list(self.cited_ids),
            "hints": [],
        }
        result["hints"] = diagnose(result)
        return result

    def save(
        self,
        diagnostics: "RetrievalDiagnostics",
        status: str | None,
        *,
        run_id: int | None,
        assistant_message_id: int | None = None,
        container: ApplicationContainer | None = None,
    ) -> None:
        """生成追踪记录并写库；状态缺失视为被取消，与运行记录的规则一致。"""
        save_trace(
            self.payload(diagnostics, status or "cancelled"),
            account_id=self.account_id,
            session_id=self.session_id,
            run_id=run_id,
            assistant_message_id=assistant_message_id,
            container=container,
        )


def diagnose(trace: ChatTracePayload) -> list[str]:
    """按已记录的事实给出问题可能出在哪个环节的提示，不额外调用模型。"""
    hints: list[str] = []
    status = trace["status"]
    if trace["skipped"]:
        hints.append("本轮由模型判断无需检索，回答没有使用知识库资料。")
    elif any(stage["status"] == "error" for stage in trace["stages"]):
        failed = next(stage["stage"] for stage in trace["stages"] if stage["status"] == "error")
        hints.append(f"在“{failed}”阶段出错，后续环节没有执行。")
    elif not trace["candidates"]:
        hints.append("召回为 0：检查文件是否已入库成功、检索筛选条件是否过严，或换个问法。")
    elif trace["rerank_status"] == "degraded":
        hints.append("重排服务不可用，本轮按召回顺序使用候选，排序质量可能下降。")
    elif trace["rerank_status"] == "success":
        top = max((doc.get("score", 0.0) for doc in trace["reranked"]), default=0.0)
        if not any(doc.get("kept") for doc in trace["reranked"]):
            hints.append(
                f"有 {len(trace['candidates'])} 条候选，但重排最高分只有 {top:.3f}，全部低于相关度下限："
                "库中可能没有答案，或召回没有找到对应段落。"
            )
        elif top < LOW_TOP_SCORE:
            hints.append(
                f"重排最高分只有 {top:.3f}：召回的片段与问题相关性不高，回答可能缺少依据。"
            )
    web = next((stage for stage in trace["stages"] if stage["stage"] == "web_search"), None)
    if web is not None and web["status"] != "ok":
        reason = web["detail"].get("reason") or "没有搜到结果"
        hints.append(f"联网搜索未提供资料：{reason}。")
    if any(span["truncated"] for span in trace["context"]):
        hints.append("上下文达到字符上限，靠后的片段被截断。")
    if status == "answered" and trace["context"] and not trace["cited_ids"]:
        hints.append("资料已送给模型，但回答没有引用任何来源：问题可能出在生成环节。")
    if status == "interrupted":
        hints.append("回答生成中途失败，内容可能不完整。")
    elif status == "error" and not any(stage["status"] == "error" for stage in trace["stages"]):
        hints.append("回答模型调用失败。")
    return hints


def _bounded_json(payload: ChatTracePayload) -> str:
    """超过大小上限时先去掉候选原文，仍超限再只保留阶段和提示。"""
    text = json.dumps(payload, ensure_ascii=False)
    if len(text.encode()) <= TRACE_MAX_BYTES:
        return text
    trimmed = dict(payload)
    for key in ("candidates", "reranked", "web_results"):
        trimmed[key] = [{**doc, "snippet": ""} for doc in payload[key]]
    text = json.dumps(trimmed, ensure_ascii=False)
    if len(text.encode()) <= TRACE_MAX_BYTES:
        return text
    return json.dumps(
        {**trimmed, "candidates": [], "reranked": [], "web_results": [], "question": payload["question"][:200]},
        ensure_ascii=False,
    )


def save_trace(
    payload: ChatTracePayload,
    *,
    account_id: int,
    session_id: str,
    run_id: int | None,
    assistant_message_id: int | None,
    container: ApplicationContainer | None = None,
) -> None:
    """写入一条追踪记录；失败只记警告，不影响回答。"""
    try:
        with application_db.connect(database_file(container)) as conn:
            conn.execute(
                "INSERT INTO chat_traces (id, created_at, account_id, kb_name, session_id, "
                "run_id, assistant_message_id, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    payload["id"],
                    payload["created_at"],
                    account_id,
                    payload["kb_name"],
                    session_id,
                    run_id,
                    assistant_message_id,
                    _bounded_json(payload),
                ),
            )
    except Exception as exc:
        logger.warning("Failed to save chat trace %s: %s", payload["id"], exc, exc_info=True)


@dataclass(frozen=True)
class StoredTrace:
    """读取到的追踪记录及其归属，供接口层做权限校验。"""

    account_id: int
    kb_name: str
    payload: ChatTracePayload


def load_trace(
    trace_id: str, *, container: ApplicationContainer | None = None
) -> StoredTrace:
    """按 ID 读取追踪记录；不存在时抛出 404 对应的错误。"""
    with application_db.connect(database_file(container)) as conn:
        row = conn.execute(
            "SELECT account_id, kb_name, payload FROM chat_traces WHERE id = ?",
            (trace_id,),
        ).fetchone()
    if row is None:
        raise ResourceNotFoundError("追踪记录不存在或已过期", code="chat_trace_not_found")
    try:
        payload = json.loads(row[2])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResourceNotFoundError("追踪记录已损坏", code="chat_trace_not_found") from exc
    return StoredTrace(int(row[0]), str(row[1]), payload)


def prune(conn: sqlite3.Connection, cutoff: str) -> int:
    """删除早于 `cutoff` 的追踪记录，与运行记录共用保留期。"""
    return conn.execute("DELETE FROM chat_traces WHERE created_at < ?", (cutoff,)).rowcount


Feedback = Literal[-1, 0, 1]
EXPORT_LIMIT_MAX = 1000


def set_feedback(
    trace_id: str,
    account_id: int,
    feedback: Feedback,
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """记录提问者对回答的评价；只有追踪记录的所属账号可以评价。"""
    with application_db.connect(database_file(container)) as conn:
        updated = conn.execute(
            "UPDATE chat_traces SET feedback = ? WHERE id = ? AND account_id = ?",
            (feedback, trace_id, account_id),
        ).rowcount
    if not updated:
        raise ResourceNotFoundError("追踪记录不存在或已过期", code="chat_trace_not_found")


def evaluation_rows(
    feedback: Feedback,
    *,
    limit: int,
    kb_names: Sequence[str] | None = None,
    container: ApplicationContainer | None = None,
) -> list[StoredTrace]:
    """按时间倒序取出指定评价的追踪记录，供导出为评测题目；损坏的记录跳过。

    `kb_names` 为 None 表示不限知识库；先按知识库过滤再取 `limit` 条。
    """
    if kb_names is not None and not kb_names:
        return []
    kb_filter = ""
    params: tuple[object, ...] = (feedback,)
    if kb_names is not None:
        kb_filter = f" AND kb_name IN ({','.join('?' for _ in kb_names)})"
        params += tuple(kb_names)
    with application_db.connect(database_file(container)) as conn:
        rows = conn.execute(
            "SELECT account_id, kb_name, payload FROM chat_traces WHERE feedback = ?"
            f"{kb_filter} ORDER BY created_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    result: list[StoredTrace] = []
    for row in rows:
        try:
            result.append(StoredTrace(int(row[0]), str(row[1]), json.loads(row[2])))
        except (json.JSONDecodeError, TypeError):
            continue
    return result


def evaluation_question(trace: ChatTracePayload) -> dict[str, object]:
    """转换为 `evaluation/questions.jsonl` 的一行；gold 先填回答引用的文件，需人工核对。"""
    cited = set(trace["cited_ids"])
    gold = sorted({span["source"] for span in trace["context"] if span["id"] in cited})
    return {"id": trace["id"], "question": trace["question"], "gold": gold, "kb_name": trace["kb_name"]}
