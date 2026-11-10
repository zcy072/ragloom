"""Operations monitoring endpoints: dependency status, chat run records, failed index jobs, and log tail."""

import json
import os
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

import app_storage
from routers import schemas
from services.chat import runs as chat_runs
from services.chat import source_access
from services.chat import trace as chat_trace
from services.chat.trace import ChatTracePayload
from services.core import log_tail, readiness
from services.core.container import ApplicationContainer, get_application_container
from services.core.logging_setup import log_file_path
from services.core.settings import get_settings
from services.identity import auth as auth_service
from services.retrieval.chunk_index import get_chunk_index

router = APIRouter(prefix="/api/ops", tags=["ops"])

_MONITOR = auth_service.require_menu_permission("运行监控")
LOG_LINES_MAX = 500


def _readable_kb_ids(principal: auth_service.Principal) -> list[str] | None:
    """返回调用者能读的知识库 ID；能读全部知识库时返回 None，表示不限。"""
    return None if principal.all_knowledge_bases else list(principal.knowledge_base_ids)


@router.get("/overview", response_model=schemas.OpsOverviewResponse)
def ops_overview(
    days: int = Query(default=7, ge=1, le=90),
    principal: auth_service.Principal = Depends(_MONITOR),
) -> chat_runs.RunsOverview:
    """返回最近 N 天问答的状态分布、耗时分位数和按天趋势，只统计调用者能读的知识库。"""
    return chat_runs.overview(days, kb_ids=_readable_kb_ids(principal))


@router.get("/dependencies", response_model=list[schemas.OpsDependencyResponse])
def ops_dependencies(
    _principal: auth_service.Principal = Depends(_MONITOR),
) -> list[readiness.DependencyStatus]:
    """检查 SQLite 和 Milvus 是否可查询，返回耗时和错误详情。"""
    return readiness.check_dependencies(app_storage.DB_FILE, get_chunk_index())


@router.get("/chat-runs", response_model=schemas.OpsChatRunPageResponse)
def ops_chat_runs(
    status: Literal[
        "answered", "no_evidence", "direct", "interrupted", "cancelled", "error"
    ] | None = None,
    kb_id: str | None = Query(default=None, max_length=36),
    slow_ms: float | None = Query(default=None, ge=0),
    feedback: int | None = Query(default=None, ge=-1, le=1),
    limit: int = Query(default=20, ge=1, le=chat_runs.LIST_LIMIT_MAX),
    offset: int = Query(default=0, ge=0),
    principal: auth_service.Principal = Depends(_MONITOR),
) -> dict:
    """按时间倒序分页返回问答记录，可按状态、知识库、最小总耗时和用户评价过滤。

    只返回调用者能读的 active 知识库的记录。
    """
    total, items = chat_runs.list_runs(
        status=status, kb_id=kb_id or None, kb_ids=_readable_kb_ids(principal),
        min_total_ms=slow_ms, feedback=feedback, limit=limit, offset=offset,
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/chat-traces/export", response_class=PlainTextResponse)
def ops_export_chat_traces(
    rating: Literal["down", "up"] = "down",
    limit: int = Query(default=200, ge=1, le=chat_trace.EXPORT_LIMIT_MAX),
    principal: auth_service.Principal = Depends(_MONITOR),
    container: ApplicationContainer = Depends(get_application_container),
) -> PlainTextResponse:
    """把带评价的问答导出为评测题目（JSON Lines）。

    只导出调用者能读的 active 知识库的问答；引用来源与追踪详情一样按来源知识库过滤，
    不会带出调用者不能读的知识库中的文档 ID 和文件名。
    """
    readable = source_access.kb_reader(
        principal.can_read_knowledge_base, container.knowledge_bases.exists
    )
    lines = [
        json.dumps(
            chat_trace.evaluation_question(
                cast(
                    ChatTracePayload,
                    source_access.filter_trace_payload(
                        cast(dict[str, Any], stored.payload), readable
                    ),
                )
            ),
            ensure_ascii=False,
        )
        for stored in chat_trace.evaluation_rows(
            -1 if rating == "down" else 1, limit=limit, kb_ids=_readable_kb_ids(principal)
        )
    ]
    return PlainTextResponse(
        "".join(line + "\n" for line in lines),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": 'attachment; filename="questions.jsonl"'},
    )


@router.get("/index-jobs", response_model=list[schemas.OpsIndexJobResponse])
def ops_index_jobs(
    status: Literal[
        "queued", "running", "completed", "failed", "superseded", "interrupted"
    ] | None = "failed",
    limit: int = Query(default=30, ge=1, le=100),
    principal: auth_service.Principal = Depends(_MONITOR),
) -> list:
    """跨知识库返回最近的索引任务，默认只看失败任务；只包含调用者能读的知识库。"""
    return app_storage.index_jobs.list_recent_all(
        status, limit, kb_ids=_readable_kb_ids(principal)
    )


@router.get("/logs", response_model=schemas.OpsLogResponse)
def ops_logs(
    lines: int = Query(default=200, ge=1, le=LOG_LINES_MAX),
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
    _principal: auth_service.Principal = Depends(_MONITOR),
) -> dict:
    """返回 backend.log 末尾若干行；文件不存在时说明原因而不是报错。"""
    path = log_file_path(get_settings())
    if not os.path.isfile(path):
        return {
            "available": False,
            "path": os.path.basename(path),
            "lines": [],
            "message": "日志文件不存在，可能已通过 LOG_FILE_BACKUPS=0 关闭文件日志或目录不可写。",
        }
    return {
        "available": True,
        "path": os.path.basename(path),
        "lines": log_tail.tail_lines(path, lines, level),
        "message": "",
    }
