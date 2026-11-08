"""Operations monitoring endpoints: dependency status, chat run records, failed index jobs, and log tail."""

import json
import os
from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

import app_storage
from routers import schemas
from services.chat import runs as chat_runs
from services.chat import trace as chat_trace
from services.core import log_tail, readiness
from services.core.config import ALL_KNOWLEDGE_BASES_PERMISSION
from services.core.logging_setup import log_file_path
from services.core.settings import get_settings
from services.identity import auth as auth_service
from services.retrieval.chunk_index import get_chunk_index

router = APIRouter(prefix="/api/ops", tags=["ops"])

_MONITOR = auth_service.require_menu_permission("运行监控")
LOG_LINES_MAX = 500


@router.get("/overview", response_model=schemas.OpsOverviewResponse)
def ops_overview(
    days: int = Query(default=7, ge=1, le=90),
    _principal: auth_service.Principal = Depends(_MONITOR),
) -> chat_runs.RunsOverview:
    """返回最近 N 天问答的状态分布、耗时分位数和按天趋势。"""
    return chat_runs.overview(days)


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
    kb: str | None = Query(default=None, max_length=80),
    slow_ms: float | None = Query(default=None, ge=0),
    feedback: int | None = Query(default=None, ge=-1, le=1),
    limit: int = Query(default=20, ge=1, le=chat_runs.LIST_LIMIT_MAX),
    offset: int = Query(default=0, ge=0),
    _principal: auth_service.Principal = Depends(_MONITOR),
) -> dict:
    """按时间倒序分页返回问答记录，可按状态、知识库、最小总耗时和用户评价过滤。"""
    total, items = chat_runs.list_runs(
        status=status, kb_name=kb or None, min_total_ms=slow_ms, feedback=feedback,
        limit=limit, offset=offset,
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/chat-traces/export", response_class=PlainTextResponse)
def ops_export_chat_traces(
    rating: Literal["down", "up"] = "down",
    limit: int = Query(default=200, ge=1, le=chat_trace.EXPORT_LIMIT_MAX),
    principal: auth_service.Principal = Depends(_MONITOR),
) -> PlainTextResponse:
    """把带评价的问答导出为评测题目（JSON Lines），只包含调用者能读的知识库。"""
    readable = (
        None
        if ALL_KNOWLEDGE_BASES_PERMISSION in principal.knowledge_base_permissions
        else principal.knowledge_base_permissions
    )
    lines = [
        json.dumps(chat_trace.evaluation_question(stored.payload), ensure_ascii=False)
        for stored in chat_trace.evaluation_rows(
            -1 if rating == "down" else 1, limit=limit, kb_names=readable
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
    _principal: auth_service.Principal = Depends(_MONITOR),
) -> list:
    """跨知识库返回最近的索引任务，默认只看失败任务。"""
    return app_storage.index_jobs.list_recent_all(status, limit)


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
