"""Knowledge-base lifecycle, configuration, and dashboard endpoints."""

import logging
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException

import app_storage
from routers import dependencies, schemas
from services.core import config as application_config
from services.core.contracts import DashboardDocumentRow
from services.identity import auth as auth_service
from services.knowledge_bases import use_cases as knowledge_base_use_cases
from services.retrieval.chunk_index import get_chunk_index

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["knowledge-bases"])


def _format_size(size_bytes: int | None) -> str:
    """把字节数格式化为适合仪表盘展示的简短文本。"""
    if not size_bytes:
        return "0 KB"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


_DASHBOARD_STATUS_KEYS = {
    "ready": "success", "error": "fail", "processing": "processing", "rebuilding": "processing",
}
_DASHBOARD_TOP_FILE_TYPES = 3
_DASHBOARD_TREND_DAYS = 30


def _dashboard_visible_kbs(principal: auth_service.Principal) -> tuple[str, ...] | None:
    """返回受限用户可见的知识库名称；不受限时返回 None。"""
    if "全公司" in principal.knowledge_base_permissions:
        return None
    return tuple(dict.fromkeys(principal.knowledge_base_permissions))


def _aggregate_dashboard_stats(document_rows: list[DashboardDocumentRow]) -> dict:
    """把原始文档记录汇总为各知识库统计和全局计数。"""
    visible_kbs: list[str] = []
    visible_kb_set: set[str] = set()
    kb_stats: dict[str, dict] = {}
    totals = {"docs": 0, "size_bytes": 0, "ready": 0, "failed": 0, "processing": 0}
    for row in document_rows:
        kb_name = row.kb_name
        filename = row.filename
        raw_size = row.size
        status = row.status
        if kb_name not in visible_kb_set:
            visible_kb_set.add(kb_name)
            visible_kbs.append(kb_name)
        if filename is None:
            continue
        stats = kb_stats.setdefault(
            kb_name,
            {
                "name": kb_name,
                "docs": 0,
                "size_bytes": 0,
                "success": 0,
                "fail": 0,
                "processing": 0,
                "extensions": defaultdict(int),
            },
        )
        size = int(raw_size or 0)
        stats["docs"] += 1
        stats["size_bytes"] += size
        totals["docs"] += 1
        totals["size_bytes"] += size
        status_key = _DASHBOARD_STATUS_KEYS.get(status or "")
        if status_key:
            stats[status_key] += 1
        totals["ready"] += int(status == "ready")
        totals["failed"] += int(status == "error")
        totals["processing"] += int(status in ("processing", "rebuilding"))
        extension = filename.rsplit(".", 1)[-1].upper() if "." in filename else "UNKNOWN"
        stats["extensions"][extension] += 1
    return {"visible_kbs": visible_kbs, "kb_stats": kb_stats, "totals": totals}


def _format_dashboard_kb_list(kb_stats: dict[str, dict]) -> list[dict]:
    """将知识库汇总整理为接口结构，并保留数量最多的文件类型。"""
    kb_list = []
    for stats in kb_stats.values():
        types = sorted(
            stats["extensions"].items(), key=lambda item: item[1], reverse=True
        )[:_DASHBOARD_TOP_FILE_TYPES]
        kb_list.append(
            {
                "name": stats["name"],
                "docs": stats["docs"],
                "size": _format_size(stats["size_bytes"]),
                "success": stats["success"],
                "fail": stats["fail"],
                "processing": stats["processing"],
                "types": [{"ext": ext, "count": count} for ext, count in types],
            }
        )
    return kb_list


@router.get("/dashboard/stats", response_model=schemas.DashboardStatsResponse)
def dashboard_stats(
    principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("仪表盘统计")
    ),
) -> dict:
    """返回仪表盘所需的文档数量、容量和近 30 天问答趋势。"""
    allowed_kbs = _dashboard_visible_kbs(principal)
    document_rows = app_storage.knowledge_bases.dashboard_documents(allowed_kbs)
    aggregated = _aggregate_dashboard_stats(document_rows)
    trend_rows = app_storage.knowledge_bases.dashboard_trend(
        allowed_kbs,
        aggregated["visible_kbs"],
        days=_DASHBOARD_TREND_DAYS,
    )
    totals = aggregated["totals"]
    kb_list = _format_dashboard_kb_list(aggregated["kb_stats"])
    return {
        "kbs": len(aggregated["visible_kbs"]),
        "docs": totals["docs"],
        "size": _format_size(totals["size_bytes"]),
        "tasks": totals["processing"],
        "kb_list": kb_list,
        "pie_data": [
            {"name": "成功", "value": totals["ready"]},
            {"name": "失败", "value": totals["failed"]},
            {"name": "处理中", "value": totals["processing"]},
        ],
        "line_data": [
            {
                "name": row.date[5:],
                "提问数": row.question_count,
                "回答数": row.answer_count,
            }
            for row in trend_rows
        ],
    }


@router.get("/kbs", response_model=list[schemas.KnowledgeBaseSummary])
def list_knowledge_bases(
    principal: auth_service.Principal = Depends(auth_service.current_principal),
):
    """按当前登录主体的读取权限过滤并返回知识库列表。"""
    knowledge_bases = app_storage.knowledge_bases.list_summaries()
    return [kb for kb in knowledge_bases if principal.can_read_knowledge_base(kb["name"])]


@router.post("/kbs", response_model=schemas.SuccessResponse)
def create_knowledge_base(
    data: schemas.KnowledgeBaseNameRequest,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """使用默认分块配置创建知识库。"""
    knowledge_base_use_cases.create_knowledge_base(data.name)
    return {"status": "success"}


@router.delete("/kbs/{name}", response_model=schemas.SuccessResponse)
def delete_knowledge_base(
    name: dependencies.KnowledgeBaseNamePath,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """删除知识库的索引、关系数据、原始文件和检索缓存。"""
    knowledge_base_use_cases.delete_knowledge_base(name)
    return {"status": "success"}


@router.put("/kbs/{old_name}", response_model=schemas.SuccessResponse)
def rename_knowledge_base(
    old_name: dependencies.OldKnowledgeBaseNamePath,
    data: schemas.KnowledgeBaseNameRequest,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """重命名知识库的全部关联资源，失败时执行补偿回滚。"""
    knowledge_base_use_cases.rename_knowledge_base(old_name, data.name)
    return {"status": "success"}


@router.get(
    "/kbs/{name}/document_filters",
    response_model=schemas.DocumentFilterOptionsResponse,
)
def get_document_filters(
    name: dependencies.KnowledgeBaseNamePath,
    _principal=Depends(auth_service.require_knowledge_base_reader),
):
    """返回可用于检索筛选的去重文档元数据值。"""
    try:
        return app_storage.documents.filter_options(name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/kbs/{name}/index_jobs", response_model=list[schemas.IndexJobResponse])
def get_index_jobs(
    name: dependencies.KnowledgeBaseNamePath,
    _principal=Depends(auth_service.require_knowledge_base_reader),
):
    """返回指定知识库最近的持久化索引任务。"""
    try:
        return app_storage.index_jobs.list_recent(name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/kbs/{name}/chunk_config", response_model=schemas.ChunkConfigResponse)
def get_kb_chunk_config(
    name: dependencies.KnowledgeBaseNamePath,
    _principal=Depends(auth_service.require_knowledge_base_reader),
):
    """返回当前生效的父子分块配置。"""
    try:
        defaults = application_config.default_chunk_config()
        chunk_size, chunk_overlap = app_storage.knowledge_base_settings.chunk_config(
            name,
            default_size=defaults.chunk_size,
            default_overlap=defaults.chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    needs_rebuild = False
    kb_id = app_storage.knowledge_bases.kb_id_for(name)
    if kb_id and app_storage.documents.has_documents(name):
        try:
            needs_rebuild = not get_chunk_index().available_routes(kb_id, None)
        except Exception:
            logger.warning("Chunk index probe failed for %s", name, exc_info=True)
    return {
        # chunk_config 已把缺失值换成默认值；这里不能再用真值判断，否则重叠 0 会变回默认。
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "index_needs_rebuild": needs_rebuild,
    }


@router.get(
    "/kbs/{name}/retrieval_config",
    response_model=schemas.RetrievalConfigResponse,
)
def get_kb_retrieval_config(
    name: dependencies.KnowledgeBaseNamePath,
    _principal=Depends(auth_service.require_knowledge_base_reader),
):
    """返回影响查询检索行为的知识库配置。"""
    try:
        return {
            "query_rewrite_enabled": app_storage.knowledge_base_settings.query_rewrite_enabled(name)
        }
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.put(
    "/kbs/{name}/retrieval_config",
    response_model=schemas.RetrievalConfigResponse,
)
def save_kb_retrieval_config(
    name: dependencies.KnowledgeBaseNamePath,
    data: schemas.RetrievalConfigRequest,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """保存不需要重建索引的检索选项。"""
    try:
        return {
            "query_rewrite_enabled": app_storage.knowledge_base_settings.set_query_rewrite_enabled(
                name, data.query_rewrite_enabled
            )
        }
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.put("/kbs/{name}/chunk_config", response_model=schemas.RebuildResponse)
def save_kb_chunk_config(
    name: dependencies.KnowledgeBaseNamePath,
    data: schemas.ChunkConfigRequest,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """保存分块大小，并安排完整索引重建任务。"""
    chunk_size = data.chunk_size
    chunk_overlap = data.chunk_overlap
    try:
        app_storage.knowledge_base_settings.set_chunk_config(name, chunk_size, chunk_overlap)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    job_id = knowledge_base_use_cases.schedule_rebuild(name)
    return {"status": "success", "rebuilding": True, "job_id": job_id}


@router.post("/kbs/{name}/rebuild", response_model=schemas.RebuildResponse)
def rebuild_knowledge_base(
    name: dependencies.KnowledgeBaseNamePath,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """显式安排 Milvus 检索索引的完整重建。"""
    job_id = knowledge_base_use_cases.schedule_rebuild(name)
    return {"status": "success", "rebuilding": True, "job_id": job_id}
