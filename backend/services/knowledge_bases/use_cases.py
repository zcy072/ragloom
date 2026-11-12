"""Application use cases for knowledge-base lifecycle operations."""

import logging
import os
import shutil

from services.core import config as application_config
from services.core.container import get_application_container
from services.core.contracts import KnowledgeBase
from services.core.errors import (
    InvalidRequestError,
    InvalidStateError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from services.core.knowledge_bases import KnowledgeBaseNameConflict
from services.documents import mineru as mineru_client
from services.indexing import pipeline as indexing_pipeline
from services.indexing import worker as index_worker
from services.retrieval import retriever
from services.retrieval.chunk_index import get_chunk_index

logger = logging.getLogger(__name__)


def _validated_name(name: str) -> str:
    """校验知识库名称，并把值错误转换为统一的应用错误。"""
    try:
        return application_config.validate_kb_name(name)
    except ValueError as exc:
        raise InvalidRequestError(
            str(exc), code="invalid_knowledge_base_name"
        ) from exc


def _name_conflict() -> InvalidStateError:
    return InvalidStateError(
        "知识库名称已存在", code="knowledge_base_already_exists"
    )


def create_knowledge_base(name: str, *, grant_role_id: int | None = None) -> KnowledgeBase:
    """Create a knowledge base with the application's default chunk settings.

    grant_role_id 是只管理部分知识库的创建者角色，新知识库同时授予它，否则创建者自己看不到。
    """
    knowledge_bases = get_application_container().knowledge_bases
    name = _validated_name(name)
    try:
        return knowledge_bases.create(
            name,
            application_config.DEFAULT_CHUNK_SIZE,
            application_config.DEFAULT_CHUNK_OVERLAP,
            grant_role_id=grant_role_id,
        )
    except KnowledgeBaseNameConflict as exc:
        raise _name_conflict() from exc


def schedule_rebuild(kb_id: str, *, chunk_config: tuple[int, int] | None = None) -> str:
    """Create the latest durable rebuild job and wake the index worker.

    `chunk_config` is saved in the same transaction as the job, so a failed schedule leaves
    the old configuration in place.
    """
    index_jobs = get_application_container().index_jobs()
    try:
        job_id = index_jobs.prepare_rebuild(kb_id, chunk_config=chunk_config)
    except ValueError as exc:
        raise ResourceNotFoundError(
            str(exc), code="knowledge_base_not_found"
        ) from exc
    retriever.drop_cached_retriever(kb_id, get_application_container())
    index_worker.enqueue(job_id)
    return job_id


def _reject_common_knowledge_base(kb_id: str) -> None:
    """通用知识库只在空库初始化时创建，删除后部门知识库的补充检索会静默失效。"""
    if kb_id == get_application_container().knowledge_bases.common_kb_id():
        raise InvalidStateError(
            "通用知识库不能删除", code="common_knowledge_base_protected"
        )


def rename_knowledge_base(kb_id: str, new_name: str) -> KnowledgeBase:
    """只改显示名称：身份是 kb_id，文件、索引、任务、权限和缓存都不受影响。"""
    knowledge_bases = get_application_container().knowledge_bases
    new_name = _validated_name(new_name)
    try:
        return knowledge_bases.rename(kb_id, new_name)
    except KnowledgeBaseNameConflict as exc:
        raise _name_conflict() from exc
    except LookupError as exc:
        raise ResourceNotFoundError(
            "知识库不存在", code="knowledge_base_not_found"
        ) from exc


def _cleanup_external_resources(kb_id: str) -> None:
    """清理 Milvus、原文件目录和 MinerU 结果；每一步都可重复执行。"""
    dependencies = get_application_container()
    get_chunk_index().delete_kb(kb_id)
    directory = dependencies.documents.knowledge_base_path(kb_id)
    if os.path.exists(directory):
        shutil.rmtree(directory)
    mineru_client.remove_saved_results_for_knowledge_base(kb_id)


def _finish_delete(kb_id: str) -> None:
    """删除一个已处于 deleting 状态的知识库；调用方须持有它的整库锁。"""
    dependencies = get_application_container()
    # 先让排队和执行中的任务失效，再清理外部资源；清理失败时也不会留下可执行的任务。
    dependencies.index_jobs().supersede(kb_id, "知识库已删除")
    _cleanup_external_resources(kb_id)
    dependencies.knowledge_bases.delete_relational_data(kb_id)
    retriever.drop_cached_retriever(kb_id, get_application_container())
    # 不在请求线程里 VACUUM：它会锁住整个数据库的写入，交给启动时和每日清理压缩。
    dependencies.index_jobs().redact_orphaned_filenames(kb_id)


def delete_knowledge_base(kb_id: str) -> None:
    """先标记 deleting 使其对用户和新任务不可见，再清理外部资源，最后级联删除记录。

    任何一步失败都保留 deleting 状态，由用户重试或下次启动继续清理。
    """
    dependencies = get_application_container()
    try:
        kb_id = application_config.validate_kb_id(kb_id)
    except ValueError as exc:
        raise ResourceNotFoundError("知识库不存在", code="knowledge_base_not_found") from exc
    _reject_common_knowledge_base(kb_id)
    with indexing_pipeline.knowledge_base_lock(kb_id):
        if dependencies.knowledge_bases.mark_deleting(kb_id) is None:
            raise ResourceNotFoundError(
                "知识库不存在", code="knowledge_base_not_found"
            )
        try:
            _finish_delete(kb_id)
        except Exception as exc:
            logger.exception("Knowledge base %s deletion is incomplete", kb_id)
            raise ServiceUnavailableError(
                f"知识库清理未完成，可稍后重试：{exc}",
                code="knowledge_base_cleanup_failed",
            ) from exc


def resume_pending_deletions() -> int:
    """启动时继续上次未完成的删除，须在 index worker 启动之前调用；返回完成数量。"""
    dependencies = get_application_container()
    done = 0
    for kb_id in dependencies.knowledge_bases.deleting_ids():
        try:
            with indexing_pipeline.knowledge_base_lock(kb_id):
                _finish_delete(kb_id)
            done += 1
        except Exception:
            logger.exception("Failed to resume deletion of knowledge base %s", kb_id)
    return done
