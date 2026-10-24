"""Application use cases for knowledge-base lifecycle operations."""

import logging
import os
import shutil
import sqlite3
from contextlib import ExitStack

from services.core import config as application_config
from services.core.container import get_application_container
from services.core.errors import (
    InvalidRequestError,
    InvalidStateError,
    OperationFailedError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from services.documents import mineru as mineru_client
from services.indexing import pipeline as indexing_pipeline
from services.indexing import worker as index_worker
from services.retrieval import chroma as chroma_index
from services.retrieval import fts as fts_index
from services.retrieval import index_coordination, retriever, source_state

logger = logging.getLogger(__name__)


def _validated_name(name: str) -> str:
    """校验知识库名称，并把值错误转换为统一的应用错误。"""
    try:
        return application_config.validate_kb_name(name)
    except ValueError as exc:
        raise InvalidRequestError(
            str(exc), code="invalid_knowledge_base_name"
        ) from exc


def create_knowledge_base(name: str) -> None:
    """Create a knowledge base with the application's default chunk settings."""
    knowledge_bases = get_application_container().knowledge_bases
    name = _validated_name(name)
    try:
        knowledge_bases.create(
            name,
            application_config.DEFAULT_CHUNK_SIZE,
            application_config.DEFAULT_CHUNK_OVERLAP,
        )
    except sqlite3.IntegrityError as exc:
        raise InvalidStateError(
            "Knowledge base already exists", code="knowledge_base_already_exists"
        ) from exc


def schedule_rebuild(name: str) -> str:
    """Create the latest durable rebuild job and wake the index worker."""
    index_jobs = get_application_container().index_jobs()
    try:
        job_id = index_jobs.prepare_rebuild(name)
    except ValueError as exc:
        raise ResourceNotFoundError(
            str(exc), code="knowledge_base_not_found"
        ) from exc
    retriever.drop_cached_retriever(name)
    index_worker.enqueue(job_id)
    return job_id


def delete_knowledge_base(name: str) -> None:
    """Delete indexes, relational state, source files, artifacts, and caches."""
    dependencies = get_application_container()
    documents = dependencies.documents
    index_jobs = dependencies.index_jobs()
    knowledge_bases = dependencies.knowledge_bases
    with indexing_pipeline.knowledge_base_lock(name):
        if not knowledge_bases.exists(name):
            raise ResourceNotFoundError(
                "知识库不存在", code="knowledge_base_not_found"
            )
        try:
            chroma_index.chroma_delete_by_knowledge_base(name)
            fts_index.remove_fts_index(name)
        except Exception as exc:
            raise ServiceUnavailableError(
                f"索引清理失败，知识库未删除：{exc}",
                code="knowledge_base_index_cleanup_failed",
            ) from exc

        index_jobs.supersede(name, "知识库已删除")
        knowledge_bases.delete_relational_data(name)
        directory = documents.knowledge_base_path(name)
        if os.path.exists(directory):
            shutil.rmtree(directory)
        mineru_client.remove_saved_results_for_knowledge_base(name)
        if name == source_state.COMMON_KNOWLEDGE_BASE:
            retriever.clear_retriever_cache()
        else:
            retriever.drop_cached_retriever(name)

    chroma_index.reset_empty_chroma_storage(
        documents.has_documents, retriever.clear_retriever_cache
    )
    index_jobs.redact_orphaned_filenames(name)
    index_jobs.compact_database()


def rename_knowledge_base(old_name: str, new_name: str) -> None:
    """Rename every knowledge-base resource, compensating completed moves on failure."""
    dependencies = get_application_container()
    documents = dependencies.documents
    index_jobs = dependencies.index_jobs()
    knowledge_bases = dependencies.knowledge_bases
    new_name = _validated_name(new_name)
    if new_name == old_name:
        return

    old_dir = documents.knowledge_base_path(old_name)
    new_dir = documents.knowledge_base_path(new_name)
    new_index_dir = fts_index.vector_store_dir(new_name)
    old_mineru_dir = mineru_client.result_dir_for_knowledge_base(old_name)
    new_mineru_dir = mineru_client.result_dir_for_knowledge_base(new_name)
    with ExitStack() as locks:
        for lock_name in sorted({old_name, new_name}):
            locks.enter_context(indexing_pipeline.knowledge_base_lock(lock_name))
        if not knowledge_bases.exists(old_name):
            raise ResourceNotFoundError(
                "知识库不存在", code="knowledge_base_not_found"
            )
        if knowledge_bases.exists(new_name):
            raise InvalidStateError(
                "知识库名称已存在", code="knowledge_base_already_exists"
            )
        # 解析阶段不持有知识库锁；此时移动目录会让运行中的任务丢失源文件快照，
        # 文档也会停留在处理中。持锁时 running 状态是稳定的，只能等任务结束再改名。
        if index_jobs.has_running(old_name):
            raise InvalidStateError(
                "知识库有正在处理的文档，请等待处理完成后再重命名",
                code="knowledge_base_has_running_jobs",
            )
        if os.path.exists(new_dir) or os.path.exists(new_index_dir) or (
            old_mineru_dir.exists() and new_mineru_dir.exists()
        ):
            raise InvalidStateError(
                "目标知识库目录或索引已存在",
                code="knowledge_base_target_exists",
            )

        data_moved = False
        indexes_renamed = False
        mineru_moved = False
        try:
            if os.path.exists(old_dir):
                os.rename(old_dir, new_dir)
                data_moved = True
            index_coordination.rename_knowledge_base_indexes(old_name, new_name)
            indexes_renamed = True
            mineru_moved = mineru_client.rename_saved_results_for_knowledge_base(
                old_name, new_name
            )
            knowledge_bases.rename_relational_data(
                old_name, new_name, old_dir, new_dir
            )
        except Exception as exc:
            rollback_errors = []
            if mineru_moved:
                try:
                    mineru_client.rename_saved_results_for_knowledge_base(
                        new_name, old_name
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(f"MinerU: {rollback_exc}")
            if indexes_renamed:
                try:
                    index_coordination.rename_knowledge_base_indexes(
                        new_name, old_name
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(f"索引: {rollback_exc}")
            if data_moved:
                try:
                    os.rename(new_dir, old_dir)
                except Exception as rollback_exc:
                    rollback_errors.append(f"原文件: {rollback_exc}")
            if rollback_errors:
                logger.error(
                    "Knowledge-base rename rollback was incomplete for %s -> %s: %s",
                    old_name,
                    new_name,
                    "; ".join(rollback_errors),
                )
                raise OperationFailedError(
                    "知识库重命名失败，且部分文件回滚失败，请检查服务日志",
                    code="knowledge_base_rename_rollback_failed",
                ) from exc
            if isinstance(exc, sqlite3.IntegrityError):
                raise InvalidStateError(
                    "Knowledge base name already exists",
                    code="knowledge_base_already_exists",
                ) from exc
            if isinstance(exc, OSError):
                raise OperationFailedError(
                    f"知识库文件迁移失败: {exc}",
                    code="knowledge_base_file_move_failed",
                ) from exc
            raise OperationFailedError(
                "知识库重命名失败，请检查服务日志",
                code="knowledge_base_rename_failed",
            ) from exc

        retriever.rename_cached_retriever(old_name, new_name)
        index_jobs.supersede(new_name, "知识库已重命名")
