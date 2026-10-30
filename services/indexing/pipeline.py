"""知识库入库编排：解析文件、切分内容并写入父段落表与 Milvus 子块索引。"""

import logging
import os
import shutil
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from services.core import config as application_config
from services.core.container import ApplicationContainer, get_application_container
from services.core.request_context import carry_request_id
from services.core.settings import get_settings
from services.documents import loaders as document_loader
from services.documents import mineru as mineru_client
from services.documents import visual as visual_content
from services.documents.parser_modes import PDF_PARSER_NATIVE_TEXT
from services.indexing import chunking
from services.indexing.jobs import (
    INDEX_JOB_COMPLETED,
    INDEX_JOB_FAILED,
    INDEX_JOB_SUPERSEDED,
)
from services.indexing.locks import document_lock, knowledge_base_lock
from services.indexing.workspace import JobWorkspace, create_workspace
from services.retrieval import parent_store, retriever
from services.retrieval.chunk_index import embed_splits, get_chunk_index

logger = logging.getLogger(__name__)


def _get_chunk_config(
    kb_name: str, container: ApplicationContainer
) -> tuple[int, int]:
    """读取知识库当前切分配置；旧数据缺失时使用全局默认值。

    在任务真正执行时才读取配置，确保重建采用设置页当前可见的参数。
    """
    defaults = application_config.default_chunk_config()
    try:
        return container.knowledge_base_settings.chunk_config(
            kb_name,
            default_size=defaults.chunk_size,
            default_overlap=defaults.chunk_overlap,
        )
    except ValueError:
        return defaults


def _split_for_index(
    documents,
    structured_documents,
    visual_documents=None,
    *,
    chunk_size: int,
    chunk_overlap: int,
):
    """从同一批解析结果构建互补的多路检索切片。

    固定切片提供稳定召回，结构化切片保留标题与表格，视觉切片把 MinerU
    提取的图片转成可检索文本；三者一起入库，在查询阶段再加权去重。
    """
    splits = chunking.split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if structured_documents:
        splits.extend(
            chunking.split_structured_markdown_documents(
                structured_documents,
                chunk_size=chunk_size,
            )
        )
    if visual_documents:
        splits.extend(chunking.split_mineru_visual_documents(visual_documents))
    return splits


class _SupersededError(Exception):
    """解析期间任务已被同名上传、删除或新的重建替代，后续步骤不必再做。"""


@dataclass(frozen=True)
class _PreparedJob:
    """持锁准备阶段留下的任务输入：源文件快照与当时生效的配置。"""

    workspace: JobWorkspace
    parser_mode: str | None
    chunk_size: int
    chunk_overlap: int
    # 重建时复用已保存的 MinerU 结果，避免把全库重新上传云端解析。
    reuse_saved_results: bool = False


@dataclass(frozen=True)
class _ParsedDocument:
    """无锁解析阶段的产物：切片及其向量。"""

    splits: list
    vectors: list
    seconds: dict[str, float]


def _prepare_document_job(
    workspace_id: str,
    kb_name: str,
    filename: str,
    file_path: str,
    container: ApplicationContainer,
    *,
    reuse_saved_results: bool = False,
) -> _PreparedJob:
    """在持有知识库锁时固定任务输入，使后续解析不依赖会被并发修改的文件。

    快照与数据库中该文档的当前版本一致（上传在同一把锁内完成），
    之后即使同名文件被替换或删除，本任务仍只处理自己的版本。
    """
    chunk_size, chunk_overlap = _get_chunk_config(kb_name, container)
    parser_mode = container.documents.parser_modes(kb_name).get(filename)
    workspace = create_workspace(
        workspace_id,
        os.path.dirname(file_path),
        filename,
        carry_saved_results=reuse_saved_results,
    )
    return _PreparedJob(
        workspace, parser_mode, chunk_size, chunk_overlap, reuse_saved_results
    )


def _parse_and_embed_document(
    kb_name: str,
    filename: str,
    prepared: _PreparedJob,
    container: ApplicationContainer,
    is_current: Callable[[], bool],
) -> _ParsedDocument:
    """解析、切分并计算向量；全程不持有知识库锁，只读写任务私有的工作区。

    嵌入前先确认任务仍拥有该文件，已被替代就不再调用嵌入服务。这里不持锁，
    只是提前放弃；最终归属仍由发布阶段持锁的校验决定。
    """
    seconds: dict[str, float] = {}
    started = time.monotonic()
    source_path = prepared.workspace.source_path
    parser_mode = prepared.parser_mode
    routes = document_loader.load_document_routes(
        source_path,
        filename,
        parser_mode,
        reuse_mineru_result=prepared.reuse_saved_results,
    )
    documents = routes.fixed
    structured_documents = routes.structured
    visual_documents = visual_content.load_mineru_visual_documents(
        source_path, filename, parser_mode
    )
    if not any(
        document.page_content.strip()
        for document in documents + structured_documents + visual_documents
    ):
        if os.path.splitext(filename)[1].lower() == ".pdf" and parser_mode == PDF_PARSER_NATIVE_TEXT:
            raise ValueError(
                "PDF 未提取到有效文本，请改用 MinerU 云端解析或 Paddle 本地 OCR 后重新解析"
            )
        raise ValueError("文档未提取到可索引文本，请检查文件格式或 OCR 依赖")
    seconds["parse"] = time.monotonic() - started

    started = time.monotonic()
    embeddings = container.create_embeddings()
    splits = _split_for_index(
        documents,
        structured_documents,
        visual_documents,
        chunk_size=prepared.chunk_size,
        chunk_overlap=prepared.chunk_overlap,
    )
    if not splits:
        raise ValueError("文档未提取到可索引文本，请检查文件格式或 OCR 依赖")
    seconds["split"] = time.monotonic() - started
    if not is_current():
        raise _SupersededError

    started = time.monotonic()
    # 嵌入失败直接抛出，文档进入 error；没有向量的子块不会进入索引。
    vectors = embed_splits(splits, embeddings)
    seconds["embed"] = time.monotonic() - started
    return _ParsedDocument(splits, vectors, seconds)


def _publish_document(
    kb_name: str,
    filename: str,
    prepared: _PreparedJob,
    parsed: _ParsedDocument,
    container: ApplicationContainer,
) -> None:
    """把解析结果写入正式索引；调用方须持有知识库锁并已确认任务仍是最新版本。

    顺序固定为先父段落表、后 Milvus：Milvus 写入失败时文档进入 error，
    残留的父段落没有对应子块，不会被检索到，重建时会被覆盖。
    """
    kb_id = container.knowledge_bases.kb_id_for(kb_name)
    if not kb_id:
        raise ValueError("知识库不存在")
    prepared.workspace.publish_result_dir()
    parent_store.replace_source(kb_id, filename, parent_store.parent_chunk_rows(parsed.splits))
    get_chunk_index().publish_source(kb_id, filename, parsed.splits, parsed.vectors)
    retriever.drop_cached_retriever(kb_name)


def _set_document_status(
    index_jobs,
    kb_name: str,
    filename: str,
    job_id: str,
    status: str,
    error_message: str,
) -> str:
    """写入文档状态；状态写入因归属变化被拒绝时视为任务已被替代。"""
    if not index_jobs.update_document_status(
        kb_name, filename, status, error_message, job_id=job_id
    ):
        return "superseded"
    return status


def _index_document(
    job_id: str,
    kb_name: str,
    filename: str,
    file_path: str,
    container: ApplicationContainer,
    *,
    rebuild: bool,
) -> tuple[str, str]:
    """对一个文档执行“持锁准备、无锁解析、持锁发布”，完成时立即写回文档状态。

    上传任务与重建共用这一流程，所以重建中每个文档处理完就会立刻变成就绪或出错。
    重建时 ``job_id`` 是重建任务的 ID：任务本身由调用方启动和结束，这里只负责文档。
    返回 (结果, 错误信息)，结果为 ready / error / superseded / skipped。
    """
    index_jobs = container.index_jobs()
    workspace: JobWorkspace | None = None
    try:
        # 准备：持锁固定输入。同名文件的新上传会替代旧任务，即使旧任务此前已排队。
        with document_lock(kb_name, filename):
            if not rebuild and not index_jobs.start(job_id):
                return "skipped", ""
            if not index_jobs.is_current_document_job(kb_name, filename, job_id):
                return "superseded", ""
            try:
                if not os.path.isfile(file_path):
                    raise FileNotFoundError("源文件不存在")
                prepared = _prepare_document_job(
                    # 重建中多个文档共用一个任务 ID，工作区必须各用各的目录。
                    uuid.uuid4().hex if rebuild else job_id,
                    kb_name,
                    filename,
                    file_path,
                    container,
                    reuse_saved_results=rebuild,
                )
            except Exception as exc:
                logger.exception("Document processing failed for %s", filename)
                message = str(exc)[:500]
                outcome = _set_document_status(
                    index_jobs, kb_name, filename, job_id, "error", message
                )
                return outcome, message
            workspace = prepared.workspace

        # 阶段 A：解析、切分、嵌入。不持有知识库锁，同一知识库的多个文档可并行。
        parsed: _ParsedDocument | None = None
        error_message = ""
        started = time.monotonic()
        try:
            parsed = _parse_and_embed_document(
                kb_name,
                filename,
                prepared,
                container,
                lambda: index_jobs.is_current_document_job(kb_name, filename, job_id),
            )
        except _SupersededError:
            logger.info("Index job %s [%s/%s] superseded before embedding", job_id, kb_name, filename)
            return "superseded", ""
        except Exception as exc:
            logger.exception("Document processing failed for %s", filename)
            error_message = str(exc)[:500]
        phase_a_seconds = time.monotonic() - started

        # 阶段 B：持锁发布。先确认任务仍拥有该文件的最新版本，保证后上传的版本最终生效。
        waited = time.monotonic()
        with document_lock(kb_name, filename):
            lock_wait_seconds = time.monotonic() - waited
            if not index_jobs.is_current_document_job(kb_name, filename, job_id):
                return "superseded", ""
            started = time.monotonic()
            if parsed is not None:
                try:
                    _publish_document(kb_name, filename, prepared, parsed, container)
                except Exception as exc:
                    logger.exception("Document processing failed for %s", filename)
                    error_message = str(exc)[:500]
            outcome = _set_document_status(
                index_jobs,
                kb_name,
                filename,
                job_id,
                "ready" if not error_message else "error",
                error_message,
            )
            logger.info(
                "Index job %s [%s/%s] %s: phaseA=%.1fs %s lock_wait=%.1fs publish=%.1fs",
                job_id,
                kb_name,
                filename,
                outcome,
                phase_a_seconds,
                " ".join(
                    f"{stage}={value:.1f}s"
                    for stage, value in (parsed.seconds if parsed else {}).items()
                ),
                lock_wait_seconds,
                time.monotonic() - started,
            )
            return outcome, error_message
    finally:
        if workspace is not None:
            workspace.cleanup()
            _remove_leftovers_of_deleted_knowledge_base(kb_name, container)


def run_document_job(
    job_id: str,
    kb_name: str,
    filename: str,
    file_path: str,
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """仅当任务仍拥有文件最新版本时，才执行上传后的索引任务。"""
    dependencies = container or get_application_container()
    outcome, message = _index_document(
        job_id, kb_name, filename, file_path, dependencies, rebuild=False
    )
    index_jobs = dependencies.index_jobs()
    if outcome == "skipped":
        return
    if outcome == "superseded":
        index_jobs.finish(job_id, INDEX_JOB_SUPERSEDED, "文件已被更新或删除")
        return
    job_status = INDEX_JOB_COMPLETED if outcome == "ready" else INDEX_JOB_FAILED
    index_jobs.finish(job_id, job_status, message)


def _remove_leftovers_of_deleted_knowledge_base(
    kb_name: str, container: ApplicationContainer
) -> None:
    """知识库在解析期间被删除时，清除任务重新创建的目录。

    删除操作不会等待解析，解析阶段又可能在删除之后才写出 MinerU 结果目录；
    残留的 ``mineru_results/<知识库>`` 会让之后的同名创建或改名误判目标已存在。
    每个任务结束都会调用这里，绝大多数时候知识库仍在，先不持锁判断，
    避免为一次确认去排整库独占锁；判断为已删除后再持锁复核。
    """
    try:
        if container.knowledge_bases.exists(kb_name):
            return
        with knowledge_base_lock(kb_name):
            if container.knowledge_bases.exists(kb_name):
                return
            mineru_client.remove_saved_results_for_knowledge_base(kb_name)
            kb_dir = container.documents.knowledge_base_path(kb_name)
            shutil.rmtree(kb_dir, ignore_errors=True)
    except Exception as exc:
        logger.warning("Failed to clean leftovers of deleted knowledge base %s: %s", kb_name, exc)


def _rebuild_one_document(
    job_id: str,
    kb_name: str,
    filename: str,
    container: ApplicationContainer,
) -> tuple[str, str]:
    """重建一个文档；任何意外异常都只让该文档出错，不影响其余文档。"""
    try:
        return _index_document(
            job_id,
            kb_name,
            filename,
            container.documents.document_path(kb_name, filename),
            container,
            rebuild=True,
        )
    except Exception as exc:
        logger.exception("Rebuild of %s/%s failed", kb_name, filename)
        message = str(exc)[:500]
        try:
            outcome = _set_document_status(
                container.index_jobs(), kb_name, filename, job_id, "error", message
            )
        except Exception:
            logger.exception("Could not record rebuild failure of %s/%s", kb_name, filename)
            outcome = "error"
        return outcome, message


def _drop_stale_vectors(kb_name: str, container: ApplicationContainer) -> None:
    """重建结束后，删除不再属于任何就绪文档的旧子块和父段落。

    重建不预先清空索引，失败文档的旧版本留到这里才删除。
    """
    kb_id = container.knowledge_bases.kb_id_for(kb_name)
    if not kb_id:
        return
    try:
        active_sources = set(container.documents.ready_sources(kb_name))
        get_chunk_index().delete_stale_sources(kb_id, active_sources)
        parent_store.delete_stale_sources(kb_id, active_sources)
    except Exception as exc:
        # 已删除或不可用的旧版本仍会被 SQLite 允许列表排除，不会参与检索。
        logger.warning("Stale-source cleanup failed for %s: %s", kb_name, exc)


def run_rebuild_job(
    job_id: str,
    kb_name: str,
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """只执行知识库最新发起的重建请求。

    各文档与上传一样并行处理，处理完立即写回自己的状态；
    解析失败只让该文档出错，其余文档照常重建。不预先清空索引：已就绪文档的旧版本
    在新版本发布前继续参与检索，发布时由 publish_source 先写新版本再删旧版本。
    """
    dependencies = container or get_application_container()
    index_jobs = dependencies.index_jobs()
    with knowledge_base_lock(kb_name):
        if not index_jobs.start(job_id):
            return
        if not index_jobs.is_active_rebuild(kb_name, job_id):
            index_jobs.finish(
                job_id, INDEX_JOB_SUPERSEDED, "已被新的重建任务替代"
            )
            return
        filenames = [
            record["filename"]
            for record in dependencies.documents.list_documents(kb_name)
            if record["index_job_id"] == job_id
        ]
        retriever.drop_cached_retriever(kb_name)

    outcomes: list[tuple[str, str]] = []
    if filenames:
        workers = min(get_settings().index_worker_threads, len(filenames))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rebuild-document"
        ) as executor:
            outcomes = list(
                executor.map(
                    carry_request_id(
                        lambda name: _rebuild_one_document(
                            job_id, kb_name, name, dependencies
                        )
                    ),
                    filenames,
                )
            )
    ready = sum(1 for outcome, _ in outcomes if outcome == "ready")
    failed = sum(1 for outcome, _ in outcomes if outcome == "error")

    with knowledge_base_lock(kb_name):
        # 大型重建期间配置可能变化，旧任务完成时不得覆盖新重建任务的持久化状态。
        if not index_jobs.is_active_rebuild(kb_name, job_id):
            index_jobs.finish(
                job_id, INDEX_JOB_SUPERSEDED, "已被新的重建任务替代"
            )
            return
        _drop_stale_vectors(kb_name, dependencies)
        retriever.drop_cached_retriever(kb_name)
        if not filenames:
            job_status, message = INDEX_JOB_FAILED, "当前知识库没有可重建的文档"
        elif failed and not ready:
            job_status, message = INDEX_JOB_FAILED, f"{failed} 个文档解析失败"
        elif failed:
            job_status = INDEX_JOB_COMPLETED
            message = f"{failed} 个文档解析失败，其余已重建"
        else:
            job_status, message = INDEX_JOB_COMPLETED, ""
        # 兜底：仍停在处理中的文档（意外中断）记为出错，并释放重建标记。
        index_jobs.complete_rebuild(kb_name, job_id, "error", message or "重建未完成")
        index_jobs.finish(job_id, job_status, message)


def delete_document_indexes(
    kb_name: str,
    filename: str,
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """文件删除后，从所有检索索引中移除其对应数据。"""
    dependencies = container or get_application_container()
    # 这里只在源文件已删除后调用；调用方会识别重新上传，防止误删新版本的索引。
    # 知识库已被删除时 kb_id 不存在，子块和父段落已随知识库一并清理。
    kb_id = dependencies.knowledge_bases.kb_id_for(kb_name)
    index = get_chunk_index()
    if kb_id:
        index.delete_source(kb_id, filename)
        parent_store.delete_source(kb_id, filename)
    if index.reset_if_empty(dependencies.documents.has_documents):
        retriever.clear_retriever_cache()
    retriever.drop_cached_retriever(kb_name)


def run_delete_job(
    job_id: str,
    kb_name: str,
    filename: str,
    *,
    container: ApplicationContainer | None = None,
) -> None:
    """删除失效索引；若同名文件已重新上传，则放弃本次删除任务。"""
    dependencies = container or get_application_container()
    index_jobs = dependencies.index_jobs()
    with document_lock(kb_name, filename):
        if not index_jobs.start(job_id):
            return
        if dependencies.documents.document_exists(kb_name, filename):
            index_jobs.finish(
                job_id, INDEX_JOB_SUPERSEDED, "文件已重新上传"
            )
            return
        try:
            delete_document_indexes(kb_name, filename, container=dependencies)
        except Exception as exc:
            error_message = str(exc)[:500]
            index_jobs.finish(job_id, INDEX_JOB_FAILED, error_message)
            index_jobs.compact_database()
            return
        index_jobs.finish(job_id, INDEX_JOB_COMPLETED)
        index_jobs.redact_orphaned_filenames(kb_name)
        index_jobs.compact_database()
