"""知识库入库编排：解析文件、切分内容并写入父段落表与 Milvus 子块索引。"""

import logging
import os
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from services.core import config as application_config
from services.core.container import ApplicationContainer
from services.core.errors import ResourceNotFoundError
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
from services.retrieval.chunk_index import (
    PublishRolledBackError,
    embed_splits,
    get_chunk_index,
)

logger = logging.getLogger(__name__)
# 旧父段落在新子块生效后再保留这么久：查询从检索到回表加载父段落之间要经过重排
# （最长 RERANK_TIMEOUT_SECONDS），期间命中的旧子块仍要能找到父段落。
# 进程在此期间退出时旧父段落残留，下次发布会把文档已有的全部旧 parent_id 一并删除。
OLD_PARENT_GRACE_SECONDS = 60.0

# 解析与嵌入（阶段 A）的全局并行名额。上传 worker 和每个重建任务各有线程池，
# 不共享名额时两个知识库同时重建会让 OCR 并行度翻倍，CPU 和内存跟着翻倍。
_parse_slots: threading.BoundedSemaphore | None = None
_parse_slots_lock = threading.Lock()


def _get_parse_slots() -> threading.BoundedSemaphore:
    """首次使用时按 index_worker_threads 创建；只在阶段 A 持有，不与任何锁嵌套。"""
    global _parse_slots
    with _parse_slots_lock:
        if _parse_slots is None:
            _parse_slots = threading.BoundedSemaphore(get_settings().index_worker_threads)
        return _parse_slots


def _get_chunk_config(kb_id: str, container: ApplicationContainer) -> tuple[int, int]:
    """读取知识库当前切分配置；旧数据缺失时使用全局默认值。

    在任务真正执行时才读取配置，确保重建采用设置页当前可见的参数。
    """
    defaults = application_config.default_chunk_config()
    try:
        return container.knowledge_base_settings.chunk_config(
            kb_id,
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
    """解析期间任务已被重新索引、文档删除或整库重建替代，后续步骤不必再做。"""


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
    kb_id: str,
    document_id: str,
    file_path: str,
    container: ApplicationContainer,
    *,
    job_id: str | None = None,
    reuse_saved_results: bool = False,
) -> _PreparedJob:
    """在持有该文档的文档锁时固定任务输入，使后续解析不依赖会被并发修改的文件。

    快照与数据库中该文档的当前版本一致（上传在同一把文档锁内完成）。工作区不在
    文档目录内，之后即使该文档被删除，本任务仍只读取准备时的快照。
    """
    chunk_size, chunk_overlap = _get_chunk_config(kb_id, container)
    # 单文档任务（传入 job_id）用登记在任务上的解析方式；整库重建沿用文档当前生效的解析方式。
    parser_mode = (
        job_id and container.index_jobs().requested_parser_mode(job_id)
    ) or container.documents.get_document(kb_id, document_id)["parser_mode"]
    workspace = create_workspace(
        workspace_id,
        file_path,
        carry_saved_results=reuse_saved_results,
    )
    return _PreparedJob(workspace, parser_mode, chunk_size, chunk_overlap, reuse_saved_results)


def _parse_and_embed_document(
    kb_id: str,
    document_id: str,
    filename: str,
    prepared: _PreparedJob,
    container: ApplicationContainer,
    is_current: Callable[[], bool],
) -> _ParsedDocument:
    """解析、切分并计算向量；全程不持有任何锁，只读写任务私有的工作区。

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
        if (
            os.path.splitext(filename)[1].lower() == ".pdf"
            and parser_mode == PDF_PARSER_NATIVE_TEXT
        ):
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
    for split in splits:
        split.metadata.update(kb_id=kb_id, document_id=document_id)
    seconds["split"] = time.monotonic() - started
    if not is_current():
        raise _SupersededError

    started = time.monotonic()
    # 嵌入失败直接抛出，文档进入 error；没有向量的子块不会进入索引。
    vectors = embed_splits(splits, embeddings)
    seconds["embed"] = time.monotonic() - started
    return _ParsedDocument(splits, vectors, seconds)


class _PublishError(Exception):
    """发布失败；previous_index_intact 表示该文档发布前的父段落与子块均完好。"""

    def __init__(self, message: str, *, previous_index_intact: bool) -> None:
        super().__init__(message)
        self.previous_index_intact = previous_index_intact


def _publish_document(
    kb_id: str,
    document_id: str,
    prepared: _PreparedJob,
    parsed: _ParsedDocument,
    container: ApplicationContainer,
) -> None:
    """把解析结果写入正式索引；调用方须持有该文档的文档锁并已确认任务仍是最新版本。

    检索不持锁，可能随时命中 Milvus 中任一代子块再回表取父段落，所以新旧两版
    父段落在 Milvus 切换期间同时保留：先加新父段落，再发布子块，最后删旧父段落。
    新旧 parent_id 由 uuid4 生成、互不相同，旧子块始终能找到自己的父段落。
    Milvus 写入失败且已撤销时旧版本完整，新父段落保留到下次发布，失败抛出 _PublishError。
    解析结果目录只在子块发布成功后替换，回滚时旧版本与它的结果目录保持一致。
    """
    rows = parent_store.parent_chunk_rows(parsed.splits)
    try:
        previous_parent_ids = parent_store.add_document_version(kb_id, document_id, rows)
    except Exception as exc:
        raise _PublishError(str(exc), previous_index_intact=True) from exc
    try:
        get_chunk_index().publish_document(kb_id, document_id, parsed.splits, parsed.vectors)
    except PublishRolledBackError as exc:
        # 新父段落不撤回：insert 超时后服务端可能在撤销之后才写入这批子块，留着父段落，
        # 这些残留子块仍能回表取到完整段落。没有子块引用的父段落不影响检索，
        # 下次发布会把文档已有的全部旧 parent_id 一并删除，删除文档时级联删除。
        raise _PublishError(str(exc), previous_index_intact=True) from exc
    except Exception as exc:
        # 新旧两代子块可能都有残留，两版父段落都保留，以免任何残留子块丢失上下文。
        raise _PublishError(str(exc), previous_index_intact=False) from exc
    retriever.drop_cached_retriever(kb_id, container)
    try:
        # 结果目录检索时不读取，只供重建复用；替换失败会恢复旧目录。
        prepared.workspace.publish_result_dir()
    except OSError:
        # 新版本已生效；旧结果目录只影响下次整库重建复用的解析结果。
        logger.warning("Result directory replacement failed for %s/%s", kb_id, document_id)
    _schedule_old_parent_removal(kb_id, document_id, previous_parent_ids)


def _remove_old_parents(kb_id: str, document_id: str, parent_ids: set[str]) -> None:
    try:
        parent_store.delete_parents(kb_id, document_id, parent_ids)
    except Exception:
        # 新版本已生效；残留的旧父段落不再被任何子块引用，下次发布会一并删除。
        logger.warning("Removal of old parent rows failed for %s/%s", kb_id, document_id)


def _schedule_old_parent_removal(kb_id: str, document_id: str, parent_ids: set[str]) -> None:
    """宽限期后删除旧父段落；parent_id 是 uuid4，届时文档已再次发布或被删除也不会误删。"""
    if not parent_ids:
        return
    if OLD_PARENT_GRACE_SECONDS <= 0:
        _remove_old_parents(kb_id, document_id, parent_ids)
        return
    timer = threading.Timer(
        OLD_PARENT_GRACE_SECONDS, _remove_old_parents, (kb_id, document_id, parent_ids)
    )
    timer.daemon = True
    timer.start()


def _set_document_status(
    index_jobs,
    kb_id: str,
    document_id: str,
    job_id: str,
    status: str,
    error_message: str,
    *,
    previous_index_intact: bool = False,
    parser_mode: str | None = None,
) -> str:
    """写入文档状态；状态写入因归属变化被拒绝时视为任务已被替代。

    失败时尚未开始发布的 rebuilding 文档回到 ready；publishing 文档在 previous_index_intact
    （发布已完整撤销）时回到 ready，否则记为 error。返回值仍是 error，任务记为失败、重建计入失败数。
    """
    if status == "error":
        written = index_jobs.fail_document(
            kb_id,
            document_id,
            job_id,
            error_message,
            previous_index_intact=previous_index_intact,
        )
    else:
        written = index_jobs.update_document_status(
            kb_id, document_id, status, error_message, job_id=job_id, parser_mode=parser_mode
        )
    return status if written else "superseded"


def _index_document(
    job_id: str,
    kb_id: str,
    document_id: str,
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
    filename = os.path.basename(file_path)
    index_jobs = container.index_jobs()
    workspace: JobWorkspace | None = None
    try:
        # 准备：持锁固定输入。同一文档的新重索引请求会替代旧任务，即使旧任务此前已排队。
        with document_lock(kb_id, document_id):
            if not rebuild and not index_jobs.start(job_id):
                return "skipped", ""
            if not index_jobs.is_current_document_job(kb_id, document_id, job_id):
                return "superseded", ""
            try:
                if not os.path.isfile(file_path):
                    raise FileNotFoundError("源文件不存在")
                prepared = _prepare_document_job(
                    # 重建中多个文档共用一个任务 ID，工作区必须各用各的目录。
                    uuid.uuid4().hex if rebuild else job_id,
                    kb_id,
                    document_id,
                    file_path,
                    container,
                    job_id=None if rebuild else job_id,
                    reuse_saved_results=rebuild,
                )
            except Exception as exc:
                logger.exception("Document processing failed for %s", document_id)
                message = str(exc)[:500]
                # 准备阶段只读源文件、建私有快照，旧索引未被改动。
                outcome = _set_document_status(
                    index_jobs,
                    kb_id,
                    document_id,
                    job_id,
                    "error",
                    message,
                    previous_index_intact=True,
                )
                return outcome, message
            workspace = prepared.workspace

        # 阶段 A：解析、切分、嵌入。不持有知识库锁，同一知识库的多个文档可并行；
        # 全局并行数受 _parse_slots 限制，上传与重建共用。
        parsed: _ParsedDocument | None = None
        error_message = ""
        started = time.monotonic()
        try:
            with _get_parse_slots():
                parsed = _parse_and_embed_document(
                    kb_id,
                    document_id,
                    filename,
                    prepared,
                    container,
                    lambda: index_jobs.is_current_document_job(kb_id, document_id, job_id),
                )
        except _SupersededError:
            logger.info(
                "Index job %s [%s/%s] superseded before embedding", job_id, kb_id, document_id
            )
            return "superseded", ""
        except Exception as exc:
            logger.exception("Document processing failed for %s", document_id)
            error_message = str(exc)[:500]
        phase_a_seconds = time.monotonic() - started

        # 阶段 B：持锁发布。先确认任务仍拥有该文档的最新索引请求，保证最新任务最终生效。
        waited = time.monotonic()
        with document_lock(kb_id, document_id):
            lock_wait_seconds = time.monotonic() - waited
            if not index_jobs.is_current_document_job(kb_id, document_id, job_id):
                return "superseded", ""
            # 解析期间知识库可能已进入删除流程；此时不得发布任何正式文件、父段落或向量。
            if not container.knowledge_bases.exists(kb_id):
                return "superseded", ""
            started = time.monotonic()
            # 解析或嵌入失败时尚未发布，旧索引完好；发布失败时由发布步骤判断旧索引是否完好。
            previous_index_intact = True
            if parsed is not None:
                # 先标为 publishing：此后旧索引可能被改动，重启或意外异常时据此不再回到 ready。
                index_jobs.begin_publish(kb_id, document_id, job_id)
                try:
                    _publish_document(kb_id, document_id, prepared, parsed, container)
                except _PublishError as exc:
                    logger.exception("Document publish failed for %s", document_id)
                    error_message = str(exc)[:500]
                    previous_index_intact = exc.previous_index_intact
            outcome = _set_document_status(
                index_jobs,
                kb_id,
                document_id,
                job_id,
                "ready" if not error_message else "error",
                error_message,
                previous_index_intact=previous_index_intact,
                # 新版本已发布，文档改记它实际使用的解析方式。
                parser_mode=prepared.parser_mode if not error_message else None,
            )
            if parsed is not None:
                # 发布时清过一次缓存，但首次上传的文档此刻才进入就绪列表；这期间建好的
                # 检索图固定了不含它的文档列表，不再清一次会一直检索不到新文档。
                retriever.drop_cached_retriever(kb_id, container)
            logger.info(
                "Index job %s [%s/%s] %s: phaseA=%.1fs %s lock_wait=%.1fs publish=%.1fs",
                job_id,
                kb_id,
                document_id,
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
        # 工作区不在文档或知识库目录内，无锁解析期间的删除不会波及它，阶段 A 也不会在
        # 已删除的目录下重建文件；任务只需清理自己的工作区，无需再为删除复查遗留。
        if workspace is not None:
            workspace.cleanup()


def run_document_job(
    job_id: str,
    kb_id: str,
    document_id: str,
    *,
    container: ApplicationContainer,
) -> None:
    """仅当任务仍拥有文件最新版本时，才执行上传后的索引任务。"""
    index_jobs = container.index_jobs()
    try:
        # 开头的存在性检查也在 try 内：这里 SQLite 被锁超时时，worker 只会把任务标为失败，
        # 文档状态却没人写回，会一直停在处理中直到重启。
        if not container.documents.document_exists(
            kb_id, document_id
        ) or not container.knowledge_bases.exists(kb_id):
            index_jobs.finish(job_id, INDEX_JOB_SUPERSEDED, "文档或知识库已删除")
            return
        try:
            file_path = container.documents.document_path(kb_id, document_id)
        except ResourceNotFoundError:
            index_jobs.finish(job_id, INDEX_JOB_SUPERSEDED, "文档或知识库已删除")
            return
        outcome, message = _index_document(
            job_id, kb_id, document_id, file_path, container, rebuild=False
        )
    except Exception as exc:
        # 意外异常（例如 SQLite 被锁）也要写回文档状态，否则文档会一直停在处理中。
        logger.exception("Index job %s [%s/%s] failed", job_id, kb_id, document_id)
        message = str(exc)[:500]
        outcome = _set_document_status(
            index_jobs, kb_id, document_id, job_id, "error", message
        )
    if outcome == "skipped":
        return
    if outcome == "superseded":
        index_jobs.finish(job_id, INDEX_JOB_SUPERSEDED, "文件已被更新或删除")
        return
    job_status = INDEX_JOB_COMPLETED if outcome == "ready" else INDEX_JOB_FAILED
    index_jobs.finish(job_id, job_status, message)


def _rebuild_one_document(
    job_id: str,
    kb_id: str,
    document_id: str,
    container: ApplicationContainer,
) -> tuple[str, str]:
    """重建一个文档；任何意外异常都只让该文档出错，不影响其余文档。"""
    try:
        return _index_document(
            job_id,
            kb_id,
            document_id,
            container.documents.document_path(kb_id, document_id),
            container,
            rebuild=True,
        )
    except Exception as exc:
        logger.exception("Rebuild of %s/%s failed", kb_id, document_id)
        message = str(exc)[:500]
        try:
            outcome = _set_document_status(
                container.index_jobs(), kb_id, document_id, job_id, "error", message
            )
        except Exception:
            logger.exception("Could not record rebuild failure of %s/%s", kb_id, document_id)
            outcome = "error"
        return outcome, message


def _drop_stale_vectors(kb_id: str, container: ApplicationContainer) -> None:
    """重建结束后，删除不再属于任何就绪文档的旧子块和父段落。

    重建不预先清空索引，失败文档的旧版本留到这里才删除。
    """
    try:
        active_document_ids = set(container.documents.ready_document_ids(kb_id))
        get_chunk_index().delete_stale_documents(kb_id, active_document_ids)
        parent_store.delete_stale_documents(kb_id, active_document_ids)
    except Exception as exc:
        # 已删除或不可用的旧版本仍会被 SQLite 允许列表排除，不会参与检索。
        logger.warning("Stale-source cleanup failed for %s: %s", kb_id, exc)


def run_rebuild_job(
    job_id: str,
    kb_id: str,
    *,
    container: ApplicationContainer,
) -> None:
    """只执行知识库最新发起的重建请求。

    各文档与上传一样并行处理，处理完立即写回自己的状态；
    解析失败只让该文档出错，其余文档照常重建。不预先清空索引：已就绪文档的旧版本
    在新版本发布前继续参与检索，发布时由 publish_document 先写新版本再删旧版本。
    """
    index_jobs = container.index_jobs()
    with knowledge_base_lock(kb_id):
        if not index_jobs.start(job_id):
            return
        if not index_jobs.is_active_rebuild(kb_id, job_id):
            index_jobs.finish(job_id, INDEX_JOB_SUPERSEDED, "已被新的重建任务替代")
            return
        document_ids = [
            record["document_id"]
            for record in container.documents.list_documents(kb_id)
            if record["index_job_id"] == job_id
        ]
        retriever.drop_cached_retriever(kb_id, container)

    outcomes: list[tuple[str, str]] = []
    if document_ids:
        workers = min(get_settings().index_worker_threads, len(document_ids))
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="rebuild-document"
        ) as executor:
            outcomes = list(
                executor.map(
                    carry_request_id(
                        lambda name: _rebuild_one_document(job_id, kb_id, name, container)
                    ),
                    document_ids,
                )
            )
    ready = sum(1 for outcome, _ in outcomes if outcome == "ready")
    failed = sum(1 for outcome, _ in outcomes if outcome == "error")

    with knowledge_base_lock(kb_id):
        # 大型重建期间配置可能变化，旧任务完成时不得覆盖新重建任务的持久化状态。
        if not index_jobs.is_active_rebuild(kb_id, job_id):
            index_jobs.finish(job_id, INDEX_JOB_SUPERSEDED, "已被新的重建任务替代")
            return
        _drop_stale_vectors(kb_id, container)
        retriever.drop_cached_retriever(kb_id, container)
        if not document_ids:
            job_status, message = INDEX_JOB_FAILED, "当前知识库没有可重建的文档"
        elif failed and not ready:
            job_status, message = INDEX_JOB_FAILED, f"{failed} 个文档解析失败"
        elif failed:
            job_status = INDEX_JOB_COMPLETED
            message = f"{failed} 个文档解析失败，其余已重建"
        else:
            job_status, message = INDEX_JOB_COMPLETED, ""
        # 兜底：仍停在处理中的文档（意外中断）记为出错，并释放重建标记。
        index_jobs.complete_rebuild(kb_id, job_id, "error", message or "重建未完成")
        index_jobs.finish(job_id, job_status, message)


def delete_document_indexes(
    kb_id: str, document_id: str, container: ApplicationContainer
) -> None:
    get_chunk_index().delete_document(kb_id, document_id)
    parent_store.delete_document(kb_id, document_id)
    retriever.drop_cached_retriever(kb_id, container)


def run_delete_job(
    job_id: str,
    kb_id: str,
    document_id: str,
    *,
    container: ApplicationContainer,
) -> None:
    """幂等清理逻辑删除文档的全部外部资源；失败任务保留目标 ID 供恢复。"""
    index_jobs = container.index_jobs()
    with document_lock(kb_id, document_id):
        if not index_jobs.start(job_id):
            return
        try:
            delete_document_indexes(kb_id, document_id, container)
            directory = container.documents.document_directory(kb_id, document_id)
            if os.path.exists(directory):
                shutil.rmtree(directory)
            mineru_client.remove_document_results(kb_id, document_id)
        except Exception as exc:
            index_jobs.finish(job_id, INDEX_JOB_FAILED, str(exc)[:500])
            return
        index_jobs.finish(job_id, INDEX_JOB_COMPLETED)
        index_jobs.redact_orphaned_filenames(kb_id)
        index_jobs.compact_database()
