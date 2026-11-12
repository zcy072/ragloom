"""基于 SQLite 持久化任务的单进程索引 worker。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

from services.core.container import ApplicationContainer, get_application_container
from services.core.request_context import bind_request_id, reset_request_id
from services.core.settings import get_settings
from services.indexing import pipeline as knowledge_base
from services.indexing.jobs import INDEX_JOB_FAILED, INDEX_JOB_QUEUED

logger = logging.getLogger(__name__)


JobDispatcher = Callable[[str], None]
PendingJobsProvider = Callable[[], list[str]]


class IndexWorker:
    """用固定数量的线程并行执行索引任务，并始终以 SQLite 任务记录为准。

    内存队列只负责唤醒 worker。每次执行都会重新从 SQLite 读取任务，因此重复入队或
    服务重启不会凭空制造任务状态。同一文档的准备与发布由文档锁串行化，整库操作由
    知识库锁独占；不同文档可以同时解析、同时发布，线程数只决定有多少个文档并行处理。
    """

    def __init__(
        self,
        dispatcher: JobDispatcher | None = None,
        pending_job_ids: PendingJobsProvider | None = None,
        container: ApplicationContainer | None = None,
        worker_count: int | None = None,
        scan_interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """初始化内存队列和可替换的任务执行、待执行任务查询函数。

        ``worker_count`` 为空时在 ``start`` 时读取 ``INDEX_WORKER_THREADS``。
        """
        self._clock = clock
        self._scan_interval = max(0.05, scan_interval)
        self._next_scan = 0.0
        self._worker_count = worker_count
        # 模块级单例在导入时创建，那时默认容器还不能构造，不传时到用时再取。
        self._container = container
        self._dispatcher = dispatcher or (
            lambda job_id: dispatch_index_job(job_id, container=self._dependencies())
        )
        self._pending_job_ids = pending_job_ids or (
            lambda: self._dependencies().index_jobs().queued_ids()
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._queued_job_ids: set[str] = set()
        self._state_lock = threading.Lock()
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

    def _dependencies(self) -> ApplicationContainer:
        return self._container or get_application_container()

    def start(self) -> None:
        """启动后台线程，并把 SQLite 中遗留的排队任务重新放入内存队列。"""
        with self._state_lock:
            if self._started:
                return
            if any(thread.is_alive() for thread in self._threads):
                return
            count = self._worker_count or get_settings().index_worker_threads
            self._stopping.clear()
            self._next_scan = self._clock() + self._scan_interval
            self._started = True
            self._threads = [
                threading.Thread(
                    target=self._run,
                    name=f"index-worker-{number}",
                    daemon=True,
                )
                for number in range(1, count + 1)
            ]
            for thread in self._threads:
                thread.start()

        self._scan_queued(force=True)

    def stop(self) -> None:
        """通知 worker 停止，并在有限时间内等待后台线程退出。"""
        with self._state_lock:
            if not self._started:
                return
            self._started = False
            self._stopping.set()
            threads = list(self._threads)
            # 每个线程各取走一个哨兵后退出。
            for _thread in threads:
                self._queue.put(None)
        # 所有线程共用一个等待期限，避免线程数越多关停越慢。
        deadline = time.monotonic() + 1
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def enqueue(self, job_id: str) -> bool:
        """将任务放入内存队列；worker 未启动、停止中或已入队时返回 ``False``。"""
        if not job_id:
            return False
        with self._state_lock:
            if not self._started or self._stopping.is_set() or job_id in self._queued_job_ids:
                return False
            self._queued_job_ids.add(job_id)
            self._queue.put(job_id)
        return True

    def _scan_queued(self, *, force: bool = False) -> None:
        with self._state_lock:
            if self._stopping.is_set() or not self._started:
                return
            now = self._clock()
            if not force and now < self._next_scan:
                return
            self._next_scan = now + self._scan_interval
        try:
            for job_id in self._pending_job_ids():
                self.enqueue(job_id)
        except Exception:
            logger.warning("Failed to scan queued index jobs", exc_info=True)

    def _run(self) -> None:
        """持续消费任务，并保证异常后释放内存中的去重标记。"""
        while True:
            self._scan_queued()
            try:
                job_id = self._queue.get(timeout=min(self._scan_interval, 1.0))
            except queue.Empty:
                continue
            try:
                if job_id is None:
                    return
                if not self._stopping.is_set():
                    # 任务 ID 作为请求 ID，这个任务的日志行都能按任务 ID 检索。
                    token = bind_request_id(job_id)
                    try:
                        self._dispatcher(job_id)
                    finally:
                        reset_request_id(token)
            except Exception as exc:
                logger.exception("Index worker crashed while dispatching %s", job_id)
                if job_id:
                    # 标记失败本身也可能出错（例如 SQLite 被锁）；不能让异常带走线程。
                    # 任务留在 running，下次启动时 recover_interrupted 会把它标为 interrupted（删除任务则重新排队）。
                    try:
                        self._dependencies().index_jobs().finish(
                            job_id, INDEX_JOB_FAILED, str(exc)
                        )
                    except Exception:
                        logger.exception("Failed to mark index job %s as failed", job_id)
            finally:
                # 无论执行成功还是异常，都释放内存去重标记，使后续重试能重新入队。
                if job_id:
                    with self._state_lock:
                        self._queued_job_ids.discard(job_id)
                self._queue.task_done()


def dispatch_index_job(job_id: str, *, container: ApplicationContainer) -> None:
    """按任务操作类型分发到文档处理、重建或删除索引流程。"""
    index_jobs = container.index_jobs()
    job = index_jobs.get(job_id)
    if not job or job["status"] != INDEX_JOB_QUEUED:
        return

    operation = job["operation"]
    kb_id = job["kb_id"]
    document_id = job["document_id"]
    if operation == "process":
        if not document_id:
            index_jobs.finish(job_id, INDEX_JOB_FAILED, "文档索引任务缺少文档 ID")
            return
        knowledge_base.run_document_job(
            job_id,
            kb_id,
            document_id,
            container=container,
        )
        return
    if operation == "rebuild":
        knowledge_base.run_rebuild_job(job_id, kb_id, container=container)
        return
    if operation == "delete":
        if not document_id:
            index_jobs.finish(job_id, INDEX_JOB_FAILED, "删除索引任务缺少文档 ID")
            return
        knowledge_base.run_delete_job(job_id, kb_id, document_id, container=container)
        return

    index_jobs.finish(job_id, INDEX_JOB_FAILED, "未知的索引任务类型")


_worker = IndexWorker()


def start() -> None:
    """启动全局索引 worker 实例。"""
    _worker.start()


def stop() -> None:
    """停止全局索引 worker 实例。"""
    _worker.stop()


def enqueue(job_id: str) -> bool:
    """向全局索引 worker 提交一个已持久化的任务 ID。"""
    return _worker.enqueue(job_id)
