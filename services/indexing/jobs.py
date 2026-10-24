"""用 SQLite 持久化文档索引任务，保证服务重启后任务状态可追溯。"""

import datetime
import logging
import sqlite3
import uuid
from collections.abc import Callable
from typing import cast

from services.core import sqlite_connections as application_db
from services.indexing.contracts import (
    IndexJob,
    IndexJobOperation,
    IndexJobStatus,
    IndexJobSummary,
    TerminalIndexJobStatus,
)

INDEX_JOB_QUEUED: IndexJobStatus = "queued"
INDEX_JOB_RUNNING: IndexJobStatus = "running"
INDEX_JOB_COMPLETED: TerminalIndexJobStatus = "completed"
INDEX_JOB_FAILED: TerminalIndexJobStatus = "failed"
INDEX_JOB_SUPERSEDED: TerminalIndexJobStatus = "superseded"
INDEX_JOB_INTERRUPTED: TerminalIndexJobStatus = "interrupted"
INDEX_JOB_OPERATIONS: frozenset[IndexJobOperation] = frozenset(
    {"process", "rebuild", "delete"}
)


def _job_from_row(row: tuple[object, ...]) -> IndexJob:
    """Convert the worker query projection into its typed task record."""
    return {
        "id": str(row[0]),
        "kb_name": str(row[1]),
        "filename": application_db.optional_text(row[2]),
        "operation": cast(IndexJobOperation, row[3]),
        "status": cast(IndexJobStatus, row[4]),
    }


def _job_summary_from_row(row: tuple[object, ...]) -> IndexJobSummary:
    """Convert the history query projection into its typed API summary."""
    return {
        "id": str(row[0]),
        "filename": application_db.optional_text(row[1]),
        "operation": cast(IndexJobOperation, row[2]),
        "status": cast(IndexJobStatus, row[3]),
        "error_message": str(row[4] or ""),
        "created_at": str(row[5]),
        "started_at": application_db.optional_text(row[6]),
        "finished_at": application_db.optional_text(row[7]),
    }


def should_compact_database(
    free_pages: int,
    page_count: int,
    *,
    min_free_pages: int,
    min_free_ratio: float,
) -> bool:
    """根据 SQLite 空闲页的数量和占比，判断是否值得执行 VACUUM。"""
    return (
        page_count > 0
        and free_pages >= min_free_pages
        and free_pages / page_count >= min_free_ratio
    )


class IndexJobStore:
    """封装索引任务的创建、状态迁移、恢复和清理操作。"""

    def __init__(
        self,
        db_file: Callable[[], str],
        now: Callable[[], str],
        validate_kb_name: Callable[[str], str],
        validate_filename: Callable[[str], str],
        *,
        vacuum_min_free_pages: int,
        vacuum_min_free_ratio: float,
        history_retention_days: int,
        history_max_per_kb: int,
        logger: logging.Logger,
    ) -> None:
        """保存数据库连接、校验函数和数据库压缩阈值等依赖。"""
        self._db_file = db_file
        self._now = now
        self._validate_kb_name = validate_kb_name
        self._validate_filename = validate_filename
        self._vacuum_min_free_pages = vacuum_min_free_pages
        self._vacuum_min_free_ratio = vacuum_min_free_ratio
        self._history_retention_days = history_retention_days
        self._history_max_per_kb = history_max_per_kb
        self._logger = logger

    def create(
        self,
        kb_name: str,
        operation: IndexJobOperation,
        filename: str | None = None,
    ) -> str:
        """创建一条待执行任务，并返回其唯一任务 ID。"""
        kb_name = self._validate_kb_name(kb_name)
        if operation not in INDEX_JOB_OPERATIONS:
            raise ValueError("未知的索引任务类型")
        if filename is not None:
            filename = self._validate_filename(filename)
        job_id = uuid.uuid4().hex
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                """
                INSERT INTO index_jobs (id, kb_name, filename, operation, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, kb_name, filename, operation, INDEX_JOB_QUEUED, self._now()),
            )
        return job_id

    def start(self, job_id: str) -> bool:
        """原子地将排队任务切换为执行中；已被替代的任务不会启动。"""
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, started_at = ?, error_message = ''
                WHERE id = ? AND status = ?
                """,
                (INDEX_JOB_RUNNING, self._now(), job_id, INDEX_JOB_QUEUED),
            )
        return cursor.rowcount == 1

    def finish(
        self,
        job_id: str,
        status: TerminalIndexJobStatus,
        error_message: str = "",
    ) -> None:
        """结束仍在排队或执行中的任务，并保留截断后的错误信息。"""
        if status not in {
            INDEX_JOB_COMPLETED,
            INDEX_JOB_FAILED,
            INDEX_JOB_SUPERSEDED,
            INDEX_JOB_INTERRUPTED,
        }:
            raise ValueError("未知的索引任务状态")
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, error_message = ?, finished_at = ?
                WHERE id = ? AND status IN (?, ?)
                """,
                (
                    status,
                    error_message[:500],
                    self._now(),
                    job_id,
                    INDEX_JOB_QUEUED,
                    INDEX_JOB_RUNNING,
                ),
            )

    def is_current_document_job(self, kb_name: str, filename: str, job_id: str) -> bool:
        """判断任务是否仍对应文件的最新上传版本。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT index_job_id FROM documents WHERE kb_name = ? AND filename = ?",
                (kb_name, filename),
            ).fetchone()
        return bool(row and row[0] == job_id)

    def update_document_status(
        self,
        kb_name: str,
        filename: str,
        status: str,
        error_message: str = "",
        *,
        job_id: str | None = None,
    ) -> bool:
        """更新文档状态；传入 job_id 时同时校验任务归属，避免旧任务覆盖新结果。"""
        statement = "UPDATE documents SET status = ?, error_message = ? WHERE kb_name = ? AND filename = ?"
        params: list[str] = [status, error_message[:500], kb_name, filename]
        if job_id is not None:
            statement += " AND index_job_id = ?"
            params.append(job_id)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(statement, params)
        return cursor.rowcount == 1

    def attach_document_job(self, kb_name: str, filename: str, job_id: str) -> bool:
        """将文档关联到最新持久化索引任务，并标记为处理中。"""
        kb_name = self._validate_kb_name(kb_name)
        filename = self._validate_filename(filename)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                "UPDATE documents SET status = 'processing', error_message = '', "
                "index_job_id = ? WHERE kb_name = ? AND filename = ?",
                (job_id, kb_name, filename),
            )
        return cursor.rowcount == 1

    def prepare_rebuild(self, kb_name: str) -> str:
        """登记新的全库重建任务，并使同知识库的旧重建任务失效。"""
        kb_name = self._validate_kb_name(kb_name)
        job_id = uuid.uuid4().hex
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE name = ?", (kb_name,)
            ).fetchone()
            if not row:
                raise ValueError("知识库不存在")
            previous_job_id = row[0]
            if previous_job_id:
                conn.execute(
                    """
                    UPDATE index_jobs
                    SET status = ?, error_message = ?, finished_at = ?
                    WHERE id = ? AND status IN (?, ?)
                    """,
                    (
                        INDEX_JOB_SUPERSEDED,
                        "已被新的索引重建任务替代",
                        self._now(),
                        previous_job_id,
                        INDEX_JOB_QUEUED,
                        INDEX_JOB_RUNNING,
                    ),
                )
            conn.execute(
                """
                INSERT INTO index_jobs (id, kb_name, filename, operation, status, created_at)
                VALUES (?, ?, NULL, 'rebuild', ?, ?)
                """,
                (job_id, kb_name, INDEX_JOB_QUEUED, self._now()),
            )
            conn.execute(
                "UPDATE knowledge_bases SET active_rebuild_job_id = ? WHERE name = ?", (job_id, kb_name)
            )
            conn.execute(
                """
                UPDATE documents
                SET status = 'processing', error_message = '', index_job_id = ?
                WHERE kb_name = ?
                """,
                (job_id, kb_name),
            )
        return job_id

    def is_active_rebuild(self, kb_name: str, job_id: str) -> bool:
        """判断 job_id 是否仍是该知识库当前有效的重建任务。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE name = ?", (kb_name,)
            ).fetchone()
        return bool(row and row[0] == job_id)

    def complete_rebuild(
        self, kb_name: str, job_id: str, status: str, error_message: str = ""
    ) -> None:
        """收尾重建：仍停在处理中的文档记为 status，并清除知识库上的活动任务标记。

        重建中已经完成的文档（就绪或出错）保持各自的状态，不会被覆盖。
        """
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                """
                UPDATE documents SET status = ?, error_message = ?
                WHERE kb_name = ? AND index_job_id = ? AND status = 'processing'
                """,
                (status, error_message[:500], kb_name, job_id),
            )
            conn.execute(
                """
                UPDATE knowledge_bases SET active_rebuild_job_id = ''
                WHERE name = ? AND active_rebuild_job_id = ?
                """,
                (kb_name, job_id),
            )

    def redact_orphaned_filenames(self, kb_name: str | None = None) -> int:
        """从已删除文件的历史任务中抹去文件名，避免任务记录泄露旧文件信息。"""
        clauses = [
            "filename IS NOT NULL",
            "status IN (?, ?, ?, ?)",
            "(operation <> 'delete' OR status NOT IN (?, ?))",
            """
            NOT EXISTS (
                SELECT 1 FROM documents
                WHERE documents.kb_name = index_jobs.kb_name
                  AND documents.filename = index_jobs.filename
            )
            """,
        ]
        params: list[str] = [
            INDEX_JOB_COMPLETED,
            INDEX_JOB_FAILED,
            INDEX_JOB_SUPERSEDED,
            INDEX_JOB_INTERRUPTED,
            INDEX_JOB_FAILED,
            INDEX_JOB_INTERRUPTED,
        ]
        if kb_name is not None:
            clauses.append("kb_name = ?")
            params.append(kb_name)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                f"""
                UPDATE index_jobs
                SET filename = NULL,
                    error_message = REPLACE(COALESCE(error_message, ''), filename, '[已删除文件]')
                WHERE {' AND '.join(clauses)}
                """,
                params,
            )
        return cursor.rowcount

    def compact_database(self) -> bool:
        """在空闲页达到阈值时压缩 SQLite 文件；数据库忙或异常时只记录警告。"""
        try:
            with application_db.connect(self._db_file(), timeout=5) as conn:
                conn.execute("PRAGMA busy_timeout = 5000")
                free_pages = conn.execute("PRAGMA freelist_count").fetchone()[0]
                page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            if not should_compact_database(
                free_pages,
                page_count,
                min_free_pages=self._vacuum_min_free_pages,
                min_free_ratio=self._vacuum_min_free_ratio,
            ):
                return False
            with application_db.connect(self._db_file(), timeout=5) as conn:
                conn.execute("PRAGMA busy_timeout = 5000")
                conn.execute("VACUUM")
            return True
        except sqlite3.DatabaseError as exc:
            self._logger.warning("SQLite compaction skipped: %s", exc)
            return False

    def prune_history(
        self,
        *,
        retention_days: int | None = None,
        max_per_kb: int | None = None,
    ) -> int:
        """清理不可访问或过量的终态任务，同时保留仍被引用的记录。"""
        retention_days = max(1, retention_days or self._history_retention_days)
        max_per_kb = max(1, max_per_kb or self._history_max_per_kb)
        cutoff = (
            datetime.datetime.fromisoformat(self._now())
            - datetime.timedelta(days=retention_days)
        ).isoformat(timespec="seconds")
        terminal = (
            INDEX_JOB_COMPLETED,
            INDEX_JOB_FAILED,
            INDEX_JOB_SUPERSEDED,
            INDEX_JOB_INTERRUPTED,
        )
        placeholders = ", ".join("?" for _ in terminal)
        candidate_guard = f"""
            status IN ({placeholders})
            AND NOT (
                operation = 'delete' AND filename IS NOT NULL
                AND status IN (?, ?)
            )
            AND NOT EXISTS (
                SELECT 1 FROM documents WHERE documents.index_job_id = index_jobs.id
            )
            AND NOT EXISTS (
                SELECT 1 FROM knowledge_bases
                WHERE knowledge_bases.active_rebuild_job_id = index_jobs.id
            )
        """
        guard_params = (
            *terminal,
            INDEX_JOB_FAILED,
            INDEX_JOB_INTERRUPTED,
        )
        deleted = 0
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                f"""
                DELETE FROM index_jobs
                WHERE {candidate_guard}
                  AND NOT EXISTS (
                      SELECT 1 FROM knowledge_bases
                      WHERE knowledge_bases.name = index_jobs.kb_name
                  )
                """,
                guard_params,
            )
            deleted += cursor.rowcount
            cursor = conn.execute(
                f"""
                DELETE FROM index_jobs
                WHERE {candidate_guard} AND created_at < ?
                """,
                (*guard_params, cutoff),
            )
            deleted += cursor.rowcount
            cursor = conn.execute(
                f"""
                DELETE FROM index_jobs WHERE id IN (
                    SELECT id FROM (
                        SELECT id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY kb_name ORDER BY created_at DESC, id DESC
                               ) AS history_rank
                        FROM index_jobs
                        WHERE {candidate_guard}
                    ) ranked
                    WHERE history_rank > ?
                )
                """,
                (*guard_params, max_per_kb),
            )
            deleted += cursor.rowcount
        return deleted

    def list_recent(self, kb_name: str, limit: int = 20) -> list[IndexJobSummary]:
        """按创建时间倒序返回知识库最近的任务摘要。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                """
                SELECT id, filename, operation, status, error_message,
                       created_at, started_at, finished_at
                FROM index_jobs WHERE kb_name = ? ORDER BY created_at DESC LIMIT ?
                """,
                (kb_name, max(1, min(limit, 100))),
            ).fetchall()
        return [_job_summary_from_row(tuple(row)) for row in rows]

    def get(self, job_id: str) -> IndexJob | None:
        """查询单个任务的基础状态；任务不存在时返回 None。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT id, kb_name, filename, operation, status FROM index_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(tuple(row)) if row else None

    def queued_ids(self) -> list[str]:
        """按入队顺序返回所有等待执行的任务 ID。"""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT id FROM index_jobs WHERE status = ? ORDER BY created_at ASC",
                (INDEX_JOB_QUEUED,),
            ).fetchall()
        return [row[0] for row in rows]

    def has_running(self, kb_name: str) -> bool:
        """判断知识库是否有正在执行的任务；调用方持有知识库锁时结果才稳定。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT 1 FROM index_jobs WHERE kb_name = ? AND status = ? LIMIT 1",
                (kb_name, INDEX_JOB_RUNNING),
            ).fetchone()
        return row is not None

    def supersede(self, kb_name: str, reason: str) -> None:
        """使知识库当前排队或执行中的任务失效，并同步修正关联文档状态。"""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT id FROM index_jobs WHERE kb_name = ? AND status IN (?, ?)",
                (kb_name, INDEX_JOB_QUEUED, INDEX_JOB_RUNNING),
            ).fetchall()
            job_ids = [row[0] for row in rows]
            conn.execute(
                """
                UPDATE index_jobs SET status = ?, error_message = ?, finished_at = ?
                WHERE kb_name = ? AND status IN (?, ?)
                """,
                (
                    INDEX_JOB_SUPERSEDED, reason[:500], self._now(), kb_name,
                    INDEX_JOB_QUEUED, INDEX_JOB_RUNNING,
                ),
            )
            if job_ids:
                placeholders = ", ".join("?" for _ in job_ids)
                conn.execute(
                    f"UPDATE documents SET status = 'error', error_message = ? "
                    f"WHERE kb_name = ? AND status = 'processing' AND index_job_id IN ({placeholders})",
                    (reason[:500], kb_name, *job_ids),
                )

    def recover_interrupted(self) -> None:
        """服务启动时恢复可重试的删除任务，并标记其余中断任务。"""
        with application_db.connect(self._db_file()) as conn:
            cleanup_message = "上次向量索引清理未完成，服务启动后将自动重试"
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, error_message = ?, started_at = NULL, finished_at = NULL
                WHERE operation = 'delete' AND filename IS NOT NULL
                  AND status IN (?, ?, ?)
                """,
                (
                    INDEX_JOB_QUEUED, cleanup_message, INDEX_JOB_RUNNING,
                    INDEX_JOB_FAILED, INDEX_JOB_INTERRUPTED,
                ),
            )
            rows = conn.execute(
                "SELECT id FROM index_jobs WHERE status = ?", (INDEX_JOB_RUNNING,)
            ).fetchall()
            job_ids = [row[0] for row in rows]
            if job_ids:
                placeholders = ", ".join("?" for _ in job_ids)
                message = "服务重启，索引任务已中断，请重新上传或重建知识库"
                conn.execute(
                    f"UPDATE index_jobs SET status = ?, error_message = ?, finished_at = ? "
                    f"WHERE id IN ({placeholders})",
                    (INDEX_JOB_INTERRUPTED, message, self._now(), *job_ids),
                )
                conn.execute(
                    f"UPDATE documents SET status = 'error', error_message = ? "
                    f"WHERE status = 'processing' AND index_job_id IN ({placeholders})",
                    (message, *job_ids),
                )
            conn.execute(
                """
                UPDATE knowledge_bases SET active_rebuild_job_id = ''
                WHERE active_rebuild_job_id <> ''
                  AND NOT EXISTS (
                      SELECT 1 FROM index_jobs
                      WHERE index_jobs.id = knowledge_bases.active_rebuild_job_id
                        AND index_jobs.status IN (?, ?)
                  )
                """,
                (INDEX_JOB_QUEUED, INDEX_JOB_RUNNING),
            )
