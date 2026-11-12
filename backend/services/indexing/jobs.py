"""用 SQLite 持久化文档索引任务，保证服务重启后任务状态可追溯。"""

import datetime
import logging
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from typing import cast

from services.core import sqlite_connections as application_db
from services.indexing.contracts import (
    CrossKnowledgeBaseJobSummary,
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
# 发布前失败、旧索引仍可用时写在错误信息前，界面据此提示仍在使用上一版本。
PREVIOUS_VERSION_KEPT_PREFIX = "本次索引失败，仍使用上一版本："
# 文档仍在索引中的状态：processing 是首次索引，rebuilding 与 publishing 时旧版本仍可检索。
IN_PROGRESS_SQL = "'processing', 'rebuilding', 'publishing'"
INDEX_JOB_OPERATIONS: frozenset[IndexJobOperation] = frozenset({"process", "rebuild", "delete"})


def insert_job(
    conn: sqlite3.Connection,
    job_id: str,
    kb_id: str,
    document_id: str | None,
    filename: str | None,
    operation: IndexJobOperation,
    created_at: str,
    *,
    parser_mode: str | None = None,
) -> None:
    """在调用方事务内插入任务，不自行开启连接或提交。

    parser_mode 是该任务要用的解析方式，发布成功后才写回文档；失败保留旧版本时
    文档仍显示旧版本实际使用的解析方式。
    """
    conn.execute(
        "INSERT INTO index_jobs "
        "(id, kb_id, document_id, filename, operation, status, created_at, parser_mode) "
        "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
        (job_id, kb_id, document_id, filename, operation, created_at, parser_mode),
    )


def _job_from_row(row: tuple[object, ...]) -> IndexJob:
    """Convert the worker query projection into its typed task record."""
    return {
        "id": str(row[0]),
        "kb_id": str(row[1]),
        "document_id": application_db.optional_text(row[5]),
        "filename": application_db.optional_text(row[2]),
        "operation": cast(IndexJobOperation, row[3]),
        "status": cast(IndexJobStatus, row[4]),
    }


def _job_summary_from_row(row: tuple[object, ...]) -> IndexJobSummary:
    """Convert the history query projection into its typed API summary."""
    return {
        "id": str(row[0]),
        "filename": application_db.optional_text(row[1]),
        "document_id": application_db.optional_text(row[8]),
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
        validate_kb_id: Callable[[str], str],
        validate_filename: Callable[[str], str],
        *,
        vacuum_min_free_pages: int,
        vacuum_min_free_ratio: float,
        history_retention_days: Callable[[], int],
        history_max_per_kb: Callable[[], int],
        logger: logging.Logger,
    ) -> None:
        """保存数据库连接、校验函数和数据库压缩阈值等依赖。"""
        self._db_file = db_file
        self._now = now
        self._validate_kb_id = validate_kb_id
        self._validate_filename = validate_filename
        self._vacuum_min_free_pages = vacuum_min_free_pages
        self._vacuum_min_free_ratio = vacuum_min_free_ratio
        self._history_retention_days = history_retention_days
        self._history_max_per_kb = history_max_per_kb
        self._logger = logger

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

    def is_current_document_job(self, kb_id: str, document_id: str, job_id: str) -> bool:
        """判断任务是否仍拥有该文档的最新索引请求。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT index_job_id FROM documents WHERE kb_id = ? AND document_id = ?",
                (kb_id, document_id),
            ).fetchone()
        return bool(row and row[0] == job_id)

    def update_document_status(
        self,
        kb_id: str,
        document_id: str,
        status: str,
        error_message: str = "",
        *,
        job_id: str | None = None,
        parser_mode: str | None = None,
    ) -> bool:
        """更新文档状态；传入 job_id 时同时校验任务归属，避免旧任务覆盖新结果。

        传入 parser_mode 时一并写入，用于新版本发布成功后记录它实际使用的解析方式。
        """
        statement = (
            "UPDATE documents SET status = ?, error_message = ?, "
            "parser_mode = COALESCE(?, parser_mode) WHERE kb_id = ? AND document_id = ?"
        )
        params: list[str | None] = [status, error_message[:500], parser_mode, kb_id, document_id]
        if job_id is not None:
            statement += " AND index_job_id = ?"
            params.append(job_id)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(statement, params)
        return cursor.rowcount == 1

    def fail_document(
        self,
        kb_id: str,
        document_id: str,
        job_id: str,
        error_message: str,
        *,
        previous_index_intact: bool,
    ) -> bool:
        """记录文档索引失败；任务归属已变化时返回 False。

        rebuilding 文档尚未开始发布，旧索引一定完好，总是回到 ready 继续使用旧版本，
        错误信息提示本次失败。publishing 文档只有在 previous_index_intact（发布已完整撤销）
        时才回到 ready，否则记为 error。原本不是就绪的文档一律记为 error。
        """
        message = error_message[:500]
        kept = "('rebuilding', 'publishing')" if previous_index_intact else "('rebuilding')"
        status_sql = f"CASE WHEN status IN {kept} THEN 'ready' ELSE 'error' END"
        message_sql = f"CASE WHEN status IN {kept} THEN ? ELSE ? END"
        params = [f"{PREVIOUS_VERSION_KEPT_PREFIX}{message}"[:500], message]
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                f"UPDATE documents SET status = {status_sql}, error_message = {message_sql} "
                "WHERE kb_id = ? AND document_id = ? AND index_job_id = ?",
                (*params, kb_id, document_id, job_id),
            )
        return cursor.rowcount == 1

    def begin_publish(self, kb_id: str, document_id: str, job_id: str) -> None:
        """发布前把 rebuilding 文档标为 publishing；调用方须持有该文档的文档锁。

        publishing 和 rebuilding 一样参与检索，但表示旧索引可能已被改动：发布中途失败或
        进程重启时，只有仍是 rebuilding 的文档才能确定旧版本完好。
        """
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                "UPDATE documents SET status = 'publishing' "
                "WHERE kb_id = ? AND document_id = ? AND index_job_id = ? AND status = 'rebuilding'",
                (kb_id, document_id, job_id),
            )

    def prepare_rebuild(
        self, kb_id: str, *, chunk_config: tuple[int, int] | None = None
    ) -> str:
        """登记新的全库重建任务，并使同知识库的旧重建任务失效。

        `chunk_config`（分块大小, 重叠长度）与任务在同一事务里写入：登记失败时配置也不变，
        不会出现配置已改、索引没重建的情况。

        已就绪的文档标为 rebuilding：旧版本继续参与检索，直到新版本发布替换它；
        其余文档标为 processing。这里不持文档锁，单文档重新索引中的 publishing 文档
        可能正在发布，旧索引不再有保证，因此也标为 processing。
        """
        kb_id = self._validate_kb_id(kb_id)
        job_id = uuid.uuid4().hex
        with application_db.connect(self._db_file()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases "
                "WHERE kb_id = ? AND state = 'active'",
                (kb_id,),
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
                INSERT INTO index_jobs (id, kb_id, filename, operation, status, created_at)
                VALUES (?, ?, NULL, 'rebuild', ?, ?)
                """,
                (job_id, kb_id, INDEX_JOB_QUEUED, self._now()),
            )
            conn.execute(
                "UPDATE knowledge_bases SET active_rebuild_job_id = ? WHERE kb_id = ?",
                (job_id, kb_id),
            )
            if chunk_config is not None:
                conn.execute(
                    "UPDATE knowledge_bases SET chunk_size = ?, chunk_overlap = ? WHERE kb_id = ?",
                    (*chunk_config, kb_id),
                )
            # 被替代的单文档任务可能带着用户新选的解析方式；重建按文档上记录的方式解析，
            # 不先写回文档，用户的选择会被静默丢掉。
            conn.execute(
                """
                UPDATE documents
                SET parser_mode = COALESCE((
                    SELECT j.parser_mode FROM index_jobs j
                    WHERE j.id = documents.index_job_id AND j.operation = 'process'
                      AND j.status IN (?, ?)
                ), parser_mode)
                WHERE kb_id = ?
                """,
                (INDEX_JOB_QUEUED, INDEX_JOB_RUNNING, kb_id),
            )
            conn.execute(
                """
                UPDATE documents
                SET status = CASE WHEN status = 'ready' THEN 'rebuilding' ELSE 'processing' END,
                    error_message = '', index_job_id = ?
                WHERE kb_id = ?
                """,
                (job_id, kb_id),
            )
        return job_id

    def is_active_rebuild(self, kb_id: str, job_id: str) -> bool:
        """判断 job_id 是否仍是该知识库当前有效的重建任务。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE kb_id = ?", (kb_id,)
            ).fetchone()
        return bool(row and row[0] == job_id)

    def complete_rebuild(
        self, kb_id: str, job_id: str, status: str, error_message: str = ""
    ) -> None:
        """收尾重建：仍停在处理中或重建中的文档记为 status，并清除知识库上的活动任务标记。

        重建中已经完成的文档（就绪或出错）保持各自的状态，不会被覆盖。
        """
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                f"""
                UPDATE documents SET status = ?, error_message = ?
                WHERE kb_id = ? AND index_job_id = ? AND status IN ({IN_PROGRESS_SQL})
                """,
                (status, error_message[:500], kb_id, job_id),
            )
            conn.execute(
                """
                UPDATE knowledge_bases SET active_rebuild_job_id = NULL
                WHERE kb_id = ? AND active_rebuild_job_id = ?
                """,
                (kb_id, job_id),
            )

    def redact_orphaned_filenames(self, kb_id: str | None = None) -> int:
        """从已删除文件的历史任务中抹去文件名，避免任务记录泄露旧文件信息。"""
        clauses = [
            "filename IS NOT NULL",
            "status IN (?, ?, ?, ?)",
            "(operation <> 'delete' OR status NOT IN (?, ?))",
            """
            NOT EXISTS (
                SELECT 1 FROM documents
                WHERE documents.kb_id = index_jobs.kb_id
                  AND documents.document_id = index_jobs.document_id
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
        if kb_id is not None:
            clauses.append("kb_id = ?")
            params.append(kb_id)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                f"""
                UPDATE index_jobs
                SET filename = NULL,
                    error_message = REPLACE(COALESCE(error_message, ''), filename, '[已删除文件]')
                WHERE {" AND ".join(clauses)}
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
        retention_days = max(1, retention_days or self._history_retention_days())
        max_per_kb = max(1, max_per_kb or self._history_max_per_kb())
        cutoff = (
            datetime.datetime.fromisoformat(self._now()) - datetime.timedelta(days=retention_days)
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
                operation = 'delete' AND document_id IS NOT NULL
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
                                   PARTITION BY kb_id ORDER BY created_at DESC, id DESC
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

    def list_recent(self, kb_id: str, limit: int = 20) -> list[IndexJobSummary]:
        """按创建时间倒序返回知识库最近的任务摘要。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                """
                SELECT id, filename, operation, status, error_message,
                       created_at, started_at, finished_at, document_id
                FROM index_jobs WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?
                """,
                (kb_id, max(1, min(limit, 100))),
            ).fetchall()
        return [_job_summary_from_row(tuple(row)) for row in rows]

    def list_recent_all(
        self,
        status: str | None = None,
        limit: int = 50,
        *,
        kb_ids: Sequence[str] | None = None,
    ) -> list[CrossKnowledgeBaseJobSummary]:
        """跨知识库按创建时间倒序返回最近的任务，可按状态过滤，供运行监控使用。

        `kb_ids` 为 None 表示不限知识库，否则只返回其中知识库的任务。
        """
        clauses: list[str] = []
        params: list[object] = []
        if status:
            clauses.append("j.status = ?")
            params.append(status)
        if kb_ids is not None:
            clauses.append(f"j.kb_id IN ({','.join('?' for _ in kb_ids) or 'NULL'})")
            params.extend(kb_ids)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                f"""
                SELECT j.id, j.filename, j.operation, j.status, j.error_message,
                       j.created_at, j.started_at, j.finished_at, j.document_id, j.kb_id, kb.name
                FROM index_jobs j JOIN knowledge_bases kb ON kb.kb_id = j.kb_id
                {where} ORDER BY j.created_at DESC LIMIT ?
                """,
                (*params, max(1, min(limit, 100))),
            ).fetchall()
        return [
            {
                **_job_summary_from_row(tuple(row)),
                "kb_id": str(row[9]),
                "kb_name": str(row[10]),
            }
            for row in rows
        ]

    def get(self, job_id: str) -> IndexJob | None:
        """查询单个任务的基础状态；任务不存在时返回 None。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT id, kb_id, filename, operation, status, document_id FROM index_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return _job_from_row(tuple(row)) if row else None

    def requested_parser_mode(self, job_id: str) -> str | None:
        """返回单文档任务登记的解析方式；整库重建任务和未登记时返回 None。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT parser_mode FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return application_db.optional_text(row[0]) if row else None

    def queued_ids(self, limit: int = 200) -> list[str]:
        """按入队顺序返回所有等待执行的任务 ID。"""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT id FROM index_jobs WHERE status = ? ORDER BY created_at ASC, id ASC LIMIT ?",
                (INDEX_JOB_QUEUED, max(1, min(limit, 1000))),
            ).fetchall()
        return [row[0] for row in rows]

    def has_running(self, kb_id: str) -> bool:
        """判断知识库是否有正在执行的任务；调用方持有知识库锁时结果才稳定。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT 1 FROM index_jobs WHERE kb_id = ? AND status = ? LIMIT 1",
                (kb_id, INDEX_JOB_RUNNING),
            ).fetchone()
        return row is not None

    def supersede(self, kb_id: str, reason: str) -> None:
        """使知识库当前排队或执行中的任务失效，并同步修正关联文档状态。"""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT id FROM index_jobs WHERE kb_id = ? AND status IN (?, ?)",
                (kb_id, INDEX_JOB_QUEUED, INDEX_JOB_RUNNING),
            ).fetchall()
            job_ids = [row[0] for row in rows]
            conn.execute(
                """
                UPDATE index_jobs SET status = ?, error_message = ?, finished_at = ?
                WHERE kb_id = ? AND status IN (?, ?)
                """,
                (
                    INDEX_JOB_SUPERSEDED,
                    reason[:500],
                    self._now(),
                    kb_id,
                    INDEX_JOB_QUEUED,
                    INDEX_JOB_RUNNING,
                ),
            )
            if job_ids:
                placeholders = ", ".join("?" for _ in job_ids)
                conn.execute(
                    f"UPDATE documents SET status = 'error', error_message = ? "
                    f"WHERE kb_id = ? AND status IN ({IN_PROGRESS_SQL}) "
                    f"AND index_job_id IN ({placeholders})",
                    (reason[:500], kb_id, *job_ids),
                )

    def recover_interrupted(self) -> None:
        """服务启动时恢复可重试的删除任务，并标记其余中断任务。"""
        with application_db.connect(self._db_file()) as conn:
            cleanup_message = "上次向量索引清理未完成，服务启动后将自动重试"
            conn.execute(
                """
                UPDATE index_jobs
                SET status = ?, error_message = ?, started_at = NULL, finished_at = NULL
                WHERE operation = 'delete' AND document_id IS NOT NULL
                  AND status IN (?, ?, ?)
                """,
                (
                    INDEX_JOB_QUEUED,
                    cleanup_message,
                    INDEX_JOB_RUNNING,
                    INDEX_JOB_FAILED,
                    INDEX_JOB_INTERRUPTED,
                ),
            )
            rows = conn.execute(
                "SELECT id FROM index_jobs WHERE status = ?", (INDEX_JOB_RUNNING,)
            ).fetchall()
            job_ids = [row[0] for row in rows]
            if job_ids:
                placeholders = ", ".join("?" for _ in job_ids)
                conn.execute(
                    f"UPDATE index_jobs SET status = ?, error_message = ?, finished_at = ? "
                    f"WHERE id IN ({placeholders})",
                    (
                        INDEX_JOB_INTERRUPTED,
                        "服务重启，索引任务已中断，请重新索引或重建知识库",
                        self._now(),
                        *job_ids,
                    ),
                )
            # 任务已不在排队或执行却仍停在处理中的文档：中断的任务，或任务失败时文档状态没能
            # 写回（例如 SQLite 被锁）。还没开始发布的 rebuilding 文档旧版本完好，回到 ready。
            message = "服务重启，索引任务已中断，请重新索引或重建知识库"
            stale = (
                f"status IN ({IN_PROGRESS_SQL}) AND (index_job_id IS NULL OR index_job_id NOT IN "
                "(SELECT id FROM index_jobs WHERE status = ?))"
            )
            conn.execute(
                f"UPDATE documents SET status = 'ready', error_message = ? "
                f"WHERE status = 'rebuilding' AND {stale}",
                (f"{PREVIOUS_VERSION_KEPT_PREFIX}{message}"[:500], INDEX_JOB_QUEUED),
            )
            conn.execute(
                f"UPDATE documents SET status = 'error', error_message = ? WHERE {stale}",
                (message, INDEX_JOB_QUEUED),
            )
            conn.execute(
                """
                UPDATE knowledge_bases SET active_rebuild_job_id = NULL
                WHERE active_rebuild_job_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM index_jobs
                      WHERE index_jobs.id = knowledge_bases.active_rebuild_job_id
                        AND index_jobs.status IN (?, ?)
                  )
                """,
                (INDEX_JOB_QUEUED, INDEX_JOB_RUNNING),
            )
