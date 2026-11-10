import dataclasses
import os
import sqlite3
import unittest
from unittest.mock import ANY, patch

from langchain_core.documents import Document

import app_storage
from services.core import config as application_config
from services.core.container import get_application_container
from services.indexing import pipeline as knowledge_base
from services.indexing import worker as index_worker
from services.indexing.jobs import (
    INDEX_JOB_COMPLETED,
    INDEX_JOB_FAILED,
    INDEX_JOB_QUEUED,
    INDEX_JOB_SUPERSEDED,
)
from tests.support import (
    IsolatedStorageTestCase,
    add_knowledge_bases,
    create_index_job,
    fixture_document_id,
)


class IndexJobTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.kb_id = add_knowledge_bases("研发部")[0]
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, kb_id, filename, size, uploaded_at, status, index_job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fixture_document_id(self.kb_id, "方案.pdf"),
                    self.kb_id,
                    "方案.pdf",
                    1,
                    application_config.db_now(),
                    "processing",
                    "",
                ),
            )

    def _write_source(self, filename: str) -> str:
        """在知识库目录写入真实源文件；索引任务会为它创建快照。"""
        directory = app_storage.documents.document_directory(
            self.kb_id, fixture_document_id(self.kb_id, filename)
        )
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "wb") as handle:
            handle.write(b"source")
        return path

    def test_document_job_marks_latest_document_ready(self):
        job_id = create_index_job(self.kb_id, "process", "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET index_job_id = ? WHERE kb_id = ? AND filename = ?",
                (job_id, self.kb_id, "方案.pdf"),
            )

        self._write_source("方案.pdf")
        parsed = knowledge_base._ParsedDocument([], [], {})
        with (
            patch.object(knowledge_base, "_parse_and_embed_document", return_value=parsed),
            patch.object(knowledge_base, "_publish_document") as publish,
        ):
            knowledge_base.run_document_job(
                job_id, self.kb_id, fixture_document_id(self.kb_id, "方案.pdf")
            )
        publish.assert_called_once()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            status = conn.execute(
                "SELECT status FROM documents WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "方案.pdf"),
            ).fetchone()[0]
            job_status = conn.execute(
                "SELECT status FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()[0]
        self.assertEqual(status, "ready")
        self.assertEqual(job_status, INDEX_JOB_COMPLETED)

    def test_cache_is_dropped_after_new_document_becomes_ready(self):
        # 发布时清过缓存，但新文档要到状态写成 ready 才进入就绪列表；这之间建好的
        # 检索图不含新文档，状态写入后必须再清一次。
        job_id = create_index_job(self.kb_id, "process", "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET index_job_id = ? WHERE kb_id = ? AND filename = ?",
                (job_id, self.kb_id, "方案.pdf"),
            )
        self._write_source("方案.pdf")
        parsed = knowledge_base._ParsedDocument([], [], {})
        statuses_at_drop: list[str] = []

        def record_status(kb_id):
            with sqlite3.connect(app_storage.DB_FILE) as conn:
                statuses_at_drop.append(
                    conn.execute(
                        "SELECT status FROM documents WHERE kb_id = ? AND filename = ?",
                        (kb_id, "方案.pdf"),
                    ).fetchone()[0]
                )

        with (
            patch.object(knowledge_base, "_parse_and_embed_document", return_value=parsed),
            patch.object(knowledge_base, "_publish_document"),
            patch.object(
                knowledge_base.retriever, "drop_cached_retriever", side_effect=record_status
            ),
        ):
            knowledge_base.run_document_job(
                job_id, self.kb_id, fixture_document_id(self.kb_id, "方案.pdf")
            )
        self.assertEqual(statuses_at_drop[-1:], ["ready"])

    def test_rebuild_keeps_parser_mode_of_superseded_reindex(self):
        document_id = fixture_document_id(self.kb_id, "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET status = 'ready', parser_mode = 'native_text' "
                "WHERE document_id = ?",
                (document_id,),
            )
        app_storage.documents.schedule_reindex(
            self.kb_id, document_id, parser_mode="mineru_cloud"
        )

        app_storage.index_jobs.prepare_rebuild(self.kb_id)

        self.assertEqual(
            app_storage.documents.get_document(self.kb_id, document_id)["parser_mode"],
            "mineru_cloud",
        )

    def test_document_source_allowlist_contains_only_ready_versions(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                """
                INSERT INTO documents (document_id, kb_id, filename, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        fixture_document_id(self.kb_id, "就绪.pdf"),
                        self.kb_id,
                        "就绪.pdf",
                        1,
                        application_config.db_now(),
                        "ready",
                    ),
                    (
                        fixture_document_id(self.kb_id, "失败.pdf"),
                        self.kb_id,
                        "失败.pdf",
                        1,
                        application_config.db_now(),
                        "error",
                    ),
                ],
            )

        self.assertEqual(
            app_storage.documents.ready_document_ids(self.kb_id),
            [fixture_document_id(self.kb_id, "就绪.pdf")],
        )

    def test_embedding_failure_marks_the_document_error(self):
        from services.documents.types import DocumentRoutes
        from services.retrieval.chunk_index import get_chunk_index

        job_id = create_index_job(self.kb_id, "process", "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET index_job_id = ? WHERE kb_id = ? AND filename = ?",
                (job_id, self.kb_id, "方案.pdf"),
            )
        self._write_source("方案.pdf")
        routes = DocumentRoutes(
            fixed=[
                Document(
                    page_content="新制度的审批规则", metadata={"source": "方案.pdf", "page": 0}
                )
            ],
            structured=[],
        )
        with (
            patch.object(
                knowledge_base.document_loader, "load_document_routes", return_value=routes
            ),
            patch.object(
                knowledge_base, "embed_splits", side_effect=RuntimeError("embedding unavailable")
            ),
        ):
            knowledge_base.run_document_job(
                job_id,
                self.kb_id,
                fixture_document_id(self.kb_id, "方案.pdf"),
                container=dataclasses.replace(
                    get_application_container(), create_embeddings=lambda: object()
                ),
            )

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            status, error = conn.execute(
                "SELECT status, error_message FROM documents WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "方案.pdf"),
            ).fetchone()
        self.assertEqual(status, "error")
        self.assertIn("embedding unavailable", error)
        self.assertEqual(get_chunk_index().available_routes(self.kb_id, None), set())

    def test_rebuild_replaces_older_queued_job(self):
        first_job = app_storage.index_jobs.prepare_rebuild(self.kb_id)
        second_job = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            first_status = conn.execute(
                "SELECT status FROM index_jobs WHERE id = ?", (first_job,)
            ).fetchone()[0]
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE kb_id = ?", (self.kb_id,)
            ).fetchone()[0]
        self.assertEqual(first_status, INDEX_JOB_SUPERSEDED)
        self.assertEqual(active_job, second_job)

    def test_completed_rebuild_releases_its_active_marker(self):
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        app_storage.index_jobs.complete_rebuild(self.kb_id, job_id, "ready")

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE kb_id = ?", (self.kb_id,)
            ).fetchone()[0]
        self.assertIsNone(active_job)

    def test_completed_rebuild_does_not_release_a_newer_marker(self):
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)
        newer_job_id = "newer-job"
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE knowledge_bases SET active_rebuild_job_id = ? WHERE kb_id = ?",
                (newer_job_id, self.kb_id),
            )

        app_storage.index_jobs.complete_rebuild(self.kb_id, job_id, "ready")

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE kb_id = ?", (self.kb_id,)
            ).fetchone()[0]
        self.assertEqual(active_job, newer_job_id)

    def test_recovery_clears_terminal_rebuild_marker(self):
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE index_jobs SET status = ? WHERE id = ?",
                (INDEX_JOB_COMPLETED, job_id),
            )

        app_storage.index_jobs.recover_interrupted()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE kb_id = ?", (self.kb_id,)
            ).fetchone()[0]
        self.assertIsNone(active_job)

    def test_recovery_keeps_queued_jobs_for_the_worker(self):
        job_id = create_index_job(self.kb_id, "process", "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET index_job_id = ? WHERE kb_id = ? AND filename = ?",
                (job_id, self.kb_id, "方案.pdf"),
            )

        app_storage.index_jobs.recover_interrupted()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            job_status = conn.execute(
                "SELECT status FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()[0]
            document_status = conn.execute(
                "SELECT status FROM documents WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "方案.pdf"),
            ).fetchone()[0]
        self.assertEqual(job_status, INDEX_JOB_QUEUED)
        self.assertEqual(document_status, "processing")
        self.assertIn(job_id, app_storage.index_jobs.queued_ids())

    def test_job_rows_are_mapped_to_worker_and_api_contracts(self):
        job_id = create_index_job(self.kb_id, "rebuild")

        worker_job = app_storage.index_jobs.get(job_id)
        recent_job = app_storage.index_jobs.list_recent(self.kb_id, limit=1)[0]

        self.assertEqual(
            worker_job,
            {
                "id": job_id,
                "kb_id": self.kb_id,
                "document_id": None,
                "filename": None,
                "operation": "rebuild",
                "status": INDEX_JOB_QUEUED,
            },
        )
        self.assertEqual(recent_job["id"], job_id)
        self.assertIsNone(recent_job["filename"])
        self.assertEqual(recent_job["operation"], "rebuild")
        self.assertEqual(recent_job["status"], INDEX_JOB_QUEUED)
        self.assertEqual(recent_job["error_message"], "")
        self.assertIsNotNone(recent_job["created_at"])
        self.assertIsNone(recent_job["started_at"])
        self.assertIsNone(recent_job["finished_at"])

    def test_job_history_pruning_caps_terminal_rows_and_keeps_retryable_failures(self):
        completed_jobs = []
        for index in range(4):
            job_id = create_index_job(self.kb_id, "process", f"历史-{index}.pdf")
            app_storage.index_jobs.finish(job_id, INDEX_JOB_COMPLETED)
            completed_jobs.append(job_id)

        retry_job = create_index_job(self.kb_id, "delete", "待重试.pdf")
        app_storage.index_jobs.finish(retry_job, INDEX_JOB_FAILED, "index unavailable")

        deleted = app_storage.index_jobs.prune_history(
            retention_days=36500,
            max_per_kb=2,
        )

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            remaining_completed = conn.execute(
                "SELECT COUNT(*) FROM index_jobs WHERE kb_id = ? AND status = ?",
                (self.kb_id, INDEX_JOB_COMPLETED),
            ).fetchone()[0]
            retry_exists = conn.execute(
                "SELECT 1 FROM index_jobs WHERE id = ?", (retry_job,)
            ).fetchone()
        self.assertGreaterEqual(deleted, 2)
        self.assertEqual(remaining_completed, 2)
        self.assertIsNotNone(retry_exists)

    def test_dispatcher_routes_queued_document_job(self):
        job_id = create_index_job(self.kb_id, "process", "方案.pdf")
        with patch.object(index_worker.knowledge_base, "run_document_job") as run_document_job:
            index_worker.dispatch_index_job(job_id)

        run_document_job.assert_called_once_with(
            job_id,
            self.kb_id,
            fixture_document_id(self.kb_id, "方案.pdf"),
            container=ANY,
        )


if __name__ == "__main__":
    unittest.main()
