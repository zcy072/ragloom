import dataclasses
import os
import sqlite3
import unittest
from unittest.mock import ANY, patch

try:
    import app_storage
    from services.core import config as application_config
    from services.indexing.jobs import (
        INDEX_JOB_COMPLETED,
        INDEX_JOB_FAILED,
        INDEX_JOB_QUEUED,
        INDEX_JOB_SUPERSEDED,
    )
except ModuleNotFoundError:
    app_storage = None

try:
    from services.indexing import pipeline as knowledge_base
except ModuleNotFoundError:
    knowledge_base = None

try:
    from services.indexing import worker as index_worker
except ModuleNotFoundError:
    index_worker = None

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    Document = None

from services.core.container import get_application_container
from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(app_storage, "requires the SQLite application module")
class IndexJobTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at) VALUES (?, lower(hex(randomblob(16))), ?)",
                ("研发部", application_config.db_now()),
            )
            conn.execute(
                """
                INSERT INTO documents (kb_name, filename, path, size, uploaded_at, status, index_job_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "研发部",
                    "方案.pdf",
                    "/tmp/方案.pdf",
                    1,
                    application_config.db_now(),
                    "processing",
                    "",
                ),
            )

    def _write_source(self, filename: str) -> str:
        """在知识库目录写入真实源文件；索引任务会为它创建快照。"""
        directory = app_storage.documents.ensure_knowledge_base_dir("研发部")
        path = os.path.join(directory, filename)
        with open(path, "wb") as handle:
            handle.write(b"source")
        return path

    def test_document_job_marks_latest_document_ready(self):
        if knowledge_base is None:
            self.skipTest("requires the backend Python dependencies")
        job_id = app_storage.index_jobs.create("研发部", "process", "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET index_job_id = ? WHERE kb_name = ? AND filename = ?",
                (job_id, "研发部", "方案.pdf"),
            )

        file_path = self._write_source("方案.pdf")
        parsed = knowledge_base._ParsedDocument([], [], {})
        with patch.object(
            knowledge_base, "_parse_and_embed_document", return_value=parsed
        ), patch.object(knowledge_base, "_publish_document") as publish:
            knowledge_base.run_document_job(job_id, "研发部", "方案.pdf", file_path)
        publish.assert_called_once()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            status = conn.execute(
                "SELECT status FROM documents WHERE kb_name = ? AND filename = ?", ("研发部", "方案.pdf")
            ).fetchone()[0]
            job_status = conn.execute("SELECT status FROM index_jobs WHERE id = ?", (job_id,)).fetchone()[0]
        self.assertEqual(status, "ready")
        self.assertEqual(job_status, INDEX_JOB_COMPLETED)

    def test_document_source_allowlist_contains_only_ready_versions(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                """
                INSERT INTO documents (kb_name, filename, path, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "研发部",
                        "就绪.pdf",
                        "/tmp/就绪.pdf",
                        1,
                        application_config.db_now(),
                        "ready",
                    ),
                    (
                        "研发部",
                        "失败.pdf",
                        "/tmp/失败.pdf",
                        1,
                        application_config.db_now(),
                        "error",
                    ),
                ],
            )

        self.assertEqual(app_storage.documents.ready_sources("研发部"), ["就绪.pdf"])

    def test_embedding_failure_marks_the_document_error(self):
        if knowledge_base is None or Document is None:
            self.skipTest("requires the backend Python dependencies")
        from services.documents.types import DocumentRoutes
        from services.retrieval.chunk_index import get_chunk_index

        job_id = app_storage.index_jobs.create("研发部", "process", "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET index_job_id = ? WHERE kb_name = ? AND filename = ?",
                (job_id, "研发部", "方案.pdf"),
            )
        file_path = self._write_source("方案.pdf")
        routes = DocumentRoutes(
            fixed=[
                Document(page_content="新制度的审批规则", metadata={"source": "方案.pdf", "page": 0})
            ],
            structured=[],
        )
        with patch.object(
            knowledge_base.document_loader, "load_document_routes", return_value=routes
        ), patch.object(
            knowledge_base, "embed_splits", side_effect=RuntimeError("embedding unavailable")
        ):
            knowledge_base.run_document_job(
                job_id,
                "研发部",
                "方案.pdf",
                file_path,
                container=dataclasses.replace(
                    get_application_container(), create_embeddings=lambda: object()
                ),
            )

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            status, error = conn.execute(
                "SELECT status, error_message FROM documents WHERE kb_name = ? AND filename = ?",
                ("研发部", "方案.pdf"),
            ).fetchone()
        self.assertEqual(status, "error")
        self.assertIn("embedding unavailable", error)
        kb_id = app_storage.knowledge_bases.kb_id_for("研发部")
        self.assertEqual(get_chunk_index().available_routes(kb_id, None), set())

    def test_rebuild_replaces_older_queued_job(self):
        first_job = app_storage.index_jobs.prepare_rebuild("研发部")
        second_job = app_storage.index_jobs.prepare_rebuild("研发部")

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            first_status = conn.execute(
                "SELECT status FROM index_jobs WHERE id = ?", (first_job,)
            ).fetchone()[0]
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE name = ?", ("研发部",)
            ).fetchone()[0]
        self.assertEqual(first_status, INDEX_JOB_SUPERSEDED)
        self.assertEqual(active_job, second_job)

    def test_completed_rebuild_releases_its_active_marker(self):
        job_id = app_storage.index_jobs.prepare_rebuild("研发部")

        app_storage.index_jobs.complete_rebuild("研发部", job_id, "ready")

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE name = ?", ("研发部",)
            ).fetchone()[0]
        self.assertEqual(active_job, "")

    def test_completed_rebuild_does_not_release_a_newer_marker(self):
        job_id = app_storage.index_jobs.prepare_rebuild("研发部")
        newer_job_id = "newer-job"
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE knowledge_bases SET active_rebuild_job_id = ? WHERE name = ?",
                (newer_job_id, "研发部"),
            )

        app_storage.index_jobs.complete_rebuild("研发部", job_id, "ready")

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE name = ?", ("研发部",)
            ).fetchone()[0]
        self.assertEqual(active_job, newer_job_id)

    def test_recovery_clears_terminal_rebuild_marker(self):
        job_id = app_storage.index_jobs.prepare_rebuild("研发部")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE index_jobs SET status = ? WHERE id = ?",
                (INDEX_JOB_COMPLETED, job_id),
            )

        app_storage.index_jobs.recover_interrupted()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            active_job = conn.execute(
                "SELECT active_rebuild_job_id FROM knowledge_bases WHERE name = ?", ("研发部",)
            ).fetchone()[0]
        self.assertEqual(active_job, "")

    def test_recovery_keeps_queued_jobs_for_the_worker(self):
        job_id = app_storage.index_jobs.create("研发部", "process", "方案.pdf")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE documents SET index_job_id = ? WHERE kb_name = ? AND filename = ?",
                (job_id, "研发部", "方案.pdf"),
            )

        app_storage.index_jobs.recover_interrupted()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            job_status = conn.execute(
                "SELECT status FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()[0]
            document_status = conn.execute(
                "SELECT status FROM documents WHERE kb_name = ? AND filename = ?",
                ("研发部", "方案.pdf"),
            ).fetchone()[0]
        self.assertEqual(job_status, INDEX_JOB_QUEUED)
        self.assertEqual(document_status, "processing")
        self.assertIn(job_id, app_storage.index_jobs.queued_ids())

    def test_job_rows_are_mapped_to_worker_and_api_contracts(self):
        job_id = app_storage.index_jobs.create("研发部", "rebuild")

        worker_job = app_storage.index_jobs.get(job_id)
        recent_job = app_storage.index_jobs.list_recent("研发部", limit=1)[0]

        self.assertEqual(
            worker_job,
            {
                "id": job_id,
                "kb_name": "研发部",
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

    def test_job_history_pruning_caps_terminal_rows_and_removes_orphans(self):
        completed_jobs = []
        for index in range(4):
            job_id = app_storage.index_jobs.create("研发部", "process", f"历史-{index}.pdf")
            app_storage.index_jobs.finish(job_id, INDEX_JOB_COMPLETED)
            completed_jobs.append(job_id)

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at) VALUES (?, lower(hex(randomblob(16))), ?)",
                ("临时知识库", application_config.db_now()),
            )
        orphan_job = app_storage.index_jobs.create("临时知识库", "process", "旧文件.pdf")
        app_storage.index_jobs.finish(orphan_job, INDEX_JOB_COMPLETED)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("DELETE FROM knowledge_bases WHERE name = ?", ("临时知识库",))

        retry_job = app_storage.index_jobs.create("研发部", "delete", "待重试.pdf")
        app_storage.index_jobs.finish(retry_job, INDEX_JOB_FAILED, "index unavailable")

        deleted = app_storage.index_jobs.prune_history(
            retention_days=36500,
            max_per_kb=2,
        )

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            remaining_completed = conn.execute(
                "SELECT COUNT(*) FROM index_jobs "
                "WHERE kb_name = ? AND status = ?",
                ("研发部", INDEX_JOB_COMPLETED),
            ).fetchone()[0]
            orphan_exists = conn.execute(
                "SELECT 1 FROM index_jobs WHERE id = ?", (orphan_job,)
            ).fetchone()
            retry_exists = conn.execute(
                "SELECT 1 FROM index_jobs WHERE id = ?", (retry_job,)
            ).fetchone()
        self.assertGreaterEqual(deleted, 3)
        self.assertEqual(remaining_completed, 2)
        self.assertIsNone(orphan_exists)
        self.assertIsNotNone(retry_exists)

    def test_dispatcher_routes_queued_document_job(self):
        if index_worker is None:
            self.skipTest("requires the backend Python dependencies")
        job_id = app_storage.index_jobs.create("研发部", "process", "方案.pdf")
        with patch.object(index_worker.knowledge_base, "run_document_job") as run_document_job:
            index_worker.dispatch_index_job(job_id)

        run_document_job.assert_called_once_with(
            job_id,
            "研发部",
            "方案.pdf",
            app_storage.documents.document_path("研发部", "方案.pdf"),
            container=ANY,
        )


if __name__ == "__main__":
    unittest.main()
