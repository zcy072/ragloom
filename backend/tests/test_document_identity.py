"""独立文档身份、事务、进程中断恢复与同名隔离的行为测试。"""

import asyncio
import dataclasses
import io
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import app_storage
from services.core import sqlite_connections as application_db
from services.core.container import get_application_container
from services.core.errors import ResourceNotFoundError
from services.documents import recovery, use_cases
from services.indexing import pipeline, worker
from services.retrieval import context, parent_store, retriever
from services.retrieval.chunk_index import get_chunk_index
from tests.support import IsolatedStorageTestCase, add_knowledge_bases, api_client


class DocumentIdentityTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.kb_id, self.other_id = add_knowledge_bases("主库", "其他库")

    def upload(self, content=b"text", filename="same.txt"):
        with patch.object(worker, "enqueue", return_value=False):
            return use_cases.upload_document(
                self.kb_id, filename, io.BytesIO(content), metadata={}, parser_mode="native_text"
            )

    def test_concurrent_same_name_uploads_have_independent_files_and_jobs(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = list(executor.map(self.upload, (b"first", b"second")))
        self.assertNotEqual(first.document_id, second.document_id)
        self.assertNotEqual(first.job_id, second.job_id)
        records = app_storage.documents.list_documents(self.kb_id)
        self.assertEqual(len(records), 2)
        self.assertEqual({record["filename"] for record in records}, {"same.txt"})
        for result, content in ((first, b"first"), (second, b"second")):
            path = Path(app_storage.documents.document_path(self.kb_id, result.document_id))
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(path.parent.name, result.document_id)
            self.assertEqual(
                app_storage.index_jobs.get(result.job_id)["document_id"], result.document_id
            )
        with self.assertRaises(ResourceNotFoundError):
            app_storage.documents.get_document(self.other_id, first.document_id)

    def test_same_content_is_still_two_documents_and_deletion_only_cleans_one(self):
        first, second = self.upload(), self.upload()
        for result in (first, second):
            parent_store.add_document_version(
                self.kb_id, result.document_id, [("p1", "same.txt", "0", result.document_id, "")]
            )
        with patch.object(worker, "enqueue", return_value=False):
            deletion = use_cases.delete_document(self.kb_id, first.document_id)
        self.assertFalse(app_storage.documents.document_exists(self.kb_id, first.document_id))
        pipeline.run_delete_job(deletion.job_id, self.kb_id, first.document_id)
        self.assertFalse(
            Path(app_storage.documents.document_directory(self.kb_id, first.document_id)).exists()
        )
        self.assertTrue(
            Path(app_storage.documents.document_path(self.kb_id, second.document_id)).is_file()
        )
        self.assertEqual(
            parent_store.load(self.kb_id, second.document_id, ["p1"])["p1"][3], second.document_id
        )
        self.assertEqual(app_storage.index_jobs.get(deletion.job_id)["status"], "completed")

    def test_reindex_retains_identity_and_atomically_changes_ownership(self):
        original = self.upload()
        with patch.object(worker, "enqueue", return_value=False):
            refreshed = use_cases.reindex_document(self.kb_id, original.document_id)
        self.assertEqual(refreshed.document_id, original.document_id)
        self.assertNotEqual(refreshed.job_id, original.job_id)
        self.assertFalse(
            app_storage.index_jobs.is_current_document_job(
                self.kb_id, original.document_id, original.job_id
            )
        )
        self.assertTrue(
            app_storage.index_jobs.is_current_document_job(
                self.kb_id, original.document_id, refreshed.job_id
            )
        )
        self.assertEqual(len(app_storage.documents.list_documents(self.kb_id)), 1)

    def test_second_insert_failure_rolls_back_document_and_file(self):
        with patch("services.documents.store.insert_job", side_effect=RuntimeError("SQL failed")):
            with self.assertRaisesRegex(RuntimeError, "SQL failed"):
                self.upload()
        self.assertEqual(app_storage.documents.list_documents(self.kb_id), [])
        self.assertEqual(app_storage.index_jobs.queued_ids(), [])
        self.assertEqual(
            list(Path(app_storage.documents.knowledge_base_path(self.kb_id)).iterdir()), []
        )

    def test_enqueue_error_keeps_committed_upload(self):
        with patch.object(worker, "enqueue", side_effect=RuntimeError("wake failed")):
            result = use_cases.upload_document(
                self.kb_id, "same.txt", io.BytesIO(b"text"), metadata={}, parser_mode="native_text"
            )
        self.assertTrue(
            Path(app_storage.documents.document_path(self.kb_id, result.document_id)).exists()
        )
        self.assertIn(result.job_id, app_storage.index_jobs.queued_ids())

    def test_startup_sweep_preserves_registered_and_pending_deletion_and_symlinks(self):
        registered, deleting = self.upload(), self.upload()
        app_storage.documents.schedule_document_delete(self.kb_id, deleting.document_id)
        orphan_id = str(uuid.uuid4())
        orphan = Path(app_storage.documents.document_directory(self.kb_id, orphan_id))
        orphan.mkdir()
        (orphan / "file").write_text("orphan")
        outside = Path(self._storage_directory.name) / "outside"
        outside.mkdir()
        (outside / "retained").write_text("keep")
        link = Path(app_storage.documents.document_directory(self.kb_id, str(uuid.uuid4())))
        link.symlink_to(outside, target_is_directory=True)
        recovery.sweep_unregistered_documents()
        recovery.sweep_unregistered_documents()
        self.assertFalse(orphan.exists())
        self.assertTrue(
            Path(app_storage.documents.document_path(self.kb_id, registered.document_id)).exists()
        )
        self.assertTrue(
            Path(
                app_storage.documents.document_directory(self.kb_id, deleting.document_id)
            ).exists()
        )
        self.assertTrue((outside / "retained").exists())

    def test_process_exit_after_file_publication_is_cleaned_without_registering(self):
        script = """
import io, os
import app_storage
from services.documents import use_cases
app_storage.documents.create_document_with_job = lambda *a, **kw: os._exit(23)
use_cases.upload_document(os.environ['TEST_KB_ID'], 'same.txt', io.BytesIO(b'new'), metadata={}, parser_mode='native_text')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "TEST_KB_ID": self.kb_id},
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 23, result.stderr.decode())
        root = Path(app_storage.documents.knowledge_base_path(self.kb_id))
        self.assertTrue(any(root.iterdir()))
        recovery.sweep_unregistered_documents()
        self.assertEqual(list(root.iterdir()), [])
        self.assertEqual(app_storage.documents.list_documents(self.kb_id), [])

    def test_process_exit_after_commit_retains_queued_job_for_recovery(self):
        script = """
import io, os
from services.documents import use_cases
use_cases._wake_worker = lambda *a: os._exit(24)
use_cases.upload_document(os.environ['TEST_KB_ID'], 'same.txt', io.BytesIO(b'new'), metadata={}, parser_mode='native_text')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "TEST_KB_ID": self.kb_id},
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 24, result.stderr.decode())
        (record,) = app_storage.documents.list_documents(self.kb_id)
        recovery.sweep_unregistered_documents()
        self.assertTrue(
            Path(app_storage.documents.document_path(self.kb_id, record["document_id"])).exists()
        )
        app_storage.index_jobs.recover_interrupted()
        self.assertIn(record["index_job_id"], app_storage.index_jobs.queued_ids())

    def test_failed_delete_retains_target_for_restart_and_history_pruning(self):
        result = self.upload()
        job_id = app_storage.documents.schedule_document_delete(self.kb_id, result.document_id)
        with patch.object(
            get_chunk_index(), "delete_document", side_effect=RuntimeError("unavailable")
        ):
            pipeline.run_delete_job(job_id, self.kb_id, result.document_id)
        app_storage.index_jobs.prune_history(retention_days=1, max_per_kb=1)
        self.assertEqual(app_storage.index_jobs.get(job_id)["document_id"], result.document_id)
        app_storage.index_jobs.recover_interrupted()
        pipeline.run_delete_job(job_id, self.kb_id, result.document_id)
        self.assertEqual(app_storage.index_jobs.get(job_id)["status"], "completed")
        with application_db.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_identical_names_and_parent_ids_stay_distinct_through_retrieval_and_citations(self):
        from services.chat.sources import build_source_catalog, select_cited_sources

        class Embeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, _query):
                return [1.0, 0.0]

        container = dataclasses.replace(
            get_application_container(),
            create_embeddings=Embeddings,
            create_query_embeddings=Embeddings,
        )
        first, second = (
            self.upload("共同规范：审批需要负责人确认。".encode()),
            self.upload("共同规范：审批需要负责人确认。".encode()),
        )
        split_for_index = pipeline._split_for_index

        def shared_local_parent(*args, **kwargs):
            splits = split_for_index(*args, **kwargs)
            for split in splits:
                if split.metadata.get("parent_id"):
                    split.metadata["parent_id"] = "local-parent"
            return splits

        with patch.object(pipeline, "_split_for_index", side_effect=shared_local_parent):
            for result in (first, second):
                pipeline.run_document_job(
                    result.job_id, self.kb_id, result.document_id, container=container
                )
        hits = retriever.build_retriever(self.kb_id, container=container).invoke("审批")
        self.assertEqual(
            {hit.metadata["document_id"] for hit in hits}, {first.document_id, second.document_id}
        )
        self.assertEqual(len({hit.metadata["parent_id"] for hit in hits}), 1)
        spans = context.build_context_spans(self.kb_id, hits, neighbor_chunks=0)
        self.assertEqual(
            {span.document_id for span in spans}, {first.document_id, second.document_id}
        )
        cited = select_cited_sources("两份材料。[来源 1][来源 2]", build_source_catalog(spans))
        self.assertEqual(len(cited), 2)
        self.assertEqual(
            {source["document_id"] for source in cited}, {first.document_id, second.document_id}
        )
        for source in cited:
            self.assertEqual(source["evidence"][0]["document_id"], source["document_id"])

        app_storage.bootstrap_defaults()
        common_id = app_storage.knowledge_bases.common_kb_id()
        with patch.object(worker, "enqueue", return_value=False):
            common = use_cases.upload_document(
                common_id,
                "same.txt",
                io.BytesIO("通用审批规范".encode()),
                metadata={},
                parser_mode="native_text",
            )
        pipeline.run_document_job(common.job_id, common_id, common.document_id, container=container)
        filters = {"document_ids": [first.document_id]}
        for documents in (
            retriever.build_retriever(
                self.kb_id, filters, include_common=True, container=container
            ).invoke("审批"),
            retriever.retrieve_lexical_candidates(
                self.kb_id, "审批", filters, include_common=True, container=container
            ),
        ):
            self.assertEqual(
                {doc.metadata["document_id"] for doc in documents},
                {first.document_id, common.document_id},
            )

    def test_http_requires_document_uuid_and_matching_library_and_preserves_deleted_reference(self):
        app_storage.bootstrap_defaults()
        original, same_name = self.upload(b"old"), self.upload(b"new")

        async def run():
            async with api_client() as client:
                await client.post(
                    "/api/login", json={"username": "admin", "password": "test-admin-password"}
                )
                for method in ("GET", "DELETE"):
                    response = await client.request(
                        method, f"/api/documents/{self.other_id}/{original.document_id}"
                    )
                    self.assertEqual(response.status_code, 404, response.text)
                invalid = await client.get(f"/api/documents/{self.kb_id}/same.txt")
                self.assertEqual(invalid.status_code, 422)
                with patch.object(worker, "enqueue", return_value=False):
                    deleted = await client.delete(
                        f"/api/documents/{self.kb_id}/{original.document_id}"
                    )
                self.assertEqual(deleted.status_code, 200)
                old_preview = await client.get(
                    f"/api/documents/{self.kb_id}/{original.document_id}?preview=true"
                )
                self.assertEqual(old_preview.status_code, 404)
                new_preview = await client.get(
                    f"/api/documents/{self.kb_id}/{same_name.document_id}"
                )
                self.assertEqual(new_preview.content, b"new")

        asyncio.run(run())

    def test_schema_rejects_filename_identity_and_task_without_target(self):
        import sqlite3

        with application_db.connect(app_storage.DB_FILE) as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO index_jobs (id, kb_id, operation, status, created_at) "
                    "VALUES ('invalid', ?, 'process', 'queued', '')",
                    (self.kb_id,),
                )
            conn.execute(
                "CREATE UNIQUE INDEX invalid_filename_identity ON documents(kb_id, filename)"
            )
        with self.assertRaisesRegex(RuntimeError, "identity uniqueness"):
            app_storage.initialize_database()
