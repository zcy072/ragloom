import dataclasses
import os
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch

try:
    from langchain_core.documents import Document

    import app_storage
    from services.core import config as application_config
    from services.core.container import get_application_container
    from services.core.settings import get_settings
    from services.documents import mineru as mineru_client
    from services.documents.types import DocumentLoadError, DocumentRoutes
    from services.indexing import pipeline as knowledge_base
    from services.indexing import workspace as index_workspace
    from services.indexing.jobs import INDEX_JOB_COMPLETED, INDEX_JOB_FAILED, INDEX_JOB_SUPERSEDED
    from services.indexing.locks import knowledge_base_lock
    from services.retrieval import source_state
except ModuleNotFoundError:
    app_storage = None

from tests.support import IsolatedStorageTestCase

KB = "研发部"
WAIT = 5


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class BrokenEmbeddings:
    def embed_documents(self, texts):
        raise RuntimeError("embedding unavailable")


@unittest.skipUnless(app_storage, "requires the backend Python dependencies")
class IndexConcurrencyTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, created_at) VALUES (?, ?)",
                (KB, application_config.db_now()),
            )
        self.kb_dir = app_storage.documents.ensure_knowledge_base_dir(KB)
        self.on_load = lambda path, filename: None
        self.embeddings = FakeEmbeddings()

        def fake_load(path, filename, parser_mode, **kwargs):
            self.on_load(path, filename)
            return DocumentRoutes(
                fixed=[
                    Document(
                        page_content=f"{filename} 的审批规则内容",
                        metadata={"source": filename, "page": 0},
                    )
                ],
                structured=[],
            )

        for patcher in (
            patch.object(knowledge_base.document_loader, "load_document_routes", fake_load),
            patch.object(
                knowledge_base.visual_content, "load_mineru_visual_documents", return_value=[]
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        upsert = patch.object(knowledge_base.chroma_index, "chroma_upsert_splits")
        self.upsert = upsert.start()
        self.addCleanup(upsert.stop)
        self.errors: list[BaseException] = []

    # -- helpers ---------------------------------------------------------

    def _container(self):
        return dataclasses.replace(
            get_application_container(), create_embeddings=lambda: self.embeddings
        )

    def _add_document(self, filename, content=b"v1", status="processing"):
        path = os.path.join(self.kb_dir, filename)
        with open(path, "wb") as handle:
            handle.write(content)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO documents (kb_name, filename, path, size, uploaded_at, status, "
                "index_job_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (KB, filename, path, len(content), application_config.db_now(), status, ""),
            )
        if status != "processing":
            return "", path
        job_id = app_storage.index_jobs.create(KB, "process", filename)
        app_storage.index_jobs.attach_document_job(KB, filename, job_id)
        return job_id, path

    def _reupload(self, filename, content):
        """模拟上传接口：在知识库锁内替换文件并登记新任务。"""
        path = os.path.join(self.kb_dir, filename)
        with knowledge_base_lock(KB):
            temporary = path + ".upload"
            with open(temporary, "wb") as handle:
                handle.write(content)
            os.replace(temporary, path)
            job_id = app_storage.index_jobs.create(KB, "process", filename)
            app_storage.index_jobs.attach_document_job(KB, filename, job_id)
        return job_id

    def _run(self, job_id, filename, path):
        knowledge_base.run_document_job(job_id, KB, filename, path, container=self._container())

    def _start(self, job_id, filename, path):
        def target():
            try:
                self._run(job_id, filename, path)
            except BaseException as exc:  # surfaced by _join
                self.errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def _join(self, *threads):
        for thread in threads:
            thread.join(timeout=WAIT)
            self.assertFalse(thread.is_alive(), "索引任务线程未在预期时间内结束")
        self.assertEqual(self.errors, [])

    def _job_status(self, job_id):
        return app_storage.index_jobs.get(job_id)["status"]

    def _document(self, filename):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            return conn.execute(
                "SELECT status, error_message, index_job_id FROM documents "
                "WHERE kb_name = ? AND filename = ?",
                (KB, filename),
            ).fetchone()

    def _assert_workspace_removed(self, job_id):
        settings = get_settings()
        self.assertFalse(os.path.exists(os.path.join(self.kb_dir, ".index_jobs", job_id)))
        self.assertFalse(
            os.path.exists(os.path.join(settings.mineru_results_dir, KB, ".index_jobs", job_id))
        )

    # -- tests -----------------------------------------------------------

    def test_documents_in_the_same_knowledge_base_parse_in_parallel(self):
        barrier = threading.Barrier(2, timeout=WAIT)
        self.on_load = lambda path, filename: barrier.wait()
        first_id, first_path = self._add_document("甲.pdf")
        second_id, second_path = self._add_document("乙.pdf")

        threads = [
            self._start(first_id, "甲.pdf", first_path),
            self._start(second_id, "乙.pdf", second_path),
        ]
        self._join(*threads)

        # 串行执行时第一个任务会在屏障处超时失败。
        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.assertEqual(self._document("乙.pdf")[0], "ready")

    def test_knowledge_base_lock_is_free_while_a_document_parses(self):
        entered = threading.Event()
        release = threading.Event()

        def block(path, filename):
            entered.set()
            release.wait(WAIT)

        self.on_load = block
        job_id, path = self._add_document("甲.pdf")
        thread = self._start(job_id, "甲.pdf", path)
        self.assertTrue(entered.wait(WAIT))

        acquired = threading.Event()

        def probe():
            with knowledge_base_lock(KB):
                acquired.set()

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()
        try:
            self.assertTrue(acquired.wait(WAIT), "解析期间不应持有知识库锁")
        finally:
            release.set()
        self._join(thread, prober)

    def test_index_writes_are_exclusive(self):
        barrier = threading.Barrier(2, timeout=WAIT)
        self.on_load = lambda path, filename: barrier.wait()
        original = knowledge_base.fts_index.update_fts5_index
        guard = threading.Lock()
        state = {"active": 0, "peak": 0}

        def tracked(*args, **kwargs):
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                threading.Event().wait(0.1)
                return original(*args, **kwargs)
            finally:
                with guard:
                    state["active"] -= 1

        first_id, first_path = self._add_document("甲.pdf")
        second_id, second_path = self._add_document("乙.pdf")
        with patch.object(knowledge_base.fts_index, "update_fts5_index", side_effect=tracked):
            self._join(
                self._start(first_id, "甲.pdf", first_path),
                self._start(second_id, "乙.pdf", second_path),
            )

        self.assertEqual(state["peak"], 1)
        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.assertEqual(self._document("乙.pdf")[0], "ready")

    def test_reupload_during_parse_supersedes_the_old_job(self):
        entered = threading.Event()
        release = threading.Event()
        seen: list[bytes] = []

        def block_then_read(path, filename):
            entered.set()
            release.wait(WAIT)
            with open(path, "rb") as handle:
                seen.append(handle.read())

        self.on_load = block_then_read
        old_id, path = self._add_document("甲.pdf", b"v1")
        thread = self._start(old_id, "甲.pdf", path)
        self.assertTrue(entered.wait(WAIT))

        new_id = self._reupload("甲.pdf", b"v2")
        release.set()
        self._join(thread)

        # 旧任务读到的是自己的快照，而不是刚被替换的新文件。
        self.assertEqual(seen, [b"v1"])
        self.assertEqual(self._job_status(old_id), INDEX_JOB_SUPERSEDED)
        status, _error, owner = self._document("甲.pdf")
        self.assertEqual((status, owner), ("processing", new_id))
        self.assertFalse(os.path.exists(knowledge_base.fts_index.fts_db_path(KB)))
        self.upsert.assert_not_called()
        self._assert_workspace_removed(old_id)

        self._run(new_id, "甲.pdf", path)
        self.assertEqual(seen, [b"v1", b"v2"])
        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.assertEqual(self._job_status(new_id), INDEX_JOB_COMPLETED)

    def test_delete_during_parse_supersedes_without_residue(self):
        entered = threading.Event()
        release = threading.Event()

        def block_and_write_result(path, filename):
            result_dir = mineru_client.result_dir_for_document(path)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "full.md").write_text("旧结果", encoding="utf-8")
            entered.set()
            release.wait(WAIT)

        self.on_load = block_and_write_result
        job_id, path = self._add_document("甲.pdf")
        thread = self._start(job_id, "甲.pdf", path)
        self.assertTrue(entered.wait(WAIT))

        with knowledge_base_lock(KB):
            app_storage.documents.delete_document(KB, "甲.pdf")
        release.set()
        self._join(thread)

        self.assertEqual(self._job_status(job_id), INDEX_JOB_SUPERSEDED)
        self.assertIsNone(self._document("甲.pdf"))
        self.assertFalse(os.path.exists(knowledge_base.fts_index.fts_db_path(KB)))
        formal = mineru_client.result_dir_for_document(path)
        self.assertFalse(formal.exists(), "被替代的任务不应发布结果目录")
        self._assert_workspace_removed(job_id)

    def test_rename_is_refused_while_a_job_is_running(self):
        from services.core.errors import InvalidStateError
        from services.knowledge_bases import use_cases as kb_use_cases

        entered = threading.Event()
        release = threading.Event()

        def block(path, filename):
            entered.set()
            release.wait(WAIT)

        self.on_load = block
        job_id, path = self._add_document("甲.pdf")
        thread = self._start(job_id, "甲.pdf", path)
        self.assertTrue(entered.wait(WAIT))
        try:
            with self.assertRaises(InvalidStateError) as raised:
                kb_use_cases.rename_knowledge_base(KB, "新名称")
            self.assertEqual(raised.exception.code, "knowledge_base_has_running_jobs")
            self.assertTrue(os.path.isdir(self.kb_dir))
        finally:
            release.set()
        self._join(thread)

        kb_use_cases.rename_knowledge_base(KB, "新名称")
        self.assertFalse(os.path.exists(self.kb_dir))

    def test_knowledge_base_deleted_during_parse_leaves_no_directories(self):
        from services.knowledge_bases import use_cases as kb_use_cases

        entered = threading.Event()
        release = threading.Event()

        def block_and_write_result(path, filename):
            entered.set()
            release.wait(WAIT)
            result_dir = mineru_client.result_dir_for_document(path)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "full.md").write_text("解析结果", encoding="utf-8")

        self.on_load = block_and_write_result
        job_id, path = self._add_document("甲.pdf")
        thread = self._start(job_id, "甲.pdf", path)
        self.assertTrue(entered.wait(WAIT))

        kb_use_cases.delete_knowledge_base(KB)
        release.set()
        self._join(thread)

        settings = get_settings()
        self.assertFalse(os.path.exists(self.kb_dir))
        self.assertFalse(os.path.exists(os.path.join(settings.mineru_results_dir, KB)))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_SUPERSEDED)

    def test_successful_job_publishes_its_result_directory(self):
        def write_result(path, filename):
            result_dir = mineru_client.result_dir_for_document(path)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "full.md").write_text("解析结果", encoding="utf-8")

        self.on_load = write_result
        job_id, path = self._add_document("甲.pdf")
        # 正式目录里的旧结果会被新结果整体替换。
        formal = mineru_client.result_dir_for_document(path)
        formal.mkdir(parents=True, exist_ok=True)
        (formal / "stale.txt").write_text("旧", encoding="utf-8")

        self._run(job_id, "甲.pdf", path)

        self.assertEqual((formal / "full.md").read_text(encoding="utf-8"), "解析结果")
        self.assertFalse((formal / "stale.txt").exists())
        self._assert_workspace_removed(job_id)
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)

    def test_parse_failure_marks_error_and_cleans_workspace(self):
        def fail(path, filename):
            raise ValueError("解析失败")

        self.on_load = fail
        job_id, path = self._add_document("甲.pdf")

        self._run(job_id, "甲.pdf", path)

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual((status, error), ("error", "解析失败"))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)
        self._assert_workspace_removed(job_id)

    def test_embedding_failure_degrades_the_document_to_fts_only(self):
        self.embeddings = BrokenEmbeddings()
        job_id, path = self._add_document("甲.pdf")

        self._run(job_id, "甲.pdf", path)

        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.upsert.assert_not_called()
        self.assertIn("甲.pdf", source_state.chroma_disabled_sources(KB))
        self.assertTrue(os.path.exists(knowledge_base.fts_index.fts_db_path(KB)))

    def test_missing_fts_with_other_ready_documents_requires_full_rebuild(self):
        self._add_document("旧.pdf", status="ready")
        job_id, path = self._add_document("甲.pdf")

        self._run(job_id, "甲.pdf", path)

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual(status, "error")
        self.assertIn("完整重建", error)
        self.assertFalse(os.path.exists(knowledge_base.fts_index.fts_db_path(KB)))

    def test_missing_source_file_fails_the_job(self):
        job_id, path = self._add_document("甲.pdf")
        os.remove(path)

        self._run(job_id, "甲.pdf", path)

        self.assertEqual(self._document("甲.pdf")[0], "error")
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)

    def test_rebuild_isolates_a_failing_document(self):
        def fail_for_bad(path, filename):
            if filename == "坏.pdf":
                raise DocumentLoadError("PDF 共 492 页，超过上限")

        self.on_load = fail_for_bad
        self._add_document("好.pdf", status="ready")
        self._add_document("坏.pdf", status="ready")
        job_id = app_storage.index_jobs.prepare_rebuild(KB)

        knowledge_base.run_rebuild_job(job_id, KB, container=self._container())

        self.assertEqual(self._document("好.pdf")[:2], ("ready", ""))
        status, error, _owner = self._document("坏.pdf")
        self.assertEqual(status, "error")
        self.assertIn("492", error)
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)
        self.assertEqual(
            {call.args[2] for call in self.upsert.call_args_list}, {"好.pdf"}
        )
        self.assertTrue(os.path.exists(knowledge_base.fts_index.fts_db_path(KB)))

    def test_rebuild_with_only_failing_documents_fails_the_job(self):
        def fail(path, filename):
            raise DocumentLoadError("无法解析")

        self.on_load = fail
        self._add_document("坏.pdf", status="ready")
        job_id = app_storage.index_jobs.prepare_rebuild(KB)

        knowledge_base.run_rebuild_job(job_id, KB, container=self._container())

        status, error, _owner = self._document("坏.pdf")
        self.assertEqual((status, error), ("error", "无法解析"))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)
        self.upsert.assert_not_called()

    def test_rebuild_marks_each_document_ready_as_soon_as_it_finishes(self):
        release = threading.Event()

        def block_slow(path, filename):
            if filename == "慢.pdf":
                release.wait(WAIT)

        self.on_load = block_slow
        self._add_document("快.pdf", status="ready")
        self._add_document("慢.pdf", status="ready")
        job_id = app_storage.index_jobs.prepare_rebuild(KB)

        def target():
            try:
                knowledge_base.run_rebuild_job(job_id, KB, container=self._container())
            except BaseException as exc:  # surfaced by _join
                self.errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + WAIT
            while self._document("快.pdf")[0] != "ready" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(self._document("快.pdf")[0], "ready")
            self.assertEqual(self._document("慢.pdf")[0], "processing")
            self.assertEqual(self._job_status(job_id), "running")
        finally:
            release.set()
        self._join(thread)

        self.assertEqual(self._document("慢.pdf")[0], "ready")
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)

    def test_rebuild_drops_vectors_of_documents_that_are_no_longer_ready(self):
        self._add_document("好.pdf", status="ready")
        job_id = app_storage.index_jobs.prepare_rebuild(KB)

        with patch.object(
            knowledge_base.chroma_index, "chroma_delete_stale_sources"
        ) as delete_stale:
            knowledge_base.run_rebuild_job(job_id, KB, container=self._container())

        delete_stale.assert_called_once_with(KB, {"好.pdf"})

    def test_startup_sweep_removes_only_job_workspaces(self):
        settings = get_settings()
        stale_data = os.path.join(self.kb_dir, ".index_jobs", "dead-job")
        stale_results = os.path.join(settings.mineru_results_dir, KB, ".index_jobs", "dead-job")
        keep_result = os.path.join(settings.mineru_results_dir, KB, "制度.pdf.mineru")
        for directory in (stale_data, stale_results, keep_result):
            os.makedirs(directory)
        document = os.path.join(self.kb_dir, "制度.pdf")
        with open(document, "wb") as handle:
            handle.write(b"keep")

        index_workspace.sweep_stale_workspaces()

        self.assertFalse(os.path.exists(os.path.dirname(stale_data)))
        self.assertFalse(os.path.exists(os.path.dirname(stale_results)))
        self.assertTrue(os.path.isdir(keep_result))
        self.assertTrue(os.path.isfile(document))
        self.assertTrue(os.path.isfile(app_storage.DB_FILE))


if __name__ == "__main__":
    unittest.main()
