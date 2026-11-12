import dataclasses
import os
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import app_storage
from services.core import config as application_config
from services.core.container import get_application_container
from services.core.settings import get_settings
from services.documents import mineru as mineru_client
from services.documents.types import DocumentLoadError, DocumentRoutes
from services.indexing import pipeline as knowledge_base
from services.indexing import workspace as index_workspace
from services.indexing.jobs import (
    INDEX_JOB_COMPLETED,
    INDEX_JOB_FAILED,
    INDEX_JOB_INTERRUPTED,
    INDEX_JOB_SUPERSEDED,
    PREVIOUS_VERSION_KEPT_PREFIX,
)
from services.indexing.locks import knowledge_base_lock
from services.retrieval import context as context_builder
from services.retrieval import parent_store
from services.retrieval.chunk_index import get_chunk_index
from tests.support import (
    IsolatedStorageTestCase,
    add_knowledge_bases,
    bind_index_job,
    create_index_job,
    fixture_document_id,
)

WAIT = 5


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class BrokenEmbeddings:
    def embed_documents(self, texts):
        raise RuntimeError("embedding unavailable")


class IndexConcurrencyTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.kb_id = add_knowledge_bases("研发部")[0]
        self.kb_dir = app_storage.documents.knowledge_base_path(self.kb_id)
        os.makedirs(self.kb_dir, exist_ok=True)
        self.on_load = lambda path, filename: None
        self.embeddings = FakeEmbeddings()

        def fake_load(path, filename, parser_mode, **kwargs):
            self.on_load(path, filename)
            return DocumentRoutes(
                fixed=[
                    Document(
                        page_content=f"{filename} 的审批规则内容",
                        metadata={
                            "document_id": fixture_document_id("standalone", filename),
                            "source": filename,
                            "page": 0,
                        },
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
        self.index = get_chunk_index()
        publish = patch.object(self.index, "publish_document", wraps=self.index.publish_document)
        self.publish = publish.start()
        self.addCleanup(publish.stop)
        self.errors: list[BaseException] = []

    # -- helpers ---------------------------------------------------------

    def _container(self):
        return dataclasses.replace(
            get_application_container(), create_embeddings=lambda: self.embeddings
        )

    def _add_document(self, filename, content=b"v1", status="processing"):
        path = app_storage.documents.stored_path(
            self.kb_id, fixture_document_id(self.kb_id, filename), filename
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO documents (document_id, kb_id, filename, size, uploaded_at, status, index_job_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fixture_document_id(self.kb_id, filename),
                    self.kb_id,
                    filename,
                    len(content),
                    application_config.db_now(),
                    status,
                    "",
                ),
            )
        if status != "processing":
            return "", path
        job_id = create_index_job(self.kb_id, "process", filename)
        bind_index_job(self.kb_id, fixture_document_id(self.kb_id, filename), job_id)
        return job_id, path

    def _reindex(self, filename):
        """相同文档重新索引切换任务归属，原文件保持不变。"""
        return app_storage.documents.schedule_reindex(
            self.kb_id, fixture_document_id(self.kb_id, filename)
        )

    def _run(self, job_id, filename, path):
        knowledge_base.run_document_job(
            job_id,
            self.kb_id,
            fixture_document_id(self.kb_id, filename),
            container=self._container(),
        )

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
                "WHERE kb_id = ? AND filename = ?",
                (self.kb_id, filename),
            ).fetchone()

    def _assert_workspace_removed(self, job_id):
        settings = get_settings()
        for root in (settings.data_dir, settings.mineru_results_dir):
            self.assertFalse(os.path.exists(os.path.join(root, ".index_jobs", job_id)))

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

    def test_parse_slots_cap_parallel_parsing_across_jobs(self):
        """上传与重建共用全局解析名额，各自的线程池不会让解析并行度叠加。"""
        active = 0
        peak = 0
        counter_lock = threading.Lock()

        def track(path, filename):
            nonlocal active, peak
            with counter_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with counter_lock:
                active -= 1

        self.on_load = track
        upload_id, upload_path = self._add_document("甲.pdf")
        self._add_document("乙.pdf", status="ready")
        self._add_document("丙.pdf", status="ready")
        rebuild_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        def rebuild():
            try:
                knowledge_base.run_rebuild_job(
                    rebuild_id, self.kb_id, container=self._container()
                )
            except BaseException as exc:  # surfaced by _join
                self.errors.append(exc)

        with patch.object(knowledge_base, "_parse_slots", threading.BoundedSemaphore(1)):
            threads = [
                self._start(upload_id, "甲.pdf", upload_path),
                threading.Thread(target=rebuild, daemon=True),
            ]
            threads[1].start()
            self._join(*threads)

        self.assertEqual(peak, 1)
        for filename in ("甲.pdf", "乙.pdf", "丙.pdf"):
            self.assertEqual(self._document(filename)[0], "ready")

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
            with knowledge_base_lock(self.kb_id):
                acquired.set()

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()
        try:
            self.assertTrue(acquired.wait(WAIT), "解析期间不应持有知识库锁")
        finally:
            release.set()
        self._join(thread, prober)

    def test_different_files_publish_concurrently(self):
        # 两个发布都要到达屏障才能继续；仍按知识库串行时第一个会超时出错。
        barrier = threading.Barrier(2, timeout=WAIT)
        original = self.index.publish_document

        def meet_then_publish(*args, **kwargs):
            barrier.wait()
            return original(*args, **kwargs)

        first_id, first_path = self._add_document("甲.pdf")
        second_id, second_path = self._add_document("乙.pdf")
        with patch.object(self.index, "publish_document", side_effect=meet_then_publish):
            self._join(
                self._start(first_id, "甲.pdf", first_path),
                self._start(second_id, "乙.pdf", second_path),
            )

        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.assertEqual(self._document("乙.pdf")[0], "ready")

    def test_upload_of_another_file_is_not_blocked_by_a_publish(self):
        entered = threading.Event()
        release = threading.Event()
        original = self.index.publish_document

        def block_then_publish(*args, **kwargs):
            entered.set()
            release.wait(WAIT)
            return original(*args, **kwargs)

        job_id, path = self._add_document("甲.pdf")
        with patch.object(self.index, "publish_document", side_effect=block_then_publish):
            thread = self._start(job_id, "甲.pdf", path)
            self.assertTrue(entered.wait(WAIT))
            try:
                acquired = threading.Event()

                def upload_other_file():
                    from services.indexing.locks import document_lock

                    with document_lock(self.kb_id, "乙.pdf"):
                        acquired.set()

                prober = threading.Thread(target=upload_other_file, daemon=True)
                prober.start()
                self.assertTrue(acquired.wait(WAIT), "发布甲时不应阻塞乙的上传")
            finally:
                release.set()
            self._join(thread, prober)
        self.assertEqual(self._document("甲.pdf")[0], "ready")

    def test_finished_job_does_not_take_the_knowledge_base_lock_when_it_still_exists(self):
        job_id, path = self._add_document("甲.pdf")
        with patch.object(knowledge_base, "knowledge_base_lock") as kb_lock:
            self._run(job_id, "甲.pdf", path)

        kb_lock.assert_not_called()
        self.assertEqual(self._document("甲.pdf")[0], "ready")

    def test_reindex_during_parse_supersedes_the_old_job(self):
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

        new_id = self._reindex("甲.pdf")
        release.set()
        self._join(thread)

        # 旧任务读到的是自己的快照，而不是刚被替换的新文件。
        self.assertEqual(seen, [b"v1"])
        self.assertEqual(self._job_status(old_id), INDEX_JOB_SUPERSEDED)
        status, _error, owner = self._document("甲.pdf")
        self.assertEqual((status, owner), ("processing", new_id))
        self.assertEqual(self.index.available_routes(self.kb_id, None), set())
        self.publish.assert_not_called()
        self._assert_workspace_removed(old_id)

        self._run(new_id, "甲.pdf", path)
        self.assertEqual(seen, [b"v1", b"v1"])
        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.assertEqual(self._job_status(new_id), INDEX_JOB_COMPLETED)

    def test_job_superseded_during_parse_skips_embedding(self):
        entered = threading.Event()
        release = threading.Event()
        embedded: list[int] = []

        class CountingEmbeddings(FakeEmbeddings):
            def embed_documents(self, texts):
                embedded.append(len(texts))
                return super().embed_documents(texts)

        def block(path, filename):
            entered.set()
            release.wait(WAIT)

        self.embeddings = CountingEmbeddings()
        self.on_load = block
        old_id, path = self._add_document("甲.pdf", b"v1")
        thread = self._start(old_id, "甲.pdf", path)
        self.assertTrue(entered.wait(WAIT))

        new_id = self._reindex("甲.pdf")
        self.on_load = lambda path, filename: None
        release.set()
        self._join(thread)

        self.assertEqual(embedded, [], "被替代的任务不应再调用嵌入服务")
        self.assertEqual(self._job_status(old_id), INDEX_JOB_SUPERSEDED)
        self.assertEqual(self._document("甲.pdf")[2], new_id)
        self._assert_workspace_removed(old_id)

        self._run(new_id, "甲.pdf", path)
        self.assertEqual(len(embedded), 1)
        self.assertEqual(self._document("甲.pdf")[0], "ready")

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

        with knowledge_base_lock(self.kb_id):
            app_storage.documents.schedule_document_delete(
                self.kb_id, fixture_document_id(self.kb_id, "甲.pdf")
            )
        release.set()
        self._join(thread)

        self.assertEqual(self._job_status(job_id), INDEX_JOB_SUPERSEDED)
        self.assertIsNone(self._document("甲.pdf"))
        self.assertEqual(self.index.available_routes(self.kb_id, None), set())
        formal = mineru_client.result_dir_for_document(path)
        self.assertFalse(formal.exists(), "被替代的任务不应发布结果目录")
        self._assert_workspace_removed(job_id)

    def test_delete_job_during_parse_keeps_the_snapshot_and_leaves_no_residue(self):
        entered = threading.Event()
        release = threading.Event()
        seen: list[bytes] = []

        def block_then_read_and_write(path, filename):
            entered.set()
            release.wait(WAIT)
            # 删除任务已整目录清理该文档，任务仍读取自己的快照并写入私有结果目录。
            with open(path, "rb") as handle:
                seen.append(handle.read())
            result_dir = mineru_client.result_dir_for_document(path)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "full.md").write_text("解析结果", encoding="utf-8")

        self.on_load = block_then_read_and_write
        job_id, path = self._add_document("甲.pdf", b"v1")
        document_id = fixture_document_id(self.kb_id, "甲.pdf")
        with self.assertNoLogs(knowledge_base.logger, level="ERROR"):
            thread = self._start(job_id, "甲.pdf", path)
            self.assertTrue(entered.wait(WAIT))
            with knowledge_base_lock(self.kb_id):
                delete_id = app_storage.documents.schedule_document_delete(
                    self.kb_id, document_id
                )
            knowledge_base.run_delete_job(
                delete_id, self.kb_id, document_id, container=self._container()
            )
            self.assertEqual(self._job_status(delete_id), INDEX_JOB_COMPLETED)
            release.set()
            self._join(thread)

        self.assertEqual(seen, [b"v1"])
        self.assertEqual(self._job_status(job_id), INDEX_JOB_SUPERSEDED)
        self.publish.assert_not_called()
        settings = get_settings()
        self.assertFalse(
            os.path.exists(app_storage.documents.document_directory(self.kb_id, document_id))
        )
        self.assertFalse(
            os.path.exists(os.path.join(settings.mineru_results_dir, self.kb_id, document_id))
        )
        self._assert_workspace_removed(job_id)

    def test_delete_job_after_the_job_finished_removes_all_document_files(self):
        def write_result(path, filename):
            result_dir = mineru_client.result_dir_for_document(path)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "full.md").write_text("解析结果", encoding="utf-8")

        self.on_load = write_result
        job_id, path = self._add_document("甲.pdf")
        document_id = fixture_document_id(self.kb_id, "甲.pdf")
        self._run(job_id, "甲.pdf", path)
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)
        self.assertTrue(mineru_client.result_dir_for_document(path).is_dir())

        with knowledge_base_lock(self.kb_id):
            delete_id = app_storage.documents.schedule_document_delete(self.kb_id, document_id)
        knowledge_base.run_delete_job(
            delete_id, self.kb_id, document_id, container=self._container()
        )

        settings = get_settings()
        self.assertEqual(self._job_status(delete_id), INDEX_JOB_COMPLETED)
        self.assertFalse(
            os.path.exists(app_storage.documents.document_directory(self.kb_id, document_id))
        )
        self.assertFalse(
            os.path.exists(os.path.join(settings.mineru_results_dir, self.kb_id, document_id))
        )
        self.assertEqual(self.index.available_routes(self.kb_id, None), set())
        self._assert_workspace_removed(job_id)

    def test_finished_job_does_not_take_the_document_lock_again_to_check_for_deletion(self):
        job_id, path = self._add_document("甲.pdf")
        with patch.object(
            knowledge_base, "document_lock", wraps=knowledge_base.document_lock
        ) as lock:
            self._run(job_id, "甲.pdf", path)

        # 只有准备与发布两次持锁。
        self.assertEqual(lock.call_count, 2)
        self.assertEqual(self._document("甲.pdf")[0], "ready")

    def test_rename_during_a_running_job_only_changes_the_display_name(self):
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
            kb_use_cases.rename_knowledge_base(self.kb_id, "新名称")
            self.assertTrue(os.path.isdir(self.kb_dir))
        finally:
            release.set()
        self._join(thread)

        self.assertTrue(os.path.isdir(self.kb_dir))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)

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

        kb_use_cases.delete_knowledge_base(self.kb_id)
        release.set()
        self._join(thread)

        settings = get_settings()
        self.assertFalse(os.path.exists(self.kb_dir))
        self.assertFalse(os.path.exists(os.path.join(settings.mineru_results_dir, self.kb_id)))
        # 任务行随知识库级联删除。
        self.assertIsNone(app_storage.index_jobs.get(job_id))

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

    def test_embedding_failure_marks_the_document_error(self):
        self.embeddings = BrokenEmbeddings()
        job_id, path = self._add_document("甲.pdf")

        self._run(job_id, "甲.pdf", path)

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual(status, "error")
        self.assertIn("embedding unavailable", error)
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)
        self.publish.assert_not_called()
        self.assertEqual(self.index.available_routes(self.kb_id, None), set())

    def test_successful_job_publishes_chunks_and_parent_rows(self):
        job_id, path = self._add_document("甲.pdf")

        self._run(job_id, "甲.pdf", path)

        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.publish.assert_called_once()
        self.assertTrue(
            self.index.available_routes(self.kb_id, [fixture_document_id(self.kb_id, "甲.pdf")])
        )

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
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        knowledge_base.run_rebuild_job(job_id, self.kb_id, container=self._container())

        self.assertEqual(self._document("好.pdf")[:2], ("ready", ""))
        status, error, _owner = self._document("坏.pdf")
        self.assertEqual(status, "ready")
        self.assertTrue(error.startswith(PREVIOUS_VERSION_KEPT_PREFIX))
        self.assertIn("492", error)
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)
        self.assertEqual(
            {call.args[1] for call in self.publish.call_args_list},
            {fixture_document_id(self.kb_id, "好.pdf")},
        )
        self.assertTrue(
            self.index.available_routes(self.kb_id, [fixture_document_id(self.kb_id, "好.pdf")])
        )

    def test_rebuild_with_only_failing_documents_fails_the_job(self):
        def fail(path, filename):
            raise DocumentLoadError("无法解析")

        self.on_load = fail
        self._add_document("坏.pdf", status="ready")
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        knowledge_base.run_rebuild_job(job_id, self.kb_id, container=self._container())

        status, error, _owner = self._document("坏.pdf")
        self.assertEqual((status, error), ("ready", f"{PREVIOUS_VERSION_KEPT_PREFIX}无法解析"))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)
        self.publish.assert_not_called()

    def test_rebuild_marks_each_document_ready_as_soon_as_it_finishes(self):
        release = threading.Event()

        def block_slow(path, filename):
            if filename == "慢.pdf":
                release.wait(WAIT)

        self.on_load = block_slow
        self._add_document("快.pdf", status="ready")
        self._add_document("慢.pdf", status="ready")
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        def target():
            try:
                knowledge_base.run_rebuild_job(job_id, self.kb_id, container=self._container())
            except BaseException as exc:  # surfaced by _join
                self.errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + WAIT
            while self._document("快.pdf")[0] != "ready" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(self._document("快.pdf")[0], "ready")
            self.assertEqual(self._document("慢.pdf")[0], "rebuilding")
            self.assertEqual(self._job_status(job_id), "running")
        finally:
            release.set()
        self._join(thread)

        self.assertEqual(self._document("慢.pdf")[0], "ready")
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)

    def _publish_old_version(self, filename):
        split = Document(
            page_content=f"{filename} 旧版本内容",
            metadata={
                "document_id": fixture_document_id("standalone", filename),
                "source": filename,
                "page": 0,
                "parent_id": f"old-{filename}",
                "parent_content": f"{filename} 旧父段落",
            },
        )
        parent_store.add_document_version(
            self.kb_id,
            fixture_document_id(self.kb_id, filename),
            parent_store.parent_chunk_rows([split]),
        )
        self.index.publish_document(
            self.kb_id, fixture_document_id(self.kb_id, filename), [split], [[0.1] * 8]
        )

    def _indexed_texts(self, filename):
        return [
            document.page_content
            for document in self.index.neighbors(
                self.kb_id, fixture_document_id(self.kb_id, filename), "fixed", 0, 100
            )
        ]

    def test_rebuild_keeps_the_old_version_searchable_until_the_new_one_is_published(self):
        release = threading.Event()

        def block_slow(path, filename):
            if filename == "慢.pdf":
                release.wait(WAIT)

        self.on_load = block_slow
        self._add_document("慢.pdf", status="ready")
        self._publish_old_version("慢.pdf")
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        def target():
            try:
                knowledge_base.run_rebuild_job(job_id, self.kb_id, container=self._container())
            except BaseException as exc:  # surfaced by _join
                self.errors.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + WAIT
            while self._job_status(job_id) != "running" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(self._document("慢.pdf")[0], "rebuilding")
            self.assertEqual(
                app_storage.documents.ready_document_ids(self.kb_id),
                [fixture_document_id(self.kb_id, "慢.pdf")],
            )
            self.assertEqual(self._indexed_texts("慢.pdf"), ["慢.pdf 旧版本内容"])
        finally:
            release.set()
        self._join(thread)

        self.assertEqual(self._document("慢.pdf")[0], "ready")
        self.assertEqual(self._indexed_texts("慢.pdf"), ["慢.pdf 的审批规则内容"])
        self.assertEqual(
            parent_store.load(
                self.kb_id, fixture_document_id(self.kb_id, "慢.pdf"), ["old-慢.pdf"]
            ),
            {},
        )

    def test_rebuild_keeps_the_old_version_of_a_document_that_fails_before_publish(self):
        def fail_for_bad(path, filename):
            if filename == "坏.pdf":
                raise DocumentLoadError("无法解析")

        self.on_load = fail_for_bad
        self._add_document("好.pdf", status="ready")
        self._add_document("坏.pdf", status="ready")
        self._publish_old_version("坏.pdf")
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        knowledge_base.run_rebuild_job(job_id, self.kb_id, container=self._container())

        self.assertEqual(self._document("坏.pdf")[0], "ready")
        self.assertEqual(self._indexed_texts("坏.pdf"), ["坏.pdf 旧版本内容"])
        self.assertNotEqual(
            parent_store.load(
                self.kb_id, fixture_document_id(self.kb_id, "坏.pdf"), ["old-坏.pdf"]
            ),
            {},
        )
        self.assertEqual(
            app_storage.documents.ready_document_ids(self.kb_id),
            sorted(fixture_document_id(self.kb_id, name) for name in ("好.pdf", "坏.pdf")),
        )

    def test_rebuild_removes_the_old_version_of_a_document_that_was_not_ready(self):
        def fail(path, filename):
            raise DocumentLoadError("无法解析")

        self.on_load = fail
        self._add_document("坏.pdf", status="error")
        self._publish_old_version("坏.pdf")
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        knowledge_base.run_rebuild_job(job_id, self.kb_id, container=self._container())

        self.assertEqual(self._document("坏.pdf")[:2], ("error", "无法解析"))
        self.assertEqual(self._indexed_texts("坏.pdf"), [])
        self.assertEqual(app_storage.documents.ready_document_ids(self.kb_id), [])

    # -- 单文档重新索引期间与失败后旧版本的可用性 --------------------------------

    def _ready_with_old_version(self, filename):
        self._add_document(filename, status="ready")
        self._publish_old_version(filename)

    def test_reindex_keeps_a_ready_document_searchable_while_it_runs(self):
        entered = threading.Event()
        release = threading.Event()

        def block(path, filename):
            entered.set()
            release.wait(WAIT)

        self.on_load = block
        self._ready_with_old_version("甲.pdf")
        document_id = fixture_document_id(self.kb_id, "甲.pdf")
        job_id = self._reindex("甲.pdf")
        self.assertEqual(self._document("甲.pdf")[0], "rebuilding")
        thread = self._start(job_id, "甲.pdf", "")
        try:
            self.assertTrue(entered.wait(WAIT))
            self.assertEqual(self._document("甲.pdf")[0], "rebuilding")
            self.assertEqual(app_storage.documents.ready_document_ids(self.kb_id), [document_id])
            self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 旧版本内容"])
        finally:
            release.set()
        self._join(thread)

        self.assertEqual(self._document("甲.pdf")[:2], ("ready", ""))
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 的审批规则内容"])
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)

    def test_reindex_of_a_rebuilding_document_keeps_the_old_version_searchable(self):
        self._ready_with_old_version("甲.pdf")
        self._reindex("甲.pdf")

        newer = self._reindex("甲.pdf")

        self.assertEqual(self._document("甲.pdf")[::2], ("rebuilding", newer))
        self.assertEqual(
            app_storage.documents.ready_document_ids(self.kb_id),
            [fixture_document_id(self.kb_id, "甲.pdf")],
        )

    def test_reindex_parse_failure_returns_to_ready_with_the_old_version(self):
        def fail(path, filename):
            raise DocumentLoadError("无法解析")

        self.on_load = fail
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")

        self._run(job_id, "甲.pdf", "")

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual((status, error), ("ready", f"{PREVIOUS_VERSION_KEPT_PREFIX}无法解析"))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)
        self.assertEqual(
            app_storage.documents.ready_document_ids(self.kb_id),
            [fixture_document_id(self.kb_id, "甲.pdf")],
        )
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 旧版本内容"])
        self.assertNotEqual(
            parent_store.load(
                self.kb_id, fixture_document_id(self.kb_id, "甲.pdf"), ["old-甲.pdf"]
            ),
            {},
        )
        self._assert_workspace_removed(job_id)

    def test_reindex_embedding_failure_returns_to_ready_with_the_old_version(self):
        self.embeddings = BrokenEmbeddings()
        self._ready_with_old_version("甲.pdf")
        self.publish.reset_mock()
        job_id = self._reindex("甲.pdf")

        self._run(job_id, "甲.pdf", "")

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual(status, "ready")
        self.assertTrue(error.startswith(PREVIOUS_VERSION_KEPT_PREFIX))
        self.assertIn("embedding unavailable", error)
        self.publish.assert_not_called()
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 旧版本内容"])

    def test_reindex_missing_source_returns_to_ready_with_the_old_version(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        document_id = fixture_document_id(self.kb_id, "甲.pdf")
        os.remove(app_storage.documents.document_path(self.kb_id, document_id))

        self._run(job_id, "甲.pdf", "")

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual((status, error), ("ready", f"{PREVIOUS_VERSION_KEPT_PREFIX}源文件不存在"))
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 旧版本内容"])

    def test_unrecoverable_publish_failure_marks_error_and_keeps_both_parent_versions(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        self.publish.side_effect = RuntimeError("milvus unavailable")

        self._run(job_id, "甲.pdf", "")

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual((status, error), ("error", "milvus unavailable"))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)
        self.assertEqual(app_storage.documents.ready_document_ids(self.kb_id), [])
        # 无法确认 Milvus 中残留哪一代，两版父段落都保留。
        parents = self._parent_ids("甲.pdf")
        self.assertIn("old-甲.pdf", parents)
        self.assertGreater(len(parents), 1)

    def _parent_ids(self, filename):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            rows = conn.execute(
                "SELECT parent_id FROM parent_chunks WHERE kb_id = ? AND document_id = ?",
                (self.kb_id, fixture_document_id(self.kb_id, filename)),
            ).fetchall()
        return {row[0] for row in rows}

    def test_old_children_keep_their_parents_while_milvus_switches_generations(self):
        self._ready_with_old_version("甲.pdf")
        document_id = fixture_document_id(self.kb_id, "甲.pdf")
        job_id = self._reindex("甲.pdf")
        real_publish = self.publish._mock_wraps
        seen: dict[str, set[str]] = {}

        def publish_and_observe(kb_id, doc_id, splits, vectors):
            # Milvus 切换前后，旧子块都必须能回表找到旧父段落。
            seen["before"] = self._parent_ids("甲.pdf")
            real_publish(kb_id, doc_id, splits, vectors)
            seen["after"] = self._parent_ids("甲.pdf")

        self.publish.side_effect = publish_and_observe

        self._run(job_id, "甲.pdf", "")

        self.assertEqual(self._document("甲.pdf")[:2], ("ready", ""))
        new_parents = self._parent_ids("甲.pdf")
        self.assertNotIn("old-甲.pdf", new_parents)
        self.assertTrue(new_parents)
        self.assertEqual(seen["before"], {"old-甲.pdf"} | new_parents)
        self.assertEqual(seen["after"], {"old-甲.pdf"} | new_parents)
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 的审批规则内容"])
        restored = context_builder.restore_parent_documents(
            self.kb_id,
            self.index.neighbors(self.kb_id, document_id, "fixed", 0, 100),
        )
        self.assertEqual([doc.page_content for doc in restored], ["甲.pdf 的审批规则内容"])

    def test_reindex_rolled_back_publish_returns_to_ready_with_the_old_version(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        self.index.fail_next_publish = RuntimeError("milvus insert failed")

        self._run(job_id, "甲.pdf", "")

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual(
            (status, error), ("ready", f"{PREVIOUS_VERSION_KEPT_PREFIX}milvus insert failed")
        )
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 旧版本内容"])
        # 新父段落保留：insert 超时后可能有残留子块，仍要能回表取到完整段落。
        parents = self._parent_ids("甲.pdf")
        self.assertIn("old-甲.pdf", parents)
        self.assertEqual(len(parents), 2)
        self.assertEqual(
            app_storage.documents.ready_document_ids(self.kb_id),
            [fixture_document_id(self.kb_id, "甲.pdf")],
        )
        self._assert_workspace_removed(job_id)

        # 下一次成功发布把上次保留的新父段落和旧父段落一并清掉。
        next_job = self._reindex("甲.pdf")
        self._run(next_job, "甲.pdf", "")
        remaining = self._parent_ids("甲.pdf")
        self.assertEqual(len(remaining), 1)
        self.assertFalse(remaining & parents)

    def test_rolled_back_publish_keeps_the_old_result_directory(self):
        _job_id, path = self._add_document("甲.pdf", status="ready")
        self._publish_old_version("甲.pdf")
        formal = mineru_client.result_dir_for_document(path)
        formal.mkdir(parents=True, exist_ok=True)
        (formal / "full.md").write_text("旧结果", encoding="utf-8")

        def write_result(snapshot_path, filename):
            result_dir = mineru_client.result_dir_for_document(snapshot_path)
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "full.md").write_text("新结果", encoding="utf-8")

        self.on_load = write_result
        job_id = self._reindex("甲.pdf")
        self.index.fail_next_publish = RuntimeError("milvus insert failed")
        self._run(job_id, "甲.pdf", "")

        self.assertEqual(self._document("甲.pdf")[0], "ready")
        # 回滚后保留旧版本，下次整库重建复用的解析结果也必须是旧版本的。
        self.assertEqual((formal / "full.md").read_text(encoding="utf-8"), "旧结果")
        self._assert_workspace_removed(job_id)

        job_id = self._reindex("甲.pdf")
        self._run(job_id, "甲.pdf", "")
        self.assertEqual((formal / "full.md").read_text(encoding="utf-8"), "新结果")

    def test_parser_mode_is_written_to_the_document_only_after_a_successful_publish(self):
        self._ready_with_old_version("甲.pdf")
        document_id = fixture_document_id(self.kb_id, "甲.pdf")
        modes: list[str] = []
        load = knowledge_base.document_loader.load_document_routes

        def record_mode(path, filename, parser_mode, **kwargs):
            modes.append(parser_mode)
            return load(path, filename, parser_mode, **kwargs)

        def parser_mode():
            return app_storage.documents.get_document(self.kb_id, document_id)["parser_mode"]

        with patch.object(knowledge_base.document_loader, "load_document_routes", record_mode):
            job_id = app_storage.documents.schedule_reindex(
                self.kb_id, document_id, parser_mode="paddle_ocr"
            )
            self.index.fail_next_publish = RuntimeError("milvus insert failed")
            self._run(job_id, "甲.pdf", "")
            # 失败保留旧版本，文档仍记旧版本实际使用的解析方式。
            self.assertEqual(self._document("甲.pdf")[0], "ready")
            self.assertEqual(parser_mode(), "native_text")

            job_id = app_storage.documents.schedule_reindex(
                self.kb_id, document_id, parser_mode="paddle_ocr"
            )
            self._run(job_id, "甲.pdf", "")
        self.assertEqual(modes, ["paddle_ocr", "paddle_ocr"])
        self.assertEqual(self._document("甲.pdf")[:2], ("ready", ""))
        self.assertEqual(parser_mode(), "paddle_ocr")

    def test_rolled_back_first_publish_marks_error_and_keeps_the_new_parent_rows(self):
        job_id, path = self._add_document("甲.pdf")
        self.index.fail_next_publish = RuntimeError("milvus insert failed")

        self._run(job_id, "甲.pdf", path)

        self.assertEqual(self._document("甲.pdf")[:2], ("error", "milvus insert failed"))
        self.assertEqual(len(self._parent_ids("甲.pdf")), 1)
        self.assertEqual(app_storage.documents.ready_document_ids(self.kb_id), [])

    def test_parent_write_failure_keeps_the_old_version_without_touching_milvus(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        self.publish.reset_mock()

        with patch.object(
            parent_store, "add_document_version", side_effect=sqlite3.OperationalError("locked")
        ):
            self._run(job_id, "甲.pdf", "")

        status, error, _owner = self._document("甲.pdf")
        self.assertEqual((status, error), ("ready", f"{PREVIOUS_VERSION_KEPT_PREFIX}locked"))
        self.publish.assert_not_called()
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 旧版本内容"])
        self.assertEqual(self._parent_ids("甲.pdf"), {"old-甲.pdf"})

    def test_old_parents_outlive_the_switch_for_queries_still_in_flight(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        timers: list = []

        class RecordingTimer:
            def __init__(self, interval, function, args):
                self.interval, self.function, self.args = interval, function, args
                self.daemon = False
                timers.append(self)

            def start(self):
                pass

        with patch.object(knowledge_base, "OLD_PARENT_GRACE_SECONDS", 60.0), patch.object(
            knowledge_base.threading, "Timer", RecordingTimer
        ):
            self._run(job_id, "甲.pdf", "")

        self.assertEqual(self._document("甲.pdf")[:2], ("ready", ""))
        # 发布完成后旧父段落仍在：检索后正在重排的查询回表时还要用到。
        self.assertIn("old-甲.pdf", self._parent_ids("甲.pdf"))
        self.assertEqual([timer.interval for timer in timers], [60.0])
        self.assertTrue(timers[0].daemon)
        timers[0].function(*timers[0].args)
        self.assertNotIn("old-甲.pdf", self._parent_ids("甲.pdf"))
        self.assertEqual(len(self._parent_ids("甲.pdf")), 1)

    def test_old_parent_cleanup_failure_does_not_fail_a_published_document(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        real_delete = parent_store.delete_parents

        def fail_old_cleanup(kb_id, document_id, parent_ids):
            if "old-甲.pdf" in set(parent_ids):
                raise sqlite3.OperationalError("locked")
            real_delete(kb_id, document_id, parent_ids)

        with patch.object(parent_store, "delete_parents", side_effect=fail_old_cleanup):
            self._run(job_id, "甲.pdf", "")

        self.assertEqual(self._document("甲.pdf")[:2], ("ready", ""))
        self.assertEqual(self._job_status(job_id), INDEX_JOB_COMPLETED)
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 的审批规则内容"])
        self.assertIn("old-甲.pdf", self._parent_ids("甲.pdf"))

        # 下一次发布把上次遗留的旧父段落一并清掉。
        next_job = self._reindex("甲.pdf")
        self._run(next_job, "甲.pdf", "")
        self.assertNotIn("old-甲.pdf", self._parent_ids("甲.pdf"))
        self.assertEqual(len(self._parent_ids("甲.pdf")), 1)

    def test_restart_before_publish_keeps_the_old_version(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        self.assertTrue(app_storage.index_jobs.start(job_id))

        app_storage.index_jobs.recover_interrupted()

        status, message, _ = self._document("甲.pdf")
        self.assertEqual(status, "ready")
        self.assertTrue(message.startswith(PREVIOUS_VERSION_KEPT_PREFIX))
        self.assertEqual(
            app_storage.documents.ready_document_ids(self.kb_id),
            [fixture_document_id(self.kb_id, "甲.pdf")],
        )
        self.assertEqual(self._job_status(job_id), INDEX_JOB_INTERRUPTED)

    def test_restart_during_publish_marks_the_document_error(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        self.assertTrue(app_storage.index_jobs.start(job_id))
        document_id = fixture_document_id(self.kb_id, "甲.pdf")
        app_storage.index_jobs.begin_publish(self.kb_id, document_id, job_id)
        self.assertEqual(self._document("甲.pdf")[0], "publishing")
        self.assertEqual(app_storage.documents.ready_document_ids(self.kb_id), [document_id])

        app_storage.index_jobs.recover_interrupted()

        self.assertEqual(self._document("甲.pdf")[0], "error")
        self.assertEqual(app_storage.documents.ready_document_ids(self.kb_id), [])

    def test_restart_of_a_rebuild_keeps_unpublished_documents_searchable(self):
        for filename in ("甲.pdf", "乙.pdf"):
            self._ready_with_old_version(filename)
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)
        self.assertTrue(app_storage.index_jobs.start(job_id))

        app_storage.index_jobs.recover_interrupted()

        self.assertEqual(self._document("甲.pdf")[0], "ready")
        self.assertEqual(self._document("乙.pdf")[0], "ready")
        self.assertEqual(len(app_storage.documents.ready_document_ids(self.kb_id)), 2)

    def test_unexpected_error_after_publish_records_the_document_error(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")

        with patch.object(
            knowledge_base.retriever,
            "drop_cached_retriever",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            self._run(job_id, "甲.pdf", "")

        status, message, _ = self._document("甲.pdf")
        self.assertEqual(status, "error")
        self.assertIn("database is locked", message)
        self.assertEqual(self._job_status(job_id), INDEX_JOB_FAILED)

    def test_unexpected_error_before_publish_keeps_the_old_version(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")

        with patch.object(
            app_storage.knowledge_bases,
            "exists",
            # 任务开始时的检查通过，发布前的检查遇到 SQLite 被锁。
            side_effect=[True, sqlite3.OperationalError("database is locked")],
        ):
            self._run(job_id, "甲.pdf", "")

        status, message, _ = self._document("甲.pdf")
        self.assertEqual(status, "ready")
        self.assertTrue(message.startswith(PREVIOUS_VERSION_KEPT_PREFIX))
        self.assertEqual(self._indexed_texts("甲.pdf"), ["甲.pdf 旧版本内容"])

    def test_restart_recovers_documents_left_in_progress_by_a_failed_job(self):
        self._ready_with_old_version("甲.pdf")
        job_id = self._reindex("甲.pdf")
        app_storage.index_jobs.finish(job_id, INDEX_JOB_FAILED, "worker crashed")
        self.assertEqual(self._document("甲.pdf")[0], "rebuilding")

        app_storage.index_jobs.recover_interrupted()

        self.assertEqual(self._document("甲.pdf")[0], "ready")

    def test_rebuild_drops_vectors_of_documents_that_are_no_longer_ready(self):
        self._add_document("好.pdf", status="ready")
        job_id = app_storage.index_jobs.prepare_rebuild(self.kb_id)

        with patch.object(self.index, "delete_stale_documents") as delete_stale:
            knowledge_base.run_rebuild_job(job_id, self.kb_id, container=self._container())

        delete_stale.assert_called_once_with(
            self.kb_id, {fixture_document_id(self.kb_id, "好.pdf")}
        )

    def test_startup_sweep_removes_only_job_workspaces(self):
        settings = get_settings()
        document_id = fixture_document_id(self.kb_id, "制度.pdf")
        document_dir = app_storage.documents.document_directory(self.kb_id, document_id)
        stale_data = os.path.join(settings.data_dir, ".index_jobs", "dead-job")
        stale_results = os.path.join(settings.mineru_results_dir, ".index_jobs", "dead-job")
        keep_result = os.path.join(
            settings.mineru_results_dir, self.kb_id, document_id, "制度.pdf.mineru"
        )
        for directory in (stale_data, stale_results, keep_result, document_dir):
            os.makedirs(directory)
        document = os.path.join(document_dir, "制度.pdf")
        with open(document, "wb") as handle:
            handle.write(b"keep")

        index_workspace.sweep_stale_workspaces()

        self.assertFalse(os.path.exists(os.path.dirname(stale_data)))
        self.assertFalse(os.path.exists(os.path.dirname(stale_results)))
        self.assertTrue(os.path.isdir(keep_result))
        self.assertTrue(os.path.isfile(document))
        self.assertTrue(os.path.isfile(app_storage.DB_FILE))

    def test_startup_sweep_does_not_follow_a_symlinked_workspace_root(self):
        settings = get_settings()
        outside = os.path.join(settings.data_dir, "outside")
        os.makedirs(outside)
        keep = os.path.join(outside, "keep.txt")
        with open(keep, "w", encoding="utf-8") as handle:
            handle.write("keep")
        os.symlink(outside, os.path.join(settings.data_dir, ".index_jobs"))

        index_workspace.sweep_stale_workspaces()

        self.assertTrue(os.path.isfile(keep))

if __name__ == "__main__":
    unittest.main()
