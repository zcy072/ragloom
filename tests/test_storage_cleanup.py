import json
import os
import sqlite3
import unittest
from unittest.mock import ANY, patch

try:
    from langchain_core.documents import Document

    import app_storage
    from routers import chat, knowledge_bases, schemas
    from services.core import config as application_config
    from services.core.errors import OperationFailedError, ServiceUnavailableError
    from services.documents import mineru as mineru_client
    from services.indexing import pipeline as knowledge_base
    from services.indexing.jobs import (
        INDEX_JOB_COMPLETED,
        INDEX_JOB_FAILED,
        INDEX_JOB_QUEUED,
        should_compact_database,
    )
    from services.knowledge_bases import use_cases as knowledge_base_use_cases
    from services.retrieval import parent_store
    from services.retrieval.chunk_index import get_chunk_index
except ModuleNotFoundError:
    Document = None
    app_storage = None
    knowledge_base = None

from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(
    Document and app_storage and knowledge_base,
    "requires the backend Python dependencies",
)
class StorageCleanupTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at) VALUES (?, lower(hex(randomblob(16))), ?)",
                ("研发部", application_config.db_now()),
            )

    def test_init_enables_sqlite_secure_delete(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            secure_delete = conn.execute("PRAGMA secure_delete").fetchone()[0]
        self.assertEqual(secure_delete, 1)

    def test_delete_job_redacts_filename_after_source_row_is_removed(self):
        job_id = app_storage.index_jobs.create("研发部", "delete", "制度.pdf")

        with patch.object(knowledge_base, "delete_document_indexes") as delete_indexes, patch.object(
            app_storage.index_jobs, "compact_database", return_value=True
        ) as compact_database:
            knowledge_base.run_delete_job(job_id, "研发部", "制度.pdf")

        delete_indexes.assert_called_once_with("研发部", "制度.pdf", container=ANY)
        compact_database.assert_called_once_with()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT filename, status FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(row, (None, INDEX_JOB_COMPLETED))

    def test_delete_job_failure_keeps_filename_and_is_requeued_on_restart(self):
        job_id = app_storage.index_jobs.create("研发部", "delete", "制度.pdf")

        with patch.object(
            knowledge_base, "delete_document_indexes", side_effect=RuntimeError("index unavailable")
        ), patch.object(app_storage.index_jobs, "compact_database", return_value=False):
            knowledge_base.run_delete_job(job_id, "研发部", "制度.pdf")

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            failed = conn.execute(
                "SELECT filename, status FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(failed, ("制度.pdf", INDEX_JOB_FAILED))

        app_storage.index_jobs.recover_interrupted()

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            recovered = conn.execute(
                "SELECT filename, status, started_at, finished_at FROM index_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        self.assertEqual(
            recovered,
            ("制度.pdf", INDEX_JOB_QUEUED, None, None),
        )
        self.assertIn(job_id, app_storage.index_jobs.queued_ids())

    def test_knowledge_base_delete_aborts_before_sqlite_when_index_delete_fails(self):
        with patch.object(
            get_chunk_index(),
            "delete_kb",
            side_effect=RuntimeError("index unavailable"),
        ):
            with self.assertRaises(ServiceUnavailableError) as raised:
                knowledge_base_use_cases.delete_knowledge_base("研发部")

        self.assertEqual(
            raised.exception.code, "knowledge_base_index_cleanup_failed"
        )
        self.assertTrue(app_storage.documents.knowledge_base_exists("研发部"))

    def test_knowledge_base_delete_removes_index_and_parent_rows(self):
        kb_id = app_storage.knowledge_bases.kb_id_for("研发部")
        other_id = "other-kb"
        index = get_chunk_index()
        child = Document(
            page_content="子块", metadata={"source": "制度.pdf", "parent_id": "p1"}
        )
        index.publish_source(kb_id, "制度.pdf", [child], [[1.0, 0.0]])
        index.publish_source(other_id, "制度.pdf", [child], [[1.0, 0.0]])
        for owner in (kb_id, other_id):
            with sqlite3.connect(app_storage.DB_FILE) as conn:
                conn.execute(
                    "INSERT INTO parent_chunks (kb_id, parent_id, source, page, content, "
                    "contextual_title) VALUES (?, 'p1', '制度.pdf', '1', '父块', '')",
                    (owner,),
                )

        with patch.object(
            mineru_client, "remove_saved_results_for_knowledge_base"
        ), patch.object(
            app_storage.index_jobs, "compact_database", return_value=False
        ), patch.object(app_storage.documents, "has_documents", return_value=True):
            knowledge_base_use_cases.delete_knowledge_base("研发部")

        self.assertEqual(index.available_routes(kb_id, None), set())
        self.assertTrue(index.available_routes(other_id, None))
        self.assertEqual(parent_store.load(kb_id, ["p1"]), {})
        self.assertIn("p1", parent_store.load(other_id, ["p1"]))

    def test_knowledge_base_delete_removes_sessions_and_dangling_citations(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        "deleted-session", 1, "alice", "研发部",
                        "研发部 会话", "2026-01-01", "2026-01-01",
                    ),
                    (
                        "kept-session", 1, "alice", "技术部",
                        "技术部 会话", "2026-01-01", "2026-01-01",
                    ),
                ],
            )
            conn.executemany(
                "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    ("deleted-session", "assistant", "旧回答", "[]", "2026-01-01"),
                    (
                        "kept-session",
                        "assistant",
                        "跨库回答",
                        '[{"source":"制度.pdf","kb_name":"研发部"},{"source":"通用.pdf","kb_name":"通用知识库"}]',
                        "2026-01-01",
                    ),
                ],
            )

        with patch.object(
            mineru_client, "remove_saved_results_for_knowledge_base"
        ), patch.object(app_storage.index_jobs, "compact_database", return_value=False) as compact:
            result = knowledge_bases.delete_knowledge_base("研发部")

        self.assertEqual(result, {"status": "success"})
        # VACUUM 会锁住整个数据库的写入，不在删除请求里执行。
        compact.assert_not_called()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertIsNone(
                conn.execute("SELECT 1 FROM chat_sessions WHERE id = 'deleted-session'").fetchone()
            )
            sources = conn.execute(
                "SELECT sources FROM chat_messages WHERE session_id = 'kept-session'"
            ).fetchone()[0]
            self.assertEqual(json.loads(sources), [{"source": "通用.pdf", "kb_name": "通用知识库"}])

    def test_knowledge_base_rename_updates_sessions_jobs_and_saved_citations(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "session-1", 1, "alice", "研发部",
                    "研发部 会话", "2026-01-01", "2026-01-01",
                ),
            )
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "session-1",
                    "assistant",
                    "历史回答",
                    '[{"source":"制度.pdf","kb_name":"研发部"}]',
                    "2026-01-01",
                ),
            )
        job_id = app_storage.index_jobs.create("研发部", "rebuild")

        kb_id = app_storage.knowledge_bases.kb_id_for("研发部")
        index = get_chunk_index()
        index.publish_source(
            kb_id, "制度.pdf", [Document(page_content="子块", metadata={})], [[1.0, 0.0]]
        )

        with patch.object(
            mineru_client, "rename_saved_results_for_knowledge_base"
        ):
            result = knowledge_bases.rename_knowledge_base(
                "研发部", schemas.KnowledgeBaseNameRequest(name="产品研发部")
            )

        self.assertEqual(result, {"status": "success"})
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            session = conn.execute(
                "SELECT kb_name, title FROM chat_sessions WHERE id = 'session-1'"
            ).fetchone()
            sources = conn.execute(
                "SELECT sources FROM chat_messages WHERE session_id = 'session-1'"
            ).fetchone()[0]
            job_dept = conn.execute("SELECT kb_name FROM index_jobs WHERE id = ?", (job_id,)).fetchone()[0]
        self.assertEqual(session, ("产品研发部", "产品研发部 会话"))
        self.assertEqual(json.loads(sources)[0]["kb_name"], "产品研发部")
        self.assertEqual(job_dept, "产品研发部")
        self.assertEqual(app_storage.knowledge_bases.kb_id_for("产品研发部"), kb_id)
        self.assertTrue(index.available_routes(kb_id, None))

    def test_knowledge_base_rename_rolls_back_moved_files_when_database_update_fails(self):
        old_data_dir = app_storage.documents.ensure_knowledge_base_dir("研发部")
        with open(os.path.join(old_data_dir, "制度.txt"), "w", encoding="utf-8") as output:
            output.write("原始文档")

        with patch.object(
            mineru_client, "rename_saved_results_for_knowledge_base"
        ), patch.object(
            app_storage.knowledge_bases,
            "rename_relational_data",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with self.assertRaises(OperationFailedError) as raised:
                knowledge_base_use_cases.rename_knowledge_base(
                    "研发部", "产品研发部"
                )

        self.assertEqual(raised.exception.code, "knowledge_base_rename_failed")

        self.assertTrue(os.path.isdir(old_data_dir))
        self.assertFalse(
            os.path.exists(app_storage.documents.knowledge_base_path("产品研发部"))
        )
        self.assertTrue(app_storage.documents.knowledge_base_exists("研发部"))
        self.assertFalse(app_storage.documents.knowledge_base_exists("产品研发部"))

    def test_empty_index_is_reset_when_no_documents_remain(self):
        index = get_chunk_index()
        index.publish_source(
            "kb-1", "制度.pdf", [Document(page_content="子块", metadata={})], [[1.0, 0.0]]
        )

        self.assertTrue(index.reset_if_empty(lambda: False))
        self.assertEqual(index.available_routes("kb-1", None), set())

    def test_chat_delete_removes_rows_without_immediate_compaction(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions "
                "(id, account_id, username, kb_name, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "session-1",
                    42,
                    "tester",
                    "研发部",
                    "测试会话",
                    application_config.db_now(),
                    application_config.db_now(),
                ),
            )
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "session-1",
                    "user",
                    "需要删除的内容",
                    "[]",
                    application_config.db_now(),
                ),
            )

        with patch.object(
            app_storage.index_jobs, "compact_database", return_value=True
        ) as compact_database:
            from services.identity.auth import Principal
            result = chat.delete_chat_session(
                "session-1",
                Principal(
                    username="tester",
                    role="超级管理员",
                    account_id=42,
                    menu_permissions=("问答对话",),
                    knowledge_base_permissions=("全公司",),
                ),
                chat.get_application_container(),
            )

        self.assertEqual(result, {"status": "success"})
        compact_database.assert_not_called()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0)

    def test_compaction_threshold_skips_small_fragmentation(self):
        self.assertFalse(
            should_compact_database(
                application_config.SQLITE_VACUUM_MIN_FREE_PAGES - 1,
                application_config.SQLITE_VACUUM_MIN_FREE_PAGES * 2,
                min_free_pages=application_config.SQLITE_VACUUM_MIN_FREE_PAGES,
                min_free_ratio=application_config.SQLITE_VACUUM_MIN_FREE_RATIO,
            )
        )
        self.assertFalse(
            should_compact_database(
                application_config.SQLITE_VACUUM_MIN_FREE_PAGES,
                application_config.SQLITE_VACUUM_MIN_FREE_PAGES * 10,
                min_free_pages=application_config.SQLITE_VACUUM_MIN_FREE_PAGES,
                min_free_ratio=application_config.SQLITE_VACUUM_MIN_FREE_RATIO,
            )
        )
        self.assertTrue(
            should_compact_database(
                application_config.SQLITE_VACUUM_MIN_FREE_PAGES,
                application_config.SQLITE_VACUUM_MIN_FREE_PAGES * 4,
                min_free_pages=application_config.SQLITE_VACUUM_MIN_FREE_PAGES,
                min_free_ratio=application_config.SQLITE_VACUUM_MIN_FREE_RATIO,
            )
        )


if __name__ == "__main__":
    unittest.main()
