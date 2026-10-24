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
    from services.retrieval import chroma as chroma_index
    from services.retrieval import fts as fts_index
    from services.retrieval import index_coordination as retrieval
    from services.retrieval import retriever
except ModuleNotFoundError:
    Document = None
    app_storage = None
    knowledge_base = None
    retrieval = None

from tests.support import IsolatedStorageTestCase


class FakeChromaClient:
    def __init__(self):
        self.closed = False
        self.cache_cleared = False

    def close(self):
        self.closed = True

    def clear_system_cache(self):
        self.cache_cleared = True


@unittest.skipUnless(
    Document and app_storage and knowledge_base and retrieval,
    "requires the backend Python dependencies",
)
class StorageCleanupTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, created_at) VALUES (?, ?)",
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
            knowledge_base, "delete_document_indexes", side_effect=RuntimeError("chroma unavailable")
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

    def test_knowledge_base_delete_aborts_before_sqlite_when_chroma_fails(self):
        with patch.object(
            chroma_index,
            "chroma_delete_by_knowledge_base",
            side_effect=RuntimeError("chroma unavailable"),
        ):
            with self.assertRaises(ServiceUnavailableError) as raised:
                knowledge_base_use_cases.delete_knowledge_base("研发部")

        self.assertEqual(
            raised.exception.code, "knowledge_base_index_cleanup_failed"
        )
        self.assertTrue(app_storage.documents.knowledge_base_exists("研发部"))

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
            conn.execute(
                "INSERT INTO chroma_disabled_sources VALUES (?, ?, ?, ?)",
                ("研发部", "制度.pdf", "failed", application_config.db_now()),
            )

        with patch.object(chroma_index, "chroma_delete_by_knowledge_base"), patch.object(
            fts_index, "remove_fts_index"
        ), patch.object(chroma_index, "reset_empty_chroma_storage"), patch.object(
            mineru_client, "remove_saved_results_for_knowledge_base"
        ), patch.object(app_storage.index_jobs, "compact_database", return_value=False):
            result = knowledge_bases.delete_knowledge_base("研发部")

        self.assertEqual(result, {"status": "success"})
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertIsNone(
                conn.execute("SELECT 1 FROM chat_sessions WHERE id = 'deleted-session'").fetchone()
            )
            sources = conn.execute(
                "SELECT sources FROM chat_messages WHERE session_id = 'kept-session'"
            ).fetchone()[0]
            self.assertEqual(json.loads(sources), [{"source": "通用.pdf", "kb_name": "通用知识库"}])
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM chroma_disabled_sources WHERE kb_name = '研发部'"
                ).fetchone()[0],
                0,
            )

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
            conn.execute(
                "INSERT INTO chroma_disabled_sources VALUES (?, ?, ?, ?)",
                ("研发部", "制度.pdf", "failed", application_config.db_now()),
            )
        job_id = app_storage.index_jobs.create("研发部", "rebuild")

        with patch.object(retrieval, "rename_knowledge_base_indexes"), patch.object(
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
        self.assertEqual(
            app_storage.documents.chroma_disabled_sources("产品研发部"), ["制度.pdf"]
        )

    def test_knowledge_base_rename_rolls_back_moved_files_when_database_update_fails(self):
        old_data_dir = app_storage.documents.ensure_knowledge_base_dir("研发部")
        old_index_dir = fts_index.vector_store_dir("研发部")
        os.makedirs(old_index_dir, exist_ok=True)
        with open(os.path.join(old_data_dir, "制度.txt"), "w", encoding="utf-8") as output:
            output.write("原始文档")
        with open(os.path.join(old_index_dir, "marker"), "w", encoding="utf-8") as output:
            output.write("原始索引")

        with patch.object(
            chroma_index, "rename_chroma_knowledge_base"
        ), patch.object(
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
        self.assertTrue(os.path.isdir(old_index_dir))
        self.assertFalse(os.path.exists(fts_index.vector_store_dir("产品研发部")))
        self.assertTrue(app_storage.documents.knowledge_base_exists("研发部"))
        self.assertFalse(app_storage.documents.knowledge_base_exists("产品研发部"))

    def test_index_rename_restores_fts_when_chroma_rename_fails(self):
        old_index_dir = fts_index.vector_store_dir("研发部")
        new_index_dir = fts_index.vector_store_dir("产品研发部")
        os.makedirs(old_index_dir, exist_ok=True)
        with open(os.path.join(old_index_dir, "marker"), "w", encoding="utf-8") as output:
            output.write("原始索引")

        with patch.object(
            chroma_index,
            "rename_chroma_knowledge_base",
            side_effect=RuntimeError("chroma unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "chroma unavailable"):
                retrieval.rename_knowledge_base_indexes("研发部", "产品研发部")

        self.assertTrue(os.path.isdir(old_index_dir))
        self.assertFalse(os.path.exists(new_index_dir))

    def test_empty_fts_directory_is_removed_after_document_delete(self):
        fts_path = os.path.join(fts_index.vector_store_dir("研发部"), "fts5.db")
        fts_index.build_fts5_index(
            [Document(page_content="仅用于删除测试", metadata={"source": "制度.pdf"})],
            fts_path,
        )

        fts_index.delete_from_fts5("研发部", "制度.pdf")
        self.assertTrue(
            fts_index.remove_empty_fts_index(
                "研发部", app_storage.documents.has_documents
            )
        )
        self.assertFalse(os.path.exists(fts_index.vector_store_dir("研发部")))

    def test_fts_delete_failure_is_propagated_to_the_cleanup_job(self):
        fts_path = os.path.join(fts_index.vector_store_dir("研发部"), "fts5.db")
        fts_index.build_fts5_index(
            [Document(page_content="仅用于删除测试", metadata={"source": "制度.pdf"})],
            fts_path,
        )

        with patch.object(
            fts_index.sqlite3,
            "connect",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "database is locked"):
                fts_index.delete_from_fts5("研发部", "制度.pdf")

    def test_empty_chroma_storage_is_physically_removed(self):
        os.makedirs(chroma_index.CHROMA_PATH, exist_ok=True)
        stale_file = os.path.join(chroma_index.CHROMA_PATH, "stale.bin")
        with open(stale_file, "wb") as output:
            output.write(b"stale index")
        client = FakeChromaClient()
        chroma_index._chroma_client = client

        self.assertTrue(
            chroma_index.reset_empty_chroma_storage(
                lambda: False, retriever.clear_retriever_cache
            )
        )
        self.assertTrue(client.closed)
        self.assertTrue(client.cache_cleared)
        self.assertIsNone(chroma_index._chroma_client)
        self.assertFalse(os.path.exists(chroma_index.CHROMA_PATH))

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
