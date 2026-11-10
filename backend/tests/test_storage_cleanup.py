import json
import os
import sqlite3
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

import app_storage
from routers import chat, knowledge_bases, schemas
from services.core import config as application_config
from services.core.errors import ServiceUnavailableError
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
from tests.support import (
    IsolatedStorageTestCase,
    add_knowledge_bases,
    create_index_job,
    fixture_document_id,
)


class StorageCleanupTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.kb_id, self.other_kb_id = add_knowledge_bases("研发部", "技术部")

    def test_init_enables_sqlite_secure_delete(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            secure_delete = conn.execute("PRAGMA secure_delete").fetchone()[0]
        self.assertEqual(secure_delete, 1)

    def test_delete_job_redacts_filename_after_source_row_is_removed(self):
        job_id = create_index_job(self.kb_id, "delete", "制度.pdf")

        with (
            patch.object(knowledge_base, "delete_document_indexes") as delete_indexes,
            patch.object(
                app_storage.index_jobs, "compact_database", return_value=True
            ) as compact_database,
        ):
            knowledge_base.run_delete_job(
                job_id, self.kb_id, fixture_document_id(self.kb_id, "制度.pdf")
            )

        delete_indexes.assert_called_once_with(
            self.kb_id, fixture_document_id(self.kb_id, "制度.pdf")
        )
        compact_database.assert_called_once_with()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT filename, status FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(row, (None, INDEX_JOB_COMPLETED))

    def test_delete_job_failure_keeps_filename_and_is_requeued_on_restart(self):
        job_id = create_index_job(self.kb_id, "delete", "制度.pdf")

        with (
            patch.object(
                knowledge_base,
                "delete_document_indexes",
                side_effect=RuntimeError("index unavailable"),
            ),
            patch.object(app_storage.index_jobs, "compact_database", return_value=False),
        ):
            knowledge_base.run_delete_job(
                job_id, self.kb_id, fixture_document_id(self.kb_id, "制度.pdf")
            )

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

    def test_knowledge_base_delete_failure_stays_recoverable_and_resumes_at_startup(self):
        with patch.object(
            get_chunk_index(),
            "delete_kb",
            side_effect=RuntimeError("index unavailable"),
        ):
            with self.assertRaises(ServiceUnavailableError) as raised:
                knowledge_base_use_cases.delete_knowledge_base(self.kb_id)

        self.assertEqual(raised.exception.code, "knowledge_base_cleanup_failed")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            state = conn.execute(
                "SELECT state FROM knowledge_bases WHERE kb_id = ?", (self.kb_id,)
            ).fetchone()
        self.assertEqual(state, ("deleting",))
        self.assertFalse(app_storage.documents.knowledge_base_exists(self.kb_id))

        self.assertEqual(knowledge_base_use_cases.resume_pending_deletions(), 1)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertIsNone(
                conn.execute(
                    "SELECT 1 FROM knowledge_bases WHERE kb_id = ?", (self.kb_id,)
                ).fetchone()
            )

    def test_knowledge_base_delete_removes_index_and_parent_rows(self):
        kb_id = self.kb_id
        other_id = self.other_kb_id
        index = get_chunk_index()
        child = Document(page_content="子块", metadata={"source": "制度.pdf", "parent_id": "p1"})
        index.publish_document(kb_id, fixture_document_id(kb_id, "制度.pdf"), [child], [[1.0, 0.0]])
        index.publish_document(
            other_id, fixture_document_id(other_id, "制度.pdf"), [child], [[1.0, 0.0]]
        )
        for owner in (kb_id, other_id):
            app_storage.documents.create_document_with_job(
                owner,
                fixture_document_id(owner, "制度.pdf"),
                "制度.pdf",
                size=1,
                parser_mode="native_text",
                metadata={},
            )
            with sqlite3.connect(app_storage.DB_FILE) as conn:
                conn.execute(
                    "INSERT INTO parent_chunks (document_id, kb_id, parent_id, source, page, content, "
                    "contextual_title) VALUES (?, ?, 'p1', '制度.pdf', '1', '父块', '')",
                    (fixture_document_id(owner, "制度.pdf"), owner),
                )

        with (
            patch.object(mineru_client, "remove_saved_results_for_knowledge_base"),
            patch.object(app_storage.index_jobs, "compact_database", return_value=False),
            patch.object(app_storage.documents, "has_documents", return_value=True),
        ):
            knowledge_base_use_cases.delete_knowledge_base(self.kb_id)

        self.assertEqual(index.available_routes(kb_id, None), set())
        self.assertTrue(index.available_routes(other_id, None))
        self.assertEqual(
            parent_store.load(kb_id, fixture_document_id(kb_id, "制度.pdf"), ["p1"]), {}
        )
        self.assertIn(
            "p1", parent_store.load(other_id, fixture_document_id(other_id, "制度.pdf"), ["p1"])
        )

    def test_knowledge_base_delete_removes_sessions_and_dangling_citations(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.executemany(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        "deleted-session",
                        1,
                        "alice",
                        self.kb_id,
                        "研发部 会话",
                        "2026-01-01",
                        "2026-01-01",
                    ),
                    (
                        "kept-session",
                        1,
                        "alice",
                        self.other_kb_id,
                        "技术部 会话",
                        "2026-01-01",
                        "2026-01-01",
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
                        json.dumps(
                            [
                                {"source": "制度.pdf", "kb_id": self.kb_id},
                                {"source": "通用.pdf", "kb_id": self.other_kb_id},
                            ]
                        ),
                        "2026-01-01",
                    ),
                ],
            )

        with (
            patch.object(mineru_client, "remove_saved_results_for_knowledge_base"),
            patch.object(app_storage.index_jobs, "compact_database", return_value=False) as compact,
        ):
            result = knowledge_bases.delete_knowledge_base(self.kb_id)

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
            self.assertEqual(
                json.loads(sources), [{"source": "通用.pdf", "kb_id": self.other_kb_id}]
            )

    def test_knowledge_base_rename_only_changes_the_display_name(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("session-1", 1, "alice", self.kb_id, "会话", "2026-01-01", "2026-01-01"),
            )
        job_id = create_index_job(self.kb_id, "rebuild")
        data_dir = app_storage.documents.knowledge_base_path(self.kb_id)
        os.makedirs(data_dir, exist_ok=True)
        index = get_chunk_index()
        index.publish_document(
            self.kb_id,
            fixture_document_id(self.kb_id, "制度.pdf"),
            [Document(page_content="子块", metadata={})],
            [[1.0, 0.0]],
        )

        result = knowledge_bases.rename_knowledge_base(
            self.kb_id, schemas.KnowledgeBaseNameRequest(name="产品研发部")
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["kb_id"], self.kb_id)
        self.assertEqual(result["name"], "产品研发部")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            name = conn.execute(
                "SELECT name FROM knowledge_bases WHERE kb_id = ?", (self.kb_id,)
            ).fetchone()[0]
            session_kb = conn.execute(
                "SELECT kb_id FROM chat_sessions WHERE id = 'session-1'"
            ).fetchone()[0]
            job_kb = conn.execute(
                "SELECT kb_id FROM index_jobs WHERE id = ?", (job_id,)
            ).fetchone()[0]
        self.assertEqual(name, "产品研发部")
        self.assertEqual((session_kb, job_kb), (self.kb_id, self.kb_id))
        self.assertTrue(os.path.isdir(data_dir))
        self.assertTrue(index.available_routes(self.kb_id, None))

    def test_chat_delete_removes_rows_without_immediate_compaction(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_sessions "
                "(id, account_id, username, kb_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "session-1",
                    42,
                    "tester",
                    self.kb_id,
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
                    all_knowledge_bases=True,
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
