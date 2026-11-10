"""删除与聊天写入交错时的持久化和流式终态回归。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import app_storage
from routers import knowledge_bases as kb_routes
from services.chat import runs, service, sessions, trace
from services.chat.diagnostics import RetrievalDiagnostics
from services.core import sqlite_connections as application_db
from services.core.errors import ResourceNotFoundError
from tests.support import IsolatedStorageTestCase, add_knowledge_bases


class ChatLifecycleTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.a, self.b = add_knowledge_bases("主库", "来源库")
        self.owner = sessions.SessionOwner(1, "alice")
        self.session_id = sessions.resolve_session(None, self.owner, self.a).session_id

    def source(self, kb_id):
        return {"id": 1, "kb_id": kb_id, "source": "制度.pdf", "content": "证据"}

    def test_deleting_main_kb_rejects_all_writes_and_history(self):
        app_storage.knowledge_bases.mark_deleting(self.a)
        for write in (
            lambda: sessions.resolve_session(None, self.owner, self.a),
            lambda: sessions.resolve_session(self.session_id, self.owner, self.a),
            lambda: sessions.save_user_message(self.session_id, "问题"),
            lambda: sessions.save_assistant_message(self.session_id, "回答", []),
            lambda: sessions.load_owned_session_messages(self.session_id, self.owner),
        ):
            with self.assertRaises(ResourceNotFoundError):
                write()
        self.assertFalse(
            sessions.save_conversation_summary(
                self.session_id,
                "摘要",
                expected_message_id=0,
                through_message_id=1,
            )
        )
        self.assertEqual(runs.record_run(runs.ChatRunRecorder(1, "alice", self.a)), 0)
        payload = trace.ChatTrace(1, kb_id=self.a).payload(RetrievalDiagnostics("问题"), "answered")
        trace.save_trace(
            payload,
            account_id=1,
            session_id=self.session_id,
            run_id=None,
            assistant_message_id=None,
        )
        with application_db.connect(app_storage.DB_FILE) as conn:
            for table in ("chat_messages", "chat_session_memory", "chat_runs", "chat_traces"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)

    def test_source_deleted_before_write_is_removed_from_message_and_trace(self):
        app_storage.knowledge_bases.delete_relational_data(self.b)
        sources = [self.source(self.a), self.source(self.b)]
        sources[0]["evidence"] = [self.source(self.a), self.source(self.b)]
        sessions.save_assistant_message(self.session_id, "历史正文不改", sources)
        tracker = trace.ChatTrace(1, kb_id=self.a, session_id=self.session_id)
        tracker.candidates = [{"kb_id": self.b, "snippet": "不可保留"}]
        tracker.save(RetrievalDiagnostics("问题"), "answered", run_id=None)
        with application_db.connect(app_storage.DB_FILE) as conn:
            saved_sources = json.loads(
                conn.execute("SELECT sources FROM chat_messages").fetchone()[0]
            )
            saved_trace = json.loads(conn.execute("SELECT payload FROM chat_traces").fetchone()[0])
        self.assertEqual([source["kb_id"] for source in saved_sources], [self.a])
        self.assertEqual([ev["kb_id"] for ev in saved_sources[0]["evidence"]], [self.a])
        self.assertEqual(saved_trace["candidates"], [])

    def test_final_deletion_cleans_persisted_trace_and_nested_evidence(self):
        sources = [
            self.source(self.a),
            self.source(self.b),
            {"id": 3, "url": "https://example.test"},
        ]
        sources[0]["evidence"] = [self.source(self.a), self.source(self.b)]
        sessions.save_assistant_message(self.session_id, "来源库这个词保留", sources)
        payload = trace.ChatTrace(1, kb_id=self.a).payload(RetrievalDiagnostics("问题"), "answered")
        for key in ("candidates", "reranked", "context"):
            payload[key] = [{"kb_id": self.a}, {"kb_id": self.b}]
        payload["web_results"] = [{"url": "https://example.test"}]
        trace.save_trace(
            payload,
            account_id=1,
            session_id=self.session_id,
            run_id=None,
            assistant_message_id=None,
        )
        app_storage.knowledge_bases.delete_relational_data(self.b)
        with application_db.connect(app_storage.DB_FILE) as conn:
            content, raw_sources = conn.execute(
                "SELECT content, sources FROM chat_messages"
            ).fetchone()
            saved_trace = json.loads(conn.execute("SELECT payload FROM chat_traces").fetchone()[0])
        self.assertEqual(content, "来源库这个词保留")
        self.assertNotIn(self.b, raw_sources)
        for key in ("candidates", "reranked", "context"):
            self.assertEqual(saved_trace[key], [{"kb_id": self.a}])
        self.assertEqual(saved_trace["web_results"], [{"url": "https://example.test"}])

    def test_inflight_answer_reports_deletion_before_success_terminal(self):
        async def run(delete_fully):
            started, resume = asyncio.Event(), asyncio.Event()

            class Model:
                async def astream(self, _messages):
                    yield SimpleNamespace(content="部分回答")
                    started.set()
                    await resume.wait()
                    yield SimpleNamespace(content="完成")

            async def consume():
                return [
                    event
                    async for event in service._stream_answer(
                        self.session_id,
                        {},
                        [],
                        [],
                        None,
                        None,
                        allow_answer_without_evidence=True,
                    )
                ]

            with patch.object(service, "create_answer_llm", return_value=Model()):
                task = asyncio.create_task(consume())
                await asyncio.wait_for(started.wait(), 5)
                app_storage.knowledge_bases.mark_deleting(self.a)
                if delete_fully:
                    app_storage.knowledge_bases.delete_relational_data(self.a)
                resume.set()
                events = await asyncio.wait_for(task, 5)
            terminal = json.loads(events[-1].removeprefix("data: "))
            self.assertTrue(terminal["is_end"])
            self.assertIn("已删除", terminal["error"])
            self.assertNotIn("final_answer", terminal)
            with application_db.connect(app_storage.DB_FILE) as conn:
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0], 0
                )

        asyncio.run(run(False))
        # 同一 ID 从 deleting 进入最终删除，已开始生成的回答同样不能写回。
        asyncio.run(run(True))

    def test_only_authorized_managers_can_list_deleting_libraries(self):
        app_storage.knowledge_bases.mark_deleting(self.b)
        for manager, allowed, expected in (
            (True, {self.a, self.b}, {self.a, self.b}),
            (False, {self.a, self.b}, {self.a}),
            (True, {self.a}, {self.a}),
        ):
            principal = SimpleNamespace(
                can_manage_knowledge_bases=lambda manager=manager: manager,
                can_read_knowledge_base=lambda kb_id, allowed=allowed: kb_id in allowed,
            )
            self.assertEqual(
                {kb["kb_id"] for kb in kb_routes.list_knowledge_bases(principal)}, expected
            )
