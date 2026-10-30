import asyncio
import json
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

import api
import app_storage
from services.chat import query as chat_query
from services.chat import runs as chat_runs
from services.chat import service as chat_service
from services.chat import sessions as chat_sessions
from services.chat import trace as chat_trace
from services.chat.diagnostics import RetrievalDiagnostics
from services.core import config as application_config
from services.core.errors import ResourceNotFoundError
from services.retrieval import context as context_builder
from tests.support import IsolatedStorageTestCase

MODEL_CONFIG = {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"}


def _payload(**overrides):
    trace = chat_trace.ChatTrace(1, kb_name="技术部", session_id="s")
    payload = trace.payload(RetrievalDiagnostics("问题"), "answered")
    payload.update(overrides)
    return payload


def _doc(score, kept):
    return {"position": 1, "kb_name": "技术部", "source": "a.pdf", "page": "1",
            "route": "fixed", "snippet": "x", "score": score, "kept": kept}


def _span(span_id=1, truncated=False):
    return {"id": span_id, "kb_name": "技术部", "source": "a.pdf", "page": "1",
            "routes": ["fixed"], "chars": 10, "truncated": truncated}


class DiagnoseTest(unittest.TestCase):
    def test_zero_candidates(self):
        hints = chat_trace.diagnose(_payload())
        self.assertTrue(hints[0].startswith("召回为 0"))

    def test_all_candidates_below_threshold_reports_top_score(self):
        hints = chat_trace.diagnose(_payload(
            candidates=[_doc(0.0, False)], rerank_status="success",
            reranked=[_doc(0.02, False)],
        ))
        self.assertIn("0.020", hints[0])
        self.assertIn("相关度下限", hints[0])

    def test_low_top_score_and_truncation_and_uncited_answer(self):
        hints = chat_trace.diagnose(_payload(
            candidates=[_doc(0.0, False)], rerank_status="success",
            reranked=[_doc(0.2, True)], context=[_span(truncated=True)],
        ))
        self.assertEqual(len(hints), 3)
        self.assertIn("0.200", hints[0])
        self.assertIn("截断", hints[1])
        self.assertIn("生成环节", hints[2])

    def test_stage_error_is_reported_before_retrieval_hints(self):
        hints = chat_trace.diagnose(_payload(
            status="error",
            stages=[{"stage": "rerank", "status": "error", "elapsed_ms": 1.0, "detail": {}}],
        ))
        self.assertEqual(hints, ["在“rerank”阶段出错，后续环节没有执行。"])

    def test_skipped_retrieval(self):
        hints = chat_trace.diagnose(_payload(skipped=True, status="direct"))
        self.assertEqual(len(hints), 1)
        self.assertIn("无需检索", hints[0])

    def test_oversized_payload_drops_snippets_first(self):
        doc = {**_doc(0.9, True), "snippet": "长" * 200}
        payload = _payload(candidates=[doc] * 400, reranked=[doc] * 20)
        stored = json.loads(chat_trace._bounded_json(payload))
        self.assertEqual(len(stored["candidates"]), 400)
        self.assertEqual(stored["candidates"][0]["snippet"], "")


class TraceStorageTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()

    def _save(self, payload, account_id=1, run_id=None):
        chat_trace.save_trace(
            payload, account_id=account_id, session_id="s", run_id=run_id,
            assistant_message_id=None,
        )

    def test_save_load_feedback_and_export(self):
        payload = _payload(cited_ids=[2], context=[_span(1), {**_span(2), "source": "b.pdf"}])
        self._save(payload)
        stored = chat_trace.load_trace(payload["id"])
        self.assertEqual((stored.account_id, stored.kb_name), (1, "技术部"))
        self.assertEqual(stored.payload["question"], "问题")

        with self.assertRaises(ResourceNotFoundError):
            chat_trace.set_feedback(payload["id"], 2, -1)
        chat_trace.set_feedback(payload["id"], 1, -1)
        (row,) = chat_trace.evaluation_rows(-1, limit=10)
        self.assertEqual(
            chat_trace.evaluation_question(row.payload),
            {"id": payload["id"], "question": "问题", "gold": ["b.pdf"], "kb_name": "技术部"},
        )
        self.assertEqual(chat_trace.evaluation_rows(1, limit=10), [])

    def test_export_filters_knowledge_bases_before_limit(self):
        for index in range(3):
            payload = _payload(id=f"hidden-{index}", kb_name="财务部")
            self._save(payload)
            chat_trace.set_feedback(payload["id"], 1, -1)
        visible = _payload(id="visible")
        self._save(visible)
        chat_trace.set_feedback(visible["id"], 1, -1)
        # 可见记录最旧：先 LIMIT 再过滤会只取到不可见记录。
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("UPDATE chat_traces SET created_at = '2000-01-01' WHERE id = 'visible'")

        rows = chat_trace.evaluation_rows(-1, limit=1, kb_names=("技术部",))

        self.assertEqual([row.payload["id"] for row in rows], ["visible"])
        self.assertEqual(chat_trace.evaluation_rows(-1, limit=10, kb_names=()), [])

    def test_missing_trace_raises_not_found(self):
        with self.assertRaises(ResourceNotFoundError):
            chat_trace.load_trace("nope")

    def test_runs_list_carries_trace_id_and_filters_by_feedback(self):
        recorder = chat_runs.ChatRunRecorder(1, "alice", "技术部", question="问", status="answered")
        run_id = chat_runs.record_run(recorder)
        chat_runs.record_run(chat_runs.ChatRunRecorder(1, "alice", "技术部", question="问"))
        payload = _payload()
        self._save(payload, run_id=run_id)
        chat_trace.set_feedback(payload["id"], 1, -1)
        total, items = chat_runs.list_runs(feedback=-1)
        self.assertEqual((total, items[0]["trace_id"]), (1, payload["id"]))
        total, items = chat_runs.list_runs(feedback=0)
        self.assertEqual((total, items[0]["trace_id"]), (1, ""))

    def test_prune_removes_expired_traces_with_runs(self):
        self._save(_payload(created_at="2000-01-01T00:00:00+00:00"))
        self._save(_payload(id="fresh"))
        chat_runs.prune(90)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM chat_traces")]
        self.assertEqual(ids, ["fresh"])


class ChatFlowTestCase(IsolatedStorageTestCase):
    """准备一个知识库和只有问答权限的普通账号；本身不含测试。"""

    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at, query_rewrite_enabled) "
                "VALUES ('技术部', lower(hex(randomblob(16))), ?, 0)",
                (application_config.db_now(),),
            )
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, knowledge_base_permissions, created_at) "
                "VALUES ('普通', '', '[\"问答对话\"]', '[\"全公司\"]', ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '普通'").fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (name, phone, status, role_id, created_at, password, username) "
                "VALUES ('普通用户', '', '正常', ?, ?, ?, 'plain')",
                (role_id, application_config.db_now(), application_config.hash_password("user-password")),
            )
            self.plain_id = conn.execute(
                "SELECT id FROM accounts WHERE username = 'plain'"
            ).fetchone()[0]

    def _request(self, username, password, calls):
        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                login = await client.post("/api/login", json={"username": username, "password": password})
                headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
                return [
                    await client.request(method, path, json=body, headers=headers)
                    for method, path, body in calls
                ]

        return asyncio.run(run())


class ChatTraceFlowTest(ChatFlowTestCase):
    def _run_chat(self, account_id, retrieve=None):
        request = chat_service.ChatRequest(question="怎么配置？", kb_name="技术部")
        owner = chat_service.SessionOwner(account_id, "user")
        span = context_builder.ContextSpan(
            source="/data/手册.pdf",
            content="上下文" + context_builder.CONTEXT_TRUNCATION_MARKER,
            pages=("3",), page_unit="页", rank=0, kb_name="技术部", chunk_routes=("fixed",),
        )

        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="结论。[来源 1]")

        async def consume():
            stream = await chat_service.chat(request, owner)
            return [event async for event in stream.events]

        with patch.object(
            chat_service, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=retrieve or AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[span]
        ), patch.object(chat_service, "create_answer_llm", return_value=Model()):
            return asyncio.run(consume())

    def test_stream_emits_stages_and_saves_trace_linked_to_run_and_message(self):
        events = [
            json.loads(event[len("data: "):])
            for event in self._run_chat(self.plain_id)
            if event.startswith("data: ")
        ]
        stages = [event["stage"] for event in events if "stage" in event]
        self.assertEqual(stages[0], "plan")
        self.assertIn("retrieval", stages)
        self.assertIn("context", stages)
        self.assertEqual(stages[-1], "generate")
        final = events[-1]
        self.assertTrue(final["is_end"])
        self.assertIn("diagnostics", final)

        stored = chat_trace.load_trace(final["trace_id"])
        self.assertEqual(stored.account_id, self.plain_id)
        payload = stored.payload
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(payload["cited_ids"], [1])
        self.assertEqual(payload["context"][0]["source"], "手册.pdf")
        self.assertTrue(payload["context"][0]["truncated"])
        self.assertGreater(payload["prompt_chars"], 0)
        self.assertEqual(payload["stages"][-1]["detail"]["cited"], 1)

        _, items = chat_runs.list_runs()
        self.assertEqual(items[0]["trace_id"], final["trace_id"])
        _, history = chat_sessions.load_owned_session_messages(
            final["session_id"], chat_service.SessionOwner(self.plain_id, "user")
        )
        self.assertEqual(history[-1]["trace_id"], final["trace_id"])
        self.assertEqual(history[-1]["feedback"], 0)

    def test_pre_stream_failure_saves_trace_with_failed_stage(self):
        with self.assertRaises(RuntimeError):
            self._run_chat(self.plain_id, AsyncMock(side_effect=RuntimeError("index down")))
        _, items = chat_runs.list_runs()
        payload = chat_trace.load_trace(items[0]["trace_id"]).payload
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["stages"][-1]["status"], "error")
        self.assertIn("index down", payload["stages"][-1]["detail"]["error"])

    def test_trace_access_feedback_and_export(self):
        own = _payload(id="own")
        other = _payload(id="other")
        chat_trace.save_trace(own, account_id=self.plain_id, session_id="s", run_id=None,
                              assistant_message_id=None)
        chat_trace.save_trace(other, account_id=9999, session_id="s", run_id=None,
                              assistant_message_id=None)
        read_own, read_other, rate_own, rate_other, rate_bad, export = self._request(
            "plain", "user-password", [
                ("GET", "/api/chat/traces/own", None),
                ("GET", "/api/chat/traces/other", None),
                ("PUT", "/api/chat/traces/own/feedback", {"feedback": -1}),
                ("PUT", "/api/chat/traces/other/feedback", {"feedback": -1}),
                ("PUT", "/api/chat/traces/own/feedback", {"feedback": 2}),
                ("GET", "/api/ops/chat-traces/export", None),
            ],
        )
        self.assertEqual(read_own.status_code, 200)
        self.assertEqual(read_own.json()["id"], "own")
        self.assertEqual(read_other.status_code, 404)
        self.assertEqual(rate_own.status_code, 200)
        self.assertEqual(rate_other.status_code, 404)
        self.assertEqual(rate_bad.status_code, 422)
        self.assertEqual(export.status_code, 403)

        admin_read, admin_export = self._request(
            "admin", application_config.initial_admin_password(), [
                ("GET", "/api/chat/traces/other", None),
                ("GET", "/api/ops/chat-traces/export?rating=down", None),
            ],
        )
        self.assertEqual(admin_read.status_code, 200)
        lines = admin_export.text.splitlines()
        self.assertEqual([json.loads(line)["id"] for line in lines], ["own"])


if __name__ == "__main__":
    unittest.main()


class CommonKnowledgeBasePermissionTest(ChatFlowTestCase):
    def _include_common_for(self, knowledge_base_permissions):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE roles SET knowledge_base_permissions = ? WHERE name = '普通'",
                (json.dumps(knowledge_base_permissions, ensure_ascii=False),),
            )

        async def empty_events():
            yield ""

        stream = SimpleNamespace(events=empty_events(), refresh_summary=lambda: None)
        with patch.object(chat_service, "chat", new=AsyncMock(return_value=stream)) as chat:
            responses = self._request(
                "plain", "user-password",
                [("POST", "/api/chat", {"question": "问题", "kb_name": "技术部"})],
            )
        self.assertEqual(responses[0].status_code, 200)
        return chat.await_args.kwargs["include_common"]

    def test_common_knowledge_base_is_supplemented_only_with_read_permission(self):
        self.assertFalse(self._include_common_for(["技术部"]))
        self.assertTrue(self._include_common_for(["技术部", "通用知识库"]))
        self.assertTrue(self._include_common_for(["全公司"]))
