import asyncio
import json
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import app_storage
from services.chat import query as chat_query
from services.chat import runs as chat_runs
from services.chat import service as chat_service
from services.chat import sessions as chat_sessions
from services.chat import trace as chat_trace
from services.chat.diagnostics import RetrievalDiagnostics
from services.core import config as application_config
from services.core import llm_config
from services.core.container import get_application_container
from services.core.errors import ResourceNotFoundError
from services.retrieval import context as context_builder
from tests.support import (
    IsolatedStorageTestCase,
    add_knowledge_bases,
    api_client,
)

MODEL_CONFIG = {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"}


def _payload(kb_id="kb-x", **overrides):
    trace = chat_trace.ChatTrace(1, kb_id=kb_id, session_id="s")
    payload = trace.payload(RetrievalDiagnostics("问题"), "answered")
    payload.update(overrides)
    return payload


def _doc(score, kept):
    return {
        "position": 1,
        "kb_id": "kb-x",
        "source": "a.pdf",
        "page": "1",
        "route": "fixed",
        "snippet": "x",
        "score": score,
        "kept": kept,
    }


def _span(span_id=1, truncated=False):
    return {
        "id": span_id,
        "kb_id": "kb-x",
        "source": "a.pdf",
        "page": "1",
        "routes": ["fixed"],
        "chars": 10,
        "truncated": truncated,
    }


class DiagnoseTest(unittest.TestCase):
    def test_zero_candidates(self):
        hints = chat_trace.diagnose(_payload())
        self.assertTrue(hints[0].startswith("召回为 0"))

    def test_all_candidates_below_threshold_reports_top_score(self):
        hints = chat_trace.diagnose(
            _payload(
                candidates=[_doc(0.0, False)],
                rerank_status="success",
                reranked=[_doc(0.02, False)],
            )
        )
        self.assertIn("0.020", hints[0])
        self.assertIn("相关度下限", hints[0])

    def test_low_top_score_and_truncation_and_uncited_answer(self):
        hints = chat_trace.diagnose(
            _payload(
                candidates=[_doc(0.0, False)],
                rerank_status="success",
                reranked=[_doc(0.2, True)],
                context=[_span(truncated=True)],
            )
        )
        self.assertEqual(len(hints), 3)
        self.assertIn("0.200", hints[0])
        self.assertIn("截断", hints[1])
        self.assertIn("生成环节", hints[2])

    def test_stage_error_is_reported_before_retrieval_hints(self):
        hints = chat_trace.diagnose(
            _payload(
                status="error",
                stages=[{"stage": "rerank", "status": "error", "elapsed_ms": 1.0, "detail": {}}],
            )
        )
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
        self.tech_id, self.finance_id = add_knowledge_bases("技术部", "财务部")

    def _save(self, payload, account_id=1, run_id=None):
        chat_trace.save_trace(
            payload,
            account_id=account_id,
            session_id="s",
            run_id=run_id,
            assistant_message_id=None, container=get_application_container(),
        )

    def test_save_load_feedback_and_export(self):
        payload = _payload(
            self.tech_id,
            cited_ids=[2],
            context=[
                {**_span(1), "kb_id": self.tech_id, "document_id": "doc-a"},
                {**_span(2), "kb_id": self.tech_id, "source": "b.pdf", "document_id": "doc-b"},
            ],
        )
        self._save(payload)
        stored = chat_trace.load_trace(payload["id"], container=get_application_container())
        self.assertEqual((stored.account_id, stored.kb_id), (1, self.tech_id))
        self.assertEqual(stored.payload["question"], "问题")

        with self.assertRaises(ResourceNotFoundError):
            chat_trace.set_feedback(payload["id"], 2, -1, container=get_application_container())
        chat_trace.set_feedback(payload["id"], 1, -1, container=get_application_container())
        (row,) = chat_trace.evaluation_rows(-1, limit=10, container=get_application_container())
        self.assertEqual(
            chat_trace.evaluation_question(row.payload),
            {
                "id": payload["id"],
                "question": "问题",
                "gold": ["doc-b"],
                "gold_sources": {"doc-b": "b.pdf"},
                "kb_id": self.tech_id,
            },
        )
        self.assertEqual(chat_trace.evaluation_rows(1, limit=10, container=get_application_container()), [])

    def test_export_gold_keeps_same_name_documents_apart(self):
        payload = _payload(
            self.tech_id,
            cited_ids=[1, 3, 4],
            context=[
                {**_span(1), "document_id": "doc-2"},
                {**_span(2), "document_id": "doc-1"},
                {**_span(3), "document_id": "doc-3"},
                # 网页结果没有 document_id，不能作为评测的标准文档。
                {**_span(4), "source": "网页", "url": "https://example.com"},
            ],
        )

        question = chat_trace.evaluation_question(payload)

        # 三段同名 a.pdf：只有被引用的两份文档进入 gold，未引用的 doc-1 不会被合并进来。
        self.assertEqual(question["gold"], ["doc-2", "doc-3"])
        self.assertEqual(question["gold_sources"], {"doc-2": "a.pdf", "doc-3": "a.pdf"})

    def test_export_filters_knowledge_bases_before_limit(self):
        for index in range(3):
            payload = _payload(self.finance_id, id=f"hidden-{index}")
            self._save(payload)
            chat_trace.set_feedback(payload["id"], 1, -1, container=get_application_container())
        visible = _payload(self.tech_id, id="visible")
        self._save(visible)
        chat_trace.set_feedback(visible["id"], 1, -1, container=get_application_container())
        # 可见记录最旧：先 LIMIT 再过滤会只取到不可见记录。
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("UPDATE chat_traces SET created_at = '2000-01-01' WHERE id = 'visible'")

        rows = chat_trace.evaluation_rows(-1, limit=1, kb_ids=(self.tech_id,), container=get_application_container())

        self.assertEqual([row.payload["id"] for row in rows], ["visible"])
        self.assertEqual(chat_trace.evaluation_rows(-1, limit=10, kb_ids=(), container=get_application_container()), [])

    def test_missing_trace_raises_not_found(self):
        with self.assertRaises(ResourceNotFoundError):
            chat_trace.load_trace("nope", container=get_application_container())

    def test_runs_list_carries_trace_id_and_filters_by_feedback(self):
        recorder = chat_runs.ChatRunRecorder(
            1, "alice", self.tech_id, question="问", status="answered"
        )
        run_id = chat_runs.record_run(recorder, container=get_application_container())
        chat_runs.record_run(chat_runs.ChatRunRecorder(1, "alice", self.tech_id, question="问"), container=get_application_container())
        payload = _payload(self.tech_id)
        self._save(payload, run_id=run_id)
        chat_trace.set_feedback(payload["id"], 1, -1, container=get_application_container())
        total, items = chat_runs.list_runs(feedback=-1, container=get_application_container())
        self.assertEqual((total, items[0]["trace_id"]), (1, payload["id"]))
        total, items = chat_runs.list_runs(feedback=0, container=get_application_container())
        self.assertEqual((total, items[0]["trace_id"]), (1, ""))

    def test_prune_removes_expired_traces_with_runs(self):
        self._save(_payload(self.tech_id, created_at="2000-01-01T00:00:00+00:00"))
        self._save(_payload(self.tech_id, id="fresh"))
        chat_runs.prune(app_storage.DB_FILE, 90)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM chat_traces")]
        self.assertEqual(ids, ["fresh"])


class ChatFlowTestCase(IsolatedStorageTestCase):
    """准备一个知识库和只有问答权限的普通账号；本身不含测试。"""

    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()
        self.tech_id = add_knowledge_bases("技术部")[0]
        app_storage.knowledge_base_settings.set_query_rewrite_enabled(self.tech_id, False)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, all_knowledge_bases, created_at) "
                "VALUES ('普通', '', '[\"问答对话\"]', 1, ?)",
                (application_config.db_now(),),
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '普通'").fetchone()[0]
            conn.execute(
                "INSERT INTO accounts (name, phone, status, role_id, created_at, password, username) "
                "VALUES ('普通用户', '', '正常', ?, ?, ?, 'plain')",
                (
                    role_id,
                    application_config.db_now(),
                    application_config.hash_password("user-password"),
                ),
            )
            self.plain_id = conn.execute(
                "SELECT id FROM accounts WHERE username = 'plain'"
            ).fetchone()[0]

    def _request(self, username, password, calls):
        async def run():
            async with api_client() as client:
                await client.post(
                    "/api/login", json={"username": username, "password": password}
                )
                return [
                    await client.request(method, path, json=body)
                    for method, path, body in calls
                ]

        return asyncio.run(run())


class ChatTraceFlowTest(ChatFlowTestCase):
    def _run_chat(self, account_id, retrieve=None):
        request = chat_service.ChatRequest(question="怎么配置？", kb_id=self.tech_id)
        owner = chat_service.SessionOwner(account_id, "user")
        span = context_builder.ContextSpan(
            source="/data/手册.pdf",
            content="上下文" + context_builder.CONTEXT_TRUNCATION_MARKER,
            pages=("3",),
            rank=0,
            kb_id=self.tech_id,
            chunk_routes=("fixed",),
        )

        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="结论。[来源 1]")

        async def consume():
            stream = await chat_service.chat(request, owner, container=get_application_container())
            return [event async for event in stream.events]

        with (
            patch.object(llm_config, "select_model_config", return_value=MODEL_CONFIG),
            patch.object(
                chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
            ),
            patch.object(
                chat_query, "retrieve_rerank_candidates", new=retrieve or AsyncMock(return_value=[])
            ),
            patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])),
            patch.object(chat_service.context_builder, "build_context_spans", return_value=[span]),
            patch.object(llm_config, "create_answer_llm", return_value=Model()),
        ):
            return asyncio.run(consume())

    def test_stream_emits_stages_and_saves_trace_linked_to_run_and_message(self):
        events = [
            json.loads(event[len("data: ") :])
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

        stored = chat_trace.load_trace(final["trace_id"], container=get_application_container())
        self.assertEqual(stored.account_id, self.plain_id)
        payload = stored.payload
        self.assertEqual(payload["status"], "answered")
        self.assertEqual(payload["cited_ids"], [1])
        self.assertEqual(payload["context"][0]["source"], "手册.pdf")
        self.assertTrue(payload["context"][0]["truncated"])
        self.assertGreater(payload["prompt_chars"], 0)
        self.assertEqual(payload["stages"][-1]["detail"]["cited"], 1)

        _, items = chat_runs.list_runs(container=get_application_container())
        self.assertEqual(items[0]["trace_id"], final["trace_id"])
        _, history = chat_sessions.load_owned_session_messages(
            final["session_id"], chat_service.SessionOwner(self.plain_id, "user"), container=get_application_container()
        )
        self.assertEqual(history[-1]["trace_id"], final["trace_id"])
        self.assertEqual(history[-1]["feedback"], 0)

    def test_pre_stream_failure_saves_trace_with_failed_stage(self):
        with self.assertRaises(RuntimeError):
            self._run_chat(self.plain_id, AsyncMock(side_effect=RuntimeError("index down")))
        _, items = chat_runs.list_runs(container=get_application_container())
        payload = chat_trace.load_trace(items[0]["trace_id"], container=get_application_container()).payload
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["stages"][-1]["status"], "error")
        self.assertEqual(payload["stages"][-1]["detail"]["error"], "RuntimeError")
        self.assertNotIn("index down", json.dumps(payload, ensure_ascii=False))

    def test_trace_access_feedback_and_export(self):
        own = _payload(self.tech_id, id="own")
        other = _payload(self.tech_id, id="other")
        chat_trace.save_trace(
            own, account_id=self.plain_id, session_id="s", run_id=None, assistant_message_id=None, container=get_application_container()
        )
        chat_trace.save_trace(
            other, account_id=9999, session_id="s", run_id=None, assistant_message_id=None, container=get_application_container()
        )
        read_own, read_other, rate_own, rate_other, rate_bad, export = self._request(
            "plain",
            "user-password",
            [
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
            "admin",
            application_config.initial_admin_password(),
            [
                ("GET", "/api/chat/traces/other", None),
                ("GET", "/api/ops/chat-traces/export?rating=down", None),
            ],
        )
        self.assertEqual(admin_read.status_code, 200)
        lines = admin_export.text.splitlines()
        self.assertEqual([json.loads(line)["id"] for line in lines], ["own"])


    def test_feedback_requires_current_read_permission_and_an_active_knowledge_base(self):
        chat_trace.save_trace(
            _payload(self.tech_id, id="own"),
            account_id=self.plain_id,
            session_id="s",
            run_id=None,
            assistant_message_id=None, container=get_application_container(),
        )
        common_id = app_storage.knowledge_bases.common_kb_id()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE roles SET all_knowledge_bases = 0 WHERE name = '普通'"
            )
            role_id = conn.execute("SELECT id FROM roles WHERE name = '普通'").fetchone()[0]
            conn.execute(
                "INSERT INTO role_knowledge_bases (role_id, kb_id) VALUES (?, ?)",
                (role_id, common_id),
            )
        (revoked,) = self._request(
            "plain", "user-password", [("PUT", "/api/chat/traces/own/feedback", {"feedback": -1})]
        )
        self.assertEqual(revoked.status_code, 403)

        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute("UPDATE roles SET all_knowledge_bases = 1 WHERE id = ?", (role_id,))
            conn.execute(
                "UPDATE knowledge_bases SET state = 'deleting' WHERE kb_id = ?", (self.tech_id,)
            )
        (deleting,) = self._request(
            "plain", "user-password", [("PUT", "/api/chat/traces/own/feedback", {"feedback": -1})]
        )
        self.assertEqual(deleting.status_code, 404)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            feedback = conn.execute(
                "SELECT feedback FROM chat_traces WHERE id = 'own'"
            ).fetchone()[0]
        self.assertFalse(feedback)

class CommonKnowledgeBasePermissionTest(ChatFlowTestCase):
    def _include_common_for(self, *, all_knowledge_bases, kb_ids):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            role_id = conn.execute("SELECT id FROM roles WHERE name = '普通'").fetchone()[0]
            conn.execute(
                "UPDATE roles SET all_knowledge_bases = ? WHERE id = ?",
                (int(all_knowledge_bases), role_id),
            )
            conn.execute("DELETE FROM role_knowledge_bases WHERE role_id = ?", (role_id,))
            conn.executemany(
                "INSERT INTO role_knowledge_bases (role_id, kb_id) VALUES (?, ?)",
                [(role_id, kb_id) for kb_id in kb_ids],
            )

        async def empty_events():
            yield ""

        stream = SimpleNamespace(
            events=empty_events(),
            refresh_summary=lambda: asyncio.sleep(0),
            abandon=lambda: asyncio.sleep(0),
        )
        with patch.object(chat_service, "chat", new=AsyncMock(return_value=stream)) as chat:
            responses = self._request(
                "plain",
                "user-password",
                [("POST", "/api/chat", {"question": "问题", "kb_id": self.tech_id})],
            )
        self.assertEqual(responses[0].status_code, 200)
        return chat.await_args.kwargs["include_common"]

    def test_common_knowledge_base_is_supplemented_only_with_read_permission(self):
        common_id = app_storage.knowledge_bases.common_kb_id()
        self.assertFalse(self._include_common_for(all_knowledge_bases=False, kb_ids=[self.tech_id]))
        self.assertTrue(
            self._include_common_for(all_knowledge_bases=False, kb_ids=[self.tech_id, common_id])
        )
        self.assertTrue(self._include_common_for(all_knowledge_bases=True, kb_ids=[]))


class SourceLevelPermissionTest(ChatFlowTestCase):
    """主库权限不能代替来源库权限：撤销来源库后，历史和追踪不再返回其结构化原文。"""

    def setUp(self):
        super().setUp()
        self.common_id = app_storage.knowledge_bases.common_kb_id()
        owner = chat_sessions.SessionOwner(self.plain_id, "plain")
        self.session_id = chat_sessions.resolve_session(None, owner, self.tech_id, container=get_application_container()).session_id
        chat_sessions.save_user_message(self.session_id, "问题", container=get_application_container())

        def evidence(index, kb_id, text):
            return {
                "id": index,
                "source": "x.pdf",
                "kb_id": kb_id,
                "kb_name_snapshot": "",
                "page": "",
                "page_numbers": [],
                "context_title": "",
                "content": text,
            }

        def source(index, kb_id, text):
            return {
                "id": index,
                "citation_ids": [index],
                "source": "x.pdf",
                "kb_id": kb_id,
                "kb_name_snapshot": "",
                "page": "",
                "context_title": "",
                "content": text,
                "evidence": [evidence(index, kb_id, text + "证据")],
            }

        web = {
            "id": 3,
            "citation_ids": [3],
            "source": "页面",
            "page": "",
            "context_title": "",
            "url": "https://example.test",
            "content": "网页",
        }
        chat_sessions.save_assistant_message(
            self.session_id,
            "回答",
            [
                source(1, self.tech_id, "本库原文"),
                source(2, self.common_id, "通用库原文"),
                web,
            ], container=get_application_container(),
        )

        def doc(kb_id, snippet):
            return {
                "position": 1,
                "kb_id": kb_id,
                "source": "x.pdf",
                "page": "1",
                "route": "fixed",
                "snippet": snippet,
            }

        payload = _payload(
            self.tech_id,
            id="mixed",
            candidates=[doc(self.tech_id, "本库片段"), doc(self.common_id, "通用库片段")],
            reranked=[doc(self.common_id, "通用库片段")],
            context=[{**_span(), "kb_id": self.common_id}, {**_span(2), "kb_id": self.tech_id}],
        )
        chat_trace.save_trace(
            payload,
            account_id=self.plain_id,
            session_id=self.session_id,
            run_id=None,
            assistant_message_id=None, container=get_application_container(),
        )

    def _grant(self, kb_ids):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            role_id = conn.execute("SELECT id FROM roles WHERE name = '普通'").fetchone()[0]
            conn.execute("UPDATE roles SET all_knowledge_bases = 0 WHERE id = ?", (role_id,))
            conn.execute("DELETE FROM role_knowledge_bases WHERE role_id = ?", (role_id,))
            conn.executemany(
                "INSERT INTO role_knowledge_bases (role_id, kb_id) VALUES (?, ?)",
                [(role_id, kb_id) for kb_id in kb_ids],
            )

    def _read(self):
        history, trace = self._request(
            "plain",
            "user-password",
            [
                ("GET", f"/api/chat/sessions/{self.session_id}", None),
                ("GET", "/api/chat/traces/mixed", None),
            ],
        )
        self.assertEqual(history.status_code, 200)
        self.assertEqual(trace.status_code, 200)
        return history.json(), trace.json()

    def test_revoked_source_knowledge_base_is_hidden_from_history_and_trace(self):
        self._grant([self.tech_id])
        history, trace = self._read()

        sources = history[-1]["sources"]
        self.assertEqual([s.get("kb_id") for s in sources], [self.tech_id, None])
        self.assertEqual(sources[1]["url"], "https://example.test")
        self.assertNotIn("通用库", json.dumps(history, ensure_ascii=False))
        self.assertEqual(history[-1]["content"], "回答")

        self.assertEqual([d["kb_id"] for d in trace["candidates"]], [self.tech_id])
        self.assertEqual(trace["reranked"], [])
        self.assertEqual([s["kb_id"] for s in trace["context"]], [self.tech_id])
        self.assertNotIn("通用库", json.dumps(trace, ensure_ascii=False))

    def test_granted_source_knowledge_base_is_returned(self):
        self._grant([self.tech_id, self.common_id])
        history, trace = self._read()
        self.assertEqual(
            [s.get("kb_id") for s in history[-1]["sources"]],
            [self.tech_id, self.common_id, None],
        )
        self.assertEqual(len(trace["candidates"]), 2)
        self.assertEqual(len(trace["context"]), 2)

    def test_deleting_source_knowledge_base_is_hidden_despite_permission(self):
        self._grant([self.tech_id, self.common_id])
        app_storage.knowledge_bases.mark_deleting(self.common_id)
        history, trace = self._read()
        self.assertEqual([s.get("kb_id") for s in history[-1]["sources"]], [self.tech_id, None])
        self.assertEqual([d["kb_id"] for d in trace["candidates"]], [self.tech_id])


    def test_export_hides_cited_sources_from_unreadable_knowledge_bases(self):
        self._grant([self.tech_id])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE roles SET menu_permissions = '[\"问答对话\", \"运行监控\"]' "
                "WHERE name = '普通'"
            )
            conn.execute(
                "UPDATE chat_traces SET payload = json_set(payload, '$.cited_ids', json('[1, 2]'), "
                "'$.context[0].document_id', 'common-doc', '$.context[0].source', '通用.pdf', "
                "'$.context[1].document_id', 'tech-doc') WHERE id = 'mixed'"
            )
        chat_trace.set_feedback("mixed", self.plain_id, -1, container=get_application_container())
        (export,) = self._request(
            "plain", "user-password", [("GET", "/api/ops/chat-traces/export", None)]
        )
        self.assertEqual(export.status_code, 200)
        (question,) = [json.loads(line) for line in export.text.splitlines()]
        self.assertEqual(question["gold"], ["tech-doc"])
        self.assertNotIn("通用", export.text)

        app_storage.knowledge_bases.mark_deleting(self.tech_id)
        (export,) = self._request(
            "plain", "user-password", [("GET", "/api/ops/chat-traces/export", None)]
        )
        self.assertEqual(export.text, "")


if __name__ == "__main__":
    unittest.main()
