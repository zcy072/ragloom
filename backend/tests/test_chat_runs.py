import asyncio
import json
import sqlite3
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import app_storage
from services.chat import query as chat_query
from services.chat import runs as chat_runs
from services.chat import service as chat_service
from services.chat import sessions as chat_sessions
from services.core import llm_config
from services.core.container import get_application_container
from services.retrieval import context as context_builder
from tests.support import IsolatedStorageTestCase, add_knowledge_bases

MODEL_CONFIG = {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"}


def _span():
    return context_builder.ContextSpan(
        source="手册.pdf", content="上下文", pages=("0",), rank=0
    )


class ChatRunRecordingTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.owner = chat_service.SessionOwner(1, "alice")
        self.tech_id = add_knowledge_bases("技术部")[0]
        app_storage.knowledge_base_settings.set_query_rewrite_enabled(self.tech_id, False)

    def _run_chat(self, model, spans, question="这是一个很长的问题 " * 30):
        request = chat_service.ChatRequest(question=question, kb_id=self.tech_id)

        async def consume():
            stream = await chat_service.chat(request, self.owner, container=get_application_container())
            async for _ in stream.events:
                pass

        with patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=spans
        ), patch.object(llm_config, "create_answer_llm", return_value=model):
            asyncio.run(consume())

    def _rows(self):
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute("SELECT * FROM chat_runs ORDER BY id")]

    def test_answered_run_records_timings_and_truncated_question(self):
        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="结论。[来源 1]")

        self._run_chat(Model(), [_span()])
        (row,) = self._rows()
        self.assertEqual(row["status"], "answered")
        self.assertEqual(row["username"], "alice")
        self.assertEqual(row["kb_id"], self.tech_id)
        self.assertLessEqual(len(row["question_preview"]), 200)
        self.assertEqual(row["cited_sources"], 1)
        self.assertGreater(row["answer_chars"], 0)
        self.assertIsNotNone(row["first_token_ms"])
        self.assertIsNotNone(row["retrieval_ms"])
        self.assertGreaterEqual(row["total_ms"], row["generation_ms"])
        self.assertTrue(row["session_id"])
        self.assertIsNotNone(row["user_message_id"])

    def test_no_evidence_run(self):
        self._run_chat(object(), [])
        (row,) = self._rows()
        self.assertEqual(row["status"], "no_evidence")
        self.assertIsNone(row["first_token_ms"])

    def test_generation_failure_records_error_summary_without_leaking_long_text(self):
        class Model:
            async def astream(self, _messages):
                raise RuntimeError("upstream said " + "x" * 500)
                yield

        self._run_chat(Model(), [_span()])
        (row,) = self._rows()
        self.assertEqual(row["status"], "error")
        # 只记录异常类型，上游返回的原文一个字也不进入记录。
        self.assertEqual(row["error_summary"], "RuntimeError")
        self.assertNotIn("upstream said", row["error_summary"])

    def test_upstream_exception_text_never_leaves_the_model_boundary(self):
        canary = "canary-" + uuid.uuid4().hex

        class UpstreamError(Exception):
            status_code = 401
            request_id = "req_123"

        class Model:
            async def astream(self, _messages):
                raise UpstreamError(f"Incorrect API key sk-{canary} provided")
                yield

        events = []
        request = chat_service.ChatRequest(question="问题", kb_id=self.tech_id)

        async def consume():
            stream = await chat_service.chat(request, self.owner, container=get_application_container())
            async for event in stream.events:
                events.append(event)

        with self.assertLogs(level="DEBUG") as logs, patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[_span()]
        ), patch.object(llm_config, "create_answer_llm", return_value=Model()):
            asyncio.run(consume())

        (row,) = self._rows()
        self.assertEqual(row["status"], "error")
        self.assertEqual(
            row["error_summary"], "UpstreamError (HTTP 401, request_id=req_123)"
        )
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            stored = "\n".join(
                str(value)
                for table in ("chat_runs", "chat_traces", "chat_messages")
                for record in conn.execute(f"SELECT * FROM {table}")
                for value in record
            )
        self.assertTrue(events)
        for label, text in (
            ("sse", "\n".join(str(event) for event in events)),
            ("database", stored),
            ("logs", "\n".join(logs.output)),
        ):
            with self.subTest(label=label):
                self.assertNotIn(canary, text)

    def _consume_events(self, model, spans, question="问题"):
        request = chat_service.ChatRequest(question=question, kb_id=self.tech_id)

        async def consume():
            stream = await chat_service.chat(request, self.owner, container=get_application_container())
            return [
                json.loads(line[6:])
                async for event in stream.events
                for line in event.splitlines()
                if line.startswith("data: ")
            ]

        with patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=spans
        ), patch.object(llm_config, "create_answer_llm", return_value=model):
            return asyncio.run(consume())

    def test_empty_answer_is_a_generation_failure(self):
        """推理模型把额度用在思考上、正文为空时，不能记成回答成功。"""

        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="", response_metadata={})
                yield SimpleNamespace(content="  ", response_metadata={"finish_reason": "length"})

        events = self._consume_events(Model(), [_span()])
        terminal = events[-1]
        self.assertEqual(terminal["error"], chat_service.ANSWER_ERROR_MESSAGE)
        self.assertTrue(terminal["is_end"])
        (row,) = self._rows()
        self.assertEqual(row["status"], "error")
        self.assertIn("finish_reason=length", row["error_summary"])
        # 问题随失败撤回，空会话一并删除。
        self.assertEqual(chat_sessions.list_owned_sessions(self.owner, container=get_application_container()), [])

    def test_length_truncated_answer_is_flagged(self):
        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="很长的回答", response_metadata={})
                yield SimpleNamespace(content="", response_metadata={"finish_reason": "length"})

        events = self._consume_events(Model(), [_span()])
        terminal = events[-1]
        self.assertTrue(terminal["truncated"])
        self.assertNotIn("interrupted", terminal)
        self.assertEqual(terminal["final_answer"], "很长的回答")
        (row,) = self._rows()
        self.assertEqual(row["status"], "answered")

    def test_unstarted_stream_is_recorded_as_cancelled_and_keeps_the_question(self):
        """客户端在首包前断开、回答流一次都没被迭代时，由 abandon 补写运行记录。"""
        request = chat_service.ChatRequest(question="没等到回答", kb_id=self.tech_id)

        async def abandon_without_iterating():
            stream = await chat_service.chat(request, self.owner, container=get_application_container())
            await stream.abandon()
            await stream.abandon()
            await stream.events.aclose()

        with patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[_span()]
        ):
            asyncio.run(abandon_without_iterating())

        (row,) = self._rows()
        self.assertEqual(row["status"], "cancelled")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM chat_traces").fetchone()[0], 1)
            self.assertEqual(
                conn.execute("SELECT role, content FROM chat_messages").fetchall(),
                [("user", "没等到回答")],
            )

    def test_abandon_after_the_stream_started_writes_nothing_extra(self):
        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="结论。[来源 1]")

        request = chat_service.ChatRequest(question="问题", kb_id=self.tech_id)

        async def consume_then_abandon():
            stream = await chat_service.chat(request, self.owner, container=get_application_container())
            async for _ in stream.events:
                pass
            await stream.abandon()

        with patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[_span()]
        ), patch.object(llm_config, "create_answer_llm", return_value=Model()):
            asyncio.run(consume_then_abandon())

        (row,) = self._rows()
        self.assertEqual(row["status"], "answered")

    def test_cancellation_is_recorded_as_cancelled(self):
        class Model:
            async def astream(self, _messages):
                raise asyncio.CancelledError()
                yield

        with self.assertRaises(asyncio.CancelledError):
            self._run_chat(Model(), [_span()])
        (row,) = self._rows()
        self.assertEqual(row["status"], "cancelled")

    def test_pre_stream_failure_is_recorded_as_error(self):
        request = chat_service.ChatRequest(question="会失败", kb_id=self.tech_id)
        with patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_service,
            "retrieve_candidates_with_query_fallback",
            new=AsyncMock(side_effect=RuntimeError("index down")),
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(chat_service.chat(request, self.owner, container=get_application_container()))
        (row,) = self._rows()
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_summary"], "RuntimeError")

    def test_validation_failures_before_the_stream_are_recorded(self):
        missing_kb = chat_service.ChatRequest(question="问题", kb_id="00000000-0000-4000-8000-000000000000")
        no_key = chat_service.ChatRequest(question="问题", kb_id=self.tech_id)

        with patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ):
            with self.assertRaises(chat_service.ResourceNotFoundError):
                asyncio.run(chat_service.chat(missing_kb, self.owner, container=get_application_container()))
        with patch.object(
            llm_config, "select_model_config", return_value={**MODEL_CONFIG, "api_key": ""}
        ):
            with self.assertRaises(chat_service.ServiceUnavailableError) as raised:
                asyncio.run(chat_service.chat(no_key, self.owner, container=get_application_container()))
        self.assertEqual(raised.exception.code, "model_not_configured")

        rows = self._rows()
        # 不存在的知识库没有可引用的 kb_id，外键使这条记录无法写入；其余失败照常记录。
        self.assertEqual([row["status"] for row in rows], ["error"])
        self.assertEqual({row["session_id"] for row in rows}, {""})
        self.assertEqual([row["kb_id"] for row in rows], [self.tech_id])

    def test_recording_failure_does_not_change_the_answer(self):
        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="正常回答")

        request = chat_service.ChatRequest(question="问题", kb_id=self.tech_id)
        events: list[str] = []

        async def consume():
            stream = await chat_service.chat(request, self.owner, container=get_application_container())
            async for event in stream.events:
                events.append(event)

        with patch.object(
            llm_config, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[_span()]
        ), patch.object(
            llm_config, "create_answer_llm", return_value=Model()
        ), patch.object(chat_runs, "record_run", side_effect=sqlite3.OperationalError("locked")):
            asyncio.run(consume())

        end = json.loads(next(e[6:] for e in events if '"is_end": true' in e))
        self.assertEqual(end["final_answer"], "正常回答")
        self.assertEqual(self._rows(), [])


class ChatRunQueryTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.kb_ids = dict(zip(("技术部", "财务部"), add_knowledge_bases("技术部", "财务部")))

    def _insert(self, status="answered", total_ms=100.0, kb="技术部", created_at=None):
        recorder = chat_runs.ChatRunRecorder(
            1, "alice", self.kb_ids[kb], question="问", status=status
        )
        recorder.session_id = "s"
        with patch.object(chat_runs.time, "perf_counter", return_value=recorder.started + total_ms / 1000):
            chat_runs.record_run(recorder, container=get_application_container())
        if created_at:
            with sqlite3.connect(app_storage.DB_FILE) as conn:
                conn.execute(
                    "UPDATE chat_runs SET created_at = ? WHERE id = (SELECT MAX(id) FROM chat_runs)",
                    (created_at,),
                )

    def test_percentiles_use_nearest_rank_and_skip_error_rows(self):
        for value in range(1, 11):
            self._insert(total_ms=value * 100.0)
        self._insert(status="error", total_ms=99999.0)
        result = chat_runs.overview(7, container=get_application_container())
        self.assertEqual(result["total"], 11)
        self.assertEqual(result["status_counts"]["error"], 1)
        self.assertAlmostEqual(result["error_rate"], 1 / 11, places=3)
        self.assertEqual(result["latency"]["total"]["p50"], 500.0)
        self.assertEqual(result["latency"]["total"]["p95"], 1000.0)
        self.assertEqual(len(result["daily"]), 1)

    def test_overview_without_runs_has_no_percentiles(self):
        result = chat_runs.overview(7, container=get_application_container())
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["error_rate"], 0)
        self.assertIsNone(result["latency"]["total"]["p95"])

    def test_list_filters_and_paginates(self):
        self._insert(total_ms=50.0)
        self._insert(status="error", total_ms=500.0, kb="财务部")
        self._insert(total_ms=900.0)
        total, items = chat_runs.list_runs(status="error", container=get_application_container())
        self.assertEqual((total, [i["kb_name"] for i in items]), (1, ["财务部"]))
        total, _ = chat_runs.list_runs(min_total_ms=400, container=get_application_container())
        self.assertEqual(total, 2)
        total, items = chat_runs.list_runs(limit=1, offset=1, container=get_application_container())
        self.assertEqual((total, len(items)), (3, 1))
        self.assertEqual(items[0]["total_ms"], 500.0)

    def test_prune_removes_only_expired_runs(self):
        self._insert()
        old = "2000-01-01T00:00:00+00:00"
        self._insert(created_at=old)
        self.assertEqual(chat_runs.prune(app_storage.DB_FILE, 90), 1)
        self.assertEqual(chat_runs.list_runs(container=get_application_container())[0], 1)


if __name__ == "__main__":
    unittest.main()
