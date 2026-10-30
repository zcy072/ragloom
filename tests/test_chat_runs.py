import asyncio
import json
import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    import app_storage
    from services.chat import query as chat_query
    from services.chat import runs as chat_runs
    from services.chat import service as chat_service
    from services.core import config as application_config
    from services.retrieval import context as context_builder
except ModuleNotFoundError:
    chat_service = None

from tests.support import IsolatedStorageTestCase

MODEL_CONFIG = {"api_key": "k", "base_url": "https://example.test/v1", "model": "m"}


def _span():
    return context_builder.ContextSpan(
        source="手册.pdf", content="上下文", pages=("0",), page_unit="页", rank=0
    )


@unittest.skipUnless(chat_service, "requires the backend Python dependencies")
class ChatRunRecordingTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.owner = chat_service.SessionOwner(1, "alice")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at, query_rewrite_enabled) "
                "VALUES ('技术部', lower(hex(randomblob(16))), ?, 0)",
                (application_config.db_now(),),
            )

    def _run_chat(self, model, spans, question="这是一个很长的问题 " * 30):
        request = chat_service.ChatRequest(question=question, kb_name="技术部")

        async def consume():
            stream = await chat_service.chat(request, self.owner)
            async for _ in stream.events:
                pass

        with patch.object(
            chat_service, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=spans
        ), patch.object(chat_service, "create_answer_llm", return_value=model):
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
        self.assertEqual(row["kb_name"], "技术部")
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
        self.assertTrue(row["error_summary"].startswith("RuntimeError: upstream said"))
        self.assertLessEqual(len(row["error_summary"]), 200)

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
        request = chat_service.ChatRequest(question="会失败", kb_name="技术部")
        with patch.object(
            chat_service, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_service,
            "retrieve_candidates_with_query_fallback",
            new=AsyncMock(side_effect=RuntimeError("index down")),
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(chat_service.chat(request, self.owner))
        (row,) = self._rows()
        self.assertEqual(row["status"], "error")
        self.assertIn("index down", row["error_summary"])

    def test_validation_failures_before_the_stream_are_recorded(self):
        missing_kb = chat_service.ChatRequest(question="问题", kb_name="不存在的库")
        no_key = chat_service.ChatRequest(question="问题", kb_name="技术部")

        with patch.object(
            chat_service, "select_model_config", return_value=MODEL_CONFIG
        ):
            with self.assertRaises(chat_service.ResourceNotFoundError):
                asyncio.run(chat_service.chat(missing_kb, self.owner))
        with patch.object(
            chat_service, "select_model_config", return_value={**MODEL_CONFIG, "api_key": ""}
        ):
            with self.assertRaises(chat_service.PreconditionFailedError):
                asyncio.run(chat_service.chat(no_key, self.owner))

        rows = self._rows()
        self.assertEqual([row["status"] for row in rows], ["error", "error"])
        self.assertEqual({row["session_id"] for row in rows}, {""})
        self.assertEqual([row["kb_name"] for row in rows], ["不存在的库", "技术部"])

    def test_recording_failure_does_not_change_the_answer(self):
        class Model:
            async def astream(self, _messages):
                yield SimpleNamespace(content="正常回答")

        request = chat_service.ChatRequest(question="问题", kb_name="技术部")
        events: list[str] = []

        async def consume():
            stream = await chat_service.chat(request, self.owner)
            async for event in stream.events:
                events.append(event)

        with patch.object(
            chat_service, "select_model_config", return_value=MODEL_CONFIG
        ), patch.object(
            chat_service, "prepare_chat_image", new=AsyncMock(return_value=("", "", ""))
        ), patch.object(
            chat_query, "retrieve_rerank_candidates", new=AsyncMock(return_value=[])
        ), patch.object(chat_service, "rerank", new=AsyncMock(return_value=[])), patch.object(
            chat_service.context_builder, "build_context_spans", return_value=[_span()]
        ), patch.object(
            chat_service, "create_answer_llm", return_value=Model()
        ), patch.object(chat_runs, "record_run", side_effect=sqlite3.OperationalError("locked")):
            asyncio.run(consume())

        end = json.loads(next(e[6:] for e in events if '"is_end": true' in e))
        self.assertEqual(end["final_answer"], "正常回答")
        self.assertEqual(self._rows(), [])


@unittest.skipUnless(chat_service, "requires the backend Python dependencies")
class ChatRunQueryTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()

    def _insert(self, status="answered", total_ms=100.0, kb="技术部", created_at=None):
        recorder = chat_runs.ChatRunRecorder(1, "alice", kb, question="问", status=status)
        recorder.session_id = "s"
        with patch.object(chat_runs.time, "perf_counter", return_value=recorder.started + total_ms / 1000):
            chat_runs.record_run(recorder)
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
        result = chat_runs.overview(7)
        self.assertEqual(result["total"], 11)
        self.assertEqual(result["status_counts"]["error"], 1)
        self.assertAlmostEqual(result["error_rate"], 1 / 11, places=3)
        self.assertEqual(result["latency"]["total"]["p50"], 500.0)
        self.assertEqual(result["latency"]["total"]["p95"], 1000.0)
        self.assertEqual(len(result["daily"]), 1)

    def test_overview_without_runs_has_no_percentiles(self):
        result = chat_runs.overview(7)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["error_rate"], 0)
        self.assertIsNone(result["latency"]["total"]["p95"])

    def test_list_filters_and_paginates(self):
        self._insert(total_ms=50.0)
        self._insert(status="error", total_ms=500.0, kb="财务部")
        self._insert(total_ms=900.0)
        total, items = chat_runs.list_runs(status="error")
        self.assertEqual((total, [i["kb_name"] for i in items]), (1, ["财务部"]))
        total, _ = chat_runs.list_runs(min_total_ms=400)
        self.assertEqual(total, 2)
        total, items = chat_runs.list_runs(limit=1, offset=1)
        self.assertEqual((total, len(items)), (3, 1))
        self.assertEqual(items[0]["total_ms"], 500.0)

    def test_prune_removes_only_expired_runs(self):
        self._insert()
        old = "2000-01-01T00:00:00+00:00"
        self._insert(created_at=old)
        self.assertEqual(chat_runs.prune(90), 1)
        self.assertEqual(chat_runs.list_runs()[0], 1)


if __name__ == "__main__":
    unittest.main()
