import asyncio
import io
import logging
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import httpx

import api
from services.chat.trace import ChatTrace
from services.core import log_tail, logging_setup, request_context
from services.indexing.worker import IndexWorker
from services.retrieval.chunk_index import run_searches


class _Capture:
    """把 logging 输出按 LOG_FORMAT 格式化到内存，用来检查行内的请求 ID。"""

    def __init__(self, name="services.test_request_context"):
        request_context.install_log_record_factory()
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter(logging_setup.LOG_FORMAT))
        self.logger = logging.getLogger(name)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def close(self):
        self.logger.removeHandler(self.handler)

    def lines(self):
        return self.stream.getvalue().splitlines()


class RequestContextTest(unittest.TestCase):
    def setUp(self):
        self.capture = _Capture()
        self.addCleanup(self.capture.close)

    def test_log_line_carries_bound_id_and_dash_outside_requests(self):
        self.capture.logger.info("外面")
        token = request_context.bind_request_id("abc123")
        try:
            self.capture.logger.info("里面")
            self.assertEqual(request_context.current_request_id(), "abc123")
        finally:
            request_context.reset_request_id(token)
        outside, inside = self.capture.lines()
        self.assertIn(" INFO [-] services.test_request_context: 外面", outside)
        self.assertIn(" INFO [abc123] services.test_request_context: 里面", inside)
        self.assertIsNone(request_context.current_request_id())

    def test_level_filter_of_log_tail_still_matches_new_format(self):
        token = request_context.bind_request_id("abc123")
        try:
            self.capture.logger.warning("告警")
        finally:
            request_context.reset_request_id(token)
        (line,) = self.capture.lines()
        self.assertEqual(log_tail._HEADER.match(line).group(1), "WARNING")

    def test_installing_the_record_factory_twice_keeps_one_wrapper(self):
        factory = logging.getLogRecordFactory()
        request_context.install_log_record_factory()
        self.assertIs(logging.getLogRecordFactory(), factory)

    def test_thread_pool_calls_inherit_the_submitting_request_id(self):
        seen = []

        def call():
            seen.append(request_context.current_request_id())
            return []

        token = request_context.bind_request_id("search-1")
        try:
            run_searches([call, call, call])
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(request_context.carry_request_id(lambda _: call()), range(4)))
        finally:
            request_context.reset_request_id(token)
        self.assertEqual(seen, ["search-1"] * 7)

    def test_index_worker_binds_the_job_id_while_dispatching(self):
        seen = []
        done = threading.Event()

        def dispatcher(job_id):
            seen.append(request_context.current_request_id())
            done.set()

        worker = IndexWorker(dispatcher=dispatcher, pending_job_ids=list, worker_count=1)
        worker.start()
        self.addCleanup(worker.stop)
        worker.enqueue("job-42")
        self.assertTrue(done.wait(5))
        self.assertEqual(seen, ["job-42"])

    def test_chat_trace_reuses_the_request_id(self):
        token = request_context.bind_request_id("f" * 32)
        try:
            self.assertEqual(ChatTrace(1).trace_id, "f" * 32)
        finally:
            request_context.reset_request_id(token)
        self.assertEqual(len(ChatTrace(1).trace_id), 32)


class RequestIdMiddlewareTest(unittest.TestCase):
    def test_each_response_gets_a_fresh_request_id_and_client_value_is_ignored(self):
        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first = await client.get("/api/health", headers={"X-Request-ID": "forged"})
                second = await client.get("/api/health")
                return first, second

        first, second = asyncio.run(run())
        ids = [response.headers["x-request-id"] for response in (first, second)]
        self.assertNotIn("forged", ids)
        self.assertNotEqual(ids[0], ids[1])
        self.assertTrue(all(len(value) == 32 for value in ids))
        self.assertIsNone(request_context.current_request_id())


if __name__ == "__main__":
    unittest.main()
