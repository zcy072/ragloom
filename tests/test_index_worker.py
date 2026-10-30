import asyncio
import sqlite3
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

try:
    import api
    from services.indexing.worker import IndexWorker
    from tests.support import IsolatedStorageTestCase
except ModuleNotFoundError:
    IndexWorker = None
    api = None
    IsolatedStorageTestCase = unittest.TestCase


@unittest.skipUnless(IndexWorker and api, "requires the backend Python dependencies")
class IndexWorkerTest(IsolatedStorageTestCase):
    def test_concurrent_duplicate_enqueue_dispatches_job_once(self):
        dispatch_started = threading.Event()
        release_dispatch = threading.Event()
        dispatch_finished = threading.Event()
        enqueue_barrier = threading.Barrier(12)
        processed = []
        enqueue_results = []

        def dispatch(job_id):
            processed.append(job_id)
            dispatch_started.set()
            release_dispatch.wait(timeout=1)
            dispatch_finished.set()

        worker = IndexWorker(dispatcher=dispatch, pending_job_ids=lambda: [])
        worker.start()
        try:
            def enqueue_same_job():
                enqueue_barrier.wait(timeout=1)
                enqueue_results.append(worker.enqueue("shared-job"))

            threads = [threading.Thread(target=enqueue_same_job) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=1)

            self.assertTrue(dispatch_started.wait(timeout=1))
            self.assertEqual(enqueue_results.count(True), 1)
            self.assertEqual(enqueue_results.count(False), 11)
            self.assertEqual(processed, ["shared-job"])

            release_dispatch.set()
            self.assertTrue(dispatch_finished.wait(timeout=1))
        finally:
            release_dispatch.set()
            worker.stop()

    def test_single_thread_worker_dispatches_jobs_serially(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_finished = threading.Event()
        processed = []

        def dispatch(job_id):
            processed.append(job_id)
            if job_id == "first":
                first_started.set()
                release_first.wait(timeout=1)
            else:
                second_finished.set()

        worker = IndexWorker(dispatcher=dispatch, pending_job_ids=lambda: [], worker_count=1)
        worker.start()
        try:
            self.assertTrue(worker.enqueue("first"))
            self.assertTrue(worker.enqueue("second"))
            self.assertTrue(first_started.wait(timeout=1))
            self.assertEqual(processed, ["first"])

            release_first.set()
            self.assertTrue(second_finished.wait(timeout=1))
            self.assertEqual(processed, ["first", "second"])
        finally:
            worker.stop()

    def test_worker_survives_a_failing_finish_after_a_dispatch_error(self):
        second_done = threading.Event()

        def dispatch(job_id):
            if job_id == "boom":
                raise RuntimeError("dispatch failed")
            second_done.set()

        class BrokenJobs:
            def finish(self, *args):
                raise sqlite3.OperationalError("database is locked")

        class StubContainer:
            def index_jobs(self):
                return BrokenJobs()

        worker = IndexWorker(
            dispatcher=dispatch,
            pending_job_ids=lambda: [],
            container=StubContainer(),
            worker_count=1,
        )
        worker.start()
        try:
            with self.assertLogs("services.indexing.worker", level="ERROR"):
                self.assertTrue(worker.enqueue("boom"))
                deadline = time.monotonic() + 2
                while "boom" in worker._queued_job_ids and time.monotonic() < deadline:
                    time.sleep(0.01)
            # 唯一的线程必须还活着，才能继续处理下一个任务。
            self.assertTrue(worker.enqueue("next"))
            self.assertTrue(second_done.wait(timeout=2))
        finally:
            worker.stop()

    def test_multi_thread_worker_runs_jobs_in_parallel_and_stops_cleanly(self):
        barrier = threading.Barrier(3, timeout=2)
        finished = []

        def dispatch(job_id):
            # 三个任务必须同时在执行，屏障才会放行；串行执行会超时。
            barrier.wait()
            finished.append(job_id)

        worker = IndexWorker(dispatcher=dispatch, pending_job_ids=lambda: [], worker_count=3)
        worker.start()
        try:
            for job_id in ("a", "b", "c"):
                self.assertTrue(worker.enqueue(job_id))
            deadline = threading.Event()
            for _ in range(40):
                if len(finished) == 3:
                    break
                deadline.wait(0.05)
            self.assertEqual(sorted(finished), ["a", "b", "c"])
        finally:
            worker.stop()
        self.assertFalse(any(thread.is_alive() for thread in worker._threads))

    def test_worker_thread_count_defaults_to_setting(self):
        with patch.dict("os.environ", {"INDEX_WORKER_THREADS": "2"}):
            worker = IndexWorker(dispatcher=lambda job_id: None, pending_job_ids=lambda: [])
            worker.start()
            try:
                self.assertEqual(len(worker._threads), 2)
            finally:
                worker.stop()

    def test_worker_requeues_durable_jobs_on_start(self):
        completed = threading.Event()
        processed = []

        def dispatch(job_id):
            processed.append(job_id)
            completed.set()

        worker = IndexWorker(dispatcher=dispatch, pending_job_ids=lambda: ["persisted-job"])
        worker.start()
        try:
            self.assertTrue(completed.wait(timeout=1))
            self.assertEqual(processed, ["persisted-job"])
        finally:
            worker.stop()

    def test_api_lifespan_starts_and_stops_the_worker(self):
        async def exercise_lifespan():
            with patch.object(api.index_worker, "start") as start, patch.object(
                api.index_worker, "stop"
            ) as stop, patch.object(
                api.retriever, "start_retriever_cache_warmup"
            ) as warmup, patch.object(
                api.chat_ranking, "start_rerank_client"
            ) as start_rerank, patch.object(
                api.chat_ranking, "close_rerank_client", new=AsyncMock()
            ) as close_rerank:
                async with api.lifespan(api.app):
                    start.assert_called_once_with()
                    start_rerank.assert_called_once_with()
                    warmup.assert_called_once_with()
                stop.assert_called_once_with()
                close_rerank.assert_awaited_once_with()

        asyncio.run(exercise_lifespan())

    def test_periodic_history_pruning_runs_and_survives_failures(self):
        calls = []

        def prune():
            calls.append(1)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")

        async def exercise():
            with patch.object(api.app_storage, "prune_expired_history", side_effect=prune):
                task = asyncio.create_task(api._prune_history_periodically(0.01))
                for _ in range(200):
                    if len(calls) >= 2:
                        break
                    await asyncio.sleep(0.01)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

        with self.assertLogs("api", level="ERROR"):
            asyncio.run(exercise())

        self.assertGreaterEqual(len(calls), 2)  # 第一次失败后仍会继续下一轮


if __name__ == "__main__":
    unittest.main()
