import asyncio
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import httpx

import api
import app_storage
from services.chat import runs as chat_runs
from services.core import config as application_config
from services.core import log_tail
from tests.in_memory_index import InMemoryChunkIndex
from tests.support import IsolatedStorageTestCase


class LogTailTest(unittest.TestCase):
    def _write(self, lines):
        handle = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".log")
        self.addCleanup(os.unlink, handle.name)
        handle.write("".join(f"{line}\n" for line in lines))
        handle.close()
        return handle.name

    def test_returns_last_lines_in_order_across_block_boundaries(self):
        lines = [f"2026-10-28 10:00:00,000 INFO m: 行 {i} " + "中" * 50 for i in range(2000)]
        path = self._write(lines)
        self.assertEqual(log_tail.tail_lines(path, 3), lines[-3:])
        self.assertEqual(log_tail.tail_lines(path, 5000)[0], lines[0])

    def test_level_filter_keeps_stack_continuation_lines(self):
        path = self._write([
            "2026-10-28 10:00:00,000 INFO m: ok",
            "2026-10-28 10:00:01,000 ERROR m: boom",
            "Traceback (most recent call last):",
            "ValueError: x",
            "2026-10-28 10:00:02,000 WARNING m: warn",
        ])
        self.assertEqual(
            log_tail.tail_lines(path, 10, "error"),
            ["2026-10-28 10:00:01,000 ERROR m: boom",
             "Traceback (most recent call last):", "ValueError: x"],
        )

    def test_empty_file_and_zero_count(self):
        path = self._write([])
        self.assertEqual(log_tail.tail_lines(path, 5), [])
        self.assertEqual(log_tail.tail_lines(path, 0), [])


class OpsApiTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
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
        recorder = chat_runs.ChatRunRecorder(1, "alice", "技术部", question="问题", status="answered")
        chat_runs.record_run(recorder)

    def _get(self, username, password, paths):
        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                login = await client.post("/api/login", json={"username": username, "password": password})
                headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
                return [await client.get(path, headers=headers) for path in paths]

        return asyncio.run(run())

    def _admin_get(self, *paths):
        return self._get("admin", application_config.initial_admin_password(), paths)

    def test_endpoints_require_the_monitoring_permission(self):
        paths = [
            "/api/ops/overview", "/api/ops/chat-runs", "/api/ops/index-jobs", "/api/ops/logs",
            "/api/ops/dependencies",
        ]
        for response in self._get("plain", "user-password", paths):
            self.assertEqual(response.status_code, 403)

    def test_model_check_requires_model_permission_but_listing_does_not(self):
        check, listing = self._get("plain", "user-password", ["/api/models/check", "/api/models"])
        self.assertEqual(check.status_code, 403)
        self.assertEqual(listing.status_code, 200)

    def test_health_probe_needs_no_login(self):
        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get("/api/health")

        response = asyncio.run(run())
        self.assertEqual((response.status_code, response.json()), (200, {"status": "ok"}))

    def _anonymous_get(self, path):
        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get(path)

        return asyncio.run(run())

    def test_ready_probe_needs_no_login_and_hides_error_details(self):
        response = self._anonymous_get("/api/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"status": "ready", "checks": {"sqlite": "ok", "milvus": "ok"}}
        )

        index = InMemoryChunkIndex()
        with patch.object(
            index, "check_ready", side_effect=ConnectionError("milvus:19530 refused")
        ), patch("api.get_chunk_index", return_value=index):
            response = self._anonymous_get("/api/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["checks"], {"sqlite": "ok", "milvus": "error"})
        self.assertNotIn("19530", response.text)

    def test_ready_probe_reports_a_broken_database(self):
        os.remove(app_storage.DB_FILE)
        os.mkdir(app_storage.DB_FILE)  # 目录无法作为数据库打开
        response = self._anonymous_get("/api/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["checks"]["sqlite"], "error")

    def test_monitor_sees_dependency_details(self):
        index = InMemoryChunkIndex()
        with patch.object(
            index, "check_ready", side_effect=ConnectionError("milvus:19530 refused")
        ), patch("routers.ops.get_chunk_index", return_value=index):
            (response,) = self._admin_get("/api/ops/dependencies")
        self.assertEqual(response.status_code, 200)
        sqlite_status, milvus_status = response.json()
        self.assertEqual((sqlite_status["name"], sqlite_status["ok"]), ("sqlite", True))
        self.assertEqual(milvus_status["name"], "milvus")
        self.assertFalse(milvus_status["ok"])
        self.assertIn("ConnectionError: milvus:19530 refused", milvus_status["detail"])

    def test_admin_reads_overview_and_runs(self):
        overview, runs, jobs = self._admin_get(
            "/api/ops/overview?days=7",
            "/api/ops/chat-runs?status=answered&limit=5",
            "/api/ops/index-jobs",
        )
        self.assertEqual(overview.json()["total"], 1)
        body = runs.json()
        self.assertEqual((body["total"], body["items"][0]["question_preview"]), (1, "问题"))
        self.assertEqual(jobs.json(), [])

    def test_invalid_filters_are_rejected(self):
        bad_status, bad_lines, bad_days = self._admin_get(
            "/api/ops/chat-runs?status=nope", "/api/ops/logs?lines=501", "/api/ops/overview?days=0"
        )
        self.assertEqual(
            [bad_status.status_code, bad_lines.status_code, bad_days.status_code], [422, 422, 422]
        )

    def test_logs_report_a_missing_file_and_read_an_existing_one(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("routers.ops.log_file_path", return_value=os.path.join(directory, "backend.log")):
                (missing,) = self._admin_get("/api/ops/logs")
                self.assertFalse(missing.json()["available"])
                with open(os.path.join(directory, "backend.log"), "w", encoding="utf-8") as handle:
                    handle.write("2026-10-28 10:00:00,000 INFO m: a\n2026-10-28 10:00:01,000 ERROR m: b\n")
                (found,) = self._admin_get("/api/ops/logs?level=ERROR")
        self.assertTrue(found.json()["available"])
        self.assertEqual(found.json()["path"], "backend.log")
        self.assertEqual(missing.json()["path"], "backend.log")
        self.assertEqual(found.json()["lines"], ["2026-10-28 10:00:01,000 ERROR m: b"])


if __name__ == "__main__":
    unittest.main()

