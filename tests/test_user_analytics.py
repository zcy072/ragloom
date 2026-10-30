import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx

import api
import app_storage
from services.analytics import user_activity
from services.chat import runs as chat_runs
from services.core import config as application_config
from services.identity import accounts, login_events
from tests.support import IsolatedStorageTestCase

ADMIN_PASSWORD = "test-admin-password"


def _ago(**delta) -> str:
    return (datetime.now(UTC) - timedelta(**delta)).isoformat(timespec="seconds")


class UserAnalyticsTestBase(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()
        self.db = app_storage.DB_FILE
        self.alice = self._account("alice", "爱丽丝", '["问答对话"]', '["技术部", "通用知识库"]')
        self.bob = self._account("bob", "鲍勃", '["问答对话"]', '["通用知识库"]')
        self.analyst = self._account(
            "analyst", "分析员", '["用户分析", "问答对话"]', '["通用知识库"]'
        )
        self.viewer = self._account(
            "viewer", "审阅员", '["用户分析", "查看用户对话", "问答对话"]', '["通用知识库"]'
        )

    def _account(self, username, name, menus, kbs, status="正常"):
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO roles (name, description, menu_permissions, "
                "knowledge_base_permissions, created_at) VALUES (?, '', ?, ?, ?)",
                (f"角色-{username}", menus, kbs, application_config.db_now()),
            )
            role_id = conn.execute(
                "SELECT id FROM roles WHERE name = ?", (f"角色-{username}",)
            ).fetchone()[0]
            cursor = conn.execute(
                "INSERT INTO accounts (username, name, phone, password, status, role_id, "
                "created_at) VALUES (?, ?, '', ?, ?, ?, ?)",
                (
                    username, name, application_config.hash_password("user-password"),
                    status, role_id, application_config.db_now(),
                ),
            )
            return int(cursor.lastrowid)

    def _run(self, account_id, username, kb, *, status="answered", created_at=None,
             session_id="s1", feedback=0):
        run_id = chat_runs.record_run(
            chat_runs.ChatRunRecorder(
                account_id, username, kb, session_id=session_id, question=f"{username} 的问题",
                status=status,
            )
        )
        with sqlite3.connect(self.db) as conn:
            if created_at:
                conn.execute("UPDATE chat_runs SET created_at = ? WHERE id = ?", (created_at, run_id))
            if feedback:
                conn.execute(
                    "INSERT INTO chat_traces (id, created_at, account_id, kb_name, session_id, "
                    "run_id, feedback, payload) VALUES (?, ?, ?, ?, ?, ?, ?, '{}')",
                    (f"t{run_id}", created_at or application_config.db_now(), account_id, kb,
                     session_id, run_id, feedback),
                )
        return run_id

    def _session(self, session_id, account_id, username, kb, messages):
        now = application_config.db_now()
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, account_id, username, kb_name, title, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, account_id, username, kb, f"{kb} 的会话", now, now),
            )
            for role, content in messages:
                sources = json.dumps([{"source": "制度.pdf"}, {"source": "制度.pdf"}]) \
                    if role == "assistant" else "[]"
                conn.execute(
                    "INSERT INTO chat_messages (session_id, role, content, sources, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (session_id, role, content, sources, now),
                )

    def _request(self, username, password, method, paths):
        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                login = await client.post(
                    "/api/login", json={"username": username, "password": password}
                )
                headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
                return [await client.request(method, path, headers=headers) for path in paths]

        return asyncio.run(run())

    def _get(self, username, *paths, password="user-password"):
        return self._request(username, password, "GET", paths)


class UserActivityAggregationTest(UserAnalyticsTestBase):
    def test_overview_counts_users_questions_feedback_and_logins(self):
        self._run(self.alice, "alice", "技术部", feedback=1)
        self._run(self.alice, "alice", "通用知识库", feedback=-1)
        self._run(self.bob, "bob", "通用知识库", status="error")
        # 超出窗口的记录不计入。
        self._run(self.bob, "bob", "通用知识库", created_at=_ago(days=40))
        login_events.record(self.db, username="alice", success=True)
        login_events.record(self.db, username="alice", success=False, reason="invalid_credentials")
        login_events.record(self.db, username="ghost", success=False, reason="invalid_credentials")

        result = user_activity.overview(self.db, days=30, visible_kbs=None)

        self.assertEqual(result["total_users"], 5)
        self.assertEqual(result["questions"], 3)
        self.assertEqual(result["active_users"], 2)
        self.assertEqual((result["upvotes"], result["downvotes"]), (1, 1))
        self.assertEqual((result["login_users"], result["login_failures"]), (1, 2))
        self.assertEqual(len(result["daily"]), 30)
        self.assertEqual(result["daily"][-1]["questions"], 3)
        self.assertEqual(sum(result["hourly"]), 3)
        self.assertEqual(
            [(kb["kb_name"], kb["users"], kb["questions"], kb["downvotes"])
             for kb in result["knowledge_bases"]],
            [("通用知识库", 2, 2, 1), ("技术部", 1, 1, 0)],
        )

    def test_restricted_caller_only_sees_questions_in_readable_knowledge_bases(self):
        self._run(self.alice, "alice", "技术部", feedback=-1)
        self._run(self.alice, "alice", "通用知识库")

        result = user_activity.overview(self.db, days=30, visible_kbs=("通用知识库",))
        self.assertEqual((result["questions"], result["downvotes"]), (1, 0))
        self.assertEqual([kb["kb_name"] for kb in result["knowledge_bases"]], ["通用知识库"])

        page = user_activity.list_users(
            self.db, days=30, visible_kbs=("通用知识库",), page=1, page_size=20
        )
        alice = next(item for item in page["items"] if item["username"] == "alice")
        self.assertEqual((alice["questions"], alice["downvotes"]), (1, 0))

        nothing = user_activity.overview(self.db, days=30, visible_kbs=())
        self.assertEqual(nothing["questions"], 0)

    def test_user_list_sorts_searches_and_reports_last_activity(self):
        self._run(self.alice, "alice", "通用知识库", session_id="a1")
        self._run(self.alice, "alice", "通用知识库", session_id="a2")
        self._run(self.bob, "bob", "通用知识库")
        # 窗口外的提问不计数，但仍是最后提问时间的来源。
        old = _ago(days=60)
        self._run(self.viewer, "viewer", "通用知识库", created_at=old)
        login_events.record(self.db, username="bob", success=True)

        page = user_activity.list_users(self.db, days=7, visible_kbs=None, page=1, page_size=20)
        self.assertEqual([item["username"] for item in page["items"][:2]], ["alice", "bob"])
        alice = page["items"][0]
        self.assertEqual((alice["questions"], alice["sessions"]), (2, 2))
        viewer = next(item for item in page["items"] if item["username"] == "viewer")
        self.assertEqual((viewer["questions"], viewer["last_question_at"]), (0, old))

        by_login = user_activity.list_users(
            self.db, days=7, visible_kbs=None, page=1, page_size=20, sort="last_login_at"
        )
        self.assertEqual(by_login["items"][0]["username"], "bob")
        # 从未登录的账号排在最后，不论升序降序。
        ascending = user_activity.list_users(
            self.db, days=7, visible_kbs=None, page=1, page_size=20,
            sort="last_login_at", descending=False,
        )
        self.assertEqual(ascending["items"][0]["username"], "bob")

        search = user_activity.list_users(
            self.db, days=7, visible_kbs=None, page=1, page_size=20, query="鲍"
        )
        self.assertEqual([item["username"] for item in search["items"]], ["bob"])
        self.assertEqual(search["total"], 1)

    def test_user_detail_includes_trend_sessions_questions_and_logins(self):
        self._session("a1", self.alice, "alice", "通用知识库", [("user", "问"), ("assistant", "答")])
        self._session("a2", self.alice, "alice", "技术部", [("user", "问")])
        self._run(self.alice, "alice", "通用知识库", session_id="a1", status="no_evidence")
        self._run(self.alice, "alice", "技术部", session_id="a2", feedback=-1)
        login_events.record(self.db, username="alice", success=True, client="10.0.0.1")

        detail = user_activity.user_detail(self.db, self.alice, days=7, visible_kbs=("通用知识库",))

        self.assertEqual(detail["user"]["questions"], 1)
        self.assertEqual(detail["no_evidence"], 1)
        self.assertEqual(len(detail["daily"]), 7)
        self.assertEqual([s["id"] for s in detail["recent_sessions"]], ["a1"])
        self.assertEqual(detail["recent_sessions"][0]["questions"], 1)
        self.assertEqual(
            [q["kb_name"] for q in detail["recent_questions"]], ["通用知识库"]
        )
        self.assertEqual(detail["recent_logins"][0]["client"], "10.0.0.1")

    def test_login_events_are_pruned_and_removed_with_the_account(self):
        login_events.record(self.db, username="bob", success=True)
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE login_events SET created_at = ?", (_ago(days=400),))
        login_events.record(self.db, username="bob", success=True)
        self.assertEqual(login_events.prune(self.db, 180), 1)

        accounts.delete_user(self.db, self.bob)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM login_events").fetchone()[0], 0)


class UserAnalyticsApiTest(UserAnalyticsTestBase):
    def test_login_attempts_are_recorded_with_reason(self):
        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post("/api/login", json={"username": "alice", "password": "wrong-pass"})
                await client.post("/api/login", json={"username": "alice", "password": "user-password"})
                await client.post("/api/login", json={"username": "nobody", "password": "whatever1"})

        asyncio.run(run())
        with sqlite3.connect(self.db) as conn:
            rows = conn.execute(
                "SELECT username, account_id, success, reason FROM login_events ORDER BY id"
            ).fetchall()
        self.assertEqual(
            rows,
            [
                ("alice", self.alice, 0, "invalid_credentials"),
                ("alice", self.alice, 1, ""),
                ("nobody", None, 0, "invalid_credentials"),
            ],
        )

    def test_pending_account_login_is_recorded_as_not_active(self):
        self._account("newbie", "新人", "[]", "[]", status="审核中")

        async def run():
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/api/login", json={"username": "newbie", "password": "user-password"}
                )

        self.assertEqual(asyncio.run(run()).status_code, 403)
        with sqlite3.connect(self.db) as conn:
            reason = conn.execute(
                "SELECT reason FROM login_events WHERE username = 'newbie'"
            ).fetchone()[0]
        self.assertEqual(reason, "account_not_active")

    def test_endpoints_require_the_analytics_permission(self):
        paths = [
            "/api/user-analytics/overview", "/api/user-analytics/users",
            f"/api/user-analytics/users/{self.alice}", "/api/user-analytics/audit-log",
        ]
        for response in self._get("alice", *paths):
            self.assertEqual(response.status_code, 403)
        for response in self._get("analyst", *paths):
            self.assertEqual(response.status_code, 200, response.text)

    def test_viewing_a_conversation_needs_its_own_permission_and_is_audited(self):
        self._session("b1", self.bob, "bob", "通用知识库", [("user", "年假几天"), ("assistant", "五天")])
        path = f"/api/user-analytics/users/{self.bob}/sessions/b1"

        (denied,) = self._get("analyst", path)
        self.assertEqual(denied.status_code, 403)

        (allowed, audit_log) = self._get("viewer", path, "/api/user-analytics/audit-log")
        self.assertEqual(allowed.status_code, 200, allowed.text)
        body = allowed.json()
        self.assertEqual([m["content"] for m in body["messages"]], ["年假几天", "五天"])
        self.assertEqual(body["messages"][1]["source_names"], ["制度.pdf"])
        record = audit_log.json()["items"][0]
        self.assertEqual(
            (record["actor_username"], record["action"], record["target_username"]),
            ("viewer", "view_conversation", "bob"),
        )
        self.assertIn("通用知识库", record["detail"])

    def test_conversation_outside_readable_knowledge_bases_is_not_found_and_not_audited(self):
        self._session("a1", self.alice, "alice", "技术部", [("user", "机密")])
        # alice 的角色含“技术部”，超出 viewer 的权限，属于越级访问。
        (over_scope,) = self._get("viewer", f"/api/user-analytics/users/{self.alice}/sessions/a1")
        self.assertEqual(over_scope.status_code, 403)

        bob_tech = self._account("carol", "卡罗", '["问答对话"]', '["通用知识库"]')
        self._session("c1", bob_tech, "carol", "技术部", [("user", "旧会话")])
        (hidden, wrong_owner) = self._get(
            "viewer",
            f"/api/user-analytics/users/{bob_tech}/sessions/c1",
            f"/api/user-analytics/users/{self.bob}/sessions/c1",
        )
        self.assertEqual(hidden.status_code, 404)
        self.assertEqual(wrong_owner.status_code, 404)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM admin_audit_log").fetchone()[0], 0
            )

    def test_super_admin_gets_new_permissions_and_can_view_any_conversation(self):
        self._session("a1", self.alice, "alice", "技术部", [("user", "机密")])
        responses = self._request(
            "admin", ADMIN_PASSWORD, "GET",
            ["/api/me", f"/api/user-analytics/users/{self.alice}/sessions/a1"],
        )
        self.assertIn("查看用户对话", responses[0].json()["menuPermissions"])
        self.assertEqual(responses[1].status_code, 200)
