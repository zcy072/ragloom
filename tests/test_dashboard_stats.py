import sqlite3
import unittest
from unittest.mock import patch

try:
    import app_storage
    from routers import knowledge_bases, schemas
    from services.core import sqlite_connections as application_db
    from services.identity import auth as auth_service
except ModuleNotFoundError:
    app_storage = None

from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(app_storage, "requires the SQLite application module")
class DashboardStatsTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            now = conn.execute("SELECT datetime('now')").fetchone()[0]
            conn.executemany(
                "INSERT INTO knowledge_bases (name, created_at) VALUES (?, ?)",
                [("研发知识库", now), ("财务知识库", now), ("空知识库", now)],
            )
            conn.executemany(
                """
                INSERT INTO documents
                    (kb_name, filename, path, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    ("研发知识库", "规范.pdf", "a", 1024, now, "ready"),
                    ("研发知识库", "说明.txt", "b", 2048, now, "error"),
                    ("研发知识库", "附录.PDF", "c", 512, now, "processing"),
                    ("财务知识库", "报表.docx", "d", 1024 * 1024, now, "ready"),
                    ("财务知识库", "README", "e", 0, now, "ready"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO chat_sessions
                    (id, account_id, username, kb_name, title, created_at, updated_at)
                VALUES (?, 1, 'user', ?, 'session', ?, ?)
                """,
                [("session-rd", "研发知识库", now, now), ("session-fin", "财务知识库", now, now)],
            )
            conn.executemany(
                """
                INSERT INTO chat_messages (session_id, role, content, created_at)
                VALUES (?, ?, 'content', ?)
                """,
                [
                    ("session-rd", "user", now),
                    ("session-rd", "assistant", now),
                    ("session-fin", "user", now),
                    ("session-fin", "assistant", now),
                ],
            )
        self.today = now[:10][5:]

    def _stats(self, permissions: tuple[str, ...]) -> dict:
        result = knowledge_bases.dashboard_stats(
            auth_service.Principal(
                username="viewer",
                role="user",
                menu_permissions=("仪表盘统计",),
                knowledge_base_permissions=permissions,
            )
        )
        schemas.DashboardStatsResponse.model_validate(result)
        return result

    def test_restricted_dashboard_contains_only_existing_authorized_knowledge_bases(self):
        result = self._stats(("研发知识库", "空知识库", "不存在"))

        self.assertEqual((result["kbs"], result["docs"], result["tasks"]), (2, 3, 1))
        self.assertEqual(result["size"], "3.5 KB")
        self.assertEqual([item["name"] for item in result["kb_list"]], ["研发知识库"])
        self.assertEqual(
            result["kb_list"][0],
            {
                "name": "研发知识库",
                "docs": 3,
                "size": "3.5 KB",
                "success": 1,
                "fail": 1,
                "processing": 1,
                "types": [{"ext": "PDF", "count": 2}, {"ext": "TXT", "count": 1}],
            },
        )
        self.assertEqual(
            result["pie_data"],
            [
                {"name": "成功", "value": 1},
                {"name": "失败", "value": 1},
                {"name": "处理中", "value": 1},
            ],
        )
        self.assertEqual(
            result["line_data"],
            [{"name": self.today, "会话数": 1, "回答数": 1}],
        )

    def test_company_wide_dashboard_includes_all_knowledge_bases(self):
        result = self._stats(("全公司",))

        self.assertEqual((result["kbs"], result["docs"], result["tasks"]), (3, 5, 1))
        self.assertEqual(result["size"], "1.0 MB")
        self.assertEqual(
            [item["name"] for item in result["kb_list"]],
            ["研发知识库", "财务知识库"],
        )
        self.assertEqual(
            result["pie_data"],
            [
                {"name": "成功", "value": 3},
                {"name": "失败", "value": 1},
                {"name": "处理中", "value": 1},
            ],
        )
        self.assertEqual(
            result["line_data"],
            [{"name": self.today, "会话数": 2, "回答数": 2}],
        )

    def test_dashboard_with_no_knowledge_base_permissions_is_empty(self):
        result = self._stats(())

        self.assertEqual((result["kbs"], result["docs"], result["tasks"]), (0, 0, 0))
        self.assertEqual(result["size"], "0 KB")
        self.assertEqual(result["kb_list"], [])
        self.assertEqual(result["line_data"], [])
        self.assertTrue(all(item["value"] == 0 for item in result["pie_data"]))

    def test_dashboard_uses_constant_query_count(self):
        real_connect = sqlite3.connect
        statements: list[str] = []

        def traced_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            conn.set_trace_callback(statements.append)
            return conn

        with patch.object(application_db.sqlite3, "connect", side_effect=traced_connect):
            self._stats(("研发知识库", "财务知识库"))

        selects = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
        self.assertEqual(len(selects), 2)
        self.assertFalse(any("SELECT filename FROM documents" in statement for statement in selects))

    def test_dashboard_rows_are_mapped_to_named_contracts(self):
        document = app_storage.knowledge_bases.dashboard_documents(("研发知识库",))[0]
        trend = app_storage.knowledge_bases.dashboard_trend(
            ("研发知识库",), ["研发知识库"], days=30
        )[0]

        self.assertEqual(document.kb_name, "研发知识库")
        self.assertEqual(document.filename, "规范.pdf")
        self.assertEqual(trend.date[5:], self.today)
        self.assertEqual((trend.session_count, trend.answer_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
