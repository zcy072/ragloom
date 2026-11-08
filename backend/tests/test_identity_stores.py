import sqlite3
import unittest

import app_storage
from services.core import config as application_config
from services.identity import accounts, roles
from services.identity.pagination import resolve_page_window
from tests.support import IsolatedStorageTestCase


class IdentityStoresTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        app_storage.bootstrap_defaults()

    def test_user_lifecycle_and_role_assignment(self):
        created = accounts.register_user(
            app_storage.DB_FILE,
            application_config.db_now,
            username="zhangsan",
            name="张三",
            phone="13800000000",
            password=application_config.hash_password("old-password"),
        )
        duplicate = accounts.register_user(
            app_storage.DB_FILE,
            application_config.db_now,
            username="zhangsan",
            name="另一个用户",
            phone="13900000000",
            password=application_config.hash_password("other-password"),
        )
        self.assertTrue(created)
        self.assertFalse(duplicate)

        roles.create_role(
            app_storage.DB_FILE, application_config.db_now, "研发成员"
        )
        role = next(
            role
            for role in roles.list_roles_page(
                app_storage.DB_FILE, page=1, page_size=100
            )["items"]
            if role["name"] == "研发成员"
        )
        user = next(
            item
            for item in accounts.list_users_page(
                app_storage.DB_FILE, page=1, page_size=100
            )["items"]
            if item["username"] == "zhangsan"
        )
        self.assertNotIn("email", user)
        accounts.update_user_role(app_storage.DB_FILE, user["id"], role["id"])
        accounts.update_user_status(app_storage.DB_FILE, user["id"], "正常")

        stored = accounts.find_account_by_username(app_storage.DB_FILE, "zhangsan")
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["role"], "研发成员")
        self.assertEqual(stored["status"], "正常")
        self.assertEqual(stored["knowledge_base_permissions"], [])
        self.assertEqual(stored["menu_permissions"], [])
        self.assertTrue(
            accounts.change_password(
                app_storage.DB_FILE,
                user["id"],
                "old-password",
                "new-password",
            )
        )
        self.assertFalse(
            accounts.change_password(
                app_storage.DB_FILE,
                user["id"],
                "old-password",
                "another-password",
            )
        )
        with self.assertRaisesRegex(ValueError, "仍有用户使用"):
            roles.delete_role(app_storage.DB_FILE, role["id"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO chat_runs (created_at, account_id, username, kb_name, session_id, status) "
                "VALUES ('2026-01-01', ?, 'u', 'kb', 's', 'success')",
                (user["id"],),
            )
        accounts.delete_user(app_storage.DB_FILE, user["id"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM chat_runs WHERE account_id = ?", (user["id"],)
                ).fetchone()[0],
                0,
            )
        roles.delete_role(app_storage.DB_FILE, role["id"])
        self.assertNotIn(
            "研发成员",
            {
                item["name"]
                for item in roles.list_roles_page(
                    app_storage.DB_FILE, page=1, page_size=100
                )["items"]
            },
        )

    def test_default_administrator_is_a_regular_role_account(self):
        account = accounts.find_account_by_username(
            app_storage.DB_FILE, application_config.INITIAL_ADMIN_USERNAME
        )
        self.assertIsNotNone(account)
        assert account is not None
        self.assertEqual(account["role"], "超级管理员")
        self.assertEqual(account["status"], "正常")
        self.assertIn("用户管理", account["menu_permissions"])

    def test_user_and_role_pages_filter_and_clamp_to_the_last_page(self):
        for number in range(25):
            self.assertTrue(
                accounts.register_user(
                    app_storage.DB_FILE,
                    application_config.db_now,
                    username=f"member-{number:02d}",
                    name=f"成员 {number:02d}",
                    phone="",
                    password=application_config.hash_password("password"),
                )
            )
            roles.create_role(
                app_storage.DB_FILE,
                application_config.db_now,
                f"项目角色 {number:02d}",
                f"项目成员 {number:02d} 的角色",
            )

        users = accounts.list_users_page(
            app_storage.DB_FILE, page=99, page_size=10
        )
        self.assertEqual(users["total"], 26)
        self.assertEqual(users["total_pages"], 3)
        self.assertEqual(users["page"], 3)
        self.assertEqual(len(users["items"]), 6)

        matched_users = accounts.list_users_page(
            app_storage.DB_FILE, page=1, page_size=10, query="member-2"
        )
        self.assertEqual(matched_users["total"], 5)
        self.assertTrue(
            all(item["username"].startswith("member-2") for item in matched_users["items"])
        )

        role_page = roles.list_roles_page(
            app_storage.DB_FILE, page=1, page_size=10, query="项目成员 2"
        )
        self.assertEqual(role_page["total"], 5)
        self.assertEqual(role_page["page"], 1)
        self.assertEqual(len(role_page["items"]), 5)

        first_role_page = roles.list_roles_page(
            app_storage.DB_FILE, page=1, page_size=10
        )
        self.assertEqual(first_role_page["items"][0]["name"], "超级管理员")

        last_role_page = roles.list_roles_page(
            app_storage.DB_FILE, page=99, page_size=10
        )
        self.assertEqual(last_role_page["total"], 26)
        self.assertEqual(last_role_page["total_pages"], 3)
        self.assertEqual(last_role_page["page"], 3)
        self.assertEqual(len(last_role_page["items"]), 6)

    def test_page_window_rejects_non_positive_service_arguments(self):
        with self.assertRaisesRegex(ValueError, "page 必须大于 0"):
            resolve_page_window(total=0, page=0, page_size=10)
        with self.assertRaisesRegex(ValueError, "page_size 必须大于 0"):
            resolve_page_window(total=0, page=1, page_size=0)


if __name__ == "__main__":
    unittest.main()
