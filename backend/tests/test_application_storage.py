import datetime
import importlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from services.core import config as application_config
from services.core import database_schema
from services.identity import permissions as identity_permissions
from tests.support import TEST_MODEL_API_KEY_ENCRYPTION_KEY, fixture_document_id


class ApplicationStorageTest(unittest.TestCase):
    def setUp(self):
        self.original_app_storage = sys.modules.get("app_storage")
        # 启动会校验主密钥，自建存储的用例同样需要合法的测试密钥。
        environment = patch.dict(
            os.environ, {"MODEL_API_KEY_ENCRYPTION_KEY": TEST_MODEL_API_KEY_ENCRYPTION_KEY}
        )
        environment.start()
        self.addCleanup(environment.stop)

    def tearDown(self):
        if self.original_app_storage is None:
            sys.modules.pop("app_storage", None)
        else:
            sys.modules["app_storage"] = self.original_app_storage

    def load_app_storage(self, data_dir):
        sys.modules.pop("app_storage", None)
        with patch.dict(os.environ, {"DATA_DIR": data_dir}, clear=False):
            return importlib.import_module("app_storage")

    def test_database_timestamp_has_an_explicit_utc_offset(self):
        timestamp = application_config.db_now()
        parsed = datetime.datetime.fromisoformat(timestamp)

        self.assertEqual(parsed.utcoffset(), datetime.timedelta(0))

    def test_path_components_reject_traversal(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            self.assertFalse(os.path.exists(app.DB_FILE))
            for value in ("", ".", "..", "../secret", "a/b", "a\\b"):
                with self.assertRaises(ValueError):
                    application_config.validate_filename(value)
            # 路径只由 UUID kb_id 和校验后的文件名组成，显示名称不进入路径。
            for value in ("", "研发部", "../secret", "a/b"):
                with self.assertRaises(ValueError):
                    application_config.validate_kb_id(value)
            kb_id = "0b0f1a5e-3c1d-4b57-9a0e-6f7d8c9b1a2c"
            self.assertEqual(
                app.documents.stored_path(
                    kb_id, fixture_document_id(kb_id, "方案.pdf"), "方案.pdf"
                ),
                os.path.join(data_dir, kb_id, fixture_document_id(kb_id, "方案.pdf"), "方案.pdf"),
            )
            # 显示名称可以含路径字符，但不能为空、过长或含控制字符。
            self.assertEqual(application_config.validate_kb_name(" 研发/部 "), "研发/部")
            for value in ("", "  ", "a" * 81, "a\nb"):
                with self.assertRaises(ValueError):
                    application_config.validate_kb_name(value)

    def test_existing_account_password_survives_restart(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_app_storage()
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute(
                    "UPDATE accounts SET password = ? WHERE username = ?",
                    ("changed-password-hash", "admin"),
                )

            app = self.load_app_storage(data_dir)
            app.initialize_app_storage()
            with sqlite3.connect(app.DB_FILE) as conn:
                password = conn.execute(
                    "SELECT password FROM accounts WHERE username = ?",
                    ("admin",),
                ).fetchone()[0]
            self.assertEqual(password, "changed-password-hash")

    def test_fresh_database_seeds_only_the_common_knowledge_base(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_app_storage()

            with sqlite3.connect(app.DB_FILE) as conn:
                rows = conn.execute(
                    "SELECT name, chunk_size, chunk_overlap FROM knowledge_bases"
                ).fetchall()

            self.assertEqual(
                rows,
                [
                    (
                        "通用知识库",
                        application_config.DEFAULT_CHUNK_SIZE,
                        application_config.DEFAULT_CHUNK_OVERLAP,
                    )
                ],
            )

    def test_common_knowledge_base_is_ensured_by_system_key_next_to_existing_bases(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            app.knowledge_bases.create(
                "自定义业务库",
                application_config.DEFAULT_CHUNK_SIZE,
                application_config.DEFAULT_CHUNK_OVERLAP,
            )

            app.bootstrap_defaults()
            app.bootstrap_defaults()

            with sqlite3.connect(app.DB_FILE) as conn:
                rows = conn.execute(
                    "SELECT name, system_key FROM knowledge_bases ORDER BY name"
                ).fetchall()
            self.assertEqual(rows, [("自定义业务库", None), ("通用知识库", "common")])

    def test_default_common_name_taken_by_a_regular_base_fails_startup(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            app.knowledge_bases.create(
                "通用知识库",
                application_config.DEFAULT_CHUNK_SIZE,
                application_config.DEFAULT_CHUNK_OVERLAP,
            )

            with self.assertRaisesRegex(RuntimeError, "已被普通知识库占用"):
                app.bootstrap_defaults()

    def test_unset_initial_password_is_random_and_written_to_a_private_file(self):
        from services.core import bootstrap

        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            with (
                patch.dict(os.environ, {"INITIAL_ADMIN_PASSWORD": ""}),
                self.assertLogs("services.core.bootstrap", "WARNING") as logs,
            ):
                app.bootstrap_defaults()

            path = os.path.join(data_dir, bootstrap.INITIAL_ADMIN_PASSWORD_FILENAME)
            with open(path, encoding="utf-8") as handle:
                password = handle.read().strip()
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertGreaterEqual(len(password), 16)
            self.assertNotIn(password, "\n".join(logs.output))
            with sqlite3.connect(app.DB_FILE) as conn:
                stored = conn.execute(
                    "SELECT password FROM accounts WHERE username = ?",
                    (application_config.INITIAL_ADMIN_USERNAME,),
                ).fetchone()[0]
            self.assertTrue(application_config.verify_password(password, stored))
            self.assertFalse(application_config.verify_password("admin888", stored))

            # 已有账号时不再生成或覆盖密码文件。
            os.remove(path)
            with patch.dict(os.environ, {"INITIAL_ADMIN_PASSWORD": ""}):
                app.bootstrap_defaults()
            self.assertFalse(os.path.exists(path))

    def test_fresh_database_uses_current_schema(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            with sqlite3.connect(app.DB_FILE) as conn:
                schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
                account_columns = {row[1] for row in conn.execute("PRAGMA table_info(accounts)")}
                document_columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
                session_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)")
                }
                tables = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                }
                indexes = {row[1] for row in conn.execute("PRAGMA index_list(documents)")}
                session_indexes = {
                    row[1] for row in conn.execute("PRAGMA index_list(chat_sessions)")
                }
            self.assertEqual(schema_version, database_schema.CURRENT_DATABASE_SCHEMA_VERSION)
            self.assertEqual(
                account_columns,
                {
                    "id",
                    "username",
                    "name",
                    "phone",
                    "password",
                    "status",
                    "role_id",
                    "created_at",
                },
            )
            self.assertTrue(
                {
                    "error_message",
                    "index_job_id",
                    "document_type",
                    "tags",
                    "effective_date",
                    "parser_mode",
                }.issubset(document_columns)
            )
            self.assertIn("kb_id", document_columns)
            self.assertNotIn("kb_name", document_columns)
            self.assertNotIn("path", document_columns)
            self.assertIn("kb_id", session_columns)
            self.assertNotIn("kb_name", session_columns)
            self.assertNotIn("title", document_columns)
            self.assertNotIn("owner_type", session_columns)
            self.assertIn("account_id", session_columns)
            self.assertTrue(
                {
                    "knowledge_bases",
                    "accounts",
                    "chat_session_memory",
                }.issubset(tables)
            )
            self.assertNotIn("app_settings", tables)
            self.assertIn("idx_documents_kb_status", indexes)
            self.assertIn("idx_chat_sessions_account_updated", session_indexes)

    def test_current_schema_does_not_rewrite_persisted_data(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            kb_id = app.knowledge_bases.create("current", 800, 150).kb_id
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute(
                    """
                    INSERT INTO documents
                        (document_id, kb_id, filename, size, uploaded_at, status)
                    VALUES (?, ?, 'current.txt', 0, '', 'success')
                    """,
                    (
                        fixture_document_id(kb_id, "current.txt"),
                        kb_id,
                    ),
                )

            app.initialize_database()

            with sqlite3.connect(app.DB_FILE) as conn:
                status = conn.execute(
                    "SELECT status FROM documents WHERE filename = 'current.txt'"
                ).fetchone()[0]
            self.assertEqual(status, "success")

    def test_future_database_schema_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            future_version = database_schema.CURRENT_DATABASE_SCHEMA_VERSION + 1
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute("CREATE TABLE future_marker (value TEXT NOT NULL)")
                conn.execute("INSERT INTO future_marker VALUES ('preserved')")
                conn.execute(f"PRAGMA user_version = {future_version}")

            with self.assertRaisesRegex(RuntimeError, "is not supported"):
                app.initialize_database()

            with sqlite3.connect(app.DB_FILE) as conn:
                schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
                marker = conn.execute("SELECT value FROM future_marker").fetchone()[0]
            self.assertEqual(schema_version, future_version)
            self.assertEqual(marker, "preserved")

    def test_nonempty_unversioned_database_is_rejected(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute("CREATE TABLE unknown_schema (value TEXT NOT NULL)")
                conn.execute("INSERT INTO unknown_schema VALUES ('preserved')")

            with self.assertRaisesRegex(RuntimeError, "is not supported"):
                app.initialize_database()

            with sqlite3.connect(app.DB_FILE) as conn:
                marker = conn.execute("SELECT value FROM unknown_schema").fetchone()[0]
            self.assertEqual(marker, "preserved")

    def test_older_database_schema_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute("CREATE TABLE version_marker (value TEXT NOT NULL)")
                conn.execute("INSERT INTO version_marker VALUES ('preserved')")
                conn.execute("PRAGMA user_version = 1")

            with self.assertRaisesRegex(RuntimeError, "is not supported"):
                app.initialize_database()

            with sqlite3.connect(app.DB_FILE) as conn:
                schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
                marker = conn.execute("SELECT value FROM version_marker").fetchone()[0]
            self.assertEqual(schema_version, 1)
            self.assertEqual(marker, "preserved")

    def test_system_role_keeps_all_default_permissions_after_restart(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_app_storage()
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute(
                    "UPDATE roles SET menu_permissions = ?, all_knowledge_bases = 0 WHERE name = ?",
                    (json.dumps(["问答对话"]), "超级管理员"),
                )

            app = self.load_app_storage(data_dir)
            app.initialize_app_storage()
            with sqlite3.connect(app.DB_FILE) as conn:
                menus, all_knowledge_bases = conn.execute(
                    "SELECT menu_permissions, all_knowledge_bases FROM roles WHERE name = ?",
                    ("超级管理员",),
                ).fetchone()
            self.assertTrue(set(identity_permissions.MENU_PERMISSIONS).issubset(json.loads(menus)))
            self.assertEqual(all_knowledge_bases, 1)


if __name__ == "__main__":
    unittest.main()
