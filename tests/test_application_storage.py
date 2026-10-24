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


class ApplicationStorageTest(unittest.TestCase):
    def setUp(self):
        self.original_app_storage = sys.modules.get("app_storage")

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
                    application_config.validate_kb_name(value)
                with self.assertRaises(ValueError):
                    application_config.validate_filename(value)

            self.assertEqual(
                app.documents.document_path("研发部", "方案.pdf"),
                os.path.join(data_dir, "研发部", "方案.pdf"),
            )

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
                [(
                    "通用知识库",
                    application_config.DEFAULT_CHUNK_SIZE,
                    application_config.DEFAULT_CHUNK_OVERLAP,
                )],
            )

    def test_existing_knowledge_base_prevents_default_common_base_seed(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute(
                    "INSERT INTO knowledge_bases (name, created_at) VALUES (?, ?)",
                    ("自定义业务库", application_config.db_now()),
                )

            app.bootstrap_defaults()

            with sqlite3.connect(app.DB_FILE) as conn:
                names = [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM knowledge_bases ORDER BY name"
                    ).fetchall()
                ]
            self.assertEqual(names, ["自定义业务库"])

    def test_fresh_database_uses_current_schema(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            with sqlite3.connect(app.DB_FILE) as conn:
                schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
                account_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(accounts)")
                }
                document_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(documents)")
                }
                session_columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(chat_sessions)")
                }
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                indexes = {
                    row[1] for row in conn.execute("PRAGMA index_list(documents)")
                }
                session_indexes = {
                    row[1] for row in conn.execute("PRAGMA index_list(chat_sessions)")
                }
            self.assertEqual(schema_version, database_schema.CURRENT_DATABASE_SCHEMA_VERSION)
            self.assertEqual(
                account_columns,
                {
                    "id", "username", "name", "phone", "password",
                    "status", "role_id", "created_at",
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
            self.assertIn("kb_name", document_columns)
            self.assertNotIn("title", document_columns)
            self.assertNotIn("owner_type", session_columns)
            self.assertIn("account_id", session_columns)
            self.assertTrue(
                {
                    "knowledge_bases",
                    "accounts",
                    "chat_session_memory",
                }.issubset(
                    tables
                )
            )
            self.assertIn("idx_documents_kb_status", indexes)
            self.assertIn("idx_chat_sessions_account_updated", session_indexes)

    def test_current_schema_does_not_rewrite_persisted_data(self):
        with tempfile.TemporaryDirectory() as data_dir:
            app = self.load_app_storage(data_dir)
            app.initialize_database()
            with sqlite3.connect(app.DB_FILE) as conn:
                conn.execute(
                    """
                    INSERT INTO documents
                        (kb_name, filename, path, size, uploaded_at, status)
                    VALUES ('current', 'current.txt', 'current.txt', 0, '', 'success')
                    """
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
                    "UPDATE roles SET menu_permissions = ?, knowledge_base_permissions = ? WHERE name = ?",
                    (json.dumps(["问答对话"]), json.dumps(["技术部"]), "超级管理员"),
                )

            app = self.load_app_storage(data_dir)
            app.initialize_app_storage()
            with sqlite3.connect(app.DB_FILE) as conn:
                menus, knowledge_bases = conn.execute(
                    "SELECT menu_permissions, knowledge_base_permissions FROM roles WHERE name = ?", ("超级管理员",)
                ).fetchone()
            self.assertTrue(
                set(identity_permissions.MENU_PERMISSIONS).issubset(json.loads(menus))
            )
            self.assertIn("全公司", json.loads(knowledge_bases))


if __name__ == "__main__":
    unittest.main()
