import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.core import config as application_config
from services.core import database_schema, model_api_keys
from services.documents.parser_modes import DEFAULT_PDF_PARSER_MODE


class ModelApiKeyEncryptionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db_file = Path(self.directory.name) / "app.sqlite3"

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def _create_schema(path: Path) -> None:
        """Build the real application schema so models_config cannot drift from production."""
        database_schema.initialize_schema(
            str(path),
            str(path.parent),
            default_chunk_size=application_config.DEFAULT_CHUNK_SIZE,
            default_chunk_overlap=application_config.DEFAULT_CHUNK_OVERLAP,
            default_parser_mode=DEFAULT_PDF_PARSER_MODE,
        )

    @classmethod
    def _create_models_database(cls, path: Path, api_key: str) -> None:
        cls._create_schema(path)
        with sqlite3.connect(path) as conn:
            conn.execute(
                "INSERT INTO models_config VALUES (?, ?, ?, ?, ?)",
                ("Provider", api_key, "https://example.test/v1", "model", 1),
            )

    @staticmethod
    def _stored_key(path: Path) -> str:
        with sqlite3.connect(path) as conn:
            return conn.execute(
                "SELECT api_key FROM models_config WHERE provider_name = 'Provider'"
            ).fetchone()[0]

    def test_rotation_uses_previous_key_then_reencrypts_with_current_key(self):
        old_key = "old-key-" + "b" * 32
        new_key = "new-key-" + "c" * 32
        with patch.dict(
            os.environ,
            {model_api_keys.ENCRYPTION_KEY_ENV: old_key},
            clear=True,
        ):
            old_ciphertext = model_api_keys.encrypt_api_key("rotate-me")
        self._create_models_database(self.db_file, old_ciphertext)

        with patch.dict(
            os.environ,
            {
                model_api_keys.ENCRYPTION_KEY_ENV: new_key,
                model_api_keys.PREVIOUS_KEYS_ENV: old_key,
            },
            clear=True,
        ):
            self.assertEqual(model_api_keys.rotate_database_api_keys(self.db_file), 1)
            new_ciphertext = self._stored_key(self.db_file)
            self.assertNotEqual(new_ciphertext, old_ciphertext)
            self.assertEqual(model_api_keys.decrypt_api_key(new_ciphertext), "rotate-me")

        with patch.dict(
            os.environ,
            {model_api_keys.ENCRYPTION_KEY_ENV: new_key},
            clear=True,
        ):
            self.assertEqual(model_api_keys.decrypt_api_key(new_ciphertext), "rotate-me")

    def test_rotation_requires_a_strong_configured_key(self):
        with patch.dict(
            os.environ,
            {model_api_keys.ENCRYPTION_KEY_ENV: "old-key-" + "b" * 32},
            clear=True,
        ):
            ciphertext = model_api_keys.encrypt_api_key("secret")
        self._create_models_database(self.db_file, ciphertext)

        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            model_api_keys.ModelApiKeyEncryptionError,
            model_api_keys.ENCRYPTION_KEY_ENV,
        ):
            model_api_keys.rotate_database_api_keys(self.db_file)

        with patch.dict(
            os.environ,
            {model_api_keys.ENCRYPTION_KEY_ENV: "too-short"},
            clear=True,
        ), self.assertRaisesRegex(
            model_api_keys.ModelApiKeyEncryptionError,
            "at least 32 characters",
        ):
            model_api_keys.rotate_database_api_keys(self.db_file)

    def test_write_ahead_log_retains_no_old_ciphertext_after_rotation(self):
        # VACUUM with secure_delete only scrubs the main database file. In WAL mode the
        # pre-rotation pages also sit in the -wal sidecar, and SQLite discards that only
        # when the last connection closes. A live server always holds connections open, so
        # without an explicit checkpoint the old ciphertext stays readable in storage/.
        old_key = "old-key-" + "b" * 32
        new_key = "new-key-" + "c" * 32
        with patch.dict(
            os.environ, {model_api_keys.ENCRYPTION_KEY_ENV: old_key}, clear=True
        ):
            old_ciphertext = model_api_keys.encrypt_api_key("sk-canary-9f3a2b")
        self._create_schema(self.db_file)
        writer = sqlite3.connect(self.db_file)
        self.addCleanup(writer.close)
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute(
            "INSERT INTO models_config VALUES (?, ?, ?, ?, ?)",
            ("Provider", old_ciphertext, "https://example.test/v1", "model", 1),
        )
        writer.commit()

        with patch.dict(
            os.environ,
            {
                model_api_keys.ENCRYPTION_KEY_ENV: new_key,
                model_api_keys.PREVIOUS_KEYS_ENV: old_key,
            },
            clear=True,
        ):
            self.assertEqual(model_api_keys.rotate_database_api_keys(self.db_file), 1)

        # The writer is deliberately still open here, mirroring a running server.
        leaking = [
            candidate.name
            for candidate in sorted(self.db_file.parent.iterdir())
            if candidate.is_file() and old_ciphertext.encode("utf-8") in candidate.read_bytes()
        ]
        self.assertEqual(leaking, [], f"old ciphertext found on disk in {leaking}")

    def test_local_placeholder_is_left_untouched_without_an_encryption_key(self):
        self._create_models_database(self.db_file, "ollama")

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(model_api_keys.rotate_database_api_keys(self.db_file), 0)
            self.assertEqual(self._stored_key(self.db_file), "ollama")


if __name__ == "__main__":
    unittest.main()
