import unittest
from unittest.mock import MagicMock, patch

try:
    from services.core import sqlite_connections as application_db
except ModuleNotFoundError:
    application_db = None


@unittest.skipUnless(application_db, "requires the SQLite connection module")
class DatabaseConnectionTest(unittest.TestCase):
    def test_fts5_connection_closes_without_changing_journal_mode(self):
        connection = MagicMock()
        with patch.object(
            application_db.sqlite3, "connect", return_value=connection
        ) as connect:
            with application_db.connect_fts5("fts5.db") as opened:
                self.assertIs(opened, connection)

        connect.assert_called_once_with(
            "fts5.db", timeout=application_db.FTS5_LOCK_TIMEOUT_SECONDS
        )
        connection.execute.assert_not_called()
        connection.close.assert_called_once_with()

    def test_fts5_connection_closes_after_an_exception(self):
        connection = MagicMock()
        connection.__exit__.return_value = False
        with patch.object(
            application_db.sqlite3, "connect", return_value=connection
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with application_db.connect_fts5("fts5.db"):
                    raise RuntimeError("failed")

        connection.close.assert_called_once_with()
