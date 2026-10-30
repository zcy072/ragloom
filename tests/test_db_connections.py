import unittest
from unittest.mock import MagicMock, patch

try:
    from services.core import sqlite_connections as application_db
except ModuleNotFoundError:
    application_db = None


@unittest.skipUnless(application_db, "requires the SQLite connection module")
class DatabaseConnectionTest(unittest.TestCase):
    def test_connection_closes_after_use(self):
        connection = MagicMock()
        with patch.object(
            application_db.sqlite3, "connect", return_value=connection
        ) as connect:
            with application_db.connect("app.db") as opened:
                self.assertIs(opened, connection)

        connect.assert_called_once_with(
            "app.db", timeout=application_db.LOCK_TIMEOUT_SECONDS
        )
        connection.execute.assert_not_called()
        connection.close.assert_called_once_with()

    def test_connection_closes_after_an_exception(self):
        connection = MagicMock()
        connection.__exit__.return_value = False
        with patch.object(
            application_db.sqlite3, "connect", return_value=connection
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with application_db.connect("app.db"):
                    raise RuntimeError("failed")

        connection.close.assert_called_once_with()
