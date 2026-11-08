import logging
import os
import tempfile
import unittest
from dataclasses import replace
from logging.handlers import RotatingFileHandler

from services.core import logging_setup
from services.core.settings import ApplicationSettings


def _settings(log_dir, backups=3, max_bytes=65536):
    return replace(
        ApplicationSettings.from_env(),
        log_dir=log_dir,
        log_file_backups=backups,
        log_file_max_bytes=max_bytes,
    )


class LoggingSetupTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self._remove_handlers)

    def _remove_handlers(self):
        for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
            logging_setup._remove_file_handlers(logging.getLogger(name))

    def _file_handlers(self, name=""):
        return [
            h for h in logging.getLogger(name).handlers
            if getattr(h, logging_setup._FILE_HANDLER_MARKER, False)
        ]

    def test_file_handler_uses_rotation_settings_and_writes_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            logging_setup.configure_logging(_settings(directory, backups=4, max_bytes=70000))
            (handler,) = self._file_handlers()
            self.assertIsInstance(handler, RotatingFileHandler)
            self.assertEqual((handler.backupCount, handler.maxBytes), (4, 70000))
            logging.getLogger("services.test").error("落盘测试")
            handler.flush()
            with open(os.path.join(directory, "backend.log"), encoding="utf-8") as f:
                self.assertIn("落盘测试", f.read())
            self._remove_handlers()

    def test_repeated_configuration_does_not_duplicate_handlers(self):
        with tempfile.TemporaryDirectory() as directory:
            logging_setup.configure_logging(_settings(directory))
            logging_setup.configure_logging(_settings(directory))
            self.assertEqual(len(self._file_handlers()), 1)
            self._remove_handlers()

    def test_zero_backups_disables_file_logging(self):
        with tempfile.TemporaryDirectory() as directory:
            logging_setup.configure_logging(_settings(directory, backups=0))
            self.assertEqual(self._file_handlers(), [])
            self.assertFalse(os.path.exists(os.path.join(directory, "backend.log")))

    def test_unwritable_directory_falls_back_to_stdout_only(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = os.path.join(directory, "file")
            open(blocker, "w").close()
            # 目录路径被普通文件占用，makedirs 必然失败，与只读目录同属 OSError。
            logging_setup.configure_logging(_settings(os.path.join(blocker, "logs")))
            self.assertEqual(self._file_handlers(), [])


    def test_importing_the_api_does_not_open_the_deployment_log_file(self):
        # 测试会导入 api；日志若在导入时配置，测试输出会写进 storage/logs/backend.log。
        import api  # noqa: F401

        self.assertEqual(self._file_handlers(), [])
        self.assertEqual(self._file_handlers("uvicorn.access"), [])

    def test_application_startup_configures_logging(self):
        import asyncio
        from unittest.mock import AsyncMock, patch

        import api

        async def start_and_stop():
            with patch.object(api, "configure_logging") as configure, patch.object(
                api.app_storage, "initialize_app_storage"
            ), patch.object(api, "get_chunk_index"), patch.object(
                api.index_worker, "start"
            ), patch.object(api.index_worker, "stop"), patch.object(
                api.retriever, "start_retriever_cache_warmup"
            ), patch.object(api.chat_ranking, "start_rerank_client"), patch.object(
                api.chat_ranking, "close_rerank_client", new=AsyncMock()
            ), patch.object(
                api.chat_web_search, "close_web_search_client", new=AsyncMock()
            ):
                async with api.lifespan(api.app):
                    configure.assert_called_once()

        asyncio.run(start_and_stop())


if __name__ == "__main__":
    unittest.main()
