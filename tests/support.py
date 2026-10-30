"""Shared test infrastructure for isolated application storage."""

import os
import tempfile
import unittest

import app_storage
from services.core import container as application_container
from services.identity import rate_limit
from services.retrieval import retriever
from services.retrieval.chunk_index import set_chunk_index
from tests.in_memory_index import InMemoryChunkIndex


def _reset_retrieval_caches() -> None:
    """检索器、路线和查询向量缓存都是进程级的，按知识库名为键，测试之间必须清空。"""
    retriever.clear_retriever_cache()
    retriever._query_vector_cache.clear()


class _IsolatedStorageMixin:
    def _set_up_isolated_storage(self) -> None:
        self._storage_directory = tempfile.TemporaryDirectory()
        root = self._storage_directory.name
        self._storage_environment_keys = (
            "STORAGE_DIR",
            "DATA_DIR",
            "DB_FILE",
            "MINERU_RESULTS_DIR",
            "INITIAL_ADMIN_PASSWORD",
            "LOG_FILE_BACKUPS",
        )
        self._original_storage_environment = {
            key: os.environ.get(key) for key in self._storage_environment_keys
        }
        self._original_storage_paths = (
            app_storage.STORAGE_DIR,
            app_storage.DATA_DIR,
            app_storage.DB_FILE,
        )

        app_storage.STORAGE_DIR = os.path.join(root, "storage")
        app_storage.DATA_DIR = os.path.join(app_storage.STORAGE_DIR, "data")
        app_storage.DB_FILE = os.path.join(app_storage.DATA_DIR, "app_data.sqlite3")
        set_chunk_index(InMemoryChunkIndex())
        _reset_retrieval_caches()
        application_container._default_container = None
        rate_limit.reset_auth_rate_limiter()
        os.environ.update(
            {
                "STORAGE_DIR": app_storage.STORAGE_DIR,
                "DATA_DIR": app_storage.DATA_DIR,
                "DB_FILE": app_storage.DB_FILE,
                "MINERU_RESULTS_DIR": os.path.join(
                    app_storage.STORAGE_DIR, "mineru_results"
                ),
                "INITIAL_ADMIN_PASSWORD": "test-admin-password",
                # 应用启动会配置日志；测试只输出到标准输出，不写文件日志。
                "LOG_FILE_BACKUPS": "0",
            }
        )
        os.makedirs(app_storage.DATA_DIR, exist_ok=True)
        self.addCleanup(self._tear_down_isolated_storage)

    def _tear_down_isolated_storage(self) -> None:
        (
            app_storage.STORAGE_DIR,
            app_storage.DATA_DIR,
            app_storage.DB_FILE,
        ) = self._original_storage_paths
        set_chunk_index(None)
        _reset_retrieval_caches()
        for key, value in self._original_storage_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        application_container._default_container = None
        rate_limit.reset_auth_rate_limiter()
        self._storage_directory.cleanup()


class IsolatedStorageTestCase(_IsolatedStorageMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._set_up_isolated_storage()


class IsolatedStorageAsyncTestCase(
    _IsolatedStorageMixin, unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        self._set_up_isolated_storage()
