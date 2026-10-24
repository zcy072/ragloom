"""Shared test infrastructure for isolated application storage."""

import os
import tempfile
import unittest

import app_storage
from services.core import container as application_container
from services.identity import rate_limit
from services.retrieval import chroma as chroma_index


class _IsolatedStorageMixin:
    def _set_up_isolated_storage(self) -> None:
        self._storage_directory = tempfile.TemporaryDirectory()
        root = self._storage_directory.name
        self._storage_environment_keys = (
            "STORAGE_DIR",
            "DATA_DIR",
            "DB_FILE",
            "VECTOR_STORE_DIR",
            "CHROMA_PATH",
            "MINERU_RESULTS_DIR",
        )
        self._original_storage_environment = {
            key: os.environ.get(key) for key in self._storage_environment_keys
        }
        self._original_storage_paths = (
            app_storage.STORAGE_DIR,
            app_storage.DATA_DIR,
            app_storage.DB_FILE,
            chroma_index.CHROMA_PATH,
            chroma_index._chroma_client,
        )

        app_storage.STORAGE_DIR = os.path.join(root, "storage")
        app_storage.DATA_DIR = os.path.join(app_storage.STORAGE_DIR, "data")
        app_storage.DB_FILE = os.path.join(app_storage.DATA_DIR, "app_data.sqlite3")
        chroma_index.CHROMA_PATH = os.path.join(app_storage.STORAGE_DIR, "chroma")
        chroma_index._chroma_client = None
        application_container._default_container = None
        rate_limit.reset_auth_rate_limiter()
        os.environ.update(
            {
                "STORAGE_DIR": app_storage.STORAGE_DIR,
                "DATA_DIR": app_storage.DATA_DIR,
                "DB_FILE": app_storage.DB_FILE,
                "VECTOR_STORE_DIR": os.path.join(
                    app_storage.STORAGE_DIR, "vector_store"
                ),
                "CHROMA_PATH": chroma_index.CHROMA_PATH,
                "MINERU_RESULTS_DIR": os.path.join(
                    app_storage.STORAGE_DIR, "mineru_results"
                ),
            }
        )
        os.makedirs(app_storage.DATA_DIR, exist_ok=True)
        self.addCleanup(self._tear_down_isolated_storage)

    def _tear_down_isolated_storage(self) -> None:
        current_client = chroma_index._chroma_client
        original_client = self._original_storage_paths[-1]
        if current_client is not None and current_client is not original_client:
            try:
                current_client.close()
            except Exception:
                pass

        (
            app_storage.STORAGE_DIR,
            app_storage.DATA_DIR,
            app_storage.DB_FILE,
            chroma_index.CHROMA_PATH,
            chroma_index._chroma_client,
        ) = self._original_storage_paths
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
