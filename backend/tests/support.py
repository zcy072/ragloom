"""Shared test infrastructure for isolated application storage."""

import os
import tempfile
import unittest
import unittest.mock
import uuid

import httpx

import api
import app_storage
from services.core import config as application_config
from services.core import container as application_container
from services.identity import rate_limit
from services.indexing import pipeline as indexing_pipeline
from services.retrieval import retriever
from services.retrieval.chunk_index import set_chunk_index
from tests.in_memory_index import InMemoryChunkIndex

# 测试专用的标准 Fernet 主密钥；生产部署必须用 Fernet.generate_key() 另行生成。
TEST_MODEL_API_KEY_ENCRYPTION_KEY = "dGVzdC1vbmx5LWZlcm5ldC1rZXktMzItYnl0ZXMhISE="


def fixture_document_id(kb_id: str, label: str) -> str:
    """固定夹具标签对应的显式 UUID；生产路径不调用此函数。

    一个标签表示一个已知夹具文档。测试新上传/同名共存时必须使用上传返回的随机 ID。
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragloom-test:{kb_id}:{label}"))


def create_index_job(kb_id: str, operation: str, filename: str | None = None) -> str:
    """仅供任务状态单元测试构造记录，不保留旧生产 create/attach API。"""
    from services.core import sqlite_connections as application_db
    from services.indexing.jobs import insert_job

    job_id = uuid.uuid4().hex
    document_id = fixture_document_id(kb_id, filename) if filename is not None else None
    with application_db.connect(app_storage.DB_FILE) as conn:
        insert_job(
            conn, job_id, kb_id, document_id, filename, operation, application_config.db_now()
        )
    return job_id


def bind_index_job(kb_id: str, document_id: str, job_id: str) -> bool:
    """任务 ownership 单元测试的明确 SQL 夹具。"""
    from services.core import sqlite_connections as application_db

    with application_db.connect(app_storage.DB_FILE) as conn:
        cursor = conn.execute(
            "UPDATE documents SET status = 'processing', error_message = '', index_job_id = ? "
            "WHERE kb_id = ? AND document_id = ?",
            (job_id, kb_id, document_id),
        )
    return cursor.rowcount == 1


def add_knowledge_bases(*names: str) -> list[str]:
    """通过正式存储接口创建知识库并返回各自的 kb_id；调用前需先 initialize_database。"""
    return [
        app_storage.knowledge_bases.create(
            name,
            application_config.DEFAULT_CHUNK_SIZE,
            application_config.DEFAULT_CHUNK_OVERLAP,
        ).kb_id
        for name in names
    ]


def kb_id_by_name(name: str) -> str:
    """测试里按显示名称取知识库 ID；生产代码不做这种查找。"""
    for summary in app_storage.knowledge_bases.list_summaries():
        if summary["name"] == name:
            return summary["kb_id"]
    raise AssertionError(f"测试知识库不存在：{name}")


# 所有 /api 写请求必须带的固定请求头（CSRF 防护）；它不是秘密。
WRITE_HEADERS = {"X-RAGLoom-Request": "1"}


def api_client(**kwargs) -> httpx.AsyncClient:
    """连接进程内应用的 HTTP 客户端：默认带写请求头，并像浏览器一样保存会话 Cookie。"""
    headers = {**WRITE_HEADERS, **kwargs.pop("headers", {})}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app),
        base_url="http://test",
        headers=headers,
        **kwargs,
    )


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
            "MODEL_API_KEY_ENCRYPTION_KEY",
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
        # 旧父段落默认宽限 60 秒后才删除；测试里立即删除，宽限行为单独测试。
        grace = unittest.mock.patch.object(indexing_pipeline, "OLD_PARENT_GRACE_SECONDS", 0)
        grace.start()
        self.addCleanup(grace.stop)
        os.environ.update(
            {
                "STORAGE_DIR": app_storage.STORAGE_DIR,
                "DATA_DIR": app_storage.DATA_DIR,
                "DB_FILE": app_storage.DB_FILE,
                "MINERU_RESULTS_DIR": os.path.join(app_storage.STORAGE_DIR, "mineru_results"),
                "INITIAL_ADMIN_PASSWORD": "test-admin-password",
                # 应用启动会配置日志；测试只输出到标准输出，不写文件日志。
                "LOG_FILE_BACKUPS": "0",
                "MODEL_API_KEY_ENCRYPTION_KEY": TEST_MODEL_API_KEY_ENCRYPTION_KEY,
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


class IsolatedStorageAsyncTestCase(_IsolatedStorageMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._set_up_isolated_storage()
