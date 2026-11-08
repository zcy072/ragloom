"""Application storage facade and startup initialization."""

import logging
from pathlib import Path

from services.chat import runs as chat_runs
from services.core import bootstrap as application_bootstrap
from services.core import config as application_config
from services.core import database_schema, model_api_keys
from services.core.knowledge_base_settings import KnowledgeBaseSettings
from services.core.knowledge_bases import KnowledgeBaseStore
from services.core.model_providers import ModelProviderStore
from services.core.settings import get_settings
from services.documents import parser_modes
from services.documents.store import DocumentStore
from services.identity import login_events
from services.identity import permissions as identity_permissions
from services.indexing import jobs as index_job_store
from services.indexing import workspace as index_workspace

logger = logging.getLogger(__name__)

STORAGE_DIR, DATA_DIR, DB_FILE = application_config.storage_paths(
    Path(__file__).resolve().parent
)


def initialize_database() -> None:
    """创建或迁移业务数据库结构。"""
    database_schema.initialize_schema(
        DB_FILE,
        DATA_DIR,
        default_chunk_size=application_config.DEFAULT_CHUNK_SIZE,
        default_chunk_overlap=application_config.DEFAULT_CHUNK_OVERLAP,
        default_parser_mode=parser_modes.DEFAULT_PDF_PARSER_MODE,
    )


def bootstrap_defaults() -> None:
    """为空数据库写入默认账号、角色、通用知识库和模型服务商。"""
    application_bootstrap.bootstrap_defaults(
        DB_FILE,
        application_config.db_now,
        application_bootstrap.ApplicationDefaults(
            initial_admin_username=application_config.INITIAL_ADMIN_USERNAME,
            initial_admin_password=application_config.initial_admin_password(),
            role_menu_options=identity_permissions.MENU_PERMISSIONS,
            default_chunk_size=application_config.DEFAULT_CHUNK_SIZE,
            default_chunk_overlap=application_config.DEFAULT_CHUNK_OVERLAP,
            siliconflow_base_url=application_config.DEFAULT_SILICONFLOW_BASE_URL,
            siliconflow_model=application_config.DEFAULT_SILICONFLOW_MODEL,
        ),
    )


# Factories receive path callbacks instead of frozen strings. Tests and alternate deployments
# can switch DB_FILE/DATA_DIR after import without rebuilding every service object.
index_jobs = index_job_store.IndexJobStore(
    lambda: DB_FILE,
    application_config.db_now,
    application_config.validate_kb_name,
    application_config.validate_filename,
    vacuum_min_free_pages=application_config.SQLITE_VACUUM_MIN_FREE_PAGES,
    vacuum_min_free_ratio=application_config.SQLITE_VACUUM_MIN_FREE_RATIO,
    history_retention_days=lambda: get_settings().index_job_history_retention_days,
    history_max_per_kb=lambda: get_settings().index_job_history_max_per_kb,
    logger=logger,
)

documents = DocumentStore(
    lambda: DB_FILE,
    lambda: DATA_DIR,
    application_config.db_now,
    application_config.validate_kb_name,
    application_config.validate_filename,
    max_type_length=application_config.DOCUMENT_METADATA_MAX_TYPE_LENGTH,
    max_tags=application_config.DOCUMENT_METADATA_MAX_TAGS,
    max_tag_length=application_config.DOCUMENT_METADATA_MAX_TAG_LENGTH,
)

knowledge_base_settings = KnowledgeBaseSettings(
    lambda: DB_FILE,
    application_config.validate_kb_name,
)

model_providers = ModelProviderStore(lambda: DB_FILE)
knowledge_bases = KnowledgeBaseStore(
    lambda: DB_FILE,
    application_config.db_now,
    application_config.validate_kb_name,
)

def initialize_app_storage() -> None:
    """在开始处理请求前完成数据库、密钥和持久化任务的启动维护。"""
    # Validate the schema first, then reconcile durable job state before seeding defaults.
    # Filename redaction runs last because it relies on both document and job tables.
    initialize_database()
    rotated_keys = model_api_keys.rotate_database_api_keys(DB_FILE)
    if rotated_keys:
        logger.info("Re-encrypted %d stored model API keys with the current key", rotated_keys)
    index_jobs.recover_interrupted()
    # 被中断的任务已标记为 interrupted，其快照与私有结果目录不会再被使用。
    index_workspace.sweep_stale_workspaces()
    bootstrap_defaults()
    index_jobs.redact_orphaned_filenames()
    prune_expired_history()


def prune_expired_history() -> None:
    """清理超过保留期的任务历史、问答运行记录和登录记录；启动时和运行期间定时调用。

    最后尝试压缩数据库，回收删除知识库等操作留下的空闲页；空闲页不足阈值时跳过。
    """
    index_jobs.prune_history()
    chat_runs.prune(db_file=DB_FILE)
    login_events.prune(DB_FILE)
    index_jobs.compact_database()
