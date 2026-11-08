"""Typed access to environment-backed application settings."""

import os
from dataclasses import dataclass
from pathlib import Path


def _flag(name: str, default: bool) -> bool:
    """读取布尔环境变量；缺失时使用默认值。"""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(
    name: str, default: int, minimum: int | None = None, maximum: int | None = None
) -> int:
    """读取整数环境变量，并在解析失败时回退及限制取值范围。"""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


@dataclass(frozen=True)
class ApplicationSettings:
    """Validated runtime configuration loaded from the process environment."""

    log_level: str
    log_dir: str
    log_file_max_bytes: int
    log_file_backups: int
    chat_run_retention_days: int
    login_event_retention_days: int
    storage_dir: str
    data_dir: str
    db_file: str
    milvus_uri: str
    milvus_token: str
    milvus_collection_prefix: str
    embedding_dimension: int
    initial_admin_password: str
    auth_secret_key: str
    auth_token_ttl_seconds: int
    auth_login_max_failures: int
    auth_login_window_seconds: int
    auth_client_max_failures: int
    auth_client_window_seconds: int
    auth_register_max_attempts: int
    auth_register_window_seconds: int
    auth_trust_x_real_ip: bool
    siliconflow_api_key: str
    model_api_key_encryption_key: str
    model_api_key_previous_encryption_keys: str
    index_job_history_retention_days: int
    index_job_history_max_per_kb: int
    retriever_cache_max_size: int
    query_embedding_cache_size: int
    chat_retrieval_planning: bool
    chat_query_variant: bool
    chat_agentic_retrieval: bool
    web_search_daily_limit: int
    web_search_user_per_minute: int
    chat_user_max_concurrent: int
    chat_user_per_minute: int
    mineru_api_token: str
    mineru_api_base_url: str
    mineru_model_version: str
    mineru_language: str
    mineru_results_dir: str
    mineru_timeout_seconds: int
    mineru_poll_interval_seconds: int
    mineru_download_attempts: int
    mineru_enable_formula: bool
    mineru_enable_table: bool
    image_analysis_mode: str
    vision_provider: str
    vision_api_key: str
    vision_base_url: str
    vision_model: str
    vision_timeout_seconds: int
    rerank_base_url: str
    rerank_model: str
    rerank_api_key: str
    rerank_timeout_seconds: int
    index_worker_threads: int
    ocr_max_concurrency: int
    mineru_max_concurrency: int
    embedding_max_concurrency: int
    vision_global_max_concurrency: int

    @property
    def is_vision_enabled(self) -> bool:
        """判断是否启用视觉大模型分析（auto 模式）。"""
        return self.image_analysis_mode == "auto"

    @property
    def is_image_analysis_disabled(self) -> bool:
        """判断是否完全禁用图片处理。"""
        return self.image_analysis_mode == "disabled"

    @classmethod
    def from_env(cls, project_dir: Path | None = None) -> "ApplicationSettings":
        """从环境变量构建完整运行配置，并补齐存储路径等默认值。"""
        project_dir = project_dir or Path(__file__).resolve().parents[2]
        storage_dir = os.getenv("STORAGE_DIR", str(project_dir / "storage"))
        data_dir = os.getenv("DATA_DIR", os.path.join(storage_dir, "data"))
        db_file = os.getenv("DB_FILE", os.path.join(data_dir, "app_data.sqlite3"))

        # 读取图片分析配置
        image_analysis = os.getenv("RAG_IMAGE_ANALYSIS", "auto").lower().strip()
        if image_analysis not in ("auto", "ocr", "disabled"):
            image_analysis = "auto"

        return cls(
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_dir=os.getenv("LOG_DIR", os.path.join(storage_dir, "logs")).strip(),
            log_file_max_bytes=_integer(
                "LOG_FILE_MAX_BYTES", 10 * 1024 * 1024, 64 * 1024, 1024 * 1024 * 1024
            ),
            # 备份数设为 0 即关闭文件日志，只保留标准输出。
            log_file_backups=_integer("LOG_FILE_BACKUPS", 5, 0, 50),
            chat_run_retention_days=_integer("CHAT_RUN_RETENTION_DAYS", 90, 1),
            login_event_retention_days=_integer("LOGIN_EVENT_RETENTION_DAYS", 180, 1),
            storage_dir=storage_dir, data_dir=data_dir, db_file=db_file,
            milvus_uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530").strip(),
            milvus_token=os.getenv("MILVUS_TOKEN", "").strip(),
            milvus_collection_prefix=os.getenv("MILVUS_COLLECTION_PREFIX", "ragloom_").strip(),
            embedding_dimension=_integer("EMBEDDING_DIMENSION", 1024, 1),
            initial_admin_password=os.getenv("INITIAL_ADMIN_PASSWORD", ""),
            auth_secret_key=os.getenv("AUTH_SECRET_KEY", "").strip(),
            auth_token_ttl_seconds=_integer("AUTH_TOKEN_TTL_SECONDS", 28800, 1),
            # 次数或窗口设为 0 即关闭对应限制。
            auth_login_max_failures=_integer("AUTH_LOGIN_MAX_FAILURES", 5, 0),
            auth_login_window_seconds=_integer("AUTH_LOGIN_WINDOW_SECONDS", 900, 0),
            auth_client_max_failures=_integer("AUTH_CLIENT_MAX_FAILURES", 20, 0),
            auth_client_window_seconds=_integer("AUTH_CLIENT_WINDOW_SECONDS", 300, 0),
            auth_register_max_attempts=_integer("AUTH_REGISTER_MAX_ATTEMPTS", 5, 0),
            auth_register_window_seconds=_integer("AUTH_REGISTER_WINDOW_SECONDS", 3600, 0),
            auth_trust_x_real_ip=_flag("AUTH_TRUST_X_REAL_IP", False),
            siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            model_api_key_encryption_key=os.getenv("MODEL_API_KEY_ENCRYPTION_KEY", "").strip(),
            model_api_key_previous_encryption_keys=os.getenv(
                "MODEL_API_KEY_PREVIOUS_ENCRYPTION_KEYS", ""
            ),
            index_job_history_retention_days=_integer("INDEX_JOB_HISTORY_RETENTION_DAYS", 90, 1),
            index_job_history_max_per_kb=_integer("INDEX_JOB_HISTORY_MAX_PER_KB", 200, 20),
            retriever_cache_max_size=_integer("RETRIEVER_CACHE_MAX_SIZE", 8, 1),
            query_embedding_cache_size=_integer("QUERY_EMBEDDING_CACHE_SIZE", 64, 1),
            chat_retrieval_planning=_flag("CHAT_RETRIEVAL_PLANNING", False),
            chat_query_variant=_flag("CHAT_QUERY_VARIANT", False),
            chat_agentic_retrieval=_flag("CHAT_AGENTIC_RETRIEVAL", False),
            # 联网搜索按次计费；设为 0 即不限。
            web_search_daily_limit=_integer("WEB_SEARCH_DAILY_LIMIT", 500, 0),
            web_search_user_per_minute=_integer("WEB_SEARCH_USER_PER_MINUTE", 5, 0),
            # 单个账号同时进行的流式问答数和每分钟提问次数；设为 0 即不限。
            chat_user_max_concurrent=_integer("CHAT_USER_MAX_CONCURRENT", 2, 0),
            chat_user_per_minute=_integer("CHAT_USER_PER_MINUTE", 10, 0),
            mineru_api_token=os.getenv("MINERU_API_TOKEN", "").strip(),
            mineru_api_base_url=os.getenv(
                "MINERU_API_BASE_URL", "https://mineru.net/api/v4"
            ).strip().rstrip("/"),
            mineru_model_version=os.getenv("MINERU_MODEL_VERSION", "vlm").strip() or "vlm",
            mineru_language=os.getenv("MINERU_LANGUAGE", "ch").strip() or "ch",
            mineru_results_dir=os.getenv(
                "MINERU_RESULTS_DIR", os.path.join(storage_dir, "mineru_results")
            ),
            mineru_timeout_seconds=_integer("MINERU_TIMEOUT_SECONDS", 600, 30, 3600),
            mineru_poll_interval_seconds=_integer("MINERU_POLL_INTERVAL_SECONDS", 3, 1, 30),
            mineru_download_attempts=_integer("MINERU_DOWNLOAD_ATTEMPTS", 3, 1, 5),
            mineru_enable_formula=_flag("MINERU_ENABLE_FORMULA", True),
            mineru_enable_table=_flag("MINERU_ENABLE_TABLE", True),
            image_analysis_mode=image_analysis,
            vision_provider=os.getenv("RAG_VISION_PROVIDER", "OpenAI").strip(),
            vision_api_key=os.getenv("RAG_VISION_API_KEY", "").strip(),
            vision_base_url=os.getenv("RAG_VISION_BASE_URL", "").strip(),
            vision_model=os.getenv("RAG_VISION_MODEL", "").strip(),
            vision_timeout_seconds=_integer("RAG_VISION_TIMEOUT_SECONDS", 60, 10, 180),
            rerank_base_url=os.getenv("RERANK_BASE_URL", "").strip().rstrip("/"),
            rerank_model=os.getenv("RERANK_MODEL", "").strip(),
            rerank_api_key=os.getenv("RERANK_API_KEY", "").strip(),
            rerank_timeout_seconds=_integer("RERANK_TIMEOUT_SECONDS", 15, 1, 120),
            index_worker_threads=_integer("INDEX_WORKER_THREADS", 3, 1, 8),
            ocr_max_concurrency=_integer("OCR_MAX_CONCURRENCY", 1, 1, 4),
            mineru_max_concurrency=_integer("MINERU_MAX_CONCURRENCY", 3, 1, 8),
            embedding_max_concurrency=_integer("EMBEDDING_MAX_CONCURRENCY", 2, 1, 8),
            vision_global_max_concurrency=_integer("VISION_GLOBAL_MAX_CONCURRENCY", 6, 1, 32),
        )


def get_settings(project_dir: Path | None = None) -> ApplicationSettings:
    """Read current settings; intentionally dynamic for tests and key rotation."""
    return ApplicationSettings.from_env(project_dir)
