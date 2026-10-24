"""创建并校验当前应用版本所需的 SQLite 数据库结构。"""

import logging
import os
import sqlite3

from services.core import sqlite_connections as application_db

logger = logging.getLogger(__name__)

CURRENT_DATABASE_SCHEMA_VERSION = 7


def _create_chat_memory_table(conn: sqlite3.Connection) -> None:
    """创建独立的会话摘要表，避免改变基础会话记录的字段顺序。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_session_memory (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            summary_message_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _create_shared_tables(conn: sqlite3.Connection) -> None:
    """创建所有数据库版本共用的聊天会话和消息表。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            account_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            kb_name TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    _create_chat_memory_table(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT DEFAULT '[]',
            created_at TEXT NOT NULL
        )
        """
    )


def _create_current_tables(
    conn: sqlite3.Connection,
    *,
    default_chunk_size: int,
    default_chunk_overlap: int,
    default_parser_mode: str,
) -> None:
    """创建当前架构版本的业务表，并注入知识库默认配置。"""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            name TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            chunk_size INTEGER DEFAULT {default_chunk_size},
            chunk_overlap INTEGER DEFAULT {default_chunk_overlap},
            active_rebuild_job_id TEXT DEFAULT '',
            query_rewrite_enabled INTEGER DEFAULT 1
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS models_config (
            provider_name TEXT PRIMARY KEY,
            api_key TEXT,
            base_url TEXT,
            default_model TEXT,
            is_active INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            phone TEXT NOT NULL DEFAULT '',
            password TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            role_id INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kb_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            size INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL,
            status TEXT DEFAULT 'ready',
            error_message TEXT DEFAULT '',
            index_job_id TEXT DEFAULT '',
            document_type TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            effective_date TEXT DEFAULT '',
            parser_mode TEXT DEFAULT '{default_parser_mode}',
            UNIQUE (kb_name, filename)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_jobs (
            id TEXT PRIMARY KEY,
            kb_name TEXT NOT NULL,
            filename TEXT,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chroma_disabled_sources (
            kb_name TEXT NOT NULL,
            source TEXT NOT NULL,
            error_message TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (kb_name, source)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            menu_permissions TEXT NOT NULL DEFAULT '[]',
            knowledge_base_permissions TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        )
        """
    )
    _create_shared_tables(conn)


def _create_current_indexes(conn: sqlite3.Connection) -> None:
    """为文档、任务和聊天的常用查询创建 SQLite 索引。"""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_kb_name ON documents(kb_name)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_kb_status "
        "ON documents(kb_name, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_kb_type_date "
        "ON documents(kb_name, document_type, effective_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_kb_created "
        "ON index_jobs(kb_name, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_status_created "
        "ON index_jobs(status, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_sessions_account_updated "
        "ON chat_sessions(account_id, updated_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_session "
        "ON chat_messages(session_id, id)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_models_one_active "
        "ON models_config(is_active) WHERE is_active = 1"
    )


def _validate_current_schema(conn: sqlite3.Connection) -> None:
    """校验当前版本标记和实际表结构是否一致，不一致时拒绝启动。"""
    required_columns = {
        "accounts": {
            "id", "username", "name", "phone", "password", "status", "role_id", "created_at",
        },
        "chat_sessions": {
            "id", "account_id", "username", "kb_name", "title",
            "created_at", "updated_at",
        },
        "chat_session_memory": {
            "session_id", "summary", "summary_message_id", "updated_at",
        },
        "models_config": {
            "provider_name", "api_key", "base_url", "default_model", "is_active",
        },
        "documents": {
            "id", "kb_name", "filename", "path", "size", "uploaded_at",
            "status", "error_message", "index_job_id", "document_type",
            "tags", "effective_date", "parser_mode",
        },
    }
    for table, required in required_columns.items():
        actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        missing = required - actual
        if missing:
            raise RuntimeError(
                f"Database schema v{CURRENT_DATABASE_SCHEMA_VERSION} is missing "
                f"{table} columns: {', '.join(sorted(missing))}"
            )


def initialize_schema(
    db_file: str,
    data_dir: str,
    *,
    default_chunk_size: int,
    default_chunk_overlap: int,
    default_parser_mode: str,
) -> None:
    """创建全新数据库，或校验当前版本数据库；其他版本的旧库需手动删除后重建。"""
    os.makedirs(os.path.dirname(db_file) or data_dir, exist_ok=True)
    with application_db.connect(db_file) as conn:
        schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        has_application_tables = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        if schema_version not in {0, CURRENT_DATABASE_SCHEMA_VERSION} or (
            schema_version == 0 and has_application_tables
        ):
            raise RuntimeError(
                f"Application database schema version {schema_version} is not supported; "
                f"delete the old database or use version {CURRENT_DATABASE_SCHEMA_VERSION}"
            )

        conn.execute("PRAGMA secure_delete = ON")
        # WAL 让读取不被写入阻塞，索引 worker 与请求线程因此不会互相锁死。
        journal_mode = application_db.enable_write_ahead_logging(conn)
        if journal_mode.lower() != "wal":
            logger.warning(
                "Could not enable SQLite WAL mode; journal mode remains %r. "
                "Concurrent index and request writes may contend for locks.",
                journal_mode,
            )
        _create_current_tables(
            conn,
            default_chunk_size=default_chunk_size,
            default_chunk_overlap=default_chunk_overlap,
            default_parser_mode=default_parser_mode,
        )
        _create_current_indexes(conn)
        if schema_version == 0:
            conn.execute(f"PRAGMA user_version = {CURRENT_DATABASE_SCHEMA_VERSION}")
        _validate_current_schema(conn)
