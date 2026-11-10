"""创建并校验当前应用版本所需的 SQLite 数据库结构。"""

import logging
import os
import re
import sqlite3

from services.core import sqlite_connections as application_db

logger = logging.getLogger(__name__)

CURRENT_DATABASE_SCHEMA_VERSION = 13


def _create_chat_memory_table(conn: sqlite3.Connection) -> None:
    """创建会话摘要表：每个会话一行，保存滚动摘要及其覆盖到的最后一条消息。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_session_memory (
            session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
            summary TEXT NOT NULL DEFAULT '',
            summary_message_id INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _create_shared_tables(conn: sqlite3.Connection) -> None:
    """创建当前数据库版本的聊天会话和消息表。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            account_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
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
            session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
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
            kb_id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL UNIQUE CHECK(length(name) BETWEEN 1 AND 80),
            system_key TEXT UNIQUE CHECK(system_key IS NULL OR system_key = 'common'),
            state TEXT NOT NULL DEFAULT 'active' CHECK(state IN ('active', 'deleting')),
            created_at TEXT NOT NULL,
            chunk_size INTEGER DEFAULT {default_chunk_size},
            chunk_overlap INTEGER DEFAULT {default_chunk_overlap},
            active_rebuild_job_id TEXT,
            query_rewrite_enabled INTEGER NOT NULL DEFAULT 1
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
            document_id TEXT PRIMARY KEY NOT NULL,
            kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
            filename TEXT NOT NULL,
            size INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing',
            error_message TEXT DEFAULT '',
            index_job_id TEXT,
            document_type TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            effective_date TEXT DEFAULT '',
            parser_mode TEXT DEFAULT '{default_parser_mode}',
            UNIQUE (kb_id, document_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_jobs (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
            filename TEXT,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            parser_mode TEXT,
            CHECK ((operation = 'rebuild' AND document_id IS NULL)
                OR (operation IN ('process', 'delete') AND document_id IS NOT NULL))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS parent_chunks (
            kb_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            parent_id TEXT NOT NULL,
            source TEXT NOT NULL,
            page TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            contextual_title TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (kb_id, document_id, parent_id),
            FOREIGN KEY (kb_id, document_id) REFERENCES documents(kb_id, document_id) ON DELETE CASCADE
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
            all_knowledge_bases INTEGER NOT NULL DEFAULT 0
                CHECK(all_knowledge_bases IN (0, 1)),
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS role_knowledge_bases (
            role_id INTEGER NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
            kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
            PRIMARY KEY (role_id, kb_id)
        )
        """
    )
    _create_shared_tables(conn)
    _create_chat_runs_table(conn)
    _create_chat_traces_table(conn)
    _create_login_events_table(conn)
    _create_admin_audit_table(conn)
    _create_auth_sessions_table(conn)


def _create_chat_runs_table(conn: sqlite3.Connection) -> None:
    """创建问答运行记录表：每次提问一行，只存耗时、状态和问题预览，不存回答正文。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            account_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL,
            user_message_id INTEGER,
            question_preview TEXT NOT NULL DEFAULT '',
            model_provider TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            rerank_status TEXT NOT NULL DEFAULT '',
            rewritten INTEGER NOT NULL DEFAULT 0,
            retrieval_skipped INTEGER NOT NULL DEFAULT 0,
            context_spans INTEGER NOT NULL DEFAULT 0,
            cited_sources INTEGER NOT NULL DEFAULT 0,
            answer_chars INTEGER NOT NULL DEFAULT 0,
            retrieval_ms REAL,
            rerank_ms REAL,
            context_ms REAL,
            first_token_ms REAL,
            generation_ms REAL,
            total_ms REAL NOT NULL DEFAULT 0,
            elapsed_extra TEXT NOT NULL DEFAULT '{}',
            error_summary TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _create_chat_traces_table(conn: sqlite3.Connection) -> None:
    """创建问答追踪表：每次提问一行，payload 保存各阶段结果、候选片段开头和上下文来源。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_traces (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            account_id INTEGER NOT NULL,
            kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL,
            run_id INTEGER,
            assistant_message_id INTEGER,
            feedback INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL
        )
        """
    )


def _create_login_events_table(conn: sqlite3.Connection) -> None:
    """创建登录记录表：每次登录尝试一行；未知用户名的尝试 account_id 为空。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS login_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            account_id INTEGER,
            username TEXT NOT NULL,
            success INTEGER NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            client TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _create_auth_sessions_table(conn: sqlite3.Connection) -> None:
    """创建登录会话表：只保存随机令牌的 SHA-256 摘要，删除账号时级联删除会话。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_sessions (
            token_hash TEXT PRIMARY KEY NOT NULL,
            account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL,
            -- 空闲过期时间，有请求时向后滑动，但不超过绝对过期时间。
            expires_at INTEGER NOT NULL,
            absolute_expires_at INTEGER NOT NULL,
            CHECK (expires_at > created_at AND expires_at <= absolute_expires_at)
        )
        """
    )


def _create_admin_audit_table(conn: sqlite3.Connection) -> None:
    """创建管理审计表：记录谁在什么时候查看了哪个用户的哪个会话。"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            actor_account_id INTEGER NOT NULL,
            actor_username TEXT NOT NULL,
            action TEXT NOT NULL,
            target_account_id INTEGER,
            target_username TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _create_current_indexes(conn: sqlite3.Connection) -> None:
    """为文档、任务和聊天的常用查询创建 SQLite 索引。"""
    # 文档组合唯一键已覆盖 kb_id 前缀。
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_kb_status ON documents(kb_id, status)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_kb_type_date "
        "ON documents(kb_id, document_type, effective_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_kb_created "
        "ON index_jobs(kb_id, document_id, created_at DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_index_jobs_status_created ON index_jobs(status, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_sessions_account_updated "
        "ON chat_sessions(account_id, updated_at DESC)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_kb ON chat_sessions(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_traces_kb ON chat_traces(kb_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_role_kbs_kb ON role_knowledge_bases(kb_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_runs_created ON chat_runs(created_at)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_runs_kb_created ON chat_runs(kb_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_runs_account_created "
        "ON chat_runs(account_id, created_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_traces_created ON chat_traces(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_traces_run ON chat_traces(run_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_traces_message ON chat_traces(assistant_message_id)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_traces_session ON chat_traces(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_traces_account ON chat_traces(account_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_login_events_created ON login_events(created_at)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_login_events_account_created "
        "ON login_events(account_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_admin_audit_created ON admin_audit_log(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_auth_sessions_account ON auth_sessions(account_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_models_one_active "
        "ON models_config(is_active) WHERE is_active = 1"
    )


def _validate_current_schema(conn: sqlite3.Connection) -> None:
    """校验当前版本标记和实际表结构是否一致，不一致时拒绝启动。"""
    required_columns = {
        "auth_sessions": {
            "token_hash", "account_id", "created_at", "expires_at", "absolute_expires_at"
        },
        "accounts": {
            "id",
            "username",
            "name",
            "phone",
            "password",
            "status",
            "role_id",
            "created_at",
        },
        "knowledge_bases": {
            "kb_id",
            "name",
            "system_key",
            "state",
            "created_at",
            "chunk_size",
            "chunk_overlap",
            "active_rebuild_job_id",
            "query_rewrite_enabled",
        },
        "roles": {
            "id",
            "name",
            "description",
            "menu_permissions",
            "all_knowledge_bases",
            "created_at",
        },
        "role_knowledge_bases": {"role_id", "kb_id"},
        "index_jobs": {
            "id",
            "kb_id",
            "document_id",
            "filename",
            "operation",
            "status",
            "error_message",
            "created_at",
            "started_at",
            "finished_at",
            "parser_mode",
        },
        "parent_chunks": {
            "kb_id",
            "document_id",
            "parent_id",
            "source",
            "page",
            "content",
            "contextual_title",
        },
        "chat_sessions": {
            "id",
            "account_id",
            "username",
            "kb_id",
            "title",
            "created_at",
            "updated_at",
        },
        "chat_session_memory": {
            "session_id",
            "summary",
            "summary_message_id",
            "updated_at",
        },
        "models_config": {
            "provider_name",
            "api_key",
            "base_url",
            "default_model",
            "is_active",
        },
        "chat_runs": {
            "id",
            "created_at",
            "account_id",
            "username",
            "kb_id",
            "session_id",
            "user_message_id",
            "question_preview",
            "model_provider",
            "status",
            "rerank_status",
            "rewritten",
            "retrieval_skipped",
            "context_spans",
            "cited_sources",
            "answer_chars",
            "retrieval_ms",
            "rerank_ms",
            "context_ms",
            "first_token_ms",
            "generation_ms",
            "total_ms",
            "elapsed_extra",
            "error_summary",
        },
        "chat_traces": {
            "id",
            "created_at",
            "account_id",
            "kb_id",
            "session_id",
            "run_id",
            "assistant_message_id",
            "feedback",
            "payload",
        },
        "login_events": {
            "id",
            "created_at",
            "account_id",
            "username",
            "success",
            "reason",
            "client",
        },
        "admin_audit_log": {
            "id",
            "created_at",
            "actor_account_id",
            "actor_username",
            "action",
            "target_account_id",
            "target_username",
            "detail",
        },
        "documents": {
            "document_id",
            "kb_id",
            "filename",
            "size",
            "uploaded_at",
            "status",
            "error_message",
            "index_job_id",
            "document_type",
            "tags",
            "effective_date",
            "parser_mode",
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
    required_foreign_keys = {
        ("documents", "kb_id"): "knowledge_bases",
        ("index_jobs", "kb_id"): "knowledge_bases",
        ("parent_chunks", "kb_id"): "documents",
        ("parent_chunks", "document_id"): "documents",
        ("chat_sessions", "kb_id"): "knowledge_bases",
        ("chat_runs", "kb_id"): "knowledge_bases",
        ("chat_traces", "kb_id"): "knowledge_bases",
        ("chat_messages", "session_id"): "chat_sessions",
        ("chat_session_memory", "session_id"): "chat_sessions",
        ("role_knowledge_bases", "role_id"): "roles",
        ("role_knowledge_bases", "kb_id"): "knowledge_bases",
        ("auth_sessions", "account_id"): "accounts",
    }
    for (table, column), target in required_foreign_keys.items():
        # PRAGMA foreign_key_list 行: (id, seq, table, from, to, on_update, on_delete, match)
        declared = {(row[3], row[2]) for row in conn.execute(f"PRAGMA foreign_key_list({table})")}
        if (column, target) not in declared:
            raise RuntimeError(
                f"Database schema v{CURRENT_DATABASE_SCHEMA_VERSION} is missing "
                f"foreign key {table}.{column} -> {target}"
            )

    for table, expected in (
        ("documents", ("document_id",)),
        ("parent_chunks", ("kb_id", "document_id", "parent_id")),
    ):
        columns = conn.execute(f"PRAGMA table_info({table})").fetchall()
        primary = tuple(row[1] for row in sorted(columns, key=lambda row: row[5]) if row[5])
        if primary != expected or any(not row[3] for row in columns if row[1] in expected):
            raise RuntimeError(f"Invalid primary key in {table}")
    unique_keys = {
        tuple(row[2] for row in conn.execute(f'PRAGMA index_info("{index[1]}")'))
        for index in conn.execute("PRAGMA index_list(documents)")
        if index[2] and not index[4]
    }
    if ("kb_id", "document_id") not in unique_keys or ("kb_id", "filename") in unique_keys:
        raise RuntimeError("Invalid document identity uniqueness")
    parent_foreign_keys: dict[int, list] = {}
    for row in conn.execute("PRAGMA foreign_key_list(parent_chunks)"):
        parent_foreign_keys.setdefault(row[0], []).append(row)
    if not any(
        [(row[3], row[4]) for row in sorted(group, key=lambda row: row[1])]
        == [("kb_id", "kb_id"), ("document_id", "document_id")]
        and all(row[2] == "documents" and row[6] == "CASCADE" for row in group)
        for group in parent_foreign_keys.values()
    ):
        raise RuntimeError("Invalid parent document composite foreign key")
    job_sql = conn.execute("SELECT sql FROM sqlite_schema WHERE name = 'index_jobs'").fetchone()[0]
    target_check = (
        "check((operation='rebuild'anddocument_idisnull)or"
        "(operationin('process','delete')anddocument_idisnotnull))"
    )
    if target_check not in re.sub(r"\s+", "", job_sql.lower()):
        raise RuntimeError("Missing index job target constraint")


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
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
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
