"""初始化应用默认账号、权限、知识库和模型配置。"""

import json
import logging
import os
import secrets
import sqlite3
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from services.core import model_api_keys
from services.core import sqlite_connections as application_db
from services.core.config import hash_password, validate_new_password
from services.core.knowledge_bases import COMMON_SYSTEM_KEY
from services.core.model_providers import ModelProviderStore
from services.core.settings import get_settings

logger = logging.getLogger(__name__)

INITIAL_ADMIN_PASSWORD_FILENAME = "initial_admin_password.txt"
COMMON_KNOWLEDGE_BASE_NAME = "通用知识库"
OPENAI_DEFAULT_MODEL = "gpt-6-sol"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"

@dataclass(frozen=True)
class ApplicationDefaults:
    """启动初始化所需的默认账号、角色、知识库和模型配置。"""
    initial_admin_username: str
    # 为空时生成随机密码，写入数据库同目录下仅属主可读的文件。
    initial_admin_password: str
    role_menu_options: Sequence[str]
    default_chunk_size: int
    default_chunk_overlap: int
    siliconflow_base_url: str
    siliconflow_model: str


def _sync_system_role_permissions(
    conn: sqlite3.Connection,
    role_menu_options: Sequence[str],
) -> None:
    """确保超级管理员角色包含新增加的后台菜单，并保持全库访问权限。"""
    row = conn.execute(
        "SELECT menu_permissions FROM roles WHERE name = ?", ("超级管理员",)
    ).fetchone()
    if not row:
        return
    try:
        menu_permissions = json.loads(row[0] or "[]")
    except json.JSONDecodeError:
        menu_permissions = []
    if not isinstance(menu_permissions, list):
        menu_permissions = []
    conn.execute(
        "UPDATE roles SET menu_permissions = ?, all_knowledge_bases = 1 WHERE name = ?",
        (
            json.dumps(
                list(dict.fromkeys([*menu_permissions, *role_menu_options])),
                ensure_ascii=False,
            ),
            "超级管理员",
        ),
    )


def _ensure_common_knowledge_base(
    conn: sqlite3.Connection, now: Callable[[], str], defaults: "ApplicationDefaults"
) -> None:
    """按 system_key 保证通用知识库存在；默认名称被普通库占用时明确失败。"""
    if conn.execute(
        "SELECT 1 FROM knowledge_bases WHERE system_key = ?", (COMMON_SYSTEM_KEY,)
    ).fetchone():
        return
    if conn.execute(
        "SELECT 1 FROM knowledge_bases WHERE name = ?", (COMMON_KNOWLEDGE_BASE_NAME,)
    ).fetchone():
        raise RuntimeError(
            f"默认名称“{COMMON_KNOWLEDGE_BASE_NAME}”已被普通知识库占用，"
            "无法创建通用知识库；请先重命名该知识库后再启动"
        )
    conn.execute(
        "INSERT INTO knowledge_bases "
        "(kb_id, name, system_key, state, created_at, chunk_size, chunk_overlap) "
        "VALUES (?, ?, ?, 'active', ?, ?, ?)",
        (
            str(uuid.uuid4()),
            COMMON_KNOWLEDGE_BASE_NAME,
            COMMON_SYSTEM_KEY,
            now(),
            defaults.default_chunk_size,
            defaults.default_chunk_overlap,
        ),
    )


def _generated_admin_password(db_file: str) -> str:
    """生成随机初始密码并写入数据目录；日志只给文件位置，避免密码出现在运行监控里。"""
    password = secrets.token_urlsafe(12)
    path = os.path.join(os.path.dirname(db_file), INITIAL_ADMIN_PASSWORD_FILENAME)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # O_CREAT 的权限只作用于新文件；已存在的旧文件也要收紧。
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(password + "\n")
    logger.warning(
        "INITIAL_ADMIN_PASSWORD 未设置，已生成随机初始密码并写入 %s；登录改密后请删除该文件",
        path,
    )
    return password


def bootstrap_defaults(
    db_file: str,
    now: Callable[[], str],
    defaults: ApplicationDefaults,
) -> None:
    """写入缺失的默认数据，但不覆盖用户已有配置。"""
    with application_db.connect(db_file) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO roles
            (name, description, menu_permissions, all_knowledge_bases, created_at)
            VALUES (?, ?, ?, 1, ?)
            """,
            (
                "超级管理员",
                "系统默认角色，拥有全部后台菜单和全部知识库权限。",
                json.dumps(list(defaults.role_menu_options), ensure_ascii=False),
                now(),
            ),
        )
        _sync_system_role_permissions(conn, defaults.role_menu_options)
        administrator_role = conn.execute(
            "SELECT id FROM roles WHERE name = ?", ("超级管理员",)
        ).fetchone()
        if administrator_role and conn.execute(
            "SELECT COUNT(*) FROM accounts"
        ).fetchone()[0] == 0:
            # 只在真正创建管理员时校验配置的初始密码；已有账号库不因残留配置拒绝启动。
            if defaults.initial_admin_password:
                validate_new_password(defaults.initial_admin_password)
            conn.execute(
                "INSERT INTO accounts "
                "(username, name, phone, password, status, role_id, created_at) "
                "VALUES (?, ?, '', ?, '正常', ?, ?)",
                (
                    defaults.initial_admin_username,
                    "系统管理员",
                    hash_password(
                        defaults.initial_admin_password
                        or _generated_admin_password(db_file)
                    ),
                    administrator_role[0],
                    now(),
                ),
            )
        _ensure_common_knowledge_base(conn, now, defaults)
        if conn.execute("SELECT COUNT(*) FROM models_config").fetchone()[0] == 0:
            default_models = [
                ("OpenAI", "", "https://api.openai.com/v1", OPENAI_DEFAULT_MODEL),
                ("DeepSeek", "", "https://api.deepseek.com/v1", DEEPSEEK_DEFAULT_MODEL),
                (
                    "通义千问 (DashScope)",
                    "",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "qwen-plus",
                ),
                (
                    "本地 Ollama",
                    "ollama",
                    "http://host.docker.internal:11434/v1",
                    "qwen2.5:7b",
                ),
                (
                    "本地 vLLM",
                    "vllm",
                    "http://host.docker.internal:8000/v1",
                    "Qwen/Qwen2.5-7B-Instruct",
                ),
                (
                    "硅基流动 (SiliconFlow)",
                    get_settings().siliconflow_api_key,
                    defaults.siliconflow_base_url,
                    defaults.siliconflow_model,
                ),
            ]
            # 本地服务的无秘密占位符也统一加密，解密端因此不需要任何明文例外。
            # SILICONFLOW_API_KEY 只在这里导入一次；之后数据库是唯一来源。
            conn.executemany(
                """
                INSERT INTO models_config
                (provider_name, api_key, base_url, default_model, is_active)
                VALUES (?, ?, ?, ?, 0)
                """,
                [
                    (model[0], model_api_keys.encrypt_api_key(model[1]), *model[2:])
                    for model in default_models
                ],
            )
            ModelProviderStore.repair_active_provider(conn, "硅基流动 (SiliconFlow)")
