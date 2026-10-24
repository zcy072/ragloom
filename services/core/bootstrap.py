"""初始化应用默认账号、权限、知识库和模型配置。"""

import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from services.core import model_api_keys
from services.core import sqlite_connections as application_db
from services.core.config import hash_password
from services.core.settings import get_settings

OPENAI_DEFAULT_MODEL = "gpt-6-sol"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"

@dataclass(frozen=True)
class ApplicationDefaults:
    """启动初始化所需的默认账号、角色、知识库和模型配置。"""
    initial_admin_username: str
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
    """确保超级管理员角色包含新增加的后台菜单和全公司知识库权限。"""
    row = conn.execute(
        "SELECT menu_permissions, knowledge_base_permissions FROM roles WHERE name = ?",
        ("超级管理员",),
    ).fetchone()
    if not row:
        return
    try:
        menu_permissions = json.loads(row[0] or "[]")
        knowledge_base_permissions = json.loads(row[1] or "[]")
    except json.JSONDecodeError:
        menu_permissions, knowledge_base_permissions = [], []
    if not isinstance(menu_permissions, list):
        menu_permissions = []
    if not isinstance(knowledge_base_permissions, list):
        knowledge_base_permissions = []
    conn.execute(
        "UPDATE roles SET menu_permissions = ?, knowledge_base_permissions = ? WHERE name = ?",
        (
            json.dumps(
                list(dict.fromkeys([*menu_permissions, *role_menu_options])),
                ensure_ascii=False,
            ),
            json.dumps(
                list(dict.fromkeys([*knowledge_base_permissions, "全公司"])),
                ensure_ascii=False,
            ),
            "超级管理员",
        ),
    )


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
            (name, description, menu_permissions, knowledge_base_permissions, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "超级管理员",
                "系统默认角色，拥有全部后台菜单和全部知识库权限。",
                json.dumps(list(defaults.role_menu_options), ensure_ascii=False),
                json.dumps(["全公司"], ensure_ascii=False),
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
            conn.execute(
                "INSERT INTO accounts "
                "(username, name, phone, password, status, role_id, created_at) "
                "VALUES (?, ?, '', ?, '正常', ?, ?)",
                (
                    defaults.initial_admin_username,
                    "系统管理员",
                    hash_password(defaults.initial_admin_password),
                    administrator_role[0],
                    now(),
                ),
            )
        if conn.execute(
            "SELECT COUNT(*) FROM knowledge_bases"
        ).fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO knowledge_bases "
                "(name, created_at, chunk_size, chunk_overlap) VALUES (?, ?, ?, ?)",
                (
                    "通用知识库",
                    now(),
                    defaults.default_chunk_size,
                    defaults.default_chunk_overlap,
                ),
            )
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
            encrypted_models = [
                (
                    model[0],
                    (
                        model[1]
                        if model[1] in model_api_keys.NON_SECRET_PLACEHOLDERS
                        else model_api_keys.encrypt_api_key(model[1])
                    ),
                    *model[2:],
                )
                for model in default_models
            ]
            conn.executemany(
                """
                INSERT INTO models_config
                (provider_name, api_key, base_url, default_model, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (*model, 1 if model[0] == "硅基流动 (SiliconFlow)" else 0)
                    for model in encrypted_models
                ],
            )
