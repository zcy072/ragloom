"""Persistence and activation rules for OpenAI-compatible model providers."""

import sqlite3
from collections.abc import Callable

from services.core import sqlite_connections as application_db
from services.core.contracts import (
    ModelProviderState,
    PublicModelProvider,
    StoredModelConfig,
)


def _provider_state_from_row(row: tuple[object, ...]) -> ModelProviderState:
    """Convert the readiness query projection into its public structure."""
    return {
        "configured": bool(row[1]),
        "provider": str(row[0]),
        "model": application_db.optional_text(row[2]),
        "is_active": bool(row[3]),
    }


def _public_provider_from_row(row: tuple[object, ...]) -> PublicModelProvider:
    """Convert a provider row into fields safe for the configuration API."""
    return {
        "provider": str(row[0]),
        "configured": bool(row[1]),
        "base_url": application_db.optional_text(row[2]),
        "default_model": application_db.optional_text(row[3]),
        "is_active": bool(row[4]),
    }


def _stored_config_from_row(row: tuple[object, ...]) -> StoredModelConfig:
    """Convert the private configuration projection into a named tuple."""
    return StoredModelConfig(
        encrypted_api_key=str(row[0] or ""),
        base_url=str(row[1] or ""),
        model=str(row[2] or ""),
    )


class ApiKeyRequiredError(ValueError):
    """保存模型配置时缺少必须重新填写的 API Key。"""


class ModelProviderStore:
    """Persist provider settings while enforcing one active provider."""

    def __init__(self, db_file: Callable[[], str]) -> None:
        """保存延迟解析的业务数据库路径。"""
        self._db_file = db_file

    def check(self, provider: str | None = None) -> ModelProviderState | None:
        """返回指定服务商或当前激活服务商的公开配置状态。"""
        with application_db.connect(self._db_file()) as conn:
            if provider:
                row = conn.execute(
                    "SELECT provider_name, api_key, default_model, is_active "
                    "FROM models_config WHERE provider_name = ?",
                    (provider,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT provider_name, api_key, default_model, is_active "
                    "FROM models_config WHERE is_active = 1 LIMIT 1"
                ).fetchone()
        return _provider_state_from_row(tuple(row)) if row else None

    def list_public(self) -> list[PublicModelProvider]:
        """列出模型服务商配置，但不暴露加密 API Key。"""
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT provider_name, api_key, base_url, default_model, is_active "
                "FROM models_config ORDER BY is_active DESC, provider_name"
            ).fetchall()
        return [_public_provider_from_row(tuple(row)) for row in rows]

    def get_config(self, provider: str) -> StoredModelConfig | None:
        """返回指定服务商保存的加密密钥、接口地址和模型名称。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT api_key, base_url, default_model FROM models_config "
                "WHERE provider_name = ?",
                (provider,),
            ).fetchone()
        return _stored_config_from_row(tuple(row)) if row else None

    def get_active_config(self) -> StoredModelConfig | None:
        """返回当前激活服务商保存的加密密钥、接口地址和模型名称。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT api_key, base_url, default_model FROM models_config "
                "WHERE is_active = 1 LIMIT 1"
            ).fetchone()
        return _stored_config_from_row(tuple(row)) if row else None

    def api_key(self, provider: str) -> str:
        """返回服务商的加密 API Key；未配置时返回空字符串。"""
        config = self.get_config(provider)
        return config.encrypted_api_key if config else ""

    @staticmethod
    def _repair_active_provider(
        conn: sqlite3.Connection, preferred_provider: str | None = None
    ) -> None:
        """存在可用密钥时，确保恰好有一个已配置服务商处于激活状态。"""
        active_has_key = conn.execute(
            "SELECT 1 FROM models_config "
            "WHERE is_active = 1 AND COALESCE(api_key, '') <> '' LIMIT 1"
        ).fetchone()
        if active_has_key:
            return
        replacement = None
        if preferred_provider:
            row = conn.execute(
                "SELECT provider_name FROM models_config "
                "WHERE provider_name = ? AND COALESCE(api_key, '') <> ''",
                (preferred_provider,),
            ).fetchone()
            replacement = row[0] if row else None
        if replacement is None:
            row = conn.execute(
                "SELECT provider_name FROM models_config "
                "WHERE COALESCE(api_key, '') <> '' ORDER BY provider_name LIMIT 1"
            ).fetchone()
            replacement = row[0] if row else None
        conn.execute("UPDATE models_config SET is_active = 0")
        if replacement:
            conn.execute(
                "UPDATE models_config SET is_active = 1 WHERE provider_name = ?",
                (replacement,),
            )

    def save(
        self,
        provider: str,
        *,
        encrypted_api_key: str | None,
        clear_api_key: bool,
        base_url: str,
        default_model: str,
    ) -> bool:
        """创建或更新服务商，并在密钥变化后修正激活状态。"""
        with application_db.connect(self._db_file()) as conn:
            existing = conn.execute(
                "SELECT api_key, base_url FROM models_config WHERE provider_name = ?",
                (provider,),
            ).fetchone()
            if existing:
                stored_key, stored_base_url = existing
                # 已保存的 key 只能发往保存它时的地址，换地址必须重新填写，避免 key 被转发到别处。
                if (
                    stored_key
                    and encrypted_api_key is None
                    and not clear_api_key
                    and base_url != stored_base_url
                ):
                    raise ApiKeyRequiredError("修改 Base URL 时需要重新填写 API Key")
                key_value = "" if clear_api_key else encrypted_api_key
                conn.execute(
                    "UPDATE models_config SET api_key = COALESCE(?, api_key), "
                    "base_url = ?, default_model = ? WHERE provider_name = ?",
                    (key_value, base_url, default_model, provider),
                )
            else:
                conn.execute(
                    "INSERT INTO models_config "
                    "(provider_name, api_key, base_url, default_model, is_active) "
                    "VALUES (?, ?, ?, ?, 0)",
                    (provider, encrypted_api_key or "", base_url, default_model),
                )
            self._repair_active_provider(conn, provider)
            row = conn.execute(
                "SELECT is_active FROM models_config WHERE provider_name = ?", (provider,)
            ).fetchone()
        return bool(row and row[0])

    def activate(self, provider: str) -> None:
        """激活一个已经配置 API Key 的模型服务商。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT api_key FROM models_config WHERE provider_name = ?", (provider,)
            ).fetchone()
            if not row:
                raise KeyError(provider)
            if not row[0]:
                raise ValueError("请先保存该模型的 API Key")
            conn.execute("UPDATE models_config SET is_active = 0")
            conn.execute(
                "UPDATE models_config SET is_active = 1 WHERE provider_name = ?",
                (provider,),
            )
