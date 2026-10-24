"""Encrypt model-provider API keys before persisting them in SQLite."""

from __future__ import annotations

import base64
import hashlib
import sqlite3
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from services.core import sqlite_connections as application_db
from services.core.settings import get_settings

ENCRYPTION_KEY_ENV = "MODEL_API_KEY_ENCRYPTION_KEY"
PREVIOUS_KEYS_ENV = "MODEL_API_KEY_PREVIOUS_ENCRYPTION_KEYS"
ENCRYPTED_PREFIX = "enc:v1:"
_MIN_KEY_LENGTH = 32
NON_SECRET_PLACEHOLDERS = frozenset({"ollama", "vllm"})


class ModelApiKeyEncryptionError(RuntimeError):
    """Raised when stored model API keys cannot be encrypted or decrypted safely."""


def _derive_fernet(secret: str, *, variable_name: str) -> Fernet:
    """使用稳定的密钥派生算法从配置密钥生成 Fernet 密钥。"""
    value = secret.strip()
    if len(value) < _MIN_KEY_LENGTH:
        raise ModelApiKeyEncryptionError(
            f"{variable_name} must contain at least {_MIN_KEY_LENGTH} characters"
        )
    derived = hashlib.sha256(value.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(derived))


def _keyring() -> list[Fernet]:
    """构建由当前加密密钥和历史轮换密钥组成的密钥环。"""
    settings = get_settings()
    current = settings.model_api_key_encryption_key
    if not current:
        raise ModelApiKeyEncryptionError(
            f"{ENCRYPTION_KEY_ENV} is required when model API keys are stored"
        )
    keys = [_derive_fernet(current, variable_name=ENCRYPTION_KEY_ENV)]
    previous = settings.model_api_key_previous_encryption_keys
    for index, value in enumerate(previous.split(","), start=1):
        if value.strip():
            keys.append(
                _derive_fernet(
                    value,
                    variable_name=f"{PREVIOUS_KEYS_ENV} item {index}",
                )
            )
    return keys


def is_encrypted(value: str | None) -> bool:
    """判断值是否使用系统约定的 API Key 加密封装格式。"""
    return str(value or "").startswith(ENCRYPTED_PREFIX)


def encrypt_api_key(value: str | None) -> str:
    """使用当前环境主密钥加密一个明文 API Key。"""
    plaintext = str(value or "")
    if not plaintext:
        return ""
    if is_encrypted(plaintext):
        plaintext = decrypt_api_key(plaintext)
    token = _keyring()[0].encrypt(plaintext.encode("utf-8")).decode("ascii")
    return f"{ENCRYPTED_PREFIX}{token}"


def _decrypt_with_keyring(value: str) -> tuple[str, int]:
    """使用首个匹配的密钥解密，并返回该密钥在轮换列表中的位置。"""
    try:
        token = value.removeprefix(ENCRYPTED_PREFIX).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ModelApiKeyEncryptionError("Stored model API key ciphertext is invalid") from exc
    for index, fernet in enumerate(_keyring()):
        try:
            return fernet.decrypt(token).decode("utf-8"), index
        except (InvalidToken, UnicodeDecodeError):
            continue
    raise ModelApiKeyEncryptionError(
        "A stored model API key cannot be decrypted with the configured current or previous keys"
    )


def decrypt_api_key(value: str | None) -> str:
    """解密密文；非密文值（如本地服务的占位符）原样返回。"""
    stored = str(value or "")
    if not stored or not is_encrypted(stored):
        return stored
    plaintext, _key_index = _decrypt_with_keyring(stored)
    return plaintext


def _truncate_write_ahead_log(conn: sqlite3.Connection) -> None:
    """把 WAL 内容写回主库并清空 sidecar 文件。

    `VACUUM` 配合 `secure_delete` 只清理主数据库文件里的旧密文残留。在 WAL 模式下，
    改写前的旧页仍可能留在 `-wal` 中，导致复制 `storage/` 时留下旧密钥加密的密文。
    TRUNCATE 检查点会把 WAL 长度归零，使清理覆盖到 sidecar。
    """
    if str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
        return
    # busy(1) 表示仍有其他连接在读，检查点未能截断；此时旧密文可能留在 sidecar 里。
    busy, _log_pages, _checkpointed = conn.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    ).fetchone()
    if busy:
        raise ModelApiKeyEncryptionError(
            "Could not truncate the SQLite write-ahead log after rotating model API "
            "keys; old ciphertext may remain in the -wal file. Retry with no other "
            "database readers active."
        )


def rotate_database_api_keys(db_file: str | Path) -> int:
    """把仍用历史轮换密钥加密的 API Key 用当前密钥重新加密，返回更新条数。"""
    path = Path(db_file)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with application_db.connect(str(path)) as conn:
        rows = conn.execute(
            "SELECT provider_name, api_key FROM models_config "
            "WHERE COALESCE(api_key, '') <> ''"
        ).fetchall()
        rows = [row for row in rows if is_encrypted(str(row[1]))]
        if not rows:
            return 0

        keyring = _keyring()
        updates: list[tuple[str, str]] = []
        for provider, stored_value in rows:
            plaintext, key_index = _decrypt_with_keyring(str(stored_value))
            if key_index == 0:
                continue
            encrypted = (
                f"{ENCRYPTED_PREFIX}"
                f"{keyring[0].encrypt(plaintext.encode('utf-8')).decode('ascii')}"
            )
            updates.append((encrypted, provider))

        if not updates:
            return 0
        conn.execute("PRAGMA secure_delete = ON")
        conn.executemany(
            "UPDATE models_config SET api_key = ? WHERE provider_name = ?",
            updates,
        )
        conn.commit()
        conn.execute("VACUUM")
        _truncate_write_ahead_log(conn)
        return len(updates)
