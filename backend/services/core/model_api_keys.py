"""Encrypt model-provider API keys before persisting them in SQLite.

只有一个标准 Fernet 主密钥（`Fernet.generate_key()` 生成），保存在数据库之外。运行时只做
四件事：校验主密钥、加密明文、严格解密密文、报告不含秘密的错误。主密钥轮换是停机维护，
不在请求或启动路径上进行。
"""

from __future__ import annotations

import binascii

from cryptography.fernet import Fernet, InvalidToken

from services.core.settings import get_settings

ENCRYPTION_KEY_ENV = "MODEL_API_KEY_ENCRYPTION_KEY"
ENCRYPTED_PREFIX = "enc:v1:"


class ModelApiKeyEncryptionError(RuntimeError):
    """Raised when stored model API keys cannot be encrypted or decrypted safely."""


def _fernet() -> Fernet:
    """用环境中的标准 Fernet 主密钥构建加解密器；错误信息不包含密钥内容。"""
    key = get_settings().model_api_key_encryption_key
    if not key:
        raise ModelApiKeyEncryptionError(f"{ENCRYPTION_KEY_ENV} is required")
    try:
        return Fernet(key)
    except (ValueError, TypeError, binascii.Error):
        raise ModelApiKeyEncryptionError(
            f"{ENCRYPTION_KEY_ENV} must be a standard Fernet key "
            "(generate one with Fernet.generate_key())"
        ) from None


def validate_master_key() -> None:
    """启动时只校验主密钥存在且格式有效，不扫描、不解密数据库中的记录。"""
    _fernet()


def encrypt_api_key(value: str | None) -> str:
    """加密一个明文 API Key；空值返回空串。输入即使形似密文也只按普通字符串处理。"""
    plaintext = str(value or "")
    if not plaintext:
        return ""
    token = _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")
    return f"{ENCRYPTED_PREFIX}{token}"


def decrypt_api_key(value: str | None) -> str:
    """解密已保存的密文；空值返回空串，其余任何非合法密文都报错而不是原样返回。"""
    stored = str(value or "")
    if not stored:
        return ""
    if not stored.startswith(ENCRYPTED_PREFIX):
        raise ModelApiKeyEncryptionError("Stored model API key is not valid ciphertext")
    fernet = _fernet()
    try:
        token = stored.removeprefix(ENCRYPTED_PREFIX).encode("ascii")
        return fernet.decrypt(token).decode("utf-8")
    except (InvalidToken, UnicodeError):
        raise ModelApiKeyEncryptionError(
            "Stored model API key cannot be decrypted with the configured master key"
        ) from None
