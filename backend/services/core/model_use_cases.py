"""Framework-independent model-provider configuration use cases."""

from urllib.parse import urlsplit

from services.core import model_api_keys
from services.core.contracts import ModelProviderState
from services.core.errors import (
    InvalidRequestError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from services.core.model_providers import ApiKeyRequiredError, ModelProviderStore


def _validate_base_url(base_url: str) -> None:
    """Base URL 必须是 http/https 绝对地址，且不含 userinfo、query 和 fragment。

    该地址会公开返回给前端，凭证不能藏在 URL 里；本地模型允许使用 HTTP。
    """
    try:
        parts = urlsplit(base_url)
        _ = parts.port  # 非法端口在这里抛 ValueError
    except ValueError:
        parts = None
    if (
        parts is None
        or parts.scheme not in {"http", "https"}
        or not parts.hostname
        or "@" in parts.netloc
        or parts.query
        or parts.fragment
        or "?" in base_url
        or "#" in base_url
        or any(char.isspace() for char in base_url)
    ):
        raise InvalidRequestError(
            "Base URL 必须是 http 或 https 地址，且不能包含用户名密码、查询参数或片段",
            code="invalid_model_configuration",
        )


def _verify_stored_key(provider: str, encrypted_api_key: str) -> None:
    """只在本地解密验证已保存的 Key，不联网；解密失败返回安全的 503。"""
    try:
        model_api_keys.decrypt_api_key(encrypted_api_key)
    except model_api_keys.ModelApiKeyEncryptionError:
        raise ServiceUnavailableError(
            f"模型「{provider}」已保存的 API Key 无法读取，请检查部署的主密钥配置，"
            "或重新填写 API Key 后保存，也可以直接清除。",
            code="model_key_unavailable",
        ) from None


def check_model_config(
    provider: str | None, model_providers: ModelProviderStore
) -> ModelProviderState:
    """检查指定或当前激活服务商的已保存配置：只解密验证，不联网、不计费。"""
    provider = (provider or "").strip() or None
    state = model_providers.check(provider)
    if state is None:
        if provider:
            raise ResourceNotFoundError("模型配置不存在", code="model_provider_not_found")
        return {"configured": False, "provider": None, "model": None, "is_active": False}
    if state["configured"]:
        stored = model_providers.get_config(state["provider"] or "")
        _verify_stored_key(state["provider"] or "", stored.encrypted_api_key if stored else "")
    return state


def list_model_providers(model_providers: ModelProviderStore):
    """Return provider configuration safe for public API responses; never decrypts keys."""
    return model_providers.list_public()


def save_model_provider(
    *,
    provider: str,
    api_key: str,
    clear_api_key: bool,
    base_url: str,
    default_model: str,
    model_providers: ModelProviderStore,
) -> bool:
    """保存服务商配置并修正激活状态；替换和清除都不解密旧值，损坏记录也能修复。"""
    provider = provider.strip()
    base_url = base_url.strip()
    default_model = default_model.strip()
    api_key = api_key.strip()
    if not provider or not base_url or not default_model:
        raise InvalidRequestError(
            "模型名称、Base URL 和默认模型不能为空", code="invalid_model_configuration"
        )
    if api_key and clear_api_key:
        raise InvalidRequestError(
            "不能同时填写新的 API Key 并清除 API Key", code="invalid_model_configuration"
        )
    _validate_base_url(base_url)

    encrypted_api_key = None
    if api_key:
        try:
            encrypted_api_key = model_api_keys.encrypt_api_key(api_key)
        except model_api_keys.ModelApiKeyEncryptionError:
            raise ServiceUnavailableError(
                "模型 API Key 加密不可用，请检查部署的主密钥配置",
                code="model_key_encryption_unavailable",
            ) from None

    try:
        return model_providers.save(
            provider,
            encrypted_api_key=encrypted_api_key,
            clear_api_key=clear_api_key,
            base_url=base_url,
            default_model=default_model,
        )
    except ApiKeyRequiredError as exc:
        raise InvalidRequestError(str(exc), code="model_api_key_required") from exc


def activate_model_provider(provider: str, model_providers: ModelProviderStore) -> str:
    """先在本地校验配置和密钥可读，通过后才激活；失败时当前激活项保持不变。"""
    provider = provider.strip()
    if not provider:
        raise InvalidRequestError("模型名称不能为空", code="invalid_model_provider")
    stored = model_providers.get_config(provider)
    if stored is None:
        raise ResourceNotFoundError("模型配置不存在", code="model_provider_not_found")
    if stored.encrypted_api_key:
        _verify_stored_key(provider, stored.encrypted_api_key)
    try:
        model_providers.activate(provider, expected_api_key=stored.encrypted_api_key)
    except KeyError as exc:
        raise ResourceNotFoundError("模型配置不存在", code="model_provider_not_found") from exc
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_model_provider") from exc
    return provider
