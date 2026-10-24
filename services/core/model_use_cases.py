"""Framework-independent model-provider configuration use cases."""

from services.core import llm_config, model_api_keys
from services.core.contracts import ModelProviderState
from services.core.errors import (
    InvalidRequestError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from services.core.model_providers import ModelProviderStore


def check_model_config(
    provider: str | None, model_providers: ModelProviderStore
) -> ModelProviderState:
    """Return the public readiness state for a provider or the active fallback."""
    state = model_providers.check(provider)
    if state is not None:
        return state
    config = llm_config.get_active_model_config(model_providers)
    return {
        "configured": bool(config["api_key"]),
        "provider": None,
        "model": config["model"],
        "is_active": False,
    }


def list_model_providers(model_providers: ModelProviderStore):
    """Return provider configuration safe for public API responses."""
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
    """Persist a model provider and repair the active provider when necessary."""
    provider = provider.strip()
    base_url = base_url.strip()
    default_model = default_model.strip()
    if not provider or not base_url or not default_model:
        raise InvalidRequestError(
            "模型名称、Base URL 和默认模型不能为空", code="invalid_model_configuration"
        )

    encrypted_api_key = None
    if api_key.strip():
        try:
            encrypted_api_key = model_api_keys.encrypt_api_key(api_key.strip())
        except model_api_keys.ModelApiKeyEncryptionError as exc:
            raise ServiceUnavailableError(
                str(exc), code="model_key_encryption_unavailable"
            ) from exc

    is_active = model_providers.save(
        provider,
        encrypted_api_key=encrypted_api_key,
        clear_api_key=clear_api_key,
        base_url=base_url,
        default_model=default_model,
    )
    llm_config.clear_api_key_cache()
    return is_active


def activate_model_provider(provider: str, model_providers: ModelProviderStore) -> str:
    """Activate a configured model provider and return its normalized name."""
    provider = provider.strip()
    if not provider:
        raise InvalidRequestError("模型名称不能为空", code="invalid_model_provider")
    try:
        model_providers.activate(provider)
    except KeyError as exc:
        raise ResourceNotFoundError("模型配置不存在", code="model_provider_not_found") from exc
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_model_provider") from exc
    return provider
