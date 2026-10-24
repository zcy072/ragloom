"""OpenAI-compatible model configuration endpoints."""

from fastapi import APIRouter, Depends

from routers import schemas
from services.core import model_use_cases
from services.core.container import get_application_container
from services.identity import auth as auth_service

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/check", response_model=schemas.ModelCheckResponse)
def check_model_config(
    provider: str | None = None,
    _principal: auth_service.Principal = Depends(auth_service.current_principal),
):
    """检查指定或当前默认模型配置，但不向前端暴露 API Key。"""
    return model_use_cases.check_model_config(
        provider, get_application_container().model_providers
    )


@router.get("", response_model=list[schemas.ModelConfigResponse])
def list_models(
    _principal: auth_service.Principal = Depends(auth_service.current_principal),
):
    """返回可公开的模型服务商配置，不暴露加密后的 API Key。"""
    return model_use_cases.list_model_providers(
        get_application_container().model_providers
    )


@router.post("", response_model=schemas.ModelSaveResponse)
def save_model(
    data: schemas.ModelConfigRequest,
    _principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("模型配置")
    ),
):
    """保存一个模型服务商配置，并修正当前激活模型状态。"""
    is_active = model_use_cases.save_model_provider(
        provider=data.provider,
        api_key=data.api_key,
        clear_api_key=data.clear_api_key,
        base_url=data.base_url,
        default_model=data.default_model,
        model_providers=get_application_container().model_providers,
    )
    return {"status": "success", "is_active": is_active}


@router.put("/{provider}/active", response_model=schemas.ModelActivationResponse)
def activate_model(
    provider: str,
    _principal: auth_service.Principal = Depends(
        auth_service.require_menu_permission("模型配置")
    ),
):
    """把配置完整的服务商设为当前聊天模型。"""
    provider = model_use_cases.activate_model_provider(
        provider, get_application_container().model_providers
    )
    return {"status": "success", "provider": provider}
