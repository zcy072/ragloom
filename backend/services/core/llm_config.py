"""Build model clients from explicitly supplied provider settings."""

import functools
import logging
from typing import TYPE_CHECKING

import httpx
from pydantic import SecretStr

from services.core import config as application_config
from services.core import model_api_keys
from services.core.contracts import ModelConfig
from services.core.errors import (
    ApplicationError,
    ResourceNotFoundError,
    ServiceUnavailableError,
)
from services.core.model_providers import ModelProviderStore
from services.core.safe_errors import exception_status_code, safe_error_summary

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

SILICONFLOW_PROVIDER_NAME = "硅基流动 (SiliconFlow)"

_ANSWER_LLM_TIMEOUT_SECONDS = 90
# 只有请求还没收到响应时 SDK 才会重试，流式输出开始后不会重放，不会重复吐字。
_ANSWER_LLM_MAX_RETRIES = 1
# 正常建连加 TLS 握手在 1 秒内；个别 CDN 节点会把握手拖到二十多秒再断开，
# 这里尽早放弃，重试时还有机会连到另一个节点。
_CHAT_CONNECT_TIMEOUT_SECONDS = 5
# 推理类模型的思考 token 也计入该上限，2000 对带长证据的回答偏紧。
_ANSWER_LLM_MAX_TOKENS = 4000
# 改写只输出一行 JSON；推理模型的思考 token 也计入上限，256 会被思考吃光、正文为空。
_REWRITE_LLM_MAX_TOKENS = 1024
_QUERY_EMBEDDING_TIMEOUT_SECONDS = 15
_QUERY_EMBEDDING_MAX_RETRIES = 1


def _model_not_configured(provider: str | None = None) -> ServiceUnavailableError:
    """模型或其 API Key 未配置；数据库是唯一来源，不回退环境变量或其他服务商。"""
    target = f"模型「{provider}」" if provider else "默认聊天模型"
    return ServiceUnavailableError(
        f"{target}尚未配置 API Key，请先在【模型配置】页面填写并保存。",
        code="model_not_configured",
    )


def _decrypt_stored_key(encrypted_key: str) -> str:
    """严格解密已保存的 Key；失败时给出不含密文和密钥的安全错误，且不发起上游请求。"""
    try:
        return model_api_keys.decrypt_api_key(encrypted_key)
    except model_api_keys.ModelApiKeyEncryptionError:
        raise ServiceUnavailableError(
            "已保存的模型 API Key 无法读取，请检查部署的主密钥配置，或在【模型配置】中重新填写。",
            code="model_key_unavailable",
        ) from None


def get_siliconflow_api_key(model_providers: ModelProviderStore) -> str:
    """读取嵌入和默认重排序所用的 SiliconFlow 密钥；只读数据库，未配置时明确报错。"""
    encrypted_key = model_providers.api_key(SILICONFLOW_PROVIDER_NAME)
    if not encrypted_key:
        raise _model_not_configured(SILICONFLOW_PROVIDER_NAME)
    return _decrypt_stored_key(encrypted_key)


def get_model_config(provider_name: str, model_providers: ModelProviderStore) -> ModelConfig:
    """按提供商名称读取并解密完整配置，不改变当前激活模型。

    不存在时报 404；存在但缺少 Key、地址或模型名时报 503，绝不改用其他服务商。
    """
    provider_name = str(provider_name or "").strip()
    row = model_providers.get_config(provider_name) if provider_name else None
    if row is None:
        raise ResourceNotFoundError("模型配置不存在", code="model_provider_not_found")
    if not row.encrypted_api_key or not row.base_url.strip() or not row.model.strip():
        raise _model_not_configured(provider_name)
    return {
        "api_key": _decrypt_stored_key(row.encrypted_api_key),
        "base_url": row.base_url,
        "model": row.model,
    }


def get_active_model_config(model_providers: ModelProviderStore) -> ModelConfig:
    """返回当前激活的聊天模型；没有激活模型时报 503，不回退任何默认配置。"""
    row = model_providers.get_active_config()
    if not row or not row.encrypted_api_key or not row.base_url.strip() or not row.model.strip():
        raise _model_not_configured()
    return {
        "api_key": _decrypt_stored_key(row.encrypted_api_key),
        "base_url": row.base_url,
        "model": row.model,
    }


def select_model_config(
    provider: str | None, model_providers: ModelProviderStore
) -> ModelConfig:
    """请求指定了提供商时只使用它（失败即报错），否则使用系统当前激活模型。"""
    if provider and provider.strip():
        return get_model_config(provider, model_providers)
    return get_active_model_config(model_providers)


def _embedding_call_error(exc: Exception) -> ApplicationError:
    """把嵌入接口的任意异常转为固定安全错误，只保留异常类别和 HTTP 状态码。"""
    if isinstance(exc, ApplicationError):
        return exc
    logger.warning("Embedding request failed: %s", safe_error_summary(exc))
    status = exception_status_code(exc)
    suffix = f"（HTTP {status}）" if status is not None else ""
    return ServiceUnavailableError(
        f"嵌入模型服务请求失败{suffix}", code="embedding_unavailable"
    )


@functools.cache
def _safe_embeddings_class():
    """延迟构建 OpenAIEmbeddings 子类：调用边界上把上游异常换成安全错误。

    上游异常正文可能回显请求内容甚至凭证，不能沿检索、索引的异常链进入日志和记录。
    """
    from langchain_openai import OpenAIEmbeddings

    class SafeOpenAIEmbeddings(OpenAIEmbeddings):
        def embed_documents(self, texts, chunk_size=None, **kwargs):
            try:
                return super().embed_documents(texts, chunk_size, **kwargs)
            except Exception as exc:
                raise _embedding_call_error(exc) from None

        def embed_query(self, text, **kwargs):
            try:
                return super().embed_query(text, **kwargs)
            except Exception as exc:
                raise _embedding_call_error(exc) from None

        async def aembed_documents(self, texts, chunk_size=None, **kwargs):
            try:
                return await super().aembed_documents(texts, chunk_size, **kwargs)
            except Exception as exc:
                raise _embedding_call_error(exc) from None

        async def aembed_query(self, text, **kwargs):
            try:
                return await super().aembed_query(text, **kwargs)
            except Exception as exc:
                raise _embedding_call_error(exc) from None

    return SafeOpenAIEmbeddings


def get_embeddings(model_providers: ModelProviderStore):
    """创建与当前 SiliconFlow 配置一致的嵌入模型客户端，供批量建索引使用。"""
    # 多个文档并行嵌入时偶发 429 会变多。客户端按指数退避并遵循 Retry-After 重试，
    # 默认只重试 2 次；超时默认 600 秒，卡住的连接会长时间占用索引线程。
    return _create_embeddings(model_providers, max_retries=5, timeout=120)


def get_query_embeddings(model_providers: ModelProviderStore):
    """创建在线提问用的嵌入客户端：与索引同一模型，但失败要尽快暴露给用户。

    用户正等着回答，不能像后台索引那样退避重试几分钟；单条短问题的嵌入远快于 15 秒。
    """
    return _create_embeddings(
        model_providers,
        max_retries=_QUERY_EMBEDDING_MAX_RETRIES,
        timeout=_QUERY_EMBEDDING_TIMEOUT_SECONDS,
    )


def _create_embeddings(
    model_providers: ModelProviderStore, *, max_retries: int, timeout: float
):
    from services.core.config import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_SILICONFLOW_BASE_URL,
    )

    return _safe_embeddings_class()(
        api_key=SecretStr(get_siliconflow_api_key(model_providers)),
        base_url=DEFAULT_SILICONFLOW_BASE_URL,
        model=DEFAULT_EMBEDDING_MODEL,
        # 默认值会先用 OpenAI 的 tiktoken 分词，再把 token ID 发给服务端；SiliconFlow 的
        # BGE-M3 使用自己的词表，必须直接发送原始文本，否则向量几乎丢失语义。
        check_embedding_ctx_length=False,
        max_retries=max_retries,
        timeout=timeout,
    )


def _create_chat_openai(
    model_config: ModelConfig, *, timeout: float, **kwargs
) -> "ChatOpenAI":
    """使用统一的提供商配置构建 OpenAI 兼容聊天客户端；timeout 是读写超时，建连另有短上限。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=SecretStr(model_config["api_key"]),
        base_url=model_config["base_url"],
        model=model_config["model"],
        timeout=httpx.Timeout(
            timeout, connect=min(timeout, _CHAT_CONNECT_TIMEOUT_SECONDS)
        ),
        **kwargs,
    )


def create_answer_llm(model_config: ModelConfig) -> "ChatOpenAI":
    """创建用于流式回答生成的聊天模型客户端。"""
    return _create_chat_openai(
        model_config,
        streaming=True,
        temperature=0.1,
        max_completion_tokens=_ANSWER_LLM_MAX_TOKENS,
        timeout=_ANSWER_LLM_TIMEOUT_SECONDS,
        max_retries=_ANSWER_LLM_MAX_RETRIES,
    )


def create_rewrite_llm(model_config: ModelConfig) -> "ChatOpenAI":
    """创建用于追问改写的聊天模型客户端。"""
    return _create_chat_openai(
        model_config,
        temperature=0,
        max_completion_tokens=_REWRITE_LLM_MAX_TOKENS,
        # 调用方用 wait_for 限时；SDK 不再自行重试，避免取消前在后台多次请求。
        timeout=application_config.CHAT_QUERY_MODEL_TIMEOUT_SECONDS,
        max_retries=0,
    )


def create_planner_llm(model_config: ModelConfig) -> "ChatOpenAI":
    """创建用于工具化检索决策的聊天模型客户端；调用方需自行绑定工具。"""
    return _create_chat_openai(
        model_config,
        temperature=0,
        max_completion_tokens=application_config.CHAT_RETRIEVAL_PLANNING_MAX_TOKENS,
        timeout=application_config.CHAT_QUERY_MODEL_TIMEOUT_SECONDS,
        max_retries=0,
    )


def create_summary_llm(model_config: ModelConfig) -> "ChatOpenAI":
    """创建用于会话记忆压缩的聊天模型客户端。"""
    return _create_chat_openai(
        model_config,
        temperature=0,
        max_completion_tokens=application_config.CHAT_MEMORY_SUMMARY_MAX_TOKENS,
        timeout=application_config.CHAT_MEMORY_SUMMARY_TIMEOUT_SECONDS,
        max_retries=0,
    )
