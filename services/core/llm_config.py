"""Build model clients from explicitly supplied provider settings."""

import time
from typing import TYPE_CHECKING

from pydantic import SecretStr

from services.core import config as application_config
from services.core import model_api_keys
from services.core.contracts import ModelConfig
from services.core.model_providers import ModelProviderStore
from services.core.settings import get_settings

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

_ANSWER_LLM_TIMEOUT_SECONDS = 90
_ANSWER_LLM_MAX_RETRIES = 0
# 推理类模型的思考 token 也计入该上限，2000 对带长证据的回答偏紧。
_ANSWER_LLM_MAX_TOKENS = 4000
_REWRITE_LLM_MAX_TOKENS = 256
_QUERY_EMBEDDING_TIMEOUT_SECONDS = 15
_QUERY_EMBEDDING_MAX_RETRIES = 1

_SILICONFLOW_KEY_TTL = 30.0
_siliconflow_key_cache: tuple[str, float] | None = None


def get_siliconflow_api_key(model_providers: ModelProviderStore) -> str:
    """读取嵌入和重排序所用 SiliconFlow 密钥；30 秒内复用缓存，避免每次调用都查 SQLite。"""
    global _siliconflow_key_cache
    now = time.monotonic()
    if _siliconflow_key_cache is not None and now - _siliconflow_key_cache[1] < _SILICONFLOW_KEY_TTL:
        return _siliconflow_key_cache[0]
    encrypted_key = model_providers.api_key("硅基流动 (SiliconFlow)")
    key = (
        model_api_keys.decrypt_api_key(encrypted_key)
        if encrypted_key
        else get_settings().siliconflow_api_key
    )
    _siliconflow_key_cache = (key, now)
    return key


def get_model_config(
    provider_name: str, model_providers: ModelProviderStore
) -> ModelConfig | None:
    """按提供商名称读取完整配置，不改变当前激活模型。"""
    provider_name = str(provider_name or "").strip()
    if not provider_name:
        return None
    row = model_providers.get_config(provider_name)
    if not row or not all(str(value or "").strip() for value in row):
        return None
    return {
        "api_key": model_api_keys.decrypt_api_key(row.encrypted_api_key),
        "base_url": row.base_url,
        "model": row.model,
    }


def get_active_model_config(model_providers: ModelProviderStore) -> ModelConfig:
    """返回当前启用的聊天模型；未配置时使用默认 SiliconFlow 兼容配置。"""
    row = model_providers.get_active_config()
    if not row or not row.encrypted_api_key:
        return {
            "api_key": get_settings().siliconflow_api_key,
            "base_url": application_config.DEFAULT_SILICONFLOW_BASE_URL,
            "model": application_config.DEFAULT_SILICONFLOW_MODEL,
        }
    return {
        "api_key": model_api_keys.decrypt_api_key(row.encrypted_api_key),
        "base_url": row.base_url,
        "model": row.model,
    }


def clear_api_key_cache() -> None:
    """模型配置变化后清除短期 SiliconFlow API Key 缓存。"""
    global _siliconflow_key_cache
    _siliconflow_key_cache = None


def select_model_config(
    provider: str | None, model_providers: ModelProviderStore
) -> ModelConfig:
    """优先读取本次请求指定的模型提供商，否则返回系统当前激活模型。"""
    if provider:
        configured = get_model_config(provider, model_providers)
        if configured is not None:
            return configured
    return get_active_model_config(model_providers)


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
    from langchain_openai import OpenAIEmbeddings

    from services.core.config import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_SILICONFLOW_BASE_URL,
    )

    return OpenAIEmbeddings(
        api_key=SecretStr(get_siliconflow_api_key(model_providers)),
        base_url=DEFAULT_SILICONFLOW_BASE_URL,
        model=DEFAULT_EMBEDDING_MODEL,
        # 默认值会先用 OpenAI 的 tiktoken 分词，再把 token ID 发给服务端；SiliconFlow 的
        # BGE-M3 使用自己的词表，必须直接发送原始文本，否则向量几乎丢失语义。
        check_embedding_ctx_length=False,
        max_retries=max_retries,
        timeout=timeout,
    )


def _create_chat_openai(model_config: ModelConfig, **kwargs) -> "ChatOpenAI":
    """使用统一的提供商配置构建 OpenAI 兼容聊天客户端。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=SecretStr(model_config["api_key"]),
        base_url=model_config["base_url"],
        model=model_config["model"],
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
