"""Explicit application dependencies shared by use-case entry points."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from services.core.contracts import ModelConfig
from services.indexing.jobs import IndexJobStore


def _default_database_path() -> str | Path:
    """延迟读取应用存储模块中的默认数据库路径。"""
    import app_storage

    return app_storage.DB_FILE


def _default_select_model(provider: str | None):
    """按提供商选择模型配置，作为容器的默认模型查找实现。"""
    from services.core import llm_config

    return llm_config.select_model_config(provider, _default_model_providers())


def _default_answer_llm(config: ModelConfig):
    """根据模型配置创建用于生成最终回答的大模型客户端。"""
    from services.core import llm_config

    return llm_config.create_answer_llm(config)


def _default_summary_llm(config: ModelConfig):
    """根据模型配置创建用于压缩会话历史的摘要模型客户端。"""
    from services.core import llm_config

    return llm_config.create_summary_llm(config)


def _default_embeddings():
    """创建默认嵌入模型，供批量建索引使用。"""
    from services.core import llm_config

    return llm_config.get_embeddings(_default_model_providers())


def _default_query_embeddings():
    """创建在线提问用的嵌入模型：模型与索引一致，超时和重试更短。"""
    from services.core import llm_config

    return llm_config.get_query_embeddings(_default_model_providers())


def _default_model_providers():
    """延迟取得应用级模型配置存储。"""
    import app_storage

    return app_storage.model_providers


def _default_index_jobs() -> IndexJobStore:
    """延迟取得应用级索引任务存储实例。"""
    import app_storage

    return app_storage.index_jobs


def _default_knowledge_bases():
    """延迟取得应用级知识库存储实例。"""
    import app_storage

    return app_storage.knowledge_bases


@dataclass(frozen=True)
class ApplicationContainer:
    """Process-scoped storage dependencies used by application services."""

    documents: Any
    knowledge_base_settings: Any
    knowledge_bases: Any = field(default_factory=_default_knowledge_bases)
    model_providers: Any = field(default_factory=_default_model_providers)
    database_path: Callable[[], str | Path] = _default_database_path
    select_model: Callable[[str | None], ModelConfig] = _default_select_model
    create_answer_llm: Callable[[ModelConfig], Any] = _default_answer_llm
    create_summary_llm: Callable[[ModelConfig], Any] = _default_summary_llm
    create_embeddings: Callable[[], Any] = _default_embeddings
    create_query_embeddings: Callable[[], Any] = _default_query_embeddings
    index_jobs: Callable[[], IndexJobStore] = _default_index_jobs


_default_container: ApplicationContainer | None = None


def get_application_container() -> ApplicationContainer:
    """Return the default process container, constructing it on first use."""
    global _default_container
    if _default_container is None:
        import app_storage

        _default_container = ApplicationContainer(
            documents=app_storage.documents,
            knowledge_base_settings=app_storage.knowledge_base_settings,
            knowledge_bases=app_storage.knowledge_bases,
            model_providers=app_storage.model_providers,
            database_path=lambda: app_storage.DB_FILE,
            index_jobs=lambda: app_storage.index_jobs,
        )
    return _default_container
