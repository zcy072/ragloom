"""Typed cross-module data structures shared by application services."""

from typing import NamedTuple, TypedDict


class ModelConfig(TypedDict):
    """Decrypted OpenAI-compatible model settings used to create LLM clients."""

    api_key: str
    base_url: str
    model: str


class StoredModelConfig(NamedTuple):
    """Encrypted provider settings read from the application database."""

    encrypted_api_key: str
    base_url: str
    model: str


class ModelProviderState(TypedDict):
    """Public readiness state for one configured model provider."""

    configured: bool
    provider: str | None
    model: str | None
    is_active: bool


class PublicModelProvider(TypedDict):
    """Provider configuration safe to expose without its API key."""

    provider: str
    configured: bool
    base_url: str | None
    default_model: str | None
    is_active: bool


class KnowledgeBase(NamedTuple):
    """One knowledge-base record; `kb_id` is the only identity, `name` is display text."""

    kb_id: str
    name: str
    system_key: str | None
    state: str
    created_at: str


class KnowledgeBaseSummary(TypedDict):
    """Knowledge-base list item exposed to API adapters."""

    kb_id: str
    name: str
    system_key: str | None
    state: str
    date: str
    docs: int


class DashboardDocumentRow(NamedTuple):
    """Document data consumed by dashboard aggregation."""

    kb_id: str
    kb_name: str
    filename: str | None
    size: int | None
    status: str | None


class DashboardTrendRow(NamedTuple):
    """Daily conversation counts consumed by dashboard charts."""

    date: str
    question_count: int
    answer_count: int


class KnowledgeBaseConfig(NamedTuple):
    """Effective parent-child chunk configuration for one knowledge base."""

    chunk_size: int
    chunk_overlap: int


RetrievalFilters = dict[str, object]


class NormalizedRetrievalFilters(TypedDict):
    """Validated document filters consumed by SQLite and retrieval backends."""

    document_ids: tuple[str, ...]
    document_types: tuple[str, ...]
    tags: tuple[str, ...]
    effective_date_from: str
    effective_date_to: str
