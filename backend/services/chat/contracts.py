"""Framework-independent input contracts for the chat use case."""

from typing import NotRequired

from pydantic import BaseModel, ConfigDict, Field, StrictBool
from typing_extensions import TypedDict

from services.core.contracts import RetrievalFilters


class RankingDiagnostic(TypedDict):
    position: int
    model_score: float
    exact_identifiers: list[str]
    rule: str


class ContextDiagnostic(TypedDict):
    id: int
    source: str
    document_id: NotRequired[str]
    kb_id: str
    kb_name_snapshot: str
    page: str
    routes: list[str]


class RetrievalDiagnosticPayload(TypedDict):
    original_question: str
    actual_query: str
    variant_query: str
    rewritten: bool
    fallback_attempted: bool
    skipped: bool
    rerank_status: str
    counts: dict[str, int]
    elapsed_ms: dict[str, float]
    ranking: list[RankingDiagnostic]
    context_sources: list[ContextDiagnostic]


class ChatRequest(BaseModel):
    """One question with optional session, image, model, and filters."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    question: str = Field(max_length=4000)
    kb_id: str
    # 不带时新建会话。
    session_id: str | None = None
    image_base64: str | None = None
    image_mime: str | None = None
    model_provider: str | None = None
    filters: RetrievalFilters | None = None
    # 同时搜索网页；需要角色有“联网搜索”权限。
    web_search: StrictBool = False
