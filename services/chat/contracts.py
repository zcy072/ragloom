"""Framework-independent input contracts for the chat use case."""

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
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
    kb_name: str
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
    kb_name: str
    session_id: str | None = None
    new_session: StrictBool = False
    image_base64: str | None = None
    image_mime: str | None = None
    model_provider: str | None = None
    filters: RetrievalFilters | None = None

    @model_validator(mode="after")
    def validate_session_choice(self):
        """Reject ambiguous requests that select and create a session together."""
        if self.new_session and self.session_id:
            raise ValueError("新会话请求不能同时指定已有会话 ID")
        return self
