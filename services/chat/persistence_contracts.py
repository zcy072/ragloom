"""Typed records returned by chat session persistence."""

from typing_extensions import NotRequired, TypedDict


class CitationEvidence(TypedDict):
    """One citation snapshot; PDF page_numbers are one-based."""

    id: int
    source: str
    kb_name: str
    page: str
    page_numbers: list[int]
    context_title: str
    content: str


class ChatSource(TypedDict):
    """Citation data shared by prompts, SSE events, and persisted history."""

    id: int
    citation_ids: list[int]
    source: str
    page: str
    context_title: str
    content: str
    kb_name: NotRequired[str]
    evidence: NotRequired[list[CitationEvidence]]


class ChatSessionSummary(TypedDict):
    """One owned conversation shown in the session list."""

    id: str
    username: str
    kb_name: str
    title: str
    date: str


class ChatHistoryMessage(TypedDict):
    """One persisted message returned by the history endpoint."""

    role: str
    content: str
    sources: list[ChatSource]
