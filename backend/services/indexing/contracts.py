"""Typed records exposed by the durable indexing task store."""

from typing import Literal, TypedDict

IndexJobOperation = Literal["process", "rebuild", "delete"]
IndexJobStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "superseded",
    "interrupted",
]
TerminalIndexJobStatus = Literal["completed", "failed", "superseded", "interrupted"]


class IndexJob(TypedDict):
    """Task fields required by the indexing worker dispatcher."""

    id: str
    kb_id: str
    document_id: str | None
    filename: str | None
    operation: IndexJobOperation
    status: IndexJobStatus


class IndexJobSummary(TypedDict):
    """Public task history fields returned by the knowledge-base API."""

    id: str
    document_id: str | None
    filename: str | None
    operation: IndexJobOperation
    status: IndexJobStatus
    error_message: str
    created_at: str
    started_at: str | None
    finished_at: str | None


class CrossKnowledgeBaseJobSummary(IndexJobSummary):
    """Task summary that identifies its knowledge base, for the ops view.

    `kb_name` is the current display name; `kb_id` is the identity.
    """

    kb_id: str
    kb_name: str
