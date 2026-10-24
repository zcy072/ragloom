"""Typed metadata structures shared by document use cases and indexing."""

from collections.abc import Mapping
from typing import TypedDict

DocumentMetadataInput = Mapping[str, object]


class DocumentMetadata(TypedDict):
    """Validated metadata persisted for a document in the application database."""

    document_type: str
    tags: list[str]
    effective_date: str


class DocumentRecord(DocumentMetadata):
    """Document management record mapped from the application database."""

    filename: str
    path: str
    size: int
    uploaded_at: str
    status: str
    error_message: str
    index_job_id: str
    parser_mode: str


class DocumentFilterOptions(TypedDict):
    """Distinct document fields available to retrieval filters."""

    sources: list[str]
    document_types: list[str]
    tags: list[str]


class DocumentParserOption(TypedDict):
    """One parser choice exposed to document upload clients."""

    id: str
    label: str
    available: bool
    unavailable_reason: str
    extensions: list[str]


class DocumentFormatOption(TypedDict):
    """One supported upload extension and its display label."""

    extension: str
    label: str
