"""Public chunking API across fixed, structured, and visual routes."""

from . import common as _common
from .common import (
    CHUNK_ROUTE_FIXED,
    CHUNK_ROUTE_STRUCTURED,
    CHUNK_ROUTE_VISUAL,
    FIXED_CHILD_MIN_CHARS,
    STRUCTURED_CHUNK_MAX_CHARS,
    STRUCTURED_CHUNK_MIN_CHARS,
    assign_chunk_indexes,
    chunk_route,
    contextualized_chunk_text,
)
from .fixed import split_documents, split_mineru_visual_documents
from .structured import split_structured_markdown_documents

# Retain direct access for focused tests without advertising these helpers as public API.
_adjacent_split_key = _common._adjacent_split_key
_merge_small_child_chunks = _common._merge_small_child_chunks
_semantic_chunk_length = _common._semantic_chunk_length

__all__ = [
    "CHUNK_ROUTE_FIXED",
    "CHUNK_ROUTE_STRUCTURED",
    "CHUNK_ROUTE_VISUAL",
    "FIXED_CHILD_MIN_CHARS",
    "STRUCTURED_CHUNK_MAX_CHARS",
    "STRUCTURED_CHUNK_MIN_CHARS",
    "assign_chunk_indexes",
    "chunk_route",
    "contextualized_chunk_text",
    "split_documents",
    "split_mineru_visual_documents",
    "split_structured_markdown_documents",
]
