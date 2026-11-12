"""Request-scoped document_id state for consistent hybrid retrieval."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from services.core.container import ApplicationContainer


@dataclass(frozen=True)
class RetrievalDocumentState:
    """Immutable request snapshot of ready sources."""

    ready_document_ids: tuple[str, ...]


_active_document_states: ContextVar[dict[str, RetrievalDocumentState] | None] = ContextVar(
    "active_retrieval_document_states", default=None
)


def ready_document_ids(
    kb_id: str, container: ApplicationContainer
) -> tuple[str, ...]:
    """返回本次请求冻结的就绪文件列表；未冻结时读取最新持久化状态。"""
    states = _active_document_states.get()
    if states is not None and kb_id in states:
        return states[kb_id].ready_document_ids
    return tuple(container.documents.ready_document_ids(kb_id))


def current_document_allowlist(
    kb_id: str,
    requested_document_ids: tuple[str, ...] | list[str] | None,
    container: ApplicationContainer,
) -> tuple[str, ...]:
    """将请求范围收敛到当前请求快照中的就绪文件。"""
    current_document_ids = ready_document_ids(kb_id, container)
    if requested_document_ids is None:
        return current_document_ids
    requested = set(requested_document_ids)
    return tuple(document_id for document_id in current_document_ids if document_id in requested)


def common_kb_id(container: ApplicationContainer) -> str | None:
    """返回通用知识库的 ID；它由 system_key 识别，与显示名称无关。"""
    return container.knowledge_bases.common_kb_id()


def supplements_common(
    kb_id: str, include_common: bool, container: ApplicationContainer
) -> str | None:
    """本次检索要补充通用知识库时返回其 ID，否则返回 None。

    调用方须已确认提问者能读取通用知识库；查询通用知识库本身时不重复补充。
    """
    if not include_common:
        return None
    common_id = common_kb_id(container)
    return common_id if common_id and common_id != kb_id else None


@contextmanager
def request_document_snapshot(
    kb_id: str,
    container: ApplicationContainer,
    *,
    include_common: bool = False,
) -> Iterator[None]:
    """在一次问答中冻结文件状态，保证多路检索看到同一份允许文件列表。"""
    documents = container.documents
    knowledge_bases = [kb_id]
    common_id = supplements_common(kb_id, include_common, container)
    if common_id:
        knowledge_bases.append(common_id)

    states = {}
    for name in knowledge_bases:
        states[name] = RetrievalDocumentState(tuple(documents.ready_document_ids(name)))

    token = _active_document_states.set(states)
    try:
        yield
    finally:
        _active_document_states.reset(token)
