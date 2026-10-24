"""Request-scoped source state for consistent hybrid retrieval."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from services.core.container import ApplicationContainer, get_application_container

COMMON_KNOWLEDGE_BASE = "通用知识库"


@dataclass(frozen=True)
class RetrievalSourceState:
    """Immutable request snapshot of ready and FTS-only sources."""
    ready_sources: tuple[str, ...]
    chroma_disabled_sources: frozenset[str]


_active_source_states: ContextVar[dict[str, RetrievalSourceState] | None] = (
    ContextVar("active_retrieval_source_states", default=None)
)


def ready_sources(
    kb_name: str, container: ApplicationContainer | None = None
) -> tuple[str, ...]:
    """返回本次请求冻结的就绪文件列表；未冻结时读取最新持久化状态。"""
    states = _active_source_states.get()
    if states is not None and kb_name in states:
        return states[kb_name].ready_sources
    return tuple((container or get_application_container()).documents.ready_sources(kb_name))


def current_source_allowlist(
    kb_name: str,
    requested_sources: tuple[str, ...] | list[str] | None,
    container: ApplicationContainer | None = None,
) -> tuple[str, ...]:
    """将请求范围收敛到当前请求快照中的就绪文件。"""
    current_sources = ready_sources(kb_name, container)
    if requested_sources is None:
        return current_sources
    requested = set(requested_sources)
    return tuple(source for source in current_sources if source in requested)


def chroma_disabled_sources(
    kb_name: str, container: ApplicationContainer | None = None
) -> frozenset[str]:
    """返回因 Chroma 写入失败而暂时只能走关键词检索的文件。"""
    states = _active_source_states.get()
    if states is not None and kb_name in states:
        return states[kb_name].chroma_disabled_sources
    return frozenset(
        (container or get_application_container()).documents.chroma_disabled_sources(kb_name)
    )


@contextmanager
def request_source_snapshot(
    kb_name: str, container: ApplicationContainer | None = None
) -> Iterator[None]:
    """在一次问答中冻结文件状态，保证多路检索看到同一份允许文件列表。"""
    documents = (container or get_application_container()).documents
    knowledge_bases = [kb_name]
    if (
        kb_name != COMMON_KNOWLEDGE_BASE
        and documents.knowledge_base_exists(COMMON_KNOWLEDGE_BASE)
    ):
        knowledge_bases.append(COMMON_KNOWLEDGE_BASE)

    states = {}
    for name in knowledge_bases:
        ready, disabled = documents.retrieval_source_state(name)
        states[name] = RetrievalSourceState(tuple(ready), frozenset(disabled))

    token = _active_source_states.set(states)
    try:
        yield
    finally:
        _active_source_states.reset(token)
