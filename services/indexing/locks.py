"""Per-knowledge-base synchronization for index mutations."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


@contextmanager
def knowledge_base_lock(kb_name: str) -> Iterator[None]:
    """Serialize index mutations for one knowledge base."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(kb_name, threading.RLock())
    with lock:
        yield
