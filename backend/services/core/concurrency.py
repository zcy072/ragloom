"""Process-wide concurrency limits for scarce external or CPU-bound resources."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_SLOTS: dict[str, tuple[int, threading.BoundedSemaphore]] = {}
_SLOTS_GUARD = threading.Lock()


@contextmanager
def resource_slot(name: str, limit: int) -> Iterator[None]:
    """限制同一资源在整个进程内的同时使用数。

    上限取自动态读取的配置，因此配置变化后会换用新的信号量；
    已持有旧信号量的调用方仍释放到它自己获取的那一个，互不影响。
    """
    limit = max(1, int(limit))
    with _SLOTS_GUARD:
        current = _SLOTS.get(name)
        if current is None or current[0] != limit:
            current = (limit, threading.BoundedSemaphore(limit))
            _SLOTS[name] = current
        semaphore = current[1]
    with semaphore:
        yield
