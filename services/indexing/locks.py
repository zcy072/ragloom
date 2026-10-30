"""知识库与文件两级锁，协调同一知识库内的文件和索引变更。

整库操作（删除、改名、重建的登记与收尾）独占知识库；单文件操作（上传、删除、
改元数据或解析方式、索引任务的准备与发布）共享知识库，再独占该文件。
于是不同文件的变更可以同时进行，同一文件的变更和整库操作仍互斥。

获取顺序固定为先知识库、后文件。两种锁都不可重入，持锁期间不要再获取同一知识库的锁。
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class _KnowledgeBaseLock:
    """写优先的读写锁：有整库操作在等待时，新的单文件操作先等它完成。"""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def acquire_shared(self) -> None:
        with self._condition:
            while self._writer or self._waiting_writers:
                self._condition.wait()
            self._readers += 1

    def release_shared(self) -> None:
        with self._condition:
            self._readers -= 1
            if not self._readers:
                self._condition.notify_all()

    def acquire_exclusive(self) -> None:
        with self._condition:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers:
                    self._condition.wait()
            finally:
                self._waiting_writers -= 1
            self._writer = True

    def release_exclusive(self) -> None:
        with self._condition:
            self._writer = False
            self._condition.notify_all()


class _Registry:
    """按键创建锁，没有持有者也没有等待者时移除，避免文件锁随文件数无限增长。"""

    def __init__(self, factory) -> None:
        self._factory = factory
        self._guard = threading.Lock()
        self._entries: dict = {}

    @contextmanager
    def borrow(self, key) -> Iterator:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = [self._factory(), 0]
            entry[1] += 1
        try:
            yield entry[0]
        finally:
            with self._guard:
                entry[1] -= 1
                if not entry[1]:
                    del self._entries[key]

    def __len__(self) -> int:
        with self._guard:
            return len(self._entries)


_KNOWLEDGE_BASE_LOCKS = _Registry(_KnowledgeBaseLock)
_DOCUMENT_LOCKS = _Registry(threading.Lock)


@contextmanager
def knowledge_base_lock(kb_name: str) -> Iterator[None]:
    """独占一个知识库，等待其中所有单文件操作结束，用于整库变更。"""
    with _KNOWLEDGE_BASE_LOCKS.borrow(kb_name) as lock:
        lock.acquire_exclusive()
        try:
            yield
        finally:
            lock.release_exclusive()


@contextmanager
def document_lock(kb_name: str, filename: str) -> Iterator[None]:
    """共享知识库并独占其中一个文件，用于单文件变更。"""
    with _KNOWLEDGE_BASE_LOCKS.borrow(kb_name) as kb_lock:
        kb_lock.acquire_shared()
        try:
            with _DOCUMENT_LOCKS.borrow((kb_name, filename)) as file_lock, file_lock:
                yield
        finally:
            kb_lock.release_shared()
