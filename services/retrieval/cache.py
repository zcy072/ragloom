"""Thread-safe bounded cache for reusable retrieval graphs."""

import collections
import threading
from collections.abc import Hashable


class RetrieverCache[T]:
    """A small LRU cache keyed by knowledge-base name or a key derived from it."""

    def __init__(self, max_size: int) -> None:
        """创建指定容量的线程安全 LRU 缓存，容量至少为一项。"""
        self._max_size = max(1, max_size)
        self._items: collections.OrderedDict[Hashable, T] = collections.OrderedDict()
        self._lock = threading.RLock()

    def get(self, name: Hashable) -> T | None:
        """读取知识库缓存；命中时将该项标记为最近使用。"""
        with self._lock:
            if name not in self._items:
                return None
            self._items.move_to_end(name)
            return self._items[name]

    def put(self, name: Hashable, value: T) -> None:
        """写入或更新缓存，并淘汰超出容量的最久未使用项。"""
        with self._lock:
            self._items[name] = value
            self._items.move_to_end(name)
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """清空全部知识库检索器缓存。"""
        with self._lock:
            self._items.clear()

    def drop(self, name: Hashable) -> None:
        """移除指定知识库的缓存；不存在时保持静默。"""
        with self._lock:
            self._items.pop(name, None)
