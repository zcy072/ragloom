import threading
import unittest

from services.indexing import locks
from services.indexing.locks import document_lock, knowledge_base_lock

WAIT = 2


def _hold(lock_factory, entered: threading.Event, release: threading.Event) -> threading.Thread:
    """在后台线程获取锁并持有到 release 被设置。"""

    def target():
        with lock_factory():
            entered.set()
            release.wait(WAIT)

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread


def _try(lock_factory) -> tuple[threading.Thread, threading.Event]:
    """在后台线程尝试获取锁，返回线程和“已获取”事件。"""
    acquired = threading.Event()

    def target():
        with lock_factory():
            acquired.set()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, acquired


class IndexLockTest(unittest.TestCase):
    def _assert_blocks(self, holder, waiter):
        """holder 持有期间 waiter 拿不到锁，holder 释放后 waiter 立刻拿到。"""
        entered, release = threading.Event(), threading.Event()
        holding = _hold(holder, entered, release)
        self.assertTrue(entered.wait(WAIT))
        waiting, acquired = _try(waiter)
        self.assertFalse(acquired.wait(0.05))
        release.set()
        self.assertTrue(acquired.wait(WAIT))
        holding.join(WAIT)
        waiting.join(WAIT)

    def _assert_does_not_block(self, holder, other):
        entered, release = threading.Event(), threading.Event()
        holding = _hold(holder, entered, release)
        self.assertTrue(entered.wait(WAIT))
        try:
            waiting, acquired = _try(other)
            self.assertTrue(acquired.wait(WAIT))
        finally:
            release.set()
        holding.join(WAIT)
        waiting.join(WAIT)

    def test_same_knowledge_base_is_serialized(self):
        self._assert_blocks(
            lambda: knowledge_base_lock("研发部"), lambda: knowledge_base_lock("研发部")
        )

    def test_different_knowledge_bases_do_not_block_each_other(self):
        self._assert_does_not_block(
            lambda: knowledge_base_lock("研发部"), lambda: knowledge_base_lock("财务部")
        )

    def test_same_file_is_serialized(self):
        self._assert_blocks(
            lambda: document_lock("研发部", "甲.pdf"), lambda: document_lock("研发部", "甲.pdf")
        )

    def test_different_files_in_one_knowledge_base_do_not_block_each_other(self):
        self._assert_does_not_block(
            lambda: document_lock("研发部", "甲.pdf"), lambda: document_lock("研发部", "乙.pdf")
        )

    def test_knowledge_base_lock_waits_for_file_locks(self):
        self._assert_blocks(
            lambda: document_lock("研发部", "甲.pdf"), lambda: knowledge_base_lock("研发部")
        )

    def test_file_locks_wait_for_knowledge_base_lock(self):
        self._assert_blocks(
            lambda: knowledge_base_lock("研发部"), lambda: document_lock("研发部", "甲.pdf")
        )

    def test_waiting_knowledge_base_lock_holds_back_new_file_locks(self):
        # 写优先：整库操作在等待时，后来的单文件操作排在它后面，不会让它一直等下去。
        first_entered, release_first = threading.Event(), threading.Event()
        first = _hold(lambda: document_lock("研发部", "甲.pdf"), first_entered, release_first)
        self.assertTrue(first_entered.wait(WAIT))

        order: list[str] = []
        kb_entered = threading.Event()

        def whole_kb():
            with knowledge_base_lock("研发部"):
                order.append("kb")
                kb_entered.set()

        kb_thread = threading.Thread(target=whole_kb, daemon=True)
        kb_thread.start()
        # 等整库锁进入等待状态。
        lock_entry = None
        for _ in range(200):
            with locks._KNOWLEDGE_BASE_LOCKS._guard:
                lock_entry = locks._KNOWLEDGE_BASE_LOCKS._entries.get("研发部")
            if lock_entry and lock_entry[0]._waiting_writers:
                break
            threading.Event().wait(0.01)
        self.assertTrue(lock_entry and lock_entry[0]._waiting_writers)

        def other_file():
            with document_lock("研发部", "乙.pdf"):
                order.append("file")

        file_thread = threading.Thread(target=other_file, daemon=True)
        file_thread.start()
        self.assertFalse(kb_entered.wait(0.05))
        release_first.set()
        for thread in (first, kb_thread, file_thread):
            thread.join(WAIT)
            self.assertFalse(thread.is_alive())
        self.assertEqual(order, ["kb", "file"])

    def test_released_locks_are_removed(self):
        with document_lock("研发部", "甲.pdf"):
            pass
        with knowledge_base_lock("研发部"):
            pass

        self.assertEqual(len(locks._DOCUMENT_LOCKS), 0)
        self.assertEqual(len(locks._KNOWLEDGE_BASE_LOCKS), 0)

    def test_exception_inside_lock_releases_it(self):
        with self.assertRaises(RuntimeError):
            with document_lock("研发部", "甲.pdf"):
                raise RuntimeError("boom")

        thread, acquired = _try(lambda: knowledge_base_lock("研发部"))
        self.assertTrue(acquired.wait(WAIT))
        thread.join(WAIT)


if __name__ == "__main__":
    unittest.main()
