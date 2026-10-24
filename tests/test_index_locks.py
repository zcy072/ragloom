import threading
import unittest

from services.indexing.locks import knowledge_base_lock


class IndexLockTest(unittest.TestCase):
    def test_same_knowledge_base_is_serialized(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def hold_first_lock():
            with knowledge_base_lock("研发部"):
                first_entered.set()
                release_first.wait(timeout=2)

        def wait_for_same_lock():
            first_entered.wait(timeout=2)
            with knowledge_base_lock("研发部"):
                second_entered.set()

        first = threading.Thread(target=hold_first_lock)
        second = threading.Thread(target=wait_for_same_lock)
        first.start()
        second.start()
        self.assertTrue(first_entered.wait(timeout=2))
        self.assertFalse(second_entered.wait(timeout=0.05))
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertTrue(second_entered.is_set())

    def test_different_knowledge_bases_do_not_block_each_other(self):
        with knowledge_base_lock("研发部"):
            entered = threading.Event()

            def acquire_other_lock():
                with knowledge_base_lock("财务部"):
                    entered.set()

            thread = threading.Thread(target=acquire_other_lock)
            thread.start()
            self.assertTrue(entered.wait(timeout=2))
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
