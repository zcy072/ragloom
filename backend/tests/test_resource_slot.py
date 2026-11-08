import threading
import unittest

from services.core.concurrency import resource_slot


class ResourceSlotTest(unittest.TestCase):
    def _peak_concurrency(self, name, limit, workers=6):
        guard = threading.Lock()
        state = {"active": 0, "peak": 0}

        def work():
            with resource_slot(name, limit):
                with guard:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                threading.Event().wait(0.05)
                with guard:
                    state["active"] -= 1

        threads = [threading.Thread(target=work) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        return state["peak"]

    def test_limits_concurrent_holders(self):
        self.assertEqual(self._peak_concurrency("test-limit", 2), 2)

    def test_limit_of_one_serializes(self):
        self.assertEqual(self._peak_concurrency("test-serial", 1), 1)

    def test_changed_limit_takes_effect_for_new_callers(self):
        self.assertEqual(self._peak_concurrency("test-change", 1), 1)
        self.assertEqual(self._peak_concurrency("test-change", 3), 3)

    def test_slot_is_released_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with resource_slot("test-exception", 1):
                raise RuntimeError("boom")
        acquired = threading.Event()

        def work():
            with resource_slot("test-exception", 1):
                acquired.set()

        thread = threading.Thread(target=work)
        thread.start()
        self.assertTrue(acquired.wait(2))
        thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
