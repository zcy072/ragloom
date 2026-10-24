import unittest

from services.retrieval.cache import RetrieverCache


class RetrieverCacheTest(unittest.TestCase):
    def test_get_refreshes_lru_order_before_eviction(self):
        cache = RetrieverCache[str](2)
        cache.put("first", "a")
        cache.put("second", "b")

        self.assertEqual(cache.get("first"), "a")
        cache.put("third", "c")

        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("first"), "a")
        self.assertEqual(cache.get("third"), "c")

    def test_put_replaces_drop_removes_and_clear_resets(self):
        cache = RetrieverCache[int](2)
        cache.put("kb", 1)
        cache.put("kb", 2)
        self.assertEqual(cache.get("kb"), 2)

        cache.drop("kb")
        cache.drop("missing")
        self.assertIsNone(cache.get("kb"))

        cache.put("one", 1)
        cache.put("two", 2)
        cache.clear()
        self.assertIsNone(cache.get("one"))
        self.assertIsNone(cache.get("two"))

    def test_nonpositive_size_still_keeps_one_item(self):
        cache = RetrieverCache[str](0)
        cache.put("first", "a")
        cache.put("second", "b")

        self.assertIsNone(cache.get("first"))
        self.assertEqual(cache.get("second"), "b")


if __name__ == "__main__":
    unittest.main()
