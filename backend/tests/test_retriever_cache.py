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


class CommonKnowledgeBaseInvalidationTest(unittest.TestCase):
    def test_common_knowledge_base_change_drops_every_cached_retriever(self):
        from unittest.mock import patch

        from services.retrieval import retriever, source_state

        common = "common-kb-id"
        cache = RetrieverCache[str](8)
        with patch.object(retriever, "_retriever_cache", cache), patch.object(
            source_state, "common_kb_id", return_value=common
        ):
            retriever.cache_retriever("dept-kb-id", "dept", include_common=True, version=retriever.retriever_cache_version())
            retriever.cache_retriever(common, "common", version=retriever.retriever_cache_version())
            retriever.drop_cached_retriever("dept-kb-id")
            self.assertIsNone(retriever.get_cached_retriever("dept-kb-id", include_common=True))
            self.assertEqual(retriever.get_cached_retriever(common), "common")

            retriever.cache_retriever("dept-kb-id", "dept", include_common=True, version=retriever.retriever_cache_version())
            retriever.drop_cached_retriever(common)
            # 部门库的检索图内含通用知识库的路线计划，也必须失效。
            self.assertIsNone(retriever.get_cached_retriever("dept-kb-id", include_common=True))
            self.assertIsNone(retriever.get_cached_retriever(common))

    def test_retriever_built_across_a_publish_is_not_cached(self):
        from unittest.mock import patch

        from services.retrieval import retriever, source_state

        cache = RetrieverCache[str](8)
        with patch.object(retriever, "_retriever_cache", cache), patch.object(
            source_state, "common_kb_id", return_value="common-kb-id"
        ):
            version = retriever.retriever_cache_version()
            # 构建检索图期间有文档发布，构建结果可能早于发布，不能写入缓存。
            retriever.drop_cached_retriever("dept-kb-id")
            retriever.cache_retriever("dept-kb-id", "stale", version=version)
            self.assertIsNone(retriever.get_cached_retriever("dept-kb-id"))

            retriever.cache_retriever(
                "dept-kb-id", "fresh", version=retriever.retriever_cache_version()
            )
            self.assertEqual(retriever.get_cached_retriever("dept-kb-id"), "fresh")
