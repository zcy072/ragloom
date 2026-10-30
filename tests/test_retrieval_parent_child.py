import asyncio
import dataclasses
import unittest
from unittest.mock import patch

try:
    from langchain_core.documents import Document

    import app_storage
    from services.core.container import get_application_container
    from services.indexing import chunking
    from services.retrieval import context as context_builder
    from services.retrieval import parent_store, retriever, source_state
    from services.retrieval.cache import RetrieverCache
    from services.retrieval.chunk_index import (
        build_filter,
        embed_splits,
        escape_filter_value,
        get_chunk_index,
    )
except ModuleNotFoundError:
    Document = None
    source_state = None

from tests.retrieval_fixtures import create_knowledge_base, mark_ready, publish
from tests.support import IsolatedStorageTestCase


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, query):
        return [1.0, 0.0]


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self):
        self.query_calls = 0

    def embed_query(self, query):
        self.query_calls += 1
        return super().embed_query(query)


@unittest.skipUnless(Document and source_state, "requires the backend Python dependencies")
class ParentChildRetrievalTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        self.kb_name = "人力资源部"
        self.child = Document(
            page_content="子块中的审批规则",
            metadata={
                "source": "制度.pdf",
                "page": 2,
                "parent_id": "parent-1",
                "parent_content": "父块完整内容，包含审批规则和适用范围。",
            },
        )

    def index(self):
        return get_chunk_index()

    def search(self, kb_id, query, sources=None, route=None, k=5):
        return self.index().search_hybrid(kb_id, query, [1.0, 0.0], route, sources, k)

    def test_filter_escapes_backslashes_and_quotes(self):
        self.assertEqual(escape_filter_value('a"b\\c'), 'a\\"b\\\\c')
        expression = build_filter(
            "kb-1",
            source='报告"终版"\\草稿.pdf',
            sources=['x" or kb_id != "', "正常.pdf"],
            route="fixed",
        )
        self.assertEqual(
            expression,
            'kb_id == "kb-1" and source == "报告\\"终版\\"\\\\草稿.pdf" '
            'and source in ["x\\" or kb_id != \\"", "正常.pdf"] '
            'and chunk_route == "fixed"',
        )

    def test_filter_with_hostile_filename_matches_only_that_source(self):
        hostile = '合同"终版"\\草稿.pdf'
        kb_id = publish(
            self.kb_name, hostile, [Document(page_content="审批条款", metadata={"source": hostile})]
        )
        publish(
            self.kb_name, "正常.pdf", [Document(page_content="审批条款", metadata={"source": "正常.pdf"})]
        )

        found = self.search(kb_id, "审批条款", sources=[hostile])

        self.assertEqual([document.metadata["source"] for document in found], [hostile])

    def test_parent_rows_keep_each_parent_once_and_scope_by_kb(self):
        second_child = Document(
            page_content="子块中的适用范围",
            metadata={"source": "制度.pdf", "page": 2, "parent_id": "parent-1"},
        )
        kb_id = create_knowledge_base(self.kb_name)
        other_kb_id = create_knowledge_base("研发部")

        rows = parent_store.parent_chunk_rows([self.child, second_child])
        parent_store.replace_source(kb_id, "制度.pdf", rows)

        self.assertEqual(
            rows,
            [("parent-1", "制度.pdf", "2", self.child.metadata["parent_content"], "")],
        )
        self.assertEqual(set(parent_store.load(kb_id, ["parent-1"])), {"parent-1"})
        self.assertEqual(parent_store.load(other_kb_id, ["parent-1"]), {})

    def test_parent_replace_and_delete(self):
        kb_id = create_knowledge_base(self.kb_name)
        parent_store.replace_source(kb_id, "制度.pdf", [("p1", "制度.pdf", "1", "旧内容", "")])
        parent_store.replace_source(kb_id, "制度.pdf", [("p2", "制度.pdf", "1", "新内容", "")])
        parent_store.replace_source(kb_id, "其他.pdf", [("p3", "其他.pdf", "1", "其他", "")])

        self.assertEqual(set(parent_store.load(kb_id, ["p1", "p2", "p3"])), {"p2", "p3"})
        parent_store.delete_source(kb_id, "制度.pdf")
        parent_store.delete_source(kb_id, "制度.pdf")
        self.assertEqual(set(parent_store.load(kb_id, ["p2", "p3"])), {"p3"})
        parent_store.delete_kb(kb_id)
        self.assertEqual(parent_store.load(kb_id, ["p3"]), {})

    def test_parent_text_is_stored_once_and_restored_after_retrieval(self):
        second_child = Document(
            page_content="子块中的适用范围",
            metadata={"source": "制度.pdf", "page": 2, "parent_id": "parent-1"},
        )
        kb_id = publish(self.kb_name, "制度.pdf", [self.child, second_child])
        parent_store.replace_source(
            kb_id, "制度.pdf", parent_store.parent_chunk_rows([self.child, second_child])
        )

        found = self.search(kb_id, "审批规则")
        self.assertEqual(found[0].page_content, self.child.page_content)
        self.assertNotIn("parent_content", found[0].metadata)

        spans = context_builder.build_context_spans(self.kb_name, found, neighbor_chunks=0)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].content, self.child.metadata["parent_content"])

        restored = context_builder.restore_parent_documents(
            self.kb_name, self.search(kb_id, "审批规则")
        )
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].page_content, self.child.metadata["parent_content"])

    def test_request_source_snapshot_is_shared_by_retrievers_and_worker_threads(self):
        states = [["现行制度.pdf", "降级方案.pdf"], ["新版制度.pdf"]]
        with patch.object(
            app_storage.documents, "knowledge_base_exists", return_value=False
        ), patch.object(
            app_storage.documents, "ready_sources", side_effect=states
        ) as ready_read:
            with source_state.request_source_snapshot(self.kb_name):
                self.assertEqual(
                    source_state.current_source_allowlist(self.kb_name, None),
                    ("现行制度.pdf", "降级方案.pdf"),
                )
                threaded_sources = asyncio.run(
                    asyncio.to_thread(source_state.ready_sources, self.kb_name)
                )
                self.assertEqual(threaded_sources, ("现行制度.pdf", "降级方案.pdf"))

            with source_state.request_source_snapshot(self.kb_name):
                self.assertEqual(
                    source_state.ready_sources(self.kb_name), ("新版制度.pdf",)
                )

        self.assertEqual(ready_read.call_count, 2)

    def _publish_department_and_common(self):
        for kb_name, source in ((self.kb_name, "部门制度.pdf"), (source_state.COMMON_KNOWLEDGE_BASE, "通用制度.pdf")):
            publish(kb_name, source, [Document(page_content="审批规则", metadata={"source": source})])
            mark_ready(kb_name, source)

    def test_common_knowledge_base_is_supplemented_only_when_requested(self):
        self._publish_department_and_common()

        container = dataclasses.replace(
            get_application_container(), create_query_embeddings=FakeEmbeddings
        )

        def retrieved_sources(include_common):
            with source_state.request_source_snapshot(
                self.kb_name, container, include_common=include_common
            ):
                dense = retriever.build_retriever(
                    self.kb_name, include_common=include_common, container=container
                ).invoke("审批规则")
                lexical = retriever.retrieve_lexical_candidates(
                    self.kb_name, "审批规则", include_common=include_common, container=container
                )
            return {document.metadata["source"] for document in dense + lexical}

        self.assertEqual(retrieved_sources(False), {"部门制度.pdf"})
        self.assertEqual(retrieved_sources(True), {"部门制度.pdf", "通用制度.pdf"})

    def test_retriever_cache_keeps_supplemented_and_plain_graphs_apart(self):
        with patch.object(retriever, "_retriever_cache", RetrieverCache(8)):
            retriever.cache_retriever(self.kb_name, "with-common", include_common=True)
            self.assertIsNone(retriever.get_cached_retriever(self.kb_name))
            retriever.cache_retriever(self.kb_name, "plain")
            self.assertEqual(
                retriever.get_cached_retriever(self.kb_name, include_common=True), "with-common"
            )
            retriever.drop_cached_retriever(self.kb_name)
            self.assertIsNone(retriever.get_cached_retriever(self.kb_name))
            self.assertIsNone(retriever.get_cached_retriever(self.kb_name, include_common=True))

    def test_query_embedding_is_shared_across_route_searches(self):
        base_embeddings = CountingEmbeddings()
        embeddings = retriever.CachedQueryEmbeddings(base_embeddings)
        kb_id = create_knowledge_base(self.kb_name)
        mark_ready(self.kb_name, "制度.pdf")
        plans = [
            retriever._RoutePlan(self.kb_name, kb_id, route, 1 / 3, None)
            for route in (
                chunking.CHUNK_ROUTE_FIXED,
                chunking.CHUNK_ROUTE_STRUCTURED,
                chunking.CHUNK_ROUTE_VISUAL,
            )
        ]
        with patch.object(self.index(), "search_hybrid", return_value=[]) as search:
            retriever.WeightedRouteRetriever(self.index(), embeddings, plans).invoke("审批规则")

        self.assertEqual(search.call_count, 3)
        self.assertEqual(base_embeddings.query_calls, 1)

    def test_query_vectors_are_shared_across_retriever_instances(self):
        base_embeddings = CountingEmbeddings()

        retriever.CachedQueryEmbeddings(base_embeddings).embed_query("审批规则")
        retriever.CachedQueryEmbeddings(base_embeddings).embed_query("审批规则")

        self.assertEqual(base_embeddings.query_calls, 1)

    def test_route_probe_is_cached_until_the_knowledge_base_changes(self):
        publish(
            self.kb_name,
            "制度.pdf",
            [Document(page_content="固定块", metadata={"source": "制度.pdf"})],
        )
        mark_ready(self.kb_name, "制度.pdf")
        index = self.index()

        with patch.object(index, "available_routes", wraps=index.available_routes) as probe:
            for _ in range(2):
                retriever._route_plans(app_storage_container(), index, self.kb_name, None, 0.7)
            self.assertEqual(probe.call_count, 1)

            retriever.drop_cached_retriever(self.kb_name)
            retriever._route_plans(app_storage_container(), index, self.kb_name, None, 0.7)
            self.assertEqual(probe.call_count, 2)

    def test_route_probe_result_is_not_cached_when_invalidated_meanwhile(self):
        cache = retriever._RouteCache()

        def probe():
            cache.drop("研发部")
            return {"fixed"}

        self.assertEqual(cache.routes("研发部", ("a.pdf",), probe), {"fixed"})
        self.assertEqual(cache.routes("研发部", ("a.pdf",), lambda: {"visual"}), {"visual"})

    def test_route_weights_renormalize_over_existing_routes(self):
        kb_id = publish(
            self.kb_name,
            "制度.pdf",
            [Document(page_content="固定块", metadata={"source": "制度.pdf"})],
        )
        mark_ready(self.kb_name, "制度.pdf")

        plans = retriever._route_plans(
            app_storage_container(), self.index(), self.kb_name, None, 0.70
        )

        self.assertEqual([plan.route for plan in plans], [chunking.CHUNK_ROUTE_FIXED])
        self.assertAlmostEqual(plans[0].weight, 0.70)
        self.assertEqual(plans[0].kb_id, kb_id)

    def test_fusion_ranks_higher_weighted_route_first(self):
        source = "制度.pdf"
        fixed = Document(
            page_content="审批规则固定块",
            metadata={"source": source, "chunk_route": chunking.CHUNK_ROUTE_FIXED},
        )
        visual = Document(
            page_content="审批规则视觉块",
            metadata={"source": source, "chunk_route": chunking.CHUNK_ROUTE_VISUAL},
        )
        kb_id = publish(self.kb_name, source, [fixed, visual])
        mark_ready(self.kb_name, source)
        plans = [
            retriever._RoutePlan(self.kb_name, kb_id, chunking.CHUNK_ROUTE_VISUAL, 0.15, None),
            retriever._RoutePlan(self.kb_name, kb_id, chunking.CHUNK_ROUTE_FIXED, 0.60, None),
        ]

        found = retriever.WeightedRouteRetriever(
            self.index(), FakeEmbeddings(), plans
        ).invoke("审批规则")

        self.assertEqual(
            [document.metadata["chunk_route"] for document in found],
            [chunking.CHUNK_ROUTE_FIXED, chunking.CHUNK_ROUTE_VISUAL],
        )
        self.assertTrue(all(document.metadata["kb_name"] == self.kb_name for document in found))

    def test_retrieval_excludes_sources_not_ready_in_sqlite(self):
        kb_id = publish(
            self.kb_name, "旧制度.pdf", [Document(page_content="审批规则", metadata={"source": "旧制度.pdf"})]
        )
        publish(
            self.kb_name, "现行制度.pdf", [Document(page_content="审批规则", metadata={"source": "现行制度.pdf"})]
        )
        mark_ready(self.kb_name, "现行制度.pdf")
        plan = retriever._RoutePlan(
            self.kb_name, kb_id, chunking.CHUNK_ROUTE_FIXED, 1.0, ("旧制度.pdf", "现行制度.pdf")
        )

        found = retriever.WeightedRouteRetriever(
            self.index(), FakeEmbeddings(), [plan]
        ).invoke("审批规则")

        self.assertEqual([document.metadata["source"] for document in found], ["现行制度.pdf"])

    def test_search_failure_propagates_instead_of_returning_empty(self):
        kb_id = create_knowledge_base(self.kb_name)
        mark_ready(self.kb_name, "制度.pdf")
        plan = retriever._RoutePlan(self.kb_name, kb_id, chunking.CHUNK_ROUTE_FIXED, 1.0, None)

        with patch.object(self.index(), "search_hybrid", side_effect=RuntimeError("milvus down")):
            with self.assertRaisesRegex(RuntimeError, "milvus down"):
                retriever.WeightedRouteRetriever(
                    self.index(), FakeEmbeddings(), [plan]
                ).invoke("审批规则")

    def test_publish_replaces_previous_generation(self):
        source = "制度.pdf"
        kb_id = publish(self.kb_name, source, [self.child])
        publish(self.kb_name, source, [Document(page_content="更新后的审批规则", metadata={"source": source})])

        found = self.search(kb_id, "审批规则", sources=[source])

        self.assertEqual([document.page_content for document in found], ["更新后的审批规则"])

    def test_failed_publish_keeps_previous_generation(self):
        source = "制度.pdf"
        kb_id = publish(self.kb_name, source, [self.child])
        self.index().fail_next_publish = RuntimeError("batch write failed")

        with self.assertRaisesRegex(RuntimeError, "batch write failed"):
            publish(self.kb_name, source, [Document(page_content="新版内容", metadata={"source": source})])

        found = self.search(kb_id, "审批规则", sources=[source])
        self.assertEqual([document.page_content for document in found], [self.child.page_content])

    def test_publish_rejects_mismatched_vector_count(self):
        kb_id = create_knowledge_base(self.kb_name)
        with self.assertRaisesRegex(ValueError, "向量数量"):
            self.index().publish_source(kb_id, "制度.pdf", [self.child], [])
        self.assertEqual(self.index().available_routes(kb_id, None), set())

    def test_publish_uses_prefetched_vectors_without_embedding(self):
        vectors = embed_splits([self.child], FakeEmbeddings())
        kb_id = create_knowledge_base(self.kb_name)

        self.index().publish_source(kb_id, "制度.pdf", [self.child], vectors)

        self.assertEqual(self.index().available_routes(kb_id, ["制度.pdf"]), {chunking.CHUNK_ROUTE_FIXED})

    def test_rename_keeps_vectors_reachable_through_stable_kb_id(self):
        kb_id = publish(self.kb_name, "制度.pdf", [self.child])
        before = self.index().available_routes(kb_id, None)

        app_storage.knowledge_bases.rename_relational_data(
            self.kb_name, "研发部", "/tmp/old", "/tmp/new"
        )

        self.assertEqual(app_storage.knowledge_bases.kb_id_for("研发部"), kb_id)
        self.assertIsNone(app_storage.knowledge_bases.kb_id_for(self.kb_name))
        self.assertEqual(self.index().available_routes(kb_id, None), before)
        self.assertEqual(
            [document.metadata["source"] for document in self.search(kb_id, "审批规则")],
            ["制度.pdf"],
        )

    def test_delete_source_kb_and_stale_cleanup(self):
        kb_id = publish(self.kb_name, "制度.pdf", [self.child])
        publish(self.kb_name, "旧制度.pdf", [Document(page_content="旧制度内容", metadata={"source": "旧制度.pdf"})])
        other_kb_id = publish("研发部", "制度.pdf", [self.child])

        removed = self.index().delete_stale_sources(kb_id, {"制度.pdf"})

        self.assertEqual(removed, 1)
        self.assertEqual(self.search(kb_id, "旧制度内容", sources=["旧制度.pdf"]), [])
        self.assertEqual(len(self.search(kb_id, "审批规则", sources=["制度.pdf"])), 1)

        self.index().delete_source(kb_id, "制度.pdf")
        self.index().delete_source(kb_id, "制度.pdf")
        self.assertEqual(self.index().available_routes(kb_id, None), set())
        self.assertEqual(len(self.search(other_kb_id, "审批规则")), 1)

        self.index().delete_kb(other_kb_id)
        self.index().delete_kb(other_kb_id)
        self.assertEqual(self.index().available_routes(other_kb_id, None), set())

    def test_reset_if_empty_only_clears_when_no_documents_remain(self):
        kb_id = publish(self.kb_name, "制度.pdf", [self.child])

        self.assertFalse(self.index().reset_if_empty(lambda: True))
        self.assertEqual(len(self.search(kb_id, "审批规则")), 1)
        self.assertTrue(self.index().reset_if_empty(lambda: False))
        self.assertEqual(self.search(kb_id, "审批规则"), [])

    def test_reset_if_empty_rechecks_before_dropping(self):
        kb_id = publish(self.kb_name, "制度.pdf", [self.child])
        answers = iter([False, True])  # 第一次判断为空，拿到锁后已有新文档登记

        self.assertFalse(self.index().reset_if_empty(lambda: next(answers)))
        self.assertEqual(len(self.search(kb_id, "审批规则")), 1)

    def test_neighbors_return_ordered_chunks_within_one_route(self):
        source = "手册.pdf"
        splits = [
            Document(page_content=f"片段 {index}", metadata={"source": source})
            for index in range(5)
        ]
        kb_id = publish(self.kb_name, source, splits)

        found = self.index().neighbors(kb_id, source, chunking.CHUNK_ROUTE_FIXED, 1, 3)

        self.assertEqual([document.page_content for document in found], ["片段 1", "片段 2", "片段 3"])

    def test_parent_child_split_keeps_parent_text_on_one_transient_child_only(self):
        source = Document(
            page_content="审批规则和适用范围。" * 120,
            metadata={"source": "制度.pdf", "page": 2},
        )
        splits = chunking.split_documents([source], chunk_size=300, chunk_overlap=0)
        parent_counts = {}
        for split in splits:
            parent_id = split.metadata["parent_id"]
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + int(
                "parent_content" in split.metadata
            )
        self.assertTrue(parent_counts)
        self.assertTrue(all(count == 1 for count in parent_counts.values()))


def app_storage_container():
    from services.core.container import get_application_container

    return get_application_container()


if __name__ == "__main__":
    unittest.main()
