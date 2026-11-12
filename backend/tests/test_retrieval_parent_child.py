import asyncio
import dataclasses
import sqlite3
import unittest
from unittest.mock import patch

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
from tests.retrieval_fixtures import common_kb_id, create_knowledge_base, mark_ready, publish
from tests.support import IsolatedStorageTestCase, fixture_document_id


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


class ParentChildRetrievalTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        self.kb_name = "人力资源部"
        self.kb_id = create_knowledge_base(self.kb_name)
        for filename in ("制度.pdf", "其他.pdf"):
            app_storage.documents.create_document_with_job(
                self.kb_id,
                fixture_document_id(self.kb_id, filename),
                filename,
                size=1,
                parser_mode="native_text",
                metadata={},
            )
        self.child = Document(
            page_content="子块中的审批规则",
            metadata={
                "document_id": fixture_document_id(self.kb_id, "制度.pdf"),
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
            document_id='报告"终版"\\草稿.pdf',
            document_ids=['x" or kb_id != "', "正常.pdf"],
            route="fixed",
        )
        self.assertEqual(
            expression,
            'kb_id == "kb-1" and document_id == "报告\\"终版\\"\\\\草稿.pdf" '
            'and document_id in ["x\\" or kb_id != \\"", "正常.pdf"] '
            'and chunk_route == "fixed"',
        )

    def test_filter_with_hostile_filename_matches_only_that_source(self):
        hostile = '合同"终版"草稿.pdf'
        kb_id = publish(
            self.kb_id,
            fixture_document_id(self.kb_id, hostile),
            [
                Document(
                    page_content="审批条款",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, hostile),
                        "source": hostile,
                    },
                )
            ],
        )
        publish(
            self.kb_id,
            fixture_document_id(self.kb_id, "正常.pdf"),
            [
                Document(
                    page_content="审批条款",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, "正常.pdf"),
                        "source": "正常.pdf",
                    },
                )
            ],
        )

        found = self.search(kb_id, "审批条款", sources=[fixture_document_id(self.kb_id, hostile)])

        self.assertEqual([document.metadata["source"] for document in found], [hostile])

    def test_parent_rows_keep_each_parent_once_and_scope_by_kb(self):
        kb_id = self.kb_id
        second_child = Document(
            page_content="子块中的适用范围",
            metadata={
                "document_id": fixture_document_id(self.kb_id, "制度.pdf"),
                "source": "制度.pdf",
                "page": 2,
                "parent_id": "parent-1",
            },
        )
        other_kb_id = create_knowledge_base("研发部")

        rows = parent_store.parent_chunk_rows([self.child, second_child])
        parent_store.add_document_version(kb_id, fixture_document_id(kb_id, "制度.pdf"), rows)

        self.assertEqual(
            rows,
            [("parent-1", "制度.pdf", "2", self.child.metadata["parent_content"], "")],
        )
        self.assertEqual(
            set(parent_store.load(kb_id, fixture_document_id(kb_id, "制度.pdf"), ["parent-1"])),
            {"parent-1"},
        )
        self.assertEqual(
            parent_store.load(
                other_kb_id, fixture_document_id(other_kb_id, "制度.pdf"), ["parent-1"]
            ),
            {},
        )

    def test_parent_versions_coexist_until_the_old_ids_are_deleted(self):
        kb_id = self.kb_id
        document_id = fixture_document_id(kb_id, "制度.pdf")
        other_id = fixture_document_id(kb_id, "其他.pdf")
        self.assertEqual(
            parent_store.add_document_version(
                kb_id, document_id, [("p1", "制度.pdf", "1", "旧内容", "")]
            ),
            set(),
        )
        parent_store.add_document_version(kb_id, other_id, [("p3", "其他.pdf", "1", "其他", "")])

        previous = parent_store.add_document_version(
            kb_id, document_id, [("p2", "制度.pdf", "1", "新内容", "")]
        )

        self.assertEqual(previous, {"p1"})
        self.assertEqual(
            set(parent_store.load(kb_id, document_id, ["p1", "p2", "p3"])), {"p1", "p2"}
        )
        parent_store.delete_parents(kb_id, document_id, previous)
        parent_store.delete_parents(kb_id, document_id, previous)
        self.assertEqual(set(parent_store.load(kb_id, document_id, ["p1", "p2"])), {"p2"})

        with self.assertRaises(sqlite3.IntegrityError):
            parent_store.add_document_version(
                kb_id,
                document_id,
                [("p4", "制度.pdf", "1", "新内容", ""), ("p2", "制度.pdf", "1", "重复", "")],
            )
        self.assertEqual(set(parent_store.load(kb_id, document_id, ["p2", "p4"])), {"p2"})

        parent_store.delete_document(kb_id, document_id)
        parent_store.delete_document(kb_id, document_id)
        self.assertEqual(parent_store.load(kb_id, document_id, ["p2"]), {})
        self.assertEqual(set(parent_store.load(kb_id, other_id, ["p3"])), {"p3"})
        parent_store.delete_kb(kb_id)
        self.assertEqual(parent_store.load(kb_id, other_id, ["p3"]), {})

    def test_parent_text_is_stored_once_and_restored_after_retrieval(self):
        second_child = Document(
            page_content="子块中的适用范围",
            metadata={
                "document_id": fixture_document_id(self.kb_id, "制度.pdf"),
                "source": "制度.pdf",
                "page": 2,
                "parent_id": "parent-1",
            },
        )
        kb_id = publish(
            self.kb_id, fixture_document_id(self.kb_id, "制度.pdf"), [self.child, second_child]
        )
        parent_store.add_document_version(
            kb_id,
            fixture_document_id(kb_id, "制度.pdf"),
            parent_store.parent_chunk_rows([self.child, second_child]),
        )

        found = self.search(kb_id, "审批规则")
        self.assertEqual(found[0].page_content, self.child.page_content)
        self.assertNotIn("parent_content", found[0].metadata)

        spans = context_builder.build_context_spans(self.kb_id, found, neighbor_chunks=0)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].content, self.child.metadata["parent_content"])

        restored = context_builder.restore_parent_documents(
            self.kb_id, self.search(kb_id, "审批规则")
        )
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].page_content, self.child.metadata["parent_content"])

    def test_request_document_snapshot_is_shared_by_retrievers_and_worker_threads(self):
        states = [["现行制度.pdf", "降级方案.pdf"], ["新版制度.pdf"]]
        with (
            patch.object(app_storage.documents, "knowledge_base_exists", return_value=False),
            patch.object(
                app_storage.documents, "ready_document_ids", side_effect=states
            ) as ready_read,
        ):
            with source_state.request_document_snapshot(self.kb_id, container=get_application_container()):
                self.assertEqual(
                    source_state.current_document_allowlist(self.kb_id, None, container=get_application_container()),
                    ("现行制度.pdf", "降级方案.pdf"),
                )
                threaded_sources = asyncio.run(
                    asyncio.to_thread(
                        source_state.ready_document_ids, self.kb_id, get_application_container()
                    )
                )
                self.assertEqual(threaded_sources, ("现行制度.pdf", "降级方案.pdf"))

            with source_state.request_document_snapshot(self.kb_id, container=get_application_container()):
                self.assertEqual(source_state.ready_document_ids(self.kb_id, container=get_application_container()), ("新版制度.pdf",))

        self.assertEqual(ready_read.call_count, 2)

    def _publish_department_and_common(self):
        for kb_id, source in ((self.kb_id, "部门制度.pdf"), (common_kb_id(), "通用制度.pdf")):
            publish(
                kb_id,
                fixture_document_id(kb_id, source),
                [
                    Document(
                        page_content="审批规则",
                        metadata={
                            "document_id": fixture_document_id(self.kb_id, source),
                            "source": source,
                        },
                    )
                ],
            )
            mark_ready(kb_id, fixture_document_id(kb_id, source))

    def test_common_knowledge_base_is_supplemented_only_when_requested(self):
        self._publish_department_and_common()

        container = dataclasses.replace(
            get_application_container(), create_query_embeddings=FakeEmbeddings
        )

        def retrieved_sources(include_common):
            with source_state.request_document_snapshot(
                self.kb_id, container, include_common=include_common
            ):
                dense = retriever.build_retriever(
                    self.kb_id, include_common=include_common, container=container
                ).invoke("审批规则")
                lexical = retriever.retrieve_lexical_candidates(
                    self.kb_id, "审批规则", include_common=include_common, container=container
                )
            return {document.metadata["source"] for document in dense + lexical}

        self.assertEqual(retrieved_sources(False), {"部门制度.pdf"})
        self.assertEqual(retrieved_sources(True), {"部门制度.pdf", "通用制度.pdf"})

    def test_retriever_cache_keeps_supplemented_and_plain_graphs_apart(self):
        with patch.object(retriever, "_retriever_cache", RetrieverCache(8)):
            retriever.cache_retriever(self.kb_id, "with-common", include_common=True, version=retriever.retriever_cache_version())
            self.assertIsNone(retriever.get_cached_retriever(self.kb_id))
            retriever.cache_retriever(self.kb_id, "plain", version=retriever.retriever_cache_version())
            self.assertEqual(
                retriever.get_cached_retriever(self.kb_id, include_common=True), "with-common"
            )
            retriever.drop_cached_retriever(self.kb_id, container=get_application_container())
            self.assertIsNone(retriever.get_cached_retriever(self.kb_id))
            self.assertIsNone(retriever.get_cached_retriever(self.kb_id, include_common=True))

    def test_query_embedding_is_shared_across_route_searches(self):
        kb_id = self.kb_id
        base_embeddings = CountingEmbeddings()
        embeddings = retriever.CachedQueryEmbeddings(base_embeddings)
        mark_ready(self.kb_id, fixture_document_id(self.kb_id, "制度.pdf"))
        plans = [
            retriever._RoutePlan(kb_id, route, 1 / 3, None)
            for route in (
                chunking.CHUNK_ROUTE_FIXED,
                chunking.CHUNK_ROUTE_STRUCTURED,
                chunking.CHUNK_ROUTE_VISUAL,
            )
        ]
        with patch.object(self.index(), "search_hybrid", return_value=[]) as search:
            retriever.WeightedRouteRetriever(self.index(), embeddings, plans, container=get_application_container()).invoke("审批规则")

        self.assertEqual(search.call_count, 3)
        self.assertEqual(base_embeddings.query_calls, 1)

    def test_query_vectors_are_shared_across_retriever_instances(self):
        base_embeddings = CountingEmbeddings()

        retriever.CachedQueryEmbeddings(base_embeddings).embed_query("审批规则")
        retriever.CachedQueryEmbeddings(base_embeddings).embed_query("审批规则")

        self.assertEqual(base_embeddings.query_calls, 1)

    def test_route_probe_is_cached_until_the_knowledge_base_changes(self):
        publish(
            self.kb_id,
            fixture_document_id(self.kb_id, "制度.pdf"),
            [
                Document(
                    page_content="固定块",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, "制度.pdf"),
                        "source": "制度.pdf",
                    },
                )
            ],
        )
        mark_ready(self.kb_id, fixture_document_id(self.kb_id, "制度.pdf"))
        index = self.index()

        with patch.object(index, "available_routes", wraps=index.available_routes) as probe:
            for _ in range(2):
                retriever._route_plans(app_storage_container(), index, self.kb_id, None, 0.7)
            self.assertEqual(probe.call_count, 1)

            retriever.drop_cached_retriever(self.kb_id, container=get_application_container())
            retriever._route_plans(app_storage_container(), index, self.kb_id, None, 0.7)
            self.assertEqual(probe.call_count, 2)

    def test_route_probe_result_is_not_cached_when_invalidated_meanwhile(self):
        cache = retriever._RouteCache()

        def probe():
            cache.drop("kb-a")
            return {"fixed"}

        self.assertEqual(cache.routes("kb-a", ("a.pdf",), probe), {"fixed"})
        self.assertEqual(cache.routes("kb-a", ("a.pdf",), lambda: {"visual"}), {"visual"})

    def test_route_weights_renormalize_over_existing_routes(self):
        kb_id = publish(
            self.kb_id,
            fixture_document_id(self.kb_id, "制度.pdf"),
            [
                Document(
                    page_content="固定块",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, "制度.pdf"),
                        "source": "制度.pdf",
                    },
                )
            ],
        )
        mark_ready(self.kb_id, fixture_document_id(self.kb_id, "制度.pdf"))

        plans = retriever._route_plans(
            app_storage_container(), self.index(), self.kb_id, None, 0.70
        )

        self.assertEqual([plan.route for plan in plans], [chunking.CHUNK_ROUTE_FIXED])
        self.assertAlmostEqual(plans[0].weight, 0.70)
        self.assertEqual(plans[0].kb_id, kb_id)

    def test_fusion_ranks_higher_weighted_route_first(self):
        source = "制度.pdf"
        fixed = Document(
            page_content="审批规则固定块",
            metadata={
                "document_id": fixture_document_id(self.kb_id, source),
                "source": source,
                "chunk_route": chunking.CHUNK_ROUTE_FIXED,
            },
        )
        visual = Document(
            page_content="审批规则视觉块",
            metadata={
                "document_id": fixture_document_id(self.kb_id, source),
                "source": source,
                "chunk_route": chunking.CHUNK_ROUTE_VISUAL,
            },
        )
        kb_id = publish(self.kb_id, fixture_document_id(self.kb_id, source), [fixed, visual])
        mark_ready(self.kb_id, fixture_document_id(self.kb_id, source))
        plans = [
            retriever._RoutePlan(kb_id, chunking.CHUNK_ROUTE_VISUAL, 0.15, None),
            retriever._RoutePlan(kb_id, chunking.CHUNK_ROUTE_FIXED, 0.60, None),
        ]

        found = retriever.WeightedRouteRetriever(self.index(), FakeEmbeddings(), plans, container=get_application_container()).invoke(
            "审批规则"
        )

        self.assertEqual(
            [document.metadata["chunk_route"] for document in found],
            [chunking.CHUNK_ROUTE_FIXED, chunking.CHUNK_ROUTE_VISUAL],
        )
        self.assertTrue(all(document.metadata["kb_id"] == self.kb_id for document in found))

    def test_retrieval_excludes_sources_not_ready_in_sqlite(self):
        kb_id = publish(
            self.kb_id,
            fixture_document_id(self.kb_id, "旧制度.pdf"),
            [
                Document(
                    page_content="审批规则",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, "旧制度.pdf"),
                        "source": "旧制度.pdf",
                    },
                )
            ],
        )
        publish(
            self.kb_id,
            fixture_document_id(self.kb_id, "现行制度.pdf"),
            [
                Document(
                    page_content="审批规则",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, "现行制度.pdf"),
                        "source": "现行制度.pdf",
                    },
                )
            ],
        )
        mark_ready(self.kb_id, fixture_document_id(self.kb_id, "现行制度.pdf"))
        plan = retriever._RoutePlan(
            kb_id,
            chunking.CHUNK_ROUTE_FIXED,
            1.0,
            (
                fixture_document_id(kb_id, "旧制度.pdf"),
                fixture_document_id(kb_id, "现行制度.pdf"),
            ),
        )

        found = retriever.WeightedRouteRetriever(self.index(), FakeEmbeddings(), [plan], container=get_application_container()).invoke(
            "审批规则"
        )

        self.assertEqual([document.metadata["source"] for document in found], ["现行制度.pdf"])

    def test_search_failure_propagates_instead_of_returning_empty(self):
        kb_id = self.kb_id
        mark_ready(self.kb_id, fixture_document_id(self.kb_id, "制度.pdf"))
        plan = retriever._RoutePlan(kb_id, chunking.CHUNK_ROUTE_FIXED, 1.0, None)

        with patch.object(self.index(), "search_hybrid", side_effect=RuntimeError("milvus down")):
            with self.assertRaisesRegex(RuntimeError, "milvus down"):
                retriever.WeightedRouteRetriever(self.index(), FakeEmbeddings(), [plan], container=get_application_container()).invoke(
                    "审批规则"
                )

    def test_publish_replaces_previous_generation(self):
        source = "制度.pdf"
        kb_id = publish(self.kb_id, fixture_document_id(self.kb_id, source), [self.child])
        publish(
            self.kb_id,
            fixture_document_id(self.kb_id, source),
            [
                Document(
                    page_content="更新后的审批规则",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, source),
                        "source": source,
                    },
                )
            ],
        )

        found = self.search(kb_id, "审批规则", sources=[fixture_document_id(self.kb_id, source)])

        self.assertEqual([document.page_content for document in found], ["更新后的审批规则"])

    def test_failed_publish_keeps_previous_generation(self):
        source = "制度.pdf"
        kb_id = publish(self.kb_id, fixture_document_id(self.kb_id, source), [self.child])
        self.index().fail_next_publish = RuntimeError("batch write failed")

        with self.assertRaisesRegex(RuntimeError, "batch write failed"):
            publish(
                self.kb_id,
                fixture_document_id(self.kb_id, source),
                [
                    Document(
                        page_content="新版内容",
                        metadata={
                            "document_id": fixture_document_id(self.kb_id, source),
                            "source": source,
                        },
                    )
                ],
            )

        found = self.search(kb_id, "审批规则", sources=[fixture_document_id(self.kb_id, source)])
        self.assertEqual([document.page_content for document in found], [self.child.page_content])

    def test_publish_rejects_mismatched_vector_count(self):
        kb_id = self.kb_id
        with self.assertRaisesRegex(ValueError, "向量数量"):
            self.index().publish_document(
                kb_id, fixture_document_id(kb_id, "制度.pdf"), [self.child], []
            )
        self.assertEqual(self.index().available_routes(kb_id, None), set())

    def test_publish_uses_prefetched_vectors_without_embedding(self):
        kb_id = self.kb_id
        vectors = embed_splits([self.child], FakeEmbeddings())

        self.index().publish_document(
            kb_id, fixture_document_id(kb_id, "制度.pdf"), [self.child], vectors
        )

        self.assertEqual(
            self.index().available_routes(kb_id, [fixture_document_id(kb_id, "制度.pdf")]),
            {chunking.CHUNK_ROUTE_FIXED},
        )

    def test_rename_keeps_vectors_reachable_through_stable_kb_id(self):
        kb_id = publish(self.kb_id, fixture_document_id(self.kb_id, "制度.pdf"), [self.child])
        before = self.index().available_routes(kb_id, None)

        app_storage.knowledge_bases.rename(kb_id, "研发部")

        self.assertEqual(app_storage.knowledge_bases.names_for([kb_id]), {kb_id: "研发部"})
        self.assertEqual(self.index().available_routes(kb_id, None), before)
        self.assertEqual(
            [document.metadata["source"] for document in self.search(kb_id, "审批规则")],
            ["制度.pdf"],
        )

    def test_delete_document_kb_and_stale_cleanup(self):
        kb_id = publish(self.kb_id, fixture_document_id(self.kb_id, "制度.pdf"), [self.child])
        publish(
            self.kb_id,
            fixture_document_id(self.kb_id, "旧制度.pdf"),
            [
                Document(
                    page_content="旧制度内容",
                    metadata={
                        "document_id": fixture_document_id(self.kb_id, "旧制度.pdf"),
                        "source": "旧制度.pdf",
                    },
                )
            ],
        )
        other_kb_id = publish(
            create_knowledge_base("研发部"),
            fixture_document_id(create_knowledge_base("研发部"), "制度.pdf"),
            [self.child],
        )

        removed = self.index().delete_stale_documents(
            kb_id, {fixture_document_id(kb_id, "制度.pdf")}
        )

        self.assertEqual(removed, 1)
        self.assertEqual(
            self.search(
                kb_id, "旧制度内容", sources=[fixture_document_id(self.kb_id, "旧制度.pdf")]
            ),
            [],
        )
        self.assertEqual(
            len(
                self.search(
                    kb_id, "审批规则", sources=[fixture_document_id(self.kb_id, "制度.pdf")]
                )
            ),
            1,
        )

        self.index().delete_document(kb_id, fixture_document_id(kb_id, "制度.pdf"))
        self.index().delete_document(kb_id, fixture_document_id(kb_id, "制度.pdf"))
        self.assertEqual(self.index().available_routes(kb_id, None), set())
        self.assertEqual(len(self.search(other_kb_id, "审批规则")), 1)

        self.index().delete_kb(other_kb_id)
        self.index().delete_kb(other_kb_id)
        self.assertEqual(self.index().available_routes(other_kb_id, None), set())

    def test_neighbors_return_ordered_chunks_within_one_route(self):
        source = "手册.pdf"
        splits = [
            Document(
                page_content=f"片段 {index}",
                metadata={"document_id": fixture_document_id(self.kb_id, source), "source": source},
            )
            for index in range(5)
        ]
        kb_id = publish(self.kb_id, fixture_document_id(self.kb_id, source), splits)

        found = self.index().neighbors(
            kb_id, fixture_document_id(kb_id, source), chunking.CHUNK_ROUTE_FIXED, 1, 3
        )

        self.assertEqual(
            [document.page_content for document in found], ["片段 1", "片段 2", "片段 3"]
        )

    def test_switch_window_keeps_neighbors_within_the_hit_generation(self):
        source = "手册.pdf"
        document_id = fixture_document_id(self.kb_id, source)

        def splits(label):
            return [
                Document(
                    page_content=f"{label}片段 {index}",
                    metadata={"document_id": document_id, "source": source},
                )
                for index in range(3)
            ]

        index = self.index()
        publish(self.kb_id, document_id, splits("旧版"))
        old_rows = list(index._rows)
        publish(self.kb_id, document_id, splits("新版"))
        # 模拟 Milvus 切换窗口：新代已写入、旧代还没删除，两代序号相同。
        index._rows.extend(old_rows)
        old_hit, new_hit = (
            next(
                document
                for document in index.neighbors(
                    self.kb_id, document_id, chunking.CHUNK_ROUTE_FIXED, 1, 1
                )
                if document.page_content.startswith(label)
            )
            for label in ("旧版", "新版")
        )
        self.assertNotEqual(old_hit.metadata["generation"], new_hit.metadata["generation"])

        found = index.neighbors(
            self.kb_id, document_id, chunking.CHUNK_ROUTE_FIXED, 0, 2,
            new_hit.metadata["generation"],
        )
        self.assertEqual(
            [document.page_content for document in found],
            ["新版片段 0", "新版片段 1", "新版片段 2"],
        )

        spans = context_builder.build_context_spans(
            self.kb_id, [new_hit, old_hit], neighbor_chunks=1
        )
        self.assertEqual(len(spans), 2)
        self.assertNotIn("旧版", spans[0].content)
        self.assertNotIn("新版", spans[1].content)

        mark_ready(self.kb_id, document_id)
        plans = [retriever._RoutePlan(self.kb_id, chunking.CHUNK_ROUTE_FIXED, 1.0, None)]
        fused = retriever.WeightedRouteRetriever(index, FakeEmbeddings(), plans, container=get_application_container()).invoke("片段")
        self.assertEqual(len(fused), 6)

    def test_parent_child_split_keeps_parent_text_on_one_transient_child_only(self):
        source = Document(
            page_content="审批规则和适用范围。" * 120,
            metadata={
                "document_id": fixture_document_id(self.kb_id, "制度.pdf"),
                "source": "制度.pdf",
                "page": 2,
            },
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
