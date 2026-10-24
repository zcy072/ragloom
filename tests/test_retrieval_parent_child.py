import asyncio
import os
import sqlite3
import unittest
import warnings
from unittest.mock import Mock, patch

try:
    import chromadb
    from langchain_core.documents import Document

    import app_storage
    from services.indexing import chunking
    from services.indexing import schema as index_schema
    from services.retrieval import chroma as chroma_index
    from services.retrieval import chroma as retrieval
    from services.retrieval import context as context_builder
    from services.retrieval import fts as fts_index
    from services.retrieval import retriever as retriever
    from services.retrieval import source_state
except ModuleNotFoundError:
    chromadb = None
    Document = None
    retrieval = None
    source_state = None

from tests.support import IsolatedStorageTestCase


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, query):
        return [1.0, 0.0]


class FailingEmbeddings:
    def embed_documents(self, texts):
        raise RuntimeError("embedding unavailable")


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self):
        self.query_calls = 0

    def embed_query(self, query):
        self.query_calls += 1
        return super().embed_query(query)


@unittest.skipUnless(
    Document and retrieval and chromadb and source_state,
    "requires the backend Python dependencies",
)
class ParentChildRetrievalTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        self.embeddings = FakeEmbeddings()
        self.child = Document(
            page_content="子块中的审批规则",
            metadata={
                "source": "制度.pdf",
                "page": 2,
                "parent_id": "parent-1",
                "parent_content": "父块完整内容，包含审批规则和适用范围。",
            },
        )

    def test_fts_query_drops_single_character_stopwords_but_keeps_model_codes(self):
        queries = fts_index._fts_match_queries("报销的流程是什么 A 型号", 10)
        text = " ".join(query for query, _ in queries)
        self.assertNotIn('"的"', text)
        self.assertNotIn('"是"', text)
        self.assertIn('"报销"', text)
        self.assertIn('"流程"', text)
        self.assertIn('"A"', text)

    def test_chroma_metadata_keeps_parent_identifier_without_parent_text(self):
        metadata = chroma_index.chroma_metadata_for_split(
            self.child, "人力资源部", "制度.pdf"
        )
        self.assertEqual(metadata["parent_id"], "parent-1")
        self.assertEqual(
            metadata[index_schema.CHROMA_SCHEMA_VERSION_FIELD],
            index_schema.CURRENT_INDEX_SCHEMA_VERSION,
        )
        self.assertNotIn("parent_content", metadata)

    def test_chroma_metadata_omits_sqlite_document_filters(self):
        split = Document(
            page_content="制度正文",
            metadata={
                "source": "制度.pdf",
                "document_type": "制度",
                "document_tags": "研发、现行",
                "effective_date": "2026-01-01",
            },
        )

        metadata = chroma_index.chroma_metadata_for_split(
            split, "人力资源部", "制度.pdf"
        )

        self.assertNotIn("document_type", metadata)
        self.assertNotIn("document_tags", metadata)
        self.assertNotIn("effective_date", metadata)

    def test_request_source_snapshot_is_shared_by_retrievers_and_worker_threads(self):
        states = [
            (["现行制度.pdf", "降级方案.pdf"], ["降级方案.pdf"]),
            (["新版制度.pdf"], []),
        ]
        with patch.object(
            app_storage.documents, "knowledge_base_exists", return_value=False
        ), patch.object(
            app_storage.documents, "retrieval_source_state", side_effect=states
        ) as source_state_read:
            with source_state.request_source_snapshot("人力资源部"):
                self.assertEqual(
                    source_state.current_source_allowlist("人力资源部", None),
                    ("现行制度.pdf", "降级方案.pdf"),
                )
                self.assertEqual(
                    chroma_index.current_chroma_source_allowlist("人力资源部", None),
                    ("现行制度.pdf",),
                )
                threaded_sources = asyncio.run(
                    asyncio.to_thread(source_state.ready_sources, "人力资源部")
                )
                self.assertEqual(
                    threaded_sources, ("现行制度.pdf", "降级方案.pdf")
                )

            with source_state.request_source_snapshot("人力资源部"):
                self.assertEqual(
                    source_state.ready_sources("人力资源部"), ("新版制度.pdf",)
                )

        self.assertEqual(source_state_read.call_count, 2)

    def test_query_embedding_is_shared_across_route_retrievers(self):
        base_embeddings = CountingEmbeddings()
        embeddings = retriever.CachedQueryEmbeddings(base_embeddings)
        collection = Mock()
        collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

        with patch.object(
            chroma_index, "get_chroma_collection", return_value=collection
        ):
            for route in (
                chunking.CHUNK_ROUTE_FIXED,
                chunking.CHUNK_ROUTE_STRUCTURED,
                chunking.CHUNK_ROUTE_VISUAL,
            ):
                chroma_index.ChromaRetriever(
                    kb_name="人力资源部",
                    embeddings=embeddings,
                    chunk_route=route,
                ).invoke("审批规则")

        self.assertEqual(collection.query.call_count, 3)
        self.assertEqual(base_embeddings.query_calls, 1)

    def test_chroma_retrieval_excludes_unversioned_records(self):
        collection = chroma_index.get_chroma_collection()
        collection.upsert(
            ids=["legacy-record"],
            embeddings=[[1.0, 0.0]],
            documents=["旧版本向量不应参与检索"],
            metadatas=[{"kb_name": "人力资源部", "source": "旧制度.pdf"}],
        )

        documents = chroma_index.ChromaRetriever(
            kb_name="人力资源部",
            embeddings=self.embeddings,
            k=5,
            allowed_sources=("旧制度.pdf",),
        ).invoke("旧版本")

        self.assertEqual(documents, [])

    def test_chroma_upsert_query_rename_and_delete(self):
        retrieval.chroma_upsert_splits([self.child], "人力资源部", "制度.pdf", self.embeddings)
        other_child = Document(
            page_content="另一份文档中的审批规则",
            metadata={"source": "其他制度.pdf", "page": 0},
        )
        retrieval.chroma_upsert_splits([other_child], "人力资源部", "其他制度.pdf", self.embeddings)
        collection = chroma_index.get_chroma_collection(create=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(collection.configuration["hnsw"]["space"], "cosine")
        records = collection.get(where={"kb_name": "人力资源部"}, include=["documents", "metadatas"])
        self.assertIn(self.child.page_content, records["documents"])
        parent_metadata = next(
            metadata for metadata in records["metadatas"] if metadata["source"] == "制度.pdf"
        )
        self.assertEqual(parent_metadata["parent_id"], "parent-1")
        self.assertEqual(
            parent_metadata[index_schema.CHROMA_SCHEMA_VERSION_FIELD],
            index_schema.CURRENT_INDEX_SCHEMA_VERSION,
        )
        self.assertNotIn("parent_content", parent_metadata)

        docs = chroma_index.ChromaRetriever(
            kb_name="人力资源部", embeddings=self.embeddings, k=5, allowed_sources=("制度.pdf",)
        ).invoke("审批规则")
        self.assertEqual(docs[0].metadata["source"], "制度.pdf")

        chroma_index.rename_chroma_knowledge_base("人力资源部", "研发部")
        self.assertEqual(collection.get(where={"kb_name": "人力资源部"})["ids"], [])
        self.assertEqual(len(collection.get(where={"kb_name": "研发部"})["ids"]), 2)

        filtered_docs = chroma_index.ChromaRetriever(
            kb_name="研发部", embeddings=self.embeddings, k=5, allowed_sources=("制度.pdf",)
        ).invoke("审批规则")
        self.assertEqual([document.metadata["source"] for document in filtered_docs], ["制度.pdf"])

        retrieval.chroma_delete_by_source("研发部", "制度.pdf")
        self.assertEqual(collection.count(), 1)

    def test_chroma_upsert_uses_prefetched_vectors_without_embedding(self):
        kb_name = "人力资源部"
        vectors = retrieval.embed_splits([self.child], self.embeddings)

        # 向量已在无锁阶段算好，写入时即使嵌入服务不可用也必须成功。
        retrieval.chroma_upsert_splits(
            [self.child], kb_name, "制度.pdf", FailingEmbeddings(), vectors=vectors
        )

        collection = chroma_index.get_chroma_collection(create=False)
        self.assertEqual(
            collection.get(where={"source": "制度.pdf"}, include=["documents"])["documents"],
            [self.child.page_content],
        )

    def test_chroma_upsert_rejects_mismatched_vector_count(self):
        with self.assertRaisesRegex(ValueError, "向量数量"):
            retrieval.chroma_upsert_splits(
                [self.child], "人力资源部", "制度.pdf", self.embeddings, vectors=[]
            )
        # 校验发生在触碰 Chroma 之前，集合不应被创建。
        self.assertIsNone(chroma_index.get_chroma_collection(create=False))

    def test_chroma_staged_upsert_keeps_previous_generation_on_failure(self):
        kb_name = "人力资源部"
        source = "制度.pdf"
        retrieval.chroma_upsert_splits([self.child], kb_name, source, self.embeddings)
        collection = chroma_index.get_chroma_collection(create=False)
        replacement = Document(
            page_content="更新后的审批规则",
            metadata={"source": source, "page": 2},
        )

        with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
            retrieval.chroma_upsert_splits([replacement], kb_name, source, FailingEmbeddings())
        self.assertEqual(
            collection.get(where={"source": source}, include=["documents"])["documents"],
            [self.child.page_content],
        )

        second_replacement = Document(
            page_content="更新后的适用范围",
            metadata={"source": source, "page": 2},
        )
        original_upsert = collection.upsert
        calls = 0

        def fail_second_batch(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return original_upsert(**kwargs)
            raise RuntimeError("batch write failed")

        with patch.object(
            chroma_index, "get_chroma_collection", return_value=collection
        ), patch.object(
            chroma_index, "_chroma_batch_size", return_value=1
        ), patch.object(collection, "upsert", side_effect=fail_second_batch):
            with self.assertRaisesRegex(RuntimeError, "batch write failed"):
                retrieval.chroma_upsert_splits(
                    [replacement, second_replacement], kb_name, source, self.embeddings
                )
        self.assertEqual(
            collection.get(where={"source": source}, include=["documents"])["documents"],
            [self.child.page_content],
        )

        retrieval.chroma_upsert_splits([replacement], kb_name, source, self.embeddings)
        self.assertEqual(
            collection.get(where={"source": source}, include=["documents"])["documents"],
            [replacement.page_content],
        )

    def test_chroma_delete_failure_is_propagated(self):
        retrieval.chroma_upsert_splits(
            [self.child], "人力资源部", "制度.pdf", self.embeddings
        )
        collection = chroma_index.get_chroma_collection(create=False)

        with patch.object(
            chroma_index, "get_chroma_collection", return_value=collection
        ), patch.object(collection, "delete", side_effect=RuntimeError("delete failed")):
            with self.assertRaisesRegex(RuntimeError, "delete failed"):
                retrieval.chroma_delete_by_source("人力资源部", "制度.pdf")

    def test_cached_chroma_retriever_excludes_sources_removed_from_sqlite(self):
        kb_name = "人力资源部"
        retrieval.chroma_upsert_splits([self.child], kb_name, "制度.pdf", self.embeddings)
        active = Document(
            page_content="仍然有效的审批规则",
            metadata={"source": "现行制度.pdf", "page": 0},
        )
        retrieval.chroma_upsert_splits([active], kb_name, "现行制度.pdf", self.embeddings)
        chroma_retriever = chroma_index.ChromaRetriever(
            kb_name=kb_name,
            embeddings=self.embeddings,
            k=5,
            allowed_sources=("制度.pdf", "现行制度.pdf"),
            enforce_current_sources=True,
        )

        with patch.object(
            app_storage.documents, "ready_sources", return_value=["现行制度.pdf"]
        ), patch.object(
            app_storage.documents, "chroma_disabled_sources", return_value=[]
        ):
            documents = chroma_retriever.invoke("审批规则")

        self.assertEqual(
            [document.metadata["source"] for document in documents], ["现行制度.pdf"]
        )

    def test_cached_chroma_retriever_excludes_fts_only_sources(self):
        kb_name = "人力资源部"
        retrieval.chroma_upsert_splits([self.child], kb_name, "制度.pdf", self.embeddings)
        active = Document(
            page_content="仍然有效的审批规则",
            metadata={"source": "现行制度.pdf", "page": 0},
        )
        retrieval.chroma_upsert_splits([active], kb_name, "现行制度.pdf", self.embeddings)
        chroma_retriever = chroma_index.ChromaRetriever(
            kb_name=kb_name,
            embeddings=self.embeddings,
            k=5,
            allowed_sources=("制度.pdf", "现行制度.pdf"),
            enforce_current_sources=True,
        )

        with patch.object(
            app_storage.documents,
            "ready_sources",
            return_value=["制度.pdf", "现行制度.pdf"],
        ), patch.object(
            app_storage.documents,
            "chroma_disabled_sources",
            return_value=["制度.pdf"],
        ):
            documents = chroma_retriever.invoke("审批规则")

        self.assertEqual(
            [document.metadata["source"] for document in documents], ["现行制度.pdf"]
        )

    def test_chroma_stale_source_cleanup_keeps_active_sources(self):
        kb_name = "人力资源部"
        retrieval.chroma_upsert_splits([self.child], kb_name, "制度.pdf", self.embeddings)
        retrieval.chroma_upsert_splits(
            [Document(page_content="旧制度内容", metadata={"source": "旧制度.pdf", "page": 0})],
            kb_name,
            "旧制度.pdf",
            self.embeddings,
        )

        removed = retrieval.chroma_delete_stale_sources(kb_name, {"制度.pdf"})
        collection = chroma_index.get_chroma_collection(create=False)

        self.assertEqual(removed, 1)
        self.assertEqual(collection.get(where={"source": "旧制度.pdf"})["ids"], [])
        self.assertEqual(len(collection.get(where={"source": "制度.pdf"})["ids"]), 1)

    def test_fts5_stores_parent_once_and_restores_it_after_retrieval(self):
        kb_name = "人力资源部"
        second_child = Document(
            page_content="子块中的适用范围",
            metadata={"source": "制度.pdf", "page": 2, "parent_id": "parent-1"},
        )
        fts_path = os.path.join(fts_index.vector_store_dir(kb_name), "fts5.db")
        fts_index.build_fts5_index([self.child, second_child], fts_path)

        with sqlite3.connect(fts_path) as conn:
            child_rows = conn.execute("SELECT original_text FROM docs ORDER BY rowid").fetchall()
            parent_rows = conn.execute(
                "SELECT parent_id, content FROM parent_chunks"
            ).fetchall()

        self.assertTrue(fts_index.fts5_index_is_current(fts_path))
        self.assertEqual([row[0] for row in child_rows], [self.child.page_content, second_child.page_content])
        self.assertEqual(parent_rows, [("parent-1", self.child.metadata["parent_content"])])

        found = fts_index.FTS5Retriever(
            db_path=fts_path, kb_name=kb_name
        ).invoke("审批规则")
        self.assertEqual(found[0].page_content, self.child.page_content)
        self.assertNotIn("parent_content", found[0].metadata)

        spans = context_builder.build_context_spans(
            kb_name,
            found,
            neighbor_chunks=0,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].content, self.child.metadata["parent_content"])

        found = fts_index.FTS5Retriever(
            db_path=fts_path, kb_name=kb_name
        ).invoke("审批规则")
        restored = context_builder.restore_parent_documents(kb_name, found)
        self.assertEqual(restored[0].page_content, self.child.metadata["parent_content"])
        self.assertEqual(len(restored), 1)

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


if __name__ == "__main__":
    unittest.main()
