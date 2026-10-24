import os
import re
import sqlite3
import unittest
from unittest.mock import patch

try:
    from langchain_core.documents import Document

    import app_storage
    from services.indexing import chunking
    from services.retrieval import context as context_builder
    from services.retrieval import fts as fts_index
    from services.retrieval import fts as retrieval
    from services.retrieval import retriever as retriever
except ModuleNotFoundError:
    Document = None
    chunking = None
    context_builder = None
    fts_index = None
    retrieval = None
    retriever = None

from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(Document and retrieval, "requires the backend Python dependencies")
class RetrievalContextTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()

    def test_context_spans_expand_merge_and_respect_budget(self):
        kb_name = "研发部"
        source = "手册.pdf"
        splits = [
            Document(page_content=f"内容片段 {index}", metadata={"source": source, "page": 0})
            for index in range(5)
        ]
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(splits, fts_path)

        hits = [
            Document(page_content="内容片段 1", metadata={"source": source, "page": 0, "chunk_index": 1}),
            Document(page_content="内容片段 3", metadata={"source": source, "page": 0, "chunk_index": 3}),
        ]
        spans = context_builder.build_context_spans(
            kb_name,
            hits,
            neighbor_chunks=1,
            max_chars=1000,
        )

        self.assertEqual(len(spans), 1)
        self.assertIn("内容片段 0", spans[0].content)
        self.assertIn("内容片段 4", spans[0].content)
        self.assertEqual(spans[0].chunk_start, 0)
        self.assertEqual(spans[0].chunk_end, 4)
        self.assertIn(
            "【来源 1：手册.pdf，第 1 页】",
            context_builder.format_context_spans(spans),
        )

        limited = context_builder.build_context_spans(
            kb_name,
            hits,
            neighbor_chunks=1,
            max_chars=12,
        )
        self.assertLessEqual(len(limited[0].content), 12)

    def test_long_structured_context_keeps_its_section_boundary(self):
        kb_name = "研发部"
        source = "技术规范.md"
        before = "相邻的前置章节。" * 80
        target = "目标章节的完整说明。" * 80
        after = "相邻的后置章节。" * 80
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(
            [
                Document(
                    page_content=before,
                    metadata={"source": source, "page": 0, "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
                Document(
                    page_content=target,
                    metadata={"source": source, "page": 0, "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
                Document(
                    page_content=after,
                    metadata={"source": source, "page": 0, "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
            ],
            fts_path,
        )

        spans = context_builder.build_context_spans(
            kb_name,
            [
                Document(
                    page_content=target,
                    metadata={
                        "source": source,
                        "page": 0,
                        "chunk_index": 1,
                        "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED,
                    },
                ),
                Document(
                    page_content=after,
                    metadata={
                        "source": source,
                        "page": 0,
                        "chunk_index": 2,
                        "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED,
                    },
                ),
            ],
            neighbor_chunks=1,
        )

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].content, target)
        self.assertEqual(spans[1].content, after)
        self.assertNotIn(before, spans[0].content)
        self.assertNotIn(after, spans[0].content)

    def test_short_structured_context_still_expands_neighbors(self):
        kb_name = "研发部"
        source = "技术规范.md"
        before = "前置内容"
        target = "短章节内容"
        after = "后置内容"
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(
            [
                Document(
                    page_content=before,
                    metadata={"source": source, "page": 0, "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
                Document(
                    page_content=target,
                    metadata={"source": source, "page": 0, "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
                Document(
                    page_content=after,
                    metadata={"source": source, "page": 0, "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
            ],
            fts_path,
        )

        spans = context_builder.build_context_spans(
            kb_name,
            [
                Document(
                    page_content=target,
                    metadata={
                        "source": source,
                        "page": 0,
                        "chunk_index": 1,
                        "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED,
                    },
                )
            ],
            neighbor_chunks=1,
        )

        self.assertEqual(len(spans), 1)
        self.assertIn(before, spans[0].content)
        self.assertIn(target, spans[0].content)
        self.assertIn(after, spans[0].content)

    def test_visual_context_never_pulls_an_adjacent_image(self):
        kb_name = "研发部"
        source = "指南.pdf"
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        documents = [
            Document(
                page_content=f"图片 {index} 的独立内容",
                metadata={
                    "source": source,
                    "page": index,
                    "chunk_route": chunking.CHUNK_ROUTE_VISUAL,
                },
            )
            for index in range(3)
        ]
        retrieval.build_fts5_index(documents, fts_path)

        spans = context_builder.build_context_spans(
            kb_name,
            [
                Document(
                    page_content="图片 1 的独立内容",
                    metadata={
                        "source": source,
                        "page": 1,
                        "chunk_index": 1,
                        "chunk_route": chunking.CHUNK_ROUTE_VISUAL,
                    },
                )
            ],
            neighbor_chunks=1,
        )

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].content, "图片 1 的独立内容")

    def test_same_filename_from_selected_and_common_kb_remains_distinct(self):
        documents = [
            Document(
                page_content="部门制度内容",
                metadata={"source": "制度.pdf", "kb_name": "研发部"},
            ),
            Document(
                page_content="通用制度内容",
                metadata={"source": "制度.pdf", "kb_name": "通用知识库"},
            ),
        ]

        deduplicated = retriever.deduplicate_chunk_routes(documents)
        spans = context_builder.build_context_spans(
            "研发部",
            deduplicated,
            neighbor_chunks=0,
        )

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(len(spans), 2)
        self.assertEqual([span.kb_name for span in spans], ["研发部", "通用知识库"])

    def test_filename_contextual_titles_are_indexed_without_changing_citation_text(self):
        kb_name = "人力资源部"
        source = "员工手册.md"
        documents = [
            Document(
                page_content="# 员工手册\n## 请假管理\n员工请假需在系统中提交申请并获得审批。",
                metadata={"source": source, "page": 0},
            ),
            Document(
                page_content="# 费用制度\n报销需要附上发票。",
                metadata={"source": "费用制度.md", "page": 0},
            ),
        ]
        splits = chunking.split_documents(documents, chunk_size=120, chunk_overlap=0)
        target = next(split for split in splits if split.metadata["source"] == source)
        self.assertIn("员工手册", target.metadata["contextual_title"])

        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(splits, fts_path)
        found = fts_index.FTS5Retriever(
            db_path=fts_path, allowed_sources=(source,)
        ).invoke("请假")

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].page_content, target.page_content)
        self.assertIn("员工手册", found[0].metadata["contextual_title"])
        spans = context_builder.build_context_spans(
            kb_name,
            found,
            neighbor_chunks=0,
        )
        self.assertIn(
            "上下文：员工手册", context_builder.format_context_spans(spans)
        )

    def test_fixed_split_preserves_short_lines_without_inventing_headings(self):
        source_text = "血压差\n\n>10\\mathrm{mmHg}\n\nHbA1c\n\n后续说明内容。"
        document = Document(page_content=source_text, metadata={"source": "指标表.pdf", "page": 0})

        splits = chunking.split_documents([document], chunk_size=120, chunk_overlap=0)

        self.assertEqual(document.page_content, source_text)
        indexed_content = "\n".join(
            [split.page_content for split in splits]
            + [split.metadata.get("parent_content", "") for split in splits]
        )
        self.assertNotIn("## >10\\mathrm{mmHg}", indexed_content)
        self.assertNotIn("## HbA1c", indexed_content)
        self.assertTrue(
            all("HbA1c" not in split.metadata.get("contextual_title", "") for split in splits)
        )

    def test_fixed_split_merges_parser_fragments_and_adjacent_overlap_copies(self):
        source_text = (
            "## 筛查对象\n\n"
            "1. 糖尿病高危人群应定期接受筛查\n"
            "^{[1-2]}\n"
            "。\n\n"
            + "成年高危人群包括年龄较大、缺乏运动以及具有糖尿病家族史等情况。" * 12
        )
        document = Document(
            page_content=source_text,
            metadata={"source": "筛查指南.pdf", "page": 3},
        )

        splits = chunking.split_documents([document], chunk_size=240, chunk_overlap=80)

        self.assertEqual(document.page_content, source_text)
        indexed_text = "\n".join(
            [split.page_content for split in splits]
            + [split.metadata.get("parent_content", "") for split in splits]
        )
        self.assertIn("筛查^{[1-2]}。", indexed_text)
        self.assertTrue(
            all(chunking._semantic_chunk_length(split.page_content) > 0 for split in splits)
        )
        self.assertTrue(
            all(
                not re.fullmatch(r"\s*(?:\^\{\[[^]]+\]\})?[。；，、,.!?！？]*\s*", split.page_content)
                for split in splits
            )
        )

        adjacent_keys = [chunking._adjacent_split_key(split) for split in splits]
        self.assertTrue(
            all(left != right for left, right in zip(adjacent_keys, adjacent_keys[1:]))
        )
        parent_content_counts = {}
        for split in splits:
            parent_id = split.metadata["parent_id"]
            parent_content_counts[parent_id] = parent_content_counts.get(parent_id, 0) + int(
                "parent_content" in split.metadata
            )
        self.assertTrue(parent_content_counts)
        self.assertTrue(all(count == 1 for count in parent_content_counts.values()))
        parent_contents = {
            split.metadata["parent_id"]: split.metadata["parent_content"]
            for split in splits
            if "parent_content" in split.metadata
        }
        self.assertTrue(
            all(
                re.sub(r"\s+", "", split.page_content)
                in re.sub(r"\s+", "", parent_contents[split.metadata["parent_id"]])
                for split in splits
            )
        )

    def test_small_child_merge_reconstructs_parent_span_without_repeating_overlap(self):
        parent_text = "prefix-" + "|".join(f"token-{index:03d}" for index in range(24))
        first = Document(
            page_content=parent_text[:54],
            metadata={"source": "指南.pdf", "start_index": 0},
        )
        second = Document(
            page_content=parent_text[32:],
            metadata={"source": "指南.pdf", "start_index": 32},
        )

        merged = chunking._merge_small_child_chunks(
            [first, second],
            parent_content=parent_text,
            parent_start=0,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].page_content, parent_text)
        self.assertEqual(merged[0].metadata["start_index"], 0)

    def test_missing_fts_index_requires_a_complete_rebuild(self):
        with patch.object(
            retriever.source_state,
            "current_source_allowlist",
            return_value=["制度.pdf"],
        ):
            with self.assertRaisesRegex(ValueError, "完整重建"):
                retriever._probe_base_capabilities("研发部", None)

    def test_unversioned_fts_index_is_recognized_as_legacy(self):
        fts_path = os.path.join(retrieval.vector_store_dir("研发部"), "fts5.db")
        os.makedirs(os.path.dirname(fts_path), exist_ok=True)
        with sqlite3.connect(fts_path) as conn:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(content, source UNINDEXED)")

        self.assertEqual(fts_index.fts5_index_version(fts_path), 0)
        self.assertFalse(fts_index.fts5_index_is_current(fts_path))
        self.assertEqual(fts_index.FTS5Retriever(db_path=fts_path).invoke("任意查询"), [])

    def test_fts_route_probe_returns_routes_for_allowed_sources(self):
        fts_path = os.path.join(retrieval.vector_store_dir("研发部"), "fts5.db")
        retrieval.build_fts5_index(
            [
                Document(
                    page_content="固定路线",
                    metadata={
                        "source": "制度.pdf",
                        "chunk_route": chunking.CHUNK_ROUTE_FIXED,
                    },
                ),
                Document(
                    page_content="结构化路线",
                    metadata={
                        "source": "制度.pdf",
                        "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED,
                    },
                ),
                Document(
                    page_content="视觉路线",
                    metadata={
                        "source": "其他.pdf",
                        "chunk_route": chunking.CHUNK_ROUTE_VISUAL,
                    },
                ),
            ],
            fts_path,
        )

        self.assertEqual(
            fts_index.fts5_available_routes(fts_path, ["制度.pdf"]),
            {chunking.CHUNK_ROUTE_FIXED, chunking.CHUNK_ROUTE_STRUCTURED},
        )

    def test_successful_fts_rebuild_replaces_legacy_index_and_sets_version(self):
        fts_path = os.path.join(retrieval.vector_store_dir("研发部"), "fts5.db")
        retrieval.build_fts5_index(
            [Document(page_content="旧索引内容", metadata={"source": "制度.pdf"})],
            fts_path,
        )
        with sqlite3.connect(fts_path) as conn:
            conn.execute("PRAGMA user_version = 0")

        retrieval.build_fts5_index(
            [Document(page_content="重建后的内容", metadata={"source": "制度.pdf"})],
            fts_path,
        )

        self.assertTrue(fts_index.fts5_index_is_current(fts_path))
        with sqlite3.connect(fts_path) as conn:
            contents = conn.execute("SELECT original_text FROM docs").fetchall()
        self.assertEqual(contents, [("重建后的内容",)])

    def test_failed_fts_rebuild_preserves_previous_index(self):
        fts_path = os.path.join(retrieval.vector_store_dir("研发部"), "fts5.db")
        retrieval.build_fts5_index(
            [Document(page_content="仍可使用的旧内容", metadata={"source": "制度.pdf"})],
            fts_path,
        )

        with patch.object(fts_index.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                retrieval.build_fts5_index(
                    [Document(page_content="未完成的新内容", metadata={"source": "制度.pdf"})],
                    fts_path,
                )

        self.assertTrue(fts_index.fts5_index_is_current(fts_path))
        with sqlite3.connect(fts_path) as conn:
            contents = conn.execute("SELECT original_text FROM docs").fetchall()
        self.assertEqual(contents, [("仍可使用的旧内容",)])

    def test_fts_strict_keywords_precede_broad_candidates(self):
        kb_name = "研发部"
        source = "术语表.md"
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(
            [
                Document(page_content="alpha beta 精确条目", metadata={"source": source, "page": 0}),
                Document(page_content="alpha alpha 宽松条目", metadata={"source": source, "page": 1}),
                Document(page_content="beta beta 宽松条目", metadata={"source": source, "page": 2}),
            ],
            fts_path,
        )

        with patch.object(
            fts_index.jieba_analyse, "extract_tags", return_value=["alpha", "beta"]
        ):
            found = fts_index.FTS5Retriever(db_path=fts_path, k=2).invoke("alpha beta 查询")

        self.assertEqual(found[0].page_content, "alpha beta 精确条目")
        self.assertEqual(len(found), 2)

    def test_fts_broad_query_fills_when_strict_keywords_do_not_cooccur(self):
        kb_name = "研发部"
        source = "术语表.md"
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(
            [Document(page_content="alpha 可用条目", metadata={"source": source, "page": 0})],
            fts_path,
        )

        with patch.object(
            fts_index.jieba_analyse, "extract_tags", return_value=["alpha", "missing"]
        ):
            found = fts_index.FTS5Retriever(db_path=fts_path, k=2).invoke("alpha missing 查询")

        self.assertEqual([document.page_content for document in found], ["alpha 可用条目"])

    def test_chroma_disabled_source_remains_available_through_fts(self):
        kb_name = "研发部"
        source = "制度.pdf"
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(
            [Document(page_content="审批制度的完整内容", metadata={"source": source})],
            fts_path,
        )
        fts_retriever = fts_index.FTS5Retriever(
            db_path=fts_path,
            kb_name=kb_name,
            allowed_sources=(source,),
            enforce_current_sources=True,
        )

        with patch.object(
            app_storage.documents, "ready_sources", return_value=[source]
        ), patch.object(
            app_storage.documents, "chroma_disabled_sources", return_value=[source]
        ):
            found = fts_retriever.invoke("审批制度")

        self.assertEqual([document.metadata["source"] for document in found], [source])

    def test_candidate_deduplication_removes_exact_and_cross_route_containment(self):
        shared = "高血压诊断标准参见《国家基层高血压防治管理指南》。" * 8
        documents = [
            Document(
                page_content=shared,
                metadata={"source": "指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_FIXED},
            ),
            Document(
                page_content=shared,
                metadata={"source": "指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
            ),
            Document(
                page_content=shared + "补充的上下文说明。",
                metadata={"source": "指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
            ),
            Document(
                page_content="胰岛素治疗需要个体化评估。",
                metadata={"source": "指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
            ),
        ]

        deduplicated = retriever.deduplicate_chunk_routes(documents)

        self.assertEqual([document.page_content for document in deduplicated], [
            shared,
            "胰岛素治疗需要个体化评估。",
        ])

    def test_lexical_candidate_pool_preserves_all_chunk_routes(self):
        kb_name = "研发部"
        source = "指南.pdf"
        fts_path = os.path.join(retrieval.vector_store_dir(kb_name), "fts5.db")
        retrieval.build_fts5_index(
            [
                Document(
                    page_content="固定块：胰岛素的种类",
                    metadata={"source": source, "chunk_route": chunking.CHUNK_ROUTE_FIXED},
                ),
                Document(
                    page_content="结构块：胰岛素的种类和作用特点",
                    metadata={"source": source, "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
                Document(
                    page_content="视觉块：胰岛素种类与作用时间对照表",
                    metadata={"source": source, "chunk_route": chunking.CHUNK_ROUTE_VISUAL},
                ),
            ],
            fts_path,
        )

        with patch.object(
            app_storage.documents, "knowledge_base_exists", return_value=True
        ), patch.object(
            app_storage.documents, "ready_sources", return_value=[source]
        ), patch.object(
            app_storage.documents, "filtered_sources", return_value=[source]
        ):
            candidates = retriever.retrieve_lexical_candidates(
                kb_name, "胰岛素的种类", per_route=1
            )

        self.assertEqual(
            {candidate.metadata["chunk_route"] for candidate in candidates},
            {
                chunking.CHUNK_ROUTE_FIXED,
                chunking.CHUNK_ROUTE_STRUCTURED,
                chunking.CHUNK_ROUTE_VISUAL,
            },
        )


if __name__ == "__main__":
    unittest.main()
