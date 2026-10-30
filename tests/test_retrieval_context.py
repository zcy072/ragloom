import re
import unittest
from unittest.mock import patch

try:
    from langchain_core.documents import Document

    import app_storage
    from services.indexing import chunking
    from services.retrieval import context as context_builder
    from services.retrieval import parent_store
    from services.retrieval import retriever as retriever
    from services.retrieval.chunk_index import get_chunk_index
except ModuleNotFoundError:
    Document = None
    chunking = None
    context_builder = None
    parent_store = None
    retriever = None

from tests.retrieval_fixtures import create_knowledge_base, mark_ready, publish
from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(Document and retriever, "requires the backend Python dependencies")
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
        publish(kb_name, source, splits)

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

    def test_neighbor_windows_are_merged_into_one_query_per_range(self):
        kb_name = "研发部"
        source = "手册.pdf"
        splits = [
            Document(page_content=f"内容片段 {index}", metadata={"source": source, "page": 0})
            for index in range(20)
        ]
        publish(kb_name, source, splits)
        hits = [
            Document(page_content=f"内容片段 {index}", metadata={"source": source, "page": 0, "chunk_index": index})
            for index in (1, 3, 15)
        ]
        index = get_chunk_index()

        with patch.object(index, "neighbors", wraps=index.neighbors) as neighbors:
            spans = context_builder.build_context_spans(kb_name, hits, neighbor_chunks=1)

        # 1 和 3 的窗口相接合并成 [0, 4]，15 相距较远单独查 [14, 16]。
        self.assertEqual(
            sorted(call.args[3:] for call in neighbors.call_args_list), [(0, 4), (14, 16)]
        )
        self.assertEqual([(span.chunk_start, span.chunk_end) for span in spans], [(0, 4), (14, 16)])

    def test_empty_spans_do_not_take_span_slots(self):
        source = "手册.pdf"
        hits = [
            Document(page_content=content, metadata={"source": source, "row": index})
            for index, content in enumerate(["  ", "第一段", "第二段"])
        ]

        spans = context_builder.build_context_spans("研发部", hits, max_spans=2)

        self.assertEqual([span.content for span in spans], ["第一段", "第二段"])

    def test_long_structured_context_keeps_its_section_boundary(self):
        kb_name = "研发部"
        source = "技术规范.md"
        before = "相邻的前置章节。" * 80
        target = "目标章节的完整说明。" * 80
        after = "相邻的后置章节。" * 80
        publish(
            kb_name,
            source,
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
        publish(
            kb_name,
            source,
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
        publish(kb_name, source, documents)

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

        publish(kb_name, source, [s for s in splits if s.metadata["source"] == source])
        publish(kb_name, "费用制度.md", [s for s in splits if s.metadata["source"] != source])
        found = get_chunk_index().search_lexical(
            app_storage.knowledge_bases.kb_id_for(kb_name), "请假", None, [source], 5
        )

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


    def test_route_probe_returns_routes_for_allowed_sources(self):
        kb_name = "研发部"
        kb_id = publish(
            kb_name,
            "制度.pdf",
            [
                Document(
                    page_content="固定路线",
                    metadata={"source": "制度.pdf", "chunk_route": chunking.CHUNK_ROUTE_FIXED},
                ),
                Document(
                    page_content="结构化路线",
                    metadata={"source": "制度.pdf", "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
                ),
            ],
        )
        publish(
            kb_name,
            "其他.pdf",
            [
                Document(
                    page_content="视觉路线",
                    metadata={"source": "其他.pdf", "chunk_route": chunking.CHUNK_ROUTE_VISUAL},
                )
            ],
        )

        self.assertEqual(
            get_chunk_index().available_routes(kb_id, ["制度.pdf"]),
            {chunking.CHUNK_ROUTE_FIXED, chunking.CHUNK_ROUTE_STRUCTURED},
        )

    def test_missing_index_requires_a_complete_rebuild(self):
        create_knowledge_base("研发部")
        mark_ready("研发部", "制度.pdf")

        from services.core.container import get_application_container

        with self.assertRaisesRegex(ValueError, "完整重建"):
            retriever._route_plans(
                get_application_container(), get_chunk_index(), "研发部", None, 0.70
            )

    def test_parent_text_is_restored_from_the_parent_store(self):
        kb_name = "研发部"
        source = "制度.pdf"
        child = Document(
            page_content="子块中的审批规则",
            metadata={
                "source": source,
                "page": 2,
                "parent_id": "parent-1",
                "parent_content": "父块完整内容，包含审批规则和适用范围。",
            },
        )
        kb_id = publish(kb_name, source, [child])
        parent_store.replace_source(kb_id, source, parent_store.parent_chunk_rows([child]))

        found = get_chunk_index().search_lexical(kb_id, "审批规则", None, [source], 5)
        self.assertNotIn("parent_content", found[0].metadata)
        spans = context_builder.build_context_spans(kb_name, found, neighbor_chunks=0)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].content, child.metadata["parent_content"])

    def test_lexical_candidate_pool_preserves_all_chunk_routes(self):
        kb_name = "研发部"
        source = "指南.pdf"
        publish(
            kb_name,
            source,
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
        )
        mark_ready(kb_name, source)

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

    def test_lexical_candidates_ignore_sources_that_are_not_ready(self):
        kb_name = "研发部"
        publish(kb_name, "草稿.pdf", [Document(page_content="审批流程草稿", metadata={"source": "草稿.pdf"})])

        self.assertEqual(retriever.retrieve_lexical_candidates(kb_name, "审批流程"), [])

    def test_lexical_candidates_failure_is_logged_not_raised(self):
        kb_name = "研发部"
        create_knowledge_base(kb_name)
        mark_ready(kb_name, "制度.pdf")

        with patch.object(
            get_chunk_index(), "search_lexical", side_effect=RuntimeError("milvus down")
        ):
            self.assertEqual(retriever.retrieve_lexical_candidates(kb_name, "审批"), [])


if __name__ == "__main__":
    unittest.main()
