"""MilvusChunkIndex 集成测试；仅在设置 MILVUS_URI 时运行，使用一次性 collection 前缀。"""

import os
import unittest
import uuid

try:
    from langchain_core.documents import Document

    from services.retrieval.chunk_index import build_filter
    from services.retrieval.milvus_index import MilvusChunkIndex, _clip
except ModuleNotFoundError:
    Document = None
    MilvusChunkIndex = None
    _clip = None

MILVUS_URI = os.getenv("MILVUS_URI", "")
DIMENSION = 8


def _vector(axis: int) -> list[float]:
    """生成指向某一维的单位向量，便于构造确定的稠密检索结果。"""
    return [1.0 if index == axis else 0.0 for index in range(DIMENSION)]


def _split(text: str, source: str, route: str = "fixed", **metadata) -> "Document":
    """构造带路线元数据的子块。"""
    values = {"source": source, "chunk_route": route, "parent_id": "p", **metadata}
    return Document(page_content=text, metadata=values)


@unittest.skipUnless(Document, "requires the backend Python dependencies")
class VarcharByteLimitTest(unittest.TestCase):
    def test_clip_counts_utf8_bytes_not_characters(self):
        text = "中" * 30000  # 30000 字符 = 90000 字节，超过 65535 字节上限

        clipped = _clip(text, 65535)

        self.assertLessEqual(len(clipped.encode("utf-8")), 65535)
        self.assertEqual(clipped, "中" * 21845)  # 不会在多字节字符中间截断

    def test_clip_keeps_short_and_ascii_values_untouched(self):
        self.assertEqual(_clip("abc", 64), "abc")
        self.assertEqual(_clip(7, 64), "7")
        self.assertEqual(_clip("a" * 100, 64), "a" * 64)

    def test_row_fields_fit_their_varchar_limits_for_long_chinese_text(self):
        split = _split(
            "文" * 40000,
            "资料.pdf",
            contextual_title="题" * 2000,
            visual_analysis="图" * 40000,
        )

        row = MilvusChunkIndex._row("kb", "资料.pdf", 0, split, [0.0] * DIMENSION, "g")

        for field, limit in (
            ("content", 65535),
            ("search_text", 65535),
            ("contextual_title", 2048),
            ("visual_analysis", 65535),
        ):
            self.assertLessEqual(len(row[field].encode("utf-8")), limit, field)


@unittest.skipUnless(MILVUS_URI and Document, "requires MILVUS_URI and backend dependencies")
class MilvusChunkIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prefix = f"test_{uuid.uuid4().hex[:8]}_"
        cls.index = MilvusChunkIndex(MILVUS_URI, "", cls.prefix, DIMENSION)
        cls.index.ensure_collection()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.index._get_client().drop_collection(cls.index._collection)

    def setUp(self) -> None:
        self.kb_id = uuid.uuid4().hex

    def tearDown(self) -> None:
        self.index.delete_kb(self.kb_id)

    def test_filter_escapes_quotes_and_backslashes(self):
        source = 'a"b\\c.pdf'
        self.index.publish_source(
            self.kb_id, source, [_split("特殊文件名内容", source)], [_vector(0)]
        )
        self.index.publish_source(
            self.kb_id, "other.pdf", [_split("别的内容", "other.pdf")], [_vector(1)]
        )

        hits = self.index.search_dense(self.kb_id, _vector(0), None, [source], 5)

        self.assertEqual([doc.metadata["source"] for doc in hits], [source])
        self.assertIn('"a\\"b\\\\c.pdf"', build_filter(self.kb_id, sources=[source]))

    def test_republish_replaces_previous_generation(self):
        source = "制度.pdf"
        self.index.publish_source(self.kb_id, source, [_split("旧版本报销规则", source)], [_vector(0)])
        self.index.publish_source(self.kb_id, source, [_split("新版本差旅规则", source)], [_vector(0)])

        hits = self.index.search_dense(self.kb_id, _vector(0), None, None, 10)

        self.assertEqual([doc.page_content for doc in hits], ["新版本差旅规则"])

    def test_republish_removes_every_old_row_beyond_one_query_batch(self):
        source = "大文件.pdf"
        old = [_split(f"旧第{i}段", source) for i in range(1200)]
        self.index.publish_source(self.kb_id, source, old, [_vector(0)] * len(old))
        new = [_split(f"新第{i}段", source) for i in range(3)]
        self.index.publish_source(self.kb_id, source, new, [_vector(0)] * len(new))

        client = self.index._get_client()
        rows = client.query(
            self.index._collection,
            filter=build_filter(self.kb_id, source=source),
            output_fields=["content"],
            limit=2000,
        )
        self.assertEqual(sorted(row["content"] for row in rows), ["新第0段", "新第1段", "新第2段"])

    def test_failed_publish_keeps_previous_version(self):
        source = "制度.pdf"
        self.index.publish_source(self.kb_id, source, [_split("旧版本报销规则", source)], [_vector(0)])

        with self.assertRaises(ValueError):
            self.index.publish_source(
                self.kb_id, source, [_split("新", source), _split("版", source)], [_vector(0)]
            )

        hits = self.index.search_dense(self.kb_id, _vector(0), None, None, 10)
        self.assertEqual([doc.page_content for doc in hits], ["旧版本报销规则"])

    def test_available_routes_and_neighbors(self):
        source = "手册.pdf"
        splits = [
            _split("第一段内容", source),
            _split("第二段内容", source),
            _split("第三段内容", source),
            _split("图片说明", source, route="visual"),
        ]
        self.index.publish_source(
            self.kb_id, source, splits, [_vector(index) for index in range(4)]
        )

        self.assertEqual(self.index.available_routes(self.kb_id, [source]), {"fixed", "visual"})
        self.assertEqual(self.index.available_routes(self.kb_id, ["不存在.pdf"]), set())
        found = self.index.neighbors(self.kb_id, source, "fixed", 1, 2)
        self.assertEqual([doc.page_content for doc in found], ["第二段内容", "第三段内容"])

    def test_lexical_and_hybrid_find_chinese_and_model_numbers(self):
        source = "设备.pdf"
        self.index.publish_source(
            self.kb_id,
            source,
            [_split("设备 AB-123 每月保养一次", source), _split("设备 CD-456 每季度检修", source)],
            [_vector(0), _vector(1)],
        )
        self.index.publish_source(
            self.kb_id, "人事.pdf", [_split("年假申请需提前三天审批", "人事.pdf")], [_vector(2)]
        )

        lexical = self.index.search_lexical(self.kb_id, "AB-123", None, None, 3)
        chinese = self.index.search_lexical(self.kb_id, "年假审批", None, None, 3)
        hybrid = self.index.search_hybrid(self.kb_id, "CD-456 检修", _vector(1), None, None, 3)

        self.assertIn("AB-123", lexical[0].page_content)
        self.assertEqual(chinese[0].metadata["source"], "人事.pdf")
        self.assertIn("CD-456", hybrid[0].page_content)

    def test_reset_if_empty_rechecks_before_dropping_the_collection(self):
        self.index.publish_source(
            self.kb_id, "制度.pdf", [_split("审批规则", "制度.pdf")], [_vector(0)]
        )
        answers = iter([False, True])  # 第一次判断为空，拿到锁后已有新文档登记

        self.assertFalse(self.index.reset_if_empty(lambda: next(answers)))
        self.assertEqual(self.index.available_routes(self.kb_id, None), {"fixed"})

    def test_deletes_are_idempotent_and_scoped(self):
        self.index.publish_source(self.kb_id, "甲.pdf", [_split("甲内容", "甲.pdf")], [_vector(0)])
        self.index.publish_source(self.kb_id, "乙.pdf", [_split("乙内容", "乙.pdf")], [_vector(1)])

        self.index.delete_source(self.kb_id, "甲.pdf")
        self.index.delete_source(self.kb_id, "甲.pdf")
        self.assertEqual(self.index.delete_stale_sources(self.kb_id, {"乙.pdf"}), 0)
        self.assertEqual(self.index.available_routes(self.kb_id, None), {"fixed"})

        self.index.delete_kb(self.kb_id)
        self.index.delete_kb(self.kb_id)
        self.assertEqual(self.index.available_routes(self.kb_id, None), set())


if __name__ == "__main__":
    unittest.main()
