import unittest

try:
    from langchain_core.documents import Document

    from services.indexing import chunking
    from tests.in_memory_index import InMemoryChunkIndex
except ModuleNotFoundError:
    Document = None
    chunking = None
    InMemoryChunkIndex = None


@unittest.skipUnless(Document and chunking and InMemoryChunkIndex, "requires backend dependencies")
class MinerUStructuredChunkingTest(unittest.TestCase):
    def split_mineru(self, document, *, chunk_size=800):
        """通过生产入口执行 MinerU 结构化切分。"""
        document.metadata["structured_profile"] = "mineru"
        return chunking.split_structured_markdown_documents(
            [document], chunk_size=chunk_size
        )

    def test_groups_adjacent_leaf_sections_and_excludes_bibliography(self):
        document = Document(
            page_content="""# 管理指南

## 8 治疗与管理

## 8.3 随访管理

随访管理应覆盖首次评估、定期复诊和异常情况处理。

## 8.3.1 一般要求

医疗机构应建立稳定的随访机制，并根据患者情况制定管理计划。

## 8.3.2 随访频率

病情稳定的患者应按计划复诊，出现异常情况时应及时增加随访。

## 8.3.3 随访内容

随访应记录症状、用药依从性、体征和必要的检查结果。

## 8.4 转诊

出现难以控制的风险因素或严重并发症时，应及时转诊。

## 参考文献

[1] 示例作者. 示例文献[J]. 2024;1(1):1-2.
""",
            metadata={"source": "管理指南.pdf"},
        )

        splits = self.split_mineru(document)
        follow_up = next(split for split in splits if "8.3.1 一般要求" in split.page_content)

        self.assertIn("8.3.2 随访频率", follow_up.page_content)
        self.assertIn("8.3.3 随访内容", follow_up.page_content)
        self.assertIn("首次评估、定期复诊", follow_up.page_content)
        self.assertIn("8.3 随访管理", follow_up.metadata["contextual_title"])
        self.assertTrue(
            all(
                split.metadata["chunk_route"] == chunking.CHUNK_ROUTE_STRUCTURED
                for split in splits
            )
        )
        self.assertTrue(all("示例文献" not in split.page_content for split in splits))
        self.assertTrue(any("8.4 转诊" in split.page_content for split in splits))

    def test_discards_leading_cover_sections_but_preserves_preface(self):
        document = Document(
            page_content="""# 示例管理标准

封面编号和发布信息。

## Example management standard

English cover metadata.

## 前言

本标准说明制定背景和适用原则。

# 示例管理标准

## 1 范围

本标准规定管理工作的适用范围。
""",
            metadata={"source": "示例管理标准.pdf"},
        )

        splits = self.split_mineru(document)
        text = "\n".join(split.page_content for split in splits)

        self.assertIn("制定背景和适用原则", text)
        self.assertIn("适用范围", text)
        self.assertNotIn("English cover metadata", text)
        self.assertNotIn("封面编号和发布信息", text)

    def test_never_groups_unrelated_top_level_sections(self):
        document = Document(
            page_content="""## 1 范围

本标准规定管理工作的适用范围和基本边界。

## 2 术语和定义

本标准定义管理工作中使用的关键术语。
""",
            metadata={"source": "示例标准.pdf"},
        )

        splits = self.split_mineru(document)

        self.assertEqual(len(splits), 2)
        self.assertIn("1 范围", splits[0].metadata["contextual_title"])
        self.assertIn("2 术语和定义", splits[1].metadata["contextual_title"])

    def test_preserves_table_rows_as_a_structured_block(self):
        document = Document(
            page_content="""## 一、随访评估

## （一）随访内容

表 1 随访项目

<table><tr><td>项目</td><td>首次评估</td></tr><tr><td>血压</td><td>是</td></tr><tr><td>用药依从性</td><td>是</td></tr></table>

注：首次评估应在建档时完成。
""",
            metadata={"source": "随访标准.pdf"},
        )

        splits = self.split_mineru(document)
        text = "\n".join(split.page_content for split in splits)

        self.assertIn("表 1 随访项目", text)
        self.assertIn("项目 | 首次评估", text)
        self.assertIn("血压 | 是", text)
        self.assertIn("表名：表 1 随访项目", text)
        self.assertIn("表头：项目 | 首次评估", text)
        self.assertIn("注：首次评估应在建档时完成。", text)
        self.assertEqual(len(splits), 1)
        self.assertNotIn("<table", text)
        self.assertNotIn("[[TABLE]]", text)
        self.assertNotIn("[[/TABLE]]", text)

    def test_expands_merged_cells_and_preserves_empty_columns_and_symbols(self):
        document = Document(
            page_content=r"""## 二、诊断

表 2 糖代谢状态分类

<table>
<tr><td rowspan="2">糖代谢状态</td><td colspan="2">静脉血浆葡萄糖(mmol/L)</td></tr>
<tr><td>空腹</td><td>糖负荷后2 h</td></tr>
<tr><td>正常血糖</td><td>&lt;6.1</td><td>&lt;7.8</td></tr>
<tr><td>IFG</td><td>$\geq 6.1$</td><td></td></tr>
<tr><td colspan="3">注：数值单位均为 mmol/L，目标达标率&gt;90%</td></tr>
</table>
""",
            metadata={"source": "诊断指南.pdf"},
        )

        splits = self.split_mineru(document)
        table = next(split.page_content for split in splits if "表头：" in split.page_content)

        self.assertIn(
            "表头：糖代谢状态 | 静脉血浆葡萄糖(mmol/L) / 空腹 | "
            "静脉血浆葡萄糖(mmol/L) / 糖负荷后2 h",
            table,
        )
        self.assertIn("正常血糖 | <6.1 | <7.8", table)
        self.assertIn("IFG | $\\geq 6.1$ |", table)
        self.assertIn("目标达标率>90%", table)
        self.assertEqual(table.count("注：数值单位均为 mmol/L"), 1)

    def test_large_table_chunks_repeat_caption_and_complete_header(self):
        rows = "".join(
            f"<tr><td>项目{index}</td><td>{'较长的评估说明' * 18}</td><td>{index}%</td></tr>"
            for index in range(12)
        )
        document = Document(
            page_content=f"""## 三、年度评估

表 9 年度评估项目

<table><tr><td>项目</td><td>评估要求</td><td>比例</td></tr>{rows}</table>
""",
            metadata={"source": "评估标准.pdf"},
        )

        splits = self.split_mineru(document)
        table_splits = [split for split in splits if "表头：" in split.page_content]

        self.assertGreater(len(table_splits), 1)
        self.assertTrue(
            all("表名：表 9 年度评估项目" in split.page_content for split in table_splits)
        )
        self.assertTrue(
            all("表头：项目 | 评估要求 | 比例" in split.page_content for split in table_splits)
        )
        combined = "\n".join(split.page_content for split in table_splits)
        for index in range(12):
            self.assertEqual(combined.count(f"项目{index} |"), 1)

    def test_heading_is_used_when_a_table_has_no_explicit_caption(self):
        document = Document(
            page_content="""## 人体测量学指标测量方法

<table><tr><td>指标</td><td>方法</td></tr><tr><td>BMI</td><td>体重/身高平方</td></tr></table>
""",
            metadata={"source": "测量指南.pdf"},
        )

        splits = self.split_mineru(document)
        table = next(split.page_content for split in splits if "表头：" in split.page_content)

        self.assertIn("表名：人体测量学指标测量方法", table)
        self.assertIn("BMI | 体重/身高平方", table)

    def test_appendix_title_and_continuation_are_bound_to_their_tables(self):
        document = Document(
            page_content="""## 七、中国传统锻炼功法

附录2 常用口服降糖药物

<table><tr><td>药物</td><td>剂量</td></tr><tr><td>二甲双胍</td><td>500 mg</td></tr></table>

续附录2

<table><tr><td>药物</td><td>剂量</td></tr><tr><td>阿卡波糖</td><td>50 mg</td></tr></table>
""",
            metadata={"source": "糖尿病指南.pdf"},
        )

        splits = self.split_mineru(document)
        table_splits = [split for split in splits if "表头：" in split.page_content]

        self.assertEqual(len(table_splits), 2)
        self.assertTrue(
            all("表名：附录2 常用口服降糖药物" in split.page_content for split in table_splits)
        )
        self.assertTrue(all("中国传统锻炼功法" not in split.page_content for split in table_splits))
        self.assertTrue(all("续附录2" not in split.page_content for split in splits))

    def test_route_filter_keeps_fixed_and_structured_results_separate(self):
        kb_id = "kb-route-filter"
        index = InMemoryChunkIndex()
        fixed = Document(
            page_content="固定路线中的高血压管理规则",
            metadata={"source": "指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_FIXED},
        )
        structured = Document(
            page_content="结构路线中的随访管理规则",
            metadata={"source": "指南.pdf", "chunk_route": chunking.CHUNK_ROUTE_STRUCTURED},
        )
        index.publish_source(kb_id, "指南.pdf", [fixed, structured], [[1.0, 0.0], [0.0, 1.0]])

        fixed_hits = index.search_lexical(
            kb_id, "高血压管理", chunking.CHUNK_ROUTE_FIXED, None, 5
        )
        structured_hits = index.search_lexical(
            kb_id, "随访管理", chunking.CHUNK_ROUTE_STRUCTURED, None, 5
        )

        self.assertEqual(fixed_hits[0].metadata["chunk_route"], chunking.CHUNK_ROUTE_FIXED)
        self.assertEqual(
            structured_hits[0].metadata["chunk_route"], chunking.CHUNK_ROUTE_STRUCTURED
        )
        # 内存实现按单字切词，固定路线可能因“管理”命中；只断言结构化块不会泄漏进来。
        for hit in index.search_lexical(kb_id, "随访管理", chunking.CHUNK_ROUTE_FIXED, None, 5):
            self.assertEqual(hit.metadata["chunk_route"], chunking.CHUNK_ROUTE_FIXED)


if __name__ == "__main__":
    unittest.main()
