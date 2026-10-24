import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from langchain_core.documents import Document
    from PIL import Image

    from services.core.settings import get_settings
    from services.documents import mineru as mineru_client
    from services.documents import visual as visual_content
    from services.documents.parser_modes import (
        IMAGE_PARSER_VISION_MODEL,
        PDF_PARSER_MINERU_CLOUD,
    )
    from services.documents.visual_cache import (
        STANDALONE_IMAGE_CACHE_FILENAME,
        VISUAL_CACHE_FILENAME,
    )
    from services.indexing import chunking
except ModuleNotFoundError:
    Image = None
    Document = None
    mineru_client = None
    chunking = None
    visual_content = None


@unittest.skipUnless(Image and visual_content, "requires the backend Python dependencies")
class MinerUVisualContentTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.storage_dir = os.path.join(self.directory.name, "storage")
        self.environment = patch.dict(
            os.environ,
            {
                "STORAGE_DIR": self.storage_dir,
                "MINERU_RESULTS_DIR": os.path.join(self.storage_dir, "mineru_results"),
                "RAG_IMAGE_ANALYSIS": "auto",
                "VISION_GLOBAL_MAX_CONCURRENCY": "4",
            },
            clear=False,
        )
        self.environment.start()
        self.source = os.path.join(self.storage_dir, "data", "研发部", "指南.pdf")
        os.makedirs(os.path.dirname(self.source), exist_ok=True)
        Path(self.source).write_bytes(b"%PDF-1.4")

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def _write_result(self, content_list, image_names):
        result_dir = mineru_client.result_dir_for_document(self.source)
        image_dir = result_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for image_name in image_names:
            Image.new("RGB", (80, 40), "white").save(image_dir / image_name)
        content_name = "result_content_list_v2.json"
        (result_dir / content_name).write_text(
            json.dumps(content_list, ensure_ascii=False), encoding="utf-8"
        )
        (result_dir / "pymodel_manifest.json").write_text(
            json.dumps({"content_list": content_name}, ensure_ascii=False), encoding="utf-8"
        )
        return result_dir

    @staticmethod
    def _image_block(path, caption=""):
        return {
            "type": "image",
            "content": {
                "image_source": {"path": path},
                "content": "",
                "image_caption": ([{"type": "text", "content": caption}] if caption else []),
                "image_footnote": [],
            },
        }

    def test_discovers_only_referenced_visuals_that_need_analysis(self):
        content_list = [
            [
                {
                    "type": "title",
                    "content": {
                        "level": 2,
                        "title_content": [{"type": "text", "content": "治疗流程"}],
                    },
                },
                {
                    "type": "table",
                    "content": {
                        "image_source": {"path": "images/normal.jpg"},
                        "html": "<table><tr><td>完整文字表格</td></tr></table>",
                        "table_caption": [],
                        "table_footnote": [],
                    },
                },
                {
                    "type": "table",
                    "content": {
                        "image_source": {"path": "images/complex.jpg"},
                        "html": (
                            '<table><tr><td><img src="images/cell.jpg"/></td>'
                            "<td>低风险</td></tr></table>"
                        ),
                        "table_caption": [
                            {"type": "text", "content": "表 2 个体化目标"}
                        ],
                        "table_footnote": [],
                    },
                },
                self._image_block("images/flow.jpg", "图 1 管理流程"),
                self._image_block("../../outside.jpg", "越界图片"),
            ]
        ]
        self._write_result(
            content_list,
            ["normal.jpg", "complex.jpg", "cell.jpg", "flow.jpg", "orphan.jpg"],
        )
        Path(self.directory.name, "outside.jpg").write_bytes(b"outside")

        candidates = visual_content.discover_visual_candidates(self.source)

        self.assertEqual(
            [candidate.relative_path for candidate in candidates],
            ["images/complex.jpg", "images/flow.jpg"],
        )
        self.assertEqual(candidates[0].caption, "表 2 个体化目标")
        self.assertEqual(candidates[0].headings, ("治疗流程",))
        self.assertEqual(candidates[0].image_path.name, "complex.jpg")

    @staticmethod
    def _paragraph_block(text):
        return {
            "type": "paragraph",
            "content": {"paragraph_content": [{"type": "text", "content": text}]},
        }

    def test_caption_is_recovered_from_the_paragraph_right_below_an_image(self):
        self._write_result(
            [
                [
                    self._image_block("images/flow.jpg"),
                    self._paragraph_block("图 1 超重和肥胖症诊疗路径"),
                    self._image_block("images/other.jpg"),
                    self._paragraph_block("图 2 的结果表明，两组差异显著。"),
                    self._image_block("images/third.jpg", "图 3 自带图注"),
                    self._paragraph_block("图 4 不属于第三张图"),
                ]
            ],
            ["flow.jpg", "other.jpg", "third.jpg"],
        )

        captions = {
            candidate.relative_path: candidate.caption
            for candidate in visual_content.discover_visual_candidates(self.source)
        }

        self.assertEqual(captions["images/flow.jpg"], "图 1 超重和肥胖症诊疗路径")
        # 带句号的正文句子不是图注。
        self.assertEqual(captions["images/other.jpg"], "")
        # MinerU 已给出图注时不被相邻段落覆盖。
        self.assertEqual(captions["images/third.jpg"], "图 3 自带图注")

    def test_table_caption_is_taken_from_the_block_above_not_below(self):
        table = {
            "type": "table",
            "content": {
                "image_source": {"path": "images/t.jpg"},
                "html": '<table><tr><td><img src="x.jpg"/></td></tr></table>',
                "table_caption": [],
                "table_footnote": [],
            },
        }
        self._write_result(
            [[self._paragraph_block("表 1 随访频率"), table, self._paragraph_block("表 2 无关")]],
            ["t.jpg"],
        )

        candidates = visual_content.discover_visual_candidates(self.source)

        self.assertEqual(candidates[0].caption, "表 1 随访频率")

    def test_vision_config_reads_the_named_provider_without_changing_active_model(self):
        configured = {
            "api_key": "openai-key",
            "base_url": "https://openai.example/v1",
            "model": "configured-vision-model",
        }
        with patch.dict(
            os.environ,
            {
                "RAG_VISION_PROVIDER": "OpenAI",
                "RAG_VISION_MODEL": "",
                "RAG_VISION_BASE_URL": "",
                "RAG_VISION_API_KEY": "",
            },
            clear=False,
        ), patch.object(
            visual_content.model_config, "get_model_config", return_value=configured
        ) as get_model, patch.object(
            visual_content.model_config, "get_active_model_config"
        ) as get_active:
            result = visual_content._vision_model_config()

        get_model.assert_called_once()
        self.assertEqual(get_model.call_args.args[0], "OpenAI")
        get_active.assert_not_called()
        self.assertEqual(result, configured)

    def test_model_result_is_cached_and_becomes_a_visual_document(self):
        self._write_result(
            [[self._image_block("images/flow.jpg", "图 A.1 高血压管理流程")]],
            ["flow.jpg"],
        )
        analysis = visual_content.VisualAnalysis(
            indexable=True,
            kind="flowchart",
            title="高血压管理流程",
            content="筛查后先测量血压；确诊后进入分级管理。",
        )
        config = {"api_key": "key", "base_url": "http://model/v1", "model": "vision"}
        with patch.object(
            visual_content, "_vision_model_config", return_value=config
        ), patch.object(
            visual_content, "_analyze_with_model", return_value=analysis
        ) as analyze, patch.object(
            visual_content, "_ocr_fallback"
        ) as ocr:
            first = visual_content.load_mineru_visual_documents(
                self.source, "指南.pdf", PDF_PARSER_MINERU_CLOUD
            )
            second = visual_content.load_mineru_visual_documents(
                self.source, "指南.pdf", PDF_PARSER_MINERU_CLOUD
            )

        analyze.assert_called_once()
        ocr.assert_not_called()
        self.assertEqual(first[0].page_content, second[0].page_content)
        self.assertIn("确诊后进入分级管理", first[0].page_content)
        self.assertEqual(first[0].metadata["chunk_route"], chunking.CHUNK_ROUTE_VISUAL)
        self.assertEqual(first[0].metadata["visual_kind"], "flowchart")
        self.assertTrue(
            (
                mineru_client.result_dir_for_document(self.source)
                / VISUAL_CACHE_FILENAME
            ).is_file()
        )

    def test_repeated_model_failure_stops_more_remote_calls_and_uses_ocr(self):
        self._write_result(
            [[
                self._image_block("images/one.jpg", "图 1"),
                self._image_block("images/two.jpg", "图 2"),
            ]],
            ["one.jpg", "two.jpg"],
        )
        fallback = visual_content.VisualAnalysis(
            indexable=True,
            kind="image",
            title="",
            content="图片 OCR 文字：本地识别结果",
        )
        with patch.dict(
            os.environ, {"VISION_GLOBAL_MAX_CONCURRENCY": "1"}, clear=False
        ), patch.object(
            visual_content,
            "_vision_model_config",
            return_value={"api_key": "key", "base_url": "http://model/v1", "model": "text-only"},
        ), patch.object(
            visual_content, "_analyze_with_model", side_effect=RuntimeError("不支持图片")
        ) as analyze, patch.object(
            visual_content, "_ocr_fallback", return_value=fallback
        ) as ocr:
            documents = visual_content.load_mineru_visual_documents(
                self.source, "指南.pdf", PDF_PARSER_MINERU_CLOUD
            )

        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(ocr.call_count, 2)
        self.assertEqual(len(documents), 2)
        self.assertTrue(all(document.metadata["visual_analysis"] == "ocr" for document in documents))

    def test_model_calls_run_concurrently_and_documents_keep_source_order(self):
        self._write_result(
            [[
                self._image_block("images/one.jpg", "图 1"),
                self._image_block("images/two.jpg", "图 2"),
            ]],
            ["one.jpg", "two.jpg"],
        )
        lock = threading.Lock()
        both_started = threading.Event()
        active = 0
        peak_active = 0

        def analyze(candidate, _image_bytes, _config):
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
                if active == 2:
                    both_started.set()
            both_started.wait(timeout=1)
            time.sleep(0.02)
            with lock:
                active -= 1
            return visual_content.VisualAnalysis(
                indexable=True,
                kind="image",
                title=candidate.caption,
                content=f"{candidate.caption} 的可检索内容",
            )

        with patch.object(
            visual_content,
            "_vision_model_config",
            return_value={"api_key": "key", "base_url": "http://model/v1", "model": "vision"},
        ), patch.object(visual_content, "_analyze_with_model", side_effect=analyze):
            documents = visual_content.load_mineru_visual_documents(
                self.source, "指南.pdf", PDF_PARSER_MINERU_CLOUD
            )

        self.assertEqual(peak_active, 2)
        self.assertEqual(
            [document.metadata["contextual_title"] for document in documents],
            ["指南 > 图 1", "指南 > 图 2"],
        )

    def test_vision_concurrency_setting_is_bounded(self):
        for value, expected in (("0", 1), ("999", 32), ("invalid", 6)):
            with patch.dict(os.environ, {"VISION_GLOBAL_MAX_CONCURRENCY": value}, clear=False):
                self.assertEqual(get_settings().vision_global_max_concurrency, expected)

    def test_transient_model_failure_is_retried_before_ocr_fallback(self):
        self._write_result(
            [[self._image_block("images/flow.jpg", "图 1")]],
            ["flow.jpg"],
        )
        analysis = visual_content.VisualAnalysis(
            indexable=True,
            kind="flowchart",
            title="诊断流程",
            content="筛查后进入诊断流程。",
        )
        with patch.object(
            visual_content,
            "_vision_model_config",
            return_value={"api_key": "key", "base_url": "http://model/v1", "model": "vision"},
        ), patch.object(
            visual_content,
            "_analyze_with_model",
            side_effect=[RuntimeError("临时响应无效"), analysis],
        ) as analyze, patch.object(
            visual_content, "_ocr_fallback"
        ) as ocr:
            documents = visual_content.load_mineru_visual_documents(
                self.source, "指南.pdf", PDF_PARSER_MINERU_CLOUD
            )

        self.assertEqual(analyze.call_count, 2)
        ocr.assert_not_called()
        self.assertEqual(documents[0].metadata["visual_analysis"], "vision")

    def test_bad_response_on_one_image_does_not_disable_model_for_the_rest(self):
        self._write_result(
            [[
                self._image_block("images/one.jpg", "图 1"),
                self._image_block("images/two.jpg", "图 2"),
            ]],
            ["one.jpg", "two.jpg"],
        )
        analysis = visual_content.VisualAnalysis(
            indexable=True,
            kind="flowchart",
            title="诊断流程",
            content="筛查后进入诊断流程。",
        )
        fallback = visual_content.VisualAnalysis(
            indexable=True,
            kind="image",
            title="",
            content="图片 OCR 文字：本地识别结果",
        )
        bad = visual_content.visual_vision.VisionResponseError("格式不合约定")
        # 并发数为 1：第一张图两次都回答错误，第二张图仍应请求模型。
        with patch.dict(
            os.environ, {"VISION_GLOBAL_MAX_CONCURRENCY": "1"}, clear=False
        ), patch.object(
            visual_content,
            "_vision_model_config",
            return_value={"api_key": "key", "base_url": "http://model/v1", "model": "vision"},
        ), patch.object(
            visual_content, "_analyze_with_model", side_effect=[bad, bad, analysis]
        ) as analyze, patch.object(
            visual_content, "_ocr_fallback", return_value=fallback
        ) as ocr:
            documents = visual_content.load_mineru_visual_documents(
                self.source, "指南.pdf", PDF_PARSER_MINERU_CLOUD
            )

        self.assertEqual(analyze.call_count, 3)
        ocr.assert_called_once()
        self.assertEqual(
            sorted(document.metadata["visual_analysis"] for document in documents),
            ["ocr", "vision"],
        )

    def test_model_configuration_failure_also_uses_ocr(self):
        self._write_result(
            [[self._image_block("images/flow.jpg", "图 1")]],
            ["flow.jpg"],
        )
        fallback = visual_content.VisualAnalysis(
            indexable=True,
            kind="image",
            title="",
            content="图片 OCR 文字：本地识别结果",
        )
        with patch.object(
            visual_content, "_vision_model_config", side_effect=RuntimeError("配置库不可用")
        ), patch.object(
            visual_content, "_analyze_with_model"
        ) as analyze, patch.object(
            visual_content, "_ocr_fallback", return_value=fallback
        ) as ocr:
            documents = visual_content.load_mineru_visual_documents(
                self.source, "指南.pdf", PDF_PARSER_MINERU_CLOUD
            )

        analyze.assert_not_called()
        ocr.assert_called_once()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["visual_analysis"], "ocr")

    def test_standalone_image_caches_vision_result_and_keeps_ocr_text(self):
        image_path = os.path.join(self.storage_dir, "data", "研发部", "审批流程.png")
        Image.new("RGB", (80, 40), "white").save(image_path)
        analysis = visual_content.VisualAnalysis(
            indexable=True,
            kind="flowchart",
            title="审批流程",
            content="提交申请后由部门负责人审批；通过后归档。",
        )
        config = {"api_key": "key", "base_url": "http://model/v1", "model": "vision"}
        with patch.object(
            visual_content, "_vision_model_config", return_value=config
        ), patch.object(
            visual_content, "_analyze_with_model", return_value=analysis
        ) as analyze, patch.object(
            visual_content, "_extract_ocr_text", return_value="申请编号 A-001"
        ) as ocr:
            first = visual_content.load_standalone_image_document(
                image_path, "审批流程.png", IMAGE_PARSER_VISION_MODEL
            )
            second = visual_content.load_standalone_image_document(
                image_path, "审批流程.png", IMAGE_PARSER_VISION_MODEL
            )

        analyze.assert_called_once()
        self.assertEqual(ocr.call_count, 2)
        self.assertIn("部门负责人审批", first.page_content)
        self.assertIn("申请编号 A-001", first.page_content)
        self.assertEqual(first.metadata["visual_analysis"], "vision+ocr")
        self.assertEqual(first.page_content, second.page_content)
        self.assertTrue(
            (
                mineru_client.result_dir_for_document(image_path)
                / STANDALONE_IMAGE_CACHE_FILENAME
            ).is_file()
        )

    def test_standalone_image_uses_ocr_when_vision_model_fails(self):
        image_path = os.path.join(self.storage_dir, "data", "研发部", "文字图片.jpg")
        Image.new("RGB", (80, 40), "white").save(image_path)
        with patch.object(
            visual_content,
            "_vision_model_config",
            return_value={"api_key": "key", "base_url": "http://model/v1", "model": "vision"},
        ), patch.object(
            visual_content, "_analyze_with_model", side_effect=RuntimeError("模型不可用")
        ), patch.object(
            visual_content, "_extract_ocr_text", return_value="本地识别文字"
        ):
            document = visual_content.load_standalone_image_document(
                image_path, "文字图片.jpg", IMAGE_PARSER_VISION_MODEL
            )

        self.assertIsNotNone(document)
        self.assertEqual(document.metadata["visual_analysis"], "ocr")
        self.assertIn("本地识别文字", document.page_content)

    def test_visual_split_is_self_contained_and_uses_source_filename(self):
        document = Document(
            page_content="视觉内容类型：表格\n\n年龄较小对应更严格的控制目标。",
            metadata={
                "source": "指南.pdf",
                "page": 5,
                "chunk_route": "visual",
                "contextual_title": "指南 > 个体化目标",
            },
        )

        splits = chunking.split_mineru_visual_documents([document])

        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0].metadata["chunk_route"], chunking.CHUNK_ROUTE_VISUAL)
        self.assertEqual(splits[0].metadata["chunk_index"], 0)
        self.assertEqual(
            splits[0].metadata["contextual_title"], "指南 > 个体化目标"
        )


if __name__ == "__main__":
    unittest.main()
