import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

try:
    from langchain_core.documents import Document

    import app_storage
    from services.documents import loaders as document_loader
    from services.documents import mineru as mineru_client
    from services.documents.mineru import client as mineru_api_client
    from services.documents.parser_modes import (
        DEFAULT_IMAGE_PARSER_MODE,
        IMAGE_PARSER_VISION_MODEL,
        PDF_PARSER_MINERU_CLOUD,
        PDF_PARSER_NATIVE_TEXT,
        PDF_PARSER_PADDLE_OCR,
        normalize_document_parser_mode,
    )
except ModuleNotFoundError:
    Document = None
    document_loader = None
    mineru_client = None


@unittest.skipUnless(document_loader and mineru_client and Document, "requires backend dependencies")
class PdfParserTest(unittest.TestCase):
    def test_pdf_mode_is_normalized_only_for_pdf_files(self):
        self.assertEqual(
            normalize_document_parser_mode("扫描件.pdf", PDF_PARSER_PADDLE_OCR),
            PDF_PARSER_PADDLE_OCR,
        )
        self.assertEqual(
            normalize_document_parser_mode("说明.txt", PDF_PARSER_MINERU_CLOUD),
            PDF_PARSER_NATIVE_TEXT,
        )
        self.assertEqual(
            normalize_document_parser_mode("制度.docx", PDF_PARSER_MINERU_CLOUD),
            PDF_PARSER_MINERU_CLOUD,
        )
        self.assertEqual(
            normalize_document_parser_mode("制度.docx", PDF_PARSER_PADDLE_OCR),
            PDF_PARSER_NATIVE_TEXT,
        )
        with self.assertRaises(ValueError):
            normalize_document_parser_mode("扫描件.pdf", "unknown")

    def test_image_modes_default_to_vision_and_keep_legacy_records_local(self):
        self.assertEqual(
            normalize_document_parser_mode("流程图.png", None),
            DEFAULT_IMAGE_PARSER_MODE,
        )
        self.assertEqual(
            normalize_document_parser_mode("历史图片.jpg", PDF_PARSER_NATIVE_TEXT),
            PDF_PARSER_PADDLE_OCR,
        )
        self.assertEqual(
            normalize_document_parser_mode("流程图.jpeg", IMAGE_PARSER_VISION_MODEL),
            IMAGE_PARSER_VISION_MODEL,
        )
        with self.assertRaises(ValueError):
            normalize_document_parser_mode("流程图.png", PDF_PARSER_MINERU_CLOUD)

    def test_mineru_content_list_preserves_page_numbers_and_skips_headers(self):
        content_list = [
            [
                {
                    "type": "page_header",
                    "content": {"page_header_content": [{"type": "text", "content": "页眉"}]},
                },
                {
                    "type": "title",
                    "content": {
                        "level": 2,
                        "title_content": [{"type": "text", "content": "诊疗建议"}],
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [{"type": "text", "content": "建议规律复诊。"}],
                    },
                },
            ],
            [
                {
                    "type": "table",
                    "content": {
                        "table_caption": [{"type": "text", "content": "随访频率"}],
                        "html": "<table><tr><td>每月</td><td>一次</td></tr></table>",
                    },
                }
            ],
        ]

        pages = mineru_client.content_list_to_page_texts(content_list)

        self.assertEqual([page for page, _ in pages], [0, 1])
        self.assertIn("## 诊疗建议", pages[0][1])
        self.assertIn("建议规律复诊", pages[0][1])
        self.assertNotIn("页眉", pages[0][1])
        self.assertIn("随访频率", pages[1][1])
        self.assertIn("每月", pages[1][1])

    def test_mineru_keeps_numbered_recommendations_and_skips_bibliography_rows(self):
        content_list = [
            [
                {
                    "type": "list",
                    "content": {
                        "list_type": "reference_list",
                        "list_items": [
                            {
                                "item_content": [
                                    {"type": "text", "content": "1. 建议每周规律运动。"}
                                ]
                            }
                        ],
                    },
                },
                {
                    "type": "list",
                    "content": {
                        "list_type": "reference_list",
                        "list_items": [
                            {
                                "item_content": [
                                    {"type": "text", "content": "[1] 示例作者. 示例文献."}
                                ]
                            }
                        ],
                    },
                },
            ],
            [
                {
                    "type": "list",
                    "content": {
                        "list_type": "reference_list",
                        "list_items": [
                            {"item_content": [{"type": "text", "content": "12-18. DOI: test"}]}
                        ],
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [{"type": "text", "content": "后续正文。"}]
                    },
                },
            ],
        ]

        pages = mineru_client.content_list_to_page_texts(content_list)

        self.assertIn("- 1. 建议每周规律运动。", pages[0][1])
        self.assertNotIn("[1] 示例作者", pages[0][1])
        self.assertEqual(pages[1], (1, "后续正文。"))

    def test_mineru_keeps_algorithm_and_equation_content(self):
        content_list = [
            [
                {
                    "type": "algorithm",
                    "content": {
                        "algorithm_content": [
                            {"type": "text", "content": "诊疗流程建议"}
                        ]
                    },
                },
                {"type": "equation_interline", "content": {"math_content": "HbA1c >= 6.5%"}},
            ]
        ]

        pages = mineru_client.content_list_to_page_texts(content_list)

        self.assertIn("诊疗流程建议", pages[0][1])
        self.assertIn("HbA1c >= 6.5%", pages[0][1])

    def test_mineru_text_keeps_comparison_operators_but_strips_html_tags(self):
        self.assertEqual(
            mineru_client.content_text("当 x < 5 且 y > 3 时触发告警"),
            "当 x < 5 且 y > 3 时触发告警",
        )
        self.assertEqual(
            mineru_client.content_text("温度<10℃或湿度>80%时停机"),
            "温度<10℃或湿度>80%时停机",
        )
        # 实体先还原再清洗，还原出的比较符同样要保留。
        self.assertEqual(mineru_client.content_text("&lt;5 天且 &gt;3 次"), "<5 天且 >3 次")
        # 真正的标签、注释和换行类标签仍按原行为处理。
        self.assertEqual(
            mineru_client.content_text("<p>第一段</p><p>第二段 <b>加粗</b></p><!-- 注释 -->"),
            "第一段\n第二段 加粗",
        )
        pages = mineru_client.content_list_to_page_texts(
            [[{"type": "paragraph", "content": {"paragraph_content": "压力<0.5MPa 且温度>80℃"}}]]
        )
        self.assertEqual(pages, [(0, "压力<0.5MPa 且温度>80℃")])

    def test_mineru_excludes_bibliography_sections_but_keeps_normative_references(self):
        content_list = [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "2 规范性引用文件"}]
                    },
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "本标准引用 GB/T 示例标准。"}
                        ]
                    },
                },
                {
                    "type": "title",
                    "content": {
                        "title_content": [{"type": "text", "content": "3. 参考文献"}]
                    },
                },
                {
                    "type": "list",
                    "content": {
                        "list_items": [
                            {"item_content": [{"type": "text", "content": "[1] 作者. 文献。"}]}
                        ]
                    },
                },
            ],
            [
                {
                    "type": "title",
                    "content": {"title_content": [{"type": "text", "content": "附录 A"}]},
                },
                {
                    "type": "paragraph",
                    "content": {
                        "paragraph_content": [{"type": "text", "content": "附录中的管理流程。"}]
                    },
                },
            ],
        ]

        pages = mineru_client.content_list_to_page_texts(content_list)
        text = "\n".join(page_text for _, page_text in pages)

        self.assertIn("规范性引用文件", text)
        self.assertIn("GB/T 示例标准", text)
        self.assertNotIn("作者. 文献", text)
        self.assertIn("附录中的管理流程", text)
        self.assertTrue(mineru_client.is_bibliography_heading("3. 参考文献"))
        self.assertFalse(mineru_client.is_bibliography_heading("2 规范性引用文件"))

    def test_selected_cloud_parser_keeps_mineru_page_metadata(self):
        with patch.object(
            document_loader.mineru_client,
            "parse_document",
            return_value=[(2, "第 3 页内容"), (5, "第 6 页内容")],
        ):
            documents = document_loader.load_document_routes(
                "/tmp/扫描件.pdf", "扫描件.pdf", PDF_PARSER_MINERU_CLOUD
            ).fixed

        self.assertEqual([document.metadata["page"] for document in documents], [2, 5])
        self.assertTrue(
            all(document.metadata["parser_mode"] == PDF_PARSER_MINERU_CLOUD for document in documents)
        )

    def test_cloud_client_uploads_with_ocr_and_reads_structured_pages(self):
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr(
                "result/job_content_list_v2.json",
                json.dumps(
                    [[{"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "云端结果"}]}}]],
                    ensure_ascii=False,
                ),
            )

        class FakeResponse:
            def __init__(self, payload=None, chunks=None):
                self.payload = payload
                self.chunks = chunks or []

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

            def iter_content(self, chunk_size):
                return iter(self.chunks)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeSession:
            def __init__(self):
                self.request_payload = None

            def post(self, url, headers, json, timeout):
                self.request_payload = json
                return FakeResponse({"code": 0, "data": {"batch_id": "batch", "file_urls": ["upload-url"]}})

            def put(self, url, data, timeout):
                return FakeResponse()

            def get(self, url, headers=None, stream=False, timeout=None):
                if stream:
                    return FakeResponse(chunks=[archive_buffer.getvalue()])
                return FakeResponse(
                    {
                        "code": 0,
                        "data": {"extract_result": [{"state": "done", "full_zip_url": "result-url"}]},
                    }
                )

            def close(self):
                return None

        session = FakeSession()
        with tempfile.TemporaryDirectory() as output_directory, tempfile.NamedTemporaryFile(
            suffix=".pdf"
        ) as source:
            result_dir = Path(output_directory) / "mineru-result"
            source.write(b"%PDF-1.4")
            source.flush()
            with patch.dict(os.environ, {"MINERU_API_TOKEN": "test-token"}, clear=False), patch.object(
                mineru_api_client.requests, "Session", return_value=session
            ):
                pages = mineru_client.parse_document(source.name, result_dir=result_dir)

            manifest = json.loads((result_dir / "pymodel_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue((result_dir / "result.zip").is_file())
            self.assertTrue(list(result_dir.rglob("*_content_list_v2.json")))
            self.assertEqual(manifest["parser"], "mineru_cloud")
            self.assertEqual(manifest["source_filename"], os.path.basename(source.name))

        self.assertTrue(session.request_payload["files"][0]["is_ocr"])
        self.assertEqual(pages, [(0, "云端结果")])

    def test_mineru_result_download_retries_after_a_transient_ssl_failure(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                return iter([b"completed archive"])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeSession:
            def __init__(self):
                self.calls = 0

            def get(self, url, stream, timeout):
                self.calls += 1
                if self.calls == 1:
                    raise mineru_api_client.requests.exceptions.SSLError("unexpected EOF")
                return FakeResponse()

        session = FakeSession()
        retry_session = FakeSession()
        retry_session.calls = 1
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MINERU_DOWNLOAD_ATTEMPTS": "3"}, clear=False
        ), patch.object(mineru_api_client.time, "sleep") as sleep, patch.object(
            mineru_api_client.requests, "Session", return_value=retry_session
        ) as new_session:
            archive_path = Path(directory) / "result.zip"
            mineru_api_client._download_result_archive(
                session, "result-url", archive_path
            )

            self.assertEqual(archive_path.read_bytes(), b"completed archive")

        self.assertEqual(session.calls, 1)
        self.assertEqual(retry_session.calls, 2)
        new_session.assert_called_once_with()
        sleep.assert_called_once_with(1)

    def test_saved_results_are_scoped_to_documents_and_can_be_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_dir = os.path.join(directory, "storage")
            environment = {
                "STORAGE_DIR": storage_dir,
                "MINERU_RESULTS_DIR": os.path.join(storage_dir, "mineru_results"),
            }
            with patch.dict(os.environ, environment, clear=False), patch.object(
                app_storage, "STORAGE_DIR", storage_dir
            ):
                source = os.path.join(storage_dir, "data", "研发部", "制度.pdf")
                result_dir = mineru_client.result_dir_for_document(source)
                result_dir.mkdir(parents=True)
                (result_dir / "result.zip").write_bytes(b"archive")

                self.assertTrue(mineru_client.remove_saved_result("研发部", "制度.pdf"))
                self.assertFalse(result_dir.exists())

    def test_result_directory_has_bundle_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            storage_dir = Path(directory) / "storage"
            results_dir = storage_dir / "mineru_results"
            source = storage_dir / "data" / "研发部" / "制度.pdf"

            with patch.dict(
                os.environ,
                {
                    "STORAGE_DIR": str(storage_dir),
                    "MINERU_RESULTS_DIR": str(results_dir),
                },
                clear=False,
            ), patch.object(app_storage, "STORAGE_DIR", str(storage_dir)):
                result_dir = mineru_client.result_dir_for_document(str(source))

            self.assertEqual(result_dir.name, "制度.pdf.mineru")

    def test_page_limit_rejects_oversized_pdf_before_cloud_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "大.pdf")
            Path(source).write_bytes(b"%PDF-1.4")
            reader = type("Reader", (), {"pages": [object()] * 201})
            with patch("pypdf.PdfReader", return_value=reader()), patch.object(
                mineru_api_client, "_parse_document"
            ) as upload, patch.object(mineru_api_client.requests, "Session") as session:
                with self.assertRaisesRegex(mineru_client.MinerUError, "201 页"):
                    mineru_api_client.parse_document(source)

            upload.assert_not_called()
            session.assert_not_called()

    def test_page_limit_allows_boundary_unreadable_and_non_pdf_files(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf = os.path.join(directory, "边界.pdf")
            Path(pdf).write_bytes(b"%PDF-1.4")
            reader = type("Reader", (), {"pages": [object()] * 200})
            with patch("pypdf.PdfReader", return_value=reader()):
                mineru_api_client.check_page_limit(pdf)
            # 损坏的 PDF 读不出页数，交给云端判断。
            mineru_api_client.check_page_limit(pdf)
            mineru_api_client.check_page_limit(os.path.join(directory, "说明.docx"))

    def test_rebuild_reuses_saved_mineru_pages_without_cloud_upload(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "STORAGE_DIR": os.path.join(directory, "storage"),
                "MINERU_RESULTS_DIR": os.path.join(directory, "mineru_results"),
            },
            clear=False,
        ), patch.object(
            app_storage, "STORAGE_DIR", os.path.join(directory, "storage")
        ):
            source = os.path.join(directory, "storage", "data", "研发部", "制度.pdf")
            os.makedirs(os.path.dirname(source), exist_ok=True)
            Path(source).write_bytes(b"%PDF-1.4")
            result_dir = mineru_client.result_dir_for_document(source)
            result_dir.mkdir(parents=True)
            content_name = "result_content_list_v2.json"
            (result_dir / content_name).write_text(
                json.dumps(
                    [[{"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "已保存页面文本"}]}}]],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (result_dir / "full.md").write_text("## 制度\n\n已保存 Markdown", encoding="utf-8")
            (result_dir / "pymodel_manifest.json").write_text(
                json.dumps({"content_list": content_name}, ensure_ascii=False), encoding="utf-8"
            )

            with patch.object(
                document_loader.mineru_client, "parse_document"
            ) as parse_document:
                pages = document_loader.load_document_routes(
                    source,
                    "制度.pdf",
                    PDF_PARSER_MINERU_CLOUD,
                    reuse_mineru_result=True,
                ).fixed

            parse_document.assert_not_called()
            self.assertEqual(pages[0].page_content, "已保存页面文本")
            structured = document_loader.load_mineru_markdown_document(
                source, "制度.pdf", PDF_PARSER_MINERU_CLOUD
            )
            self.assertEqual(structured[0].page_content, "## 制度\n\n已保存 Markdown")

    def test_selected_native_parser_is_used_without_starting_ocr(self):
        expected = [
            Document(
                page_content="原生文字",
                metadata={"source": "制度.pdf", "page": 0, "parser_mode": PDF_PARSER_NATIVE_TEXT},
            )
        ]
        with patch.object(document_loader, "_load_pdf_native_text", return_value=expected) as loader:
            documents = document_loader.load_document_routes(
                "/tmp/制度.pdf", "制度.pdf", PDF_PARSER_NATIVE_TEXT
            ).fixed

        loader.assert_called_once_with("/tmp/制度.pdf", "制度.pdf")
        self.assertEqual(documents[0].page_content, "原生文字")

    def test_paddle_ocr_renders_one_page_at_a_time(self):
        import sys
        import types

        calls = []

        def convert_from_path(path, **kwargs):
            calls.append(kwargs)
            return [f"image-{kwargs['first_page']}"]

        fake_pdf2image = types.SimpleNamespace(
            convert_from_path=convert_from_path,
            pdfinfo_from_path=lambda path: {"Pages": 3},
        )
        with (
            patch.dict(sys.modules, {"pdf2image": fake_pdf2image}),
            patch.object(document_loader, "get_ocr_engine", return_value=object()),
            patch.object(document_loader, "ocr_image", side_effect=lambda image, ocr: f"文本-{image}"),
        ):
            documents = document_loader._load_pdf_paddle_ocr("/tmp/扫描件.pdf", "扫描件.pdf")

        self.assertEqual(
            calls,
            [{"first_page": page, "last_page": page} for page in (1, 2, 3)],
        )
        self.assertEqual(
            [(doc.metadata["page"], doc.page_content) for doc in documents],
            [(0, "文本-image-1"), (1, "文本-image-2"), (2, "文本-image-3")],
        )


if __name__ == "__main__":
    unittest.main()
