import json
import os
import sqlite3
import threading
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi import UploadFile

import app_storage
from routers import documents, knowledge_bases, schemas
from services.core import config as application_config
from services.core.container import get_application_container
from services.core.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
)
from services.documents import use_cases as document_use_cases
from services.identity import auth as auth_service
from services.indexing import pipeline
from services.indexing import worker as index_worker
from services.indexing.jobs import INDEX_JOB_FAILED
from services.retrieval.chunk_index import get_chunk_index
from tests.support import IsolatedStorageTestCase, add_knowledge_bases, fixture_document_id


class DocumentMetadataApiTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        self.kb_id = add_knowledge_bases("研发部")[0]
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            document_path = app_storage.documents.stored_path(
                self.kb_id, fixture_document_id(self.kb_id, "制度.txt"), "制度.txt"
            )
            os.makedirs(os.path.dirname(document_path), exist_ok=True)
            with open(document_path, "w", encoding="utf-8") as output:
                output.write("制度原文")
            conn.execute(
                """
                INSERT INTO documents (document_id, kb_id, filename, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, 'ready')
                """,
                (
                    fixture_document_id(self.kb_id, "制度.txt"),
                    self.kb_id,
                    "制度.txt",
                    1,
                    application_config.db_now(),
                ),
            )
            image_path = app_storage.documents.stored_path(
                self.kb_id, fixture_document_id(self.kb_id, "审批流程.png"), "审批流程.png"
            )
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            with open(image_path, "wb") as output:
                output.write(b"png")
            conn.execute(
                """
                INSERT INTO documents (document_id, kb_id, filename, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, 'ready')
                """,
                (
                    fixture_document_id(self.kb_id, "审批流程.png"),
                    self.kb_id,
                    "审批流程.png",
                    3,
                    application_config.db_now(),
                ),
            )
            pdf_path = app_storage.documents.stored_path(
                self.kb_id, fixture_document_id(self.kb_id, "扫描件.pdf"), "扫描件.pdf"
            )
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(pdf_path, "wb") as output:
                output.write(b"%PDF-1.4")
            conn.execute(
                """
                INSERT INTO documents (document_id, kb_id, filename, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, 'ready')
                """,
                (
                    fixture_document_id(self.kb_id, "扫描件.pdf"),
                    self.kb_id,
                    "扫描件.pdf",
                    8,
                    application_config.db_now(),
                ),
            )

    def test_document_list_matches_the_response_model(self):
        records = app_storage.documents.list_documents(self.kb_id)

        self.assertTrue(records)
        for record in records:
            schemas.DocumentResponse.model_validate(record)

    def test_metadata_update_only_changes_application_database(self):
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.update_document_metadata(
                self.kb_id,
                fixture_document_id(self.kb_id, "制度.txt"),
                schemas.DocumentMetadataRequest(
                    document_type="制度",
                    tags=["研发"],
                    effective_date="2026-01-01",
                ),
            )

        self.assertEqual(result["metadata"]["document_type"], "制度")
        self.assertNotIn("job_id", result)
        enqueue.assert_not_called()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT document_type, tags, effective_date, status, index_job_id "
                "FROM documents WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "制度.txt"),
            ).fetchone()
            job_count = conn.execute(
                "SELECT COUNT(*) FROM index_jobs WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "制度.txt"),
            ).fetchone()[0]
        self.assertEqual(row[0], "制度")
        self.assertEqual(json.loads(row[1]), ["研发"])
        self.assertEqual(row[2], "2026-01-01")
        self.assertEqual(row[3], "ready")
        self.assertIsNone(row[4])
        self.assertEqual(job_count, 0)

    def test_upload_rejects_filename_too_long_for_the_filesystem(self):
        # 80 个汉字是 240 字节，字符数远低于 255，但临时文件名多出 41 字节后超过文件系统
        # 的 255 字节上限，过去会在写临时文件时抛 OSError 变成 500。
        too_long = "制" * 80 + ".txt"
        self.assertLessEqual(len(too_long), 255)
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            with self.assertRaises(InvalidRequestError) as error:
                documents.upload_document(
                    self.kb_id,
                    UploadFile(filename=too_long, file=BytesIO(b"x")),
                    None,
                    parser_mode=None,
                )
        enqueue.assert_not_called()
        self.assertIn("200", str(error.exception))

        accepted = "制" * 60 + ".txt"
        with patch.object(index_worker, "enqueue", return_value=True):
            result = documents.upload_document(
                self.kb_id,
                UploadFile(filename=accepted, file=BytesIO(b"x")),
                None,
                parser_mode=None,
            )
        self.assertEqual(result["status"], "success")

    def test_upload_stores_supplied_metadata_before_indexing(self):
        upload = UploadFile(filename="新制度.pdf", file=BytesIO(b"%PDF-1.4"))
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.upload_document(
                self.kb_id,
                upload,
                json.dumps(
                    {
                        "document_type": "制度",
                        "tags": ["研发", "新规"],
                        "effective_date": "2026-02-01",
                    },
                    ensure_ascii=False,
                ),
                parser_mode="paddle_ocr",
            )

        self.assertEqual(result["metadata"]["document_type"], "制度")
        enqueue.assert_called_once_with(result["job_id"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT document_type, tags, effective_date, status, parser_mode "
                "FROM documents WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "新制度.pdf"),
            ).fetchone()
        self.assertEqual(row[0], "制度")
        self.assertEqual(json.loads(row[1]), ["研发", "新规"])
        self.assertEqual(row[2], "2026-02-01")
        self.assertEqual(row[3], "processing")
        self.assertEqual(row[4], "paddle_ocr")

    def test_oversized_upload_removes_the_temporary_file(self):
        with patch.object(document_use_cases, "MAX_UPLOAD_BYTES", 4):
            with self.assertRaises(PayloadTooLargeError) as raised:
                document_use_cases.upload_document(
                    self.kb_id,
                    "过大.txt",
                    BytesIO(b"12345"),
                    metadata=None,
                    parser_mode=None,
                )

        self.assertEqual(raised.exception.code, "document_too_large")
        self.assertEqual(os.listdir(app_storage.documents.uploads_directory()), [])

    def test_new_upload_without_metadata_does_not_change_existing_same_name_document(self):
        app_storage.documents.update_metadata(
            self.kb_id,
            fixture_document_id(self.kb_id, "制度.txt"),
            {
                "document_type": "制度",
                "tags": ["旧标签"],
                "effective_date": "2026-01-01",
            },
        )

        with patch.object(index_worker, "enqueue", return_value=True):
            result = document_use_cases.upload_document(
                self.kb_id,
                "制度.txt",
                BytesIO("新版制度".encode()),
                metadata=None,
                parser_mode=None,
            )

        self.assertEqual(
            result.metadata,
            {"document_type": "", "tags": [], "effective_date": ""},
        )
        old = app_storage.documents.get_document(
            self.kb_id, fixture_document_id(self.kb_id, "制度.txt")
        )
        new = app_storage.documents.get_document(self.kb_id, result.document_id)
        self.assertEqual(old["tags"], ["旧标签"])
        self.assertEqual(old["document_type"], "制度")
        self.assertEqual(new["tags"], [])
        self.assertEqual(new["document_type"], "")
        self.assertNotEqual(old["document_id"], new["document_id"])

    def test_concurrent_same_name_uploads_publish_independent_versions(self):
        first_publishing = threading.Event()
        release_first = threading.Event()
        second_publishing = threading.Event()
        results = {}
        failures = []
        original_create = app_storage.documents.create_document_with_job

        def controlled_create(*args, **kwargs):
            document_type = kwargs["metadata"]["document_type"]
            if document_type == "第一版":
                first_publishing.set()
                release_first.wait(timeout=2)
            else:
                second_publishing.set()
            return original_create(*args, **kwargs)

        def upload(label: str, content: bytes):
            try:
                results[label] = document_use_cases.upload_document(
                    self.kb_id,
                    "并发制度.txt",
                    BytesIO(content),
                    metadata={"document_type": label},
                    parser_mode=None,
                )
            except Exception as exc:
                failures.append(exc)

        with (
            patch.object(
                app_storage.documents,
                "create_document_with_job",
                side_effect=controlled_create,
            ),
            patch.object(index_worker, "enqueue", return_value=True),
        ):
            first = threading.Thread(target=upload, args=("第一版", b"first"))
            second = threading.Thread(target=upload, args=("第二版", b"second"))
            first.start()
            self.assertTrue(first_publishing.wait(timeout=2))
            second.start()
            self.assertTrue(second_publishing.wait(timeout=2))
            release_first.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(second_publishing.is_set())
        self.assertNotEqual(results["第一版"].document_id, results["第二版"].document_id)
        for label, content in (("第一版", b"first"), ("第二版", b"second")):
            result = results[label]
            with open(
                app_storage.documents.document_path(self.kb_id, result.document_id), "rb"
            ) as source:
                self.assertEqual(source.read(), content)
            record = app_storage.documents.get_document(self.kb_id, result.document_id)
            self.assertEqual(record["document_type"], label)
            self.assertEqual(record["index_job_id"], result.job_id)

    def test_delete_failure_marks_the_durable_job_failed(self):
        document_id = fixture_document_id(self.kb_id, "制度.txt")
        with patch.object(index_worker, "enqueue", return_value=False):
            result = document_use_cases.delete_document(self.kb_id, document_id)
        with patch.object(
            pipeline.mineru_client,
            "remove_document_results",
            side_effect=RuntimeError("artifact unavailable"),
        ):
            pipeline.run_delete_job(result.job_id, self.kb_id, document_id, container=get_application_container())
        self.assertFalse(app_storage.documents.document_exists(self.kb_id, document_id))
        job = app_storage.index_jobs.get(result.job_id)
        self.assertEqual(job["status"], INDEX_JOB_FAILED)
        self.assertIn(
            "artifact unavailable",
            app_storage.index_jobs.list_recent(self.kb_id)[0]["error_message"],
        )

    def test_parser_update_schedules_a_document_reindex(self):
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.update_document_parser(
                self.kb_id,
                fixture_document_id(self.kb_id, "扫描件.pdf"),
                schemas.DocumentParserRequest(parser_mode="paddle_ocr"),
            )

        self.assertEqual(result["parser_mode"], "paddle_ocr")
        enqueue.assert_called_once_with(result["job_id"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT parser_mode, status, index_job_id FROM documents "
                "WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "扫描件.pdf"),
            ).fetchone()
        # 新解析方式发布成功后才写回文档。
        self.assertEqual(row[0], "native_text")
        self.assertEqual(app_storage.index_jobs.requested_parser_mode(row[2]), "paddle_ocr")
        # 已就绪文档重新索引期间旧版本仍可检索。
        self.assertEqual(row[1], "rebuilding")
        self.assertTrue(row[2])

    def test_image_parser_update_schedules_a_document_reindex(self):
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.update_document_parser(
                self.kb_id,
                fixture_document_id(self.kb_id, "审批流程.png"),
                schemas.DocumentParserRequest(parser_mode="vision_model"),
            )

        self.assertEqual(result["parser_mode"], "vision_model")
        enqueue.assert_called_once_with(result["job_id"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            parser_mode = conn.execute(
                "SELECT parser_mode FROM documents WHERE kb_id = ? AND filename = ?",
                (self.kb_id, "审批流程.png"),
            ).fetchone()[0]
        self.assertNotEqual(parser_mode, "vision_model")
        self.assertEqual(app_storage.index_jobs.requested_parser_mode(result["job_id"]), "vision_model")

    def test_new_knowledge_base_uses_parent_child_chunk_defaults(self):
        result = knowledge_bases.create_knowledge_base(
            schemas.KnowledgeBaseNameRequest(name="新知识库"),
            auth_service.Principal("admin", "超级管理员", all_knowledge_bases=True),
        )

        self.assertEqual(result["status"], "success")
        expected = application_config.default_chunk_config()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT chunk_size, chunk_overlap, kb_id FROM knowledge_bases WHERE name = ?",
                ("新知识库",),
            ).fetchone()
        self.assertEqual(row[0], expected.chunk_size)
        self.assertEqual(row[1], expected.chunk_overlap)
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            # 能读取全部知识库的创建者不需要授权行。
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM role_knowledge_bases WHERE kb_id = ?", (row[2],)
                ).fetchone()[0],
                0,
            )
        config = knowledge_bases.get_kb_chunk_config(row[2])
        self.assertEqual(config["chunk_size"], expected.chunk_size)
        self.assertEqual(config["chunk_overlap"], expected.chunk_overlap)

    def test_chunk_config_endpoint_reports_zero_overlap_as_saved(self):
        app_storage.knowledge_base_settings.set_chunk_config(self.kb_id, 600, 0)

        config = knowledge_bases.get_kb_chunk_config(self.kb_id)

        self.assertEqual(config["chunk_size"], 600)
        self.assertEqual(config["chunk_overlap"], 0)

    def test_chunk_config_marks_missing_index_for_rebuild(self):
        index = get_chunk_index()
        with patch.object(index, "available_routes", return_value=set()):
            legacy_config = knowledge_bases.get_kb_chunk_config(self.kb_id)
        with patch.object(index, "available_routes", return_value={"fixed"}):
            current_config = knowledge_bases.get_kb_chunk_config(self.kb_id)

        self.assertTrue(legacy_config["index_needs_rebuild"])
        self.assertFalse(current_config["index_needs_rebuild"])


if __name__ == "__main__":
    unittest.main()
