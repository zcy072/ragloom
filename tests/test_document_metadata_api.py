import json
import os
import sqlite3
import threading
import unittest
from io import BytesIO
from unittest.mock import patch

try:
    from fastapi import UploadFile

    import app_storage
    from routers import documents, knowledge_bases, schemas
    from services.core import config as application_config
    from services.core.errors import (
        InvalidRequestError,
        OperationFailedError,
        PayloadTooLargeError,
    )
    from services.documents import use_cases as document_use_cases
    from services.indexing import worker as index_worker
    from services.indexing.jobs import INDEX_JOB_FAILED
    from services.retrieval.chunk_index import get_chunk_index
except ModuleNotFoundError:
    UploadFile = None
    app_storage = None

from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(app_storage and UploadFile, "requires the backend Python dependencies")
class DocumentMetadataApiTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, kb_id, created_at) VALUES (?, ?, ?)",
                ("研发部", "kb-rd", application_config.db_now()),
            )
            document_path = app_storage.documents.document_path("研发部", "制度.txt")
            os.makedirs(os.path.dirname(document_path), exist_ok=True)
            with open(document_path, "w", encoding="utf-8") as output:
                output.write("制度原文")
            conn.execute(
                """
                INSERT INTO documents (kb_name, filename, path, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, 'ready')
                """,
                (
                    "研发部",
                    "制度.txt",
                    document_path,
                    1,
                    application_config.db_now(),
                ),
            )
            image_path = app_storage.documents.document_path("研发部", "审批流程.png")
            with open(image_path, "wb") as output:
                output.write(b"png")
            conn.execute(
                """
                INSERT INTO documents (kb_name, filename, path, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, 'ready')
                """,
                (
                    "研发部",
                    "审批流程.png",
                    image_path,
                    3,
                    application_config.db_now(),
                ),
            )
            pdf_path = app_storage.documents.document_path("研发部", "扫描件.pdf")
            with open(pdf_path, "wb") as output:
                output.write(b"%PDF-1.4")
            conn.execute(
                """
                INSERT INTO documents (kb_name, filename, path, size, uploaded_at, status)
                VALUES (?, ?, ?, ?, ?, 'ready')
                """,
                (
                    "研发部",
                    "扫描件.pdf",
                    pdf_path,
                    8,
                    application_config.db_now(),
                ),
            )

    def test_metadata_update_only_changes_application_database(self):
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.update_document_metadata(
                "研发部",
                "制度.txt",
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
                "FROM documents WHERE kb_name = ? AND filename = ?",
                ("研发部", "制度.txt"),
            ).fetchone()
            job_count = conn.execute(
                "SELECT COUNT(*) FROM index_jobs WHERE kb_name = ? AND filename = ?",
                ("研发部", "制度.txt"),
            ).fetchone()[0]
        self.assertEqual(row[0], "制度")
        self.assertEqual(json.loads(row[1]), ["研发"])
        self.assertEqual(row[2], "2026-01-01")
        self.assertEqual(row[3], "ready")
        self.assertEqual(row[4], "")
        self.assertEqual(job_count, 0)

    def test_upload_rejects_filename_too_long_for_the_filesystem(self):
        # 80 个汉字是 240 字节，字符数远低于 255，但临时文件名多出 41 字节后超过文件系统
        # 的 255 字节上限，过去会在写临时文件时抛 OSError 变成 500。
        too_long = "制" * 80 + ".txt"
        self.assertLessEqual(len(too_long), 255)
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            with self.assertRaises(InvalidRequestError) as error:
                documents.upload_document(
                    "研发部",
                    UploadFile(filename=too_long, file=BytesIO(b"x")),
                    None,
                    parser_mode=None,
                )
        enqueue.assert_not_called()
        self.assertIn("200", str(error.exception))

        accepted = "制" * 60 + ".txt"
        with patch.object(index_worker, "enqueue", return_value=True):
            result = documents.upload_document(
                "研发部",
                UploadFile(filename=accepted, file=BytesIO(b"x")),
                None,
                parser_mode=None,
            )
        self.assertEqual(result["status"], "success")

    def test_upload_stores_supplied_metadata_before_indexing(self):
        upload = UploadFile(filename="新制度.pdf", file=BytesIO(b"%PDF-1.4"))
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.upload_document(
                "研发部",
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
                "FROM documents WHERE kb_name = ? AND filename = ?",
                ("研发部", "新制度.pdf"),
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
                    "研发部",
                    "过大.txt",
                    BytesIO(b"12345"),
                    metadata=None,
                    parser_mode=None,
                )

        self.assertEqual(raised.exception.code, "document_too_large")
        target = app_storage.documents.document_path("研发部", "过大.txt")
        self.assertFalse(os.path.exists(target))
        leftovers = [
            name
            for name in os.listdir(os.path.dirname(target))
            if name.startswith(".过大.txt.") and name.endswith(".upload")
        ]
        self.assertEqual(leftovers, [])

    def test_upload_without_metadata_clears_previous_filter_values(self):
        app_storage.documents.update_metadata(
            "研发部",
            "制度.txt",
            {
                "document_type": "制度",
                "tags": ["旧标签"],
                "effective_date": "2026-01-01",
            },
        )

        with patch.object(index_worker, "enqueue", return_value=True):
            result = document_use_cases.upload_document(
                "研发部",
                "制度.txt",
                BytesIO("新版制度".encode()),
                metadata=None,
                parser_mode=None,
            )

        self.assertEqual(
            result.metadata,
            {"document_type": "", "tags": [], "effective_date": ""},
        )
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT document_type, tags, effective_date FROM documents "
                "WHERE kb_name = ? AND filename = ?",
                ("研发部", "制度.txt"),
            ).fetchone()
        self.assertEqual(row, ("", "[]", ""))

    def test_concurrent_same_name_uploads_publish_one_coherent_version(self):
        first_publishing = threading.Event()
        release_first = threading.Event()
        second_publishing = threading.Event()
        results = {}
        failures = []
        original_upsert = app_storage.documents.upsert_uploaded_document

        def controlled_upsert(*args, **kwargs):
            document_type = kwargs["metadata"]["document_type"]
            if document_type == "第一版":
                first_publishing.set()
                release_first.wait(timeout=2)
            else:
                second_publishing.set()
            return original_upsert(*args, **kwargs)

        def upload(label: str, content: bytes):
            try:
                results[label] = document_use_cases.upload_document(
                    "研发部",
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
                "upsert_uploaded_document",
                side_effect=controlled_upsert,
            ),
            patch.object(index_worker, "enqueue", return_value=True),
        ):
            first = threading.Thread(target=upload, args=("第一版", b"first"))
            second = threading.Thread(target=upload, args=("第二版", b"second"))
            first.start()
            self.assertTrue(first_publishing.wait(timeout=2))
            second.start()
            self.assertFalse(second_publishing.wait(timeout=0.05))
            release_first.set()
            first.join(timeout=2)
            second.join(timeout=2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(failures, [])
        self.assertTrue(second_publishing.is_set())
        path = app_storage.documents.document_path("研发部", "并发制度.txt")
        with open(path, "rb") as source:
            self.assertEqual(source.read(), b"second")
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT document_type, size, index_job_id FROM documents "
                "WHERE kb_name = ? AND filename = ?",
                ("研发部", "并发制度.txt"),
            ).fetchone()
        self.assertEqual(row, ("第二版", 6, results["第二版"].job_id))

    def test_delete_failure_marks_the_durable_job_failed(self):
        with patch.object(
            document_use_cases.mineru_client,
            "remove_saved_result",
            side_effect=RuntimeError("artifact unavailable"),
        ):
            with self.assertRaises(OperationFailedError) as raised:
                document_use_cases.delete_document("研发部", "制度.txt")

        self.assertEqual(raised.exception.code, "document_delete_failed")
        self.assertTrue(app_storage.documents.document_exists("研发部", "制度.txt"))
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            job = conn.execute(
                "SELECT status, error_message FROM index_jobs "
                "WHERE kb_name = ? AND filename = ? AND operation = 'delete'",
                ("研发部", "制度.txt"),
            ).fetchone()
        self.assertEqual(job[0], INDEX_JOB_FAILED)
        self.assertIn("artifact unavailable", job[1])

    def test_parser_update_schedules_a_document_reindex(self):
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.update_document_parser(
                "研发部",
                "扫描件.pdf",
                schemas.DocumentParserRequest(parser_mode="paddle_ocr"),
            )

        self.assertEqual(result["parser_mode"], "paddle_ocr")
        enqueue.assert_called_once_with(result["job_id"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT parser_mode, status, index_job_id FROM documents "
                "WHERE kb_name = ? AND filename = ?",
                ("研发部", "扫描件.pdf"),
            ).fetchone()
        self.assertEqual(row[0], "paddle_ocr")
        self.assertEqual(row[1], "processing")
        self.assertTrue(row[2])

    def test_image_parser_update_schedules_a_document_reindex(self):
        with patch.object(index_worker, "enqueue", return_value=True) as enqueue:
            result = documents.update_document_parser(
                "研发部",
                "审批流程.png",
                schemas.DocumentParserRequest(parser_mode="vision_model"),
            )

        self.assertEqual(result["parser_mode"], "vision_model")
        enqueue.assert_called_once_with(result["job_id"])
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            parser_mode = conn.execute(
                "SELECT parser_mode FROM documents WHERE kb_name = ? AND filename = ?",
                ("研发部", "审批流程.png"),
            ).fetchone()[0]
        self.assertEqual(parser_mode, "vision_model")

    def test_new_knowledge_base_uses_parent_child_chunk_defaults(self):
        result = knowledge_bases.create_knowledge_base(
            schemas.KnowledgeBaseNameRequest(name="新知识库")
        )

        self.assertEqual(result["status"], "success")
        expected = application_config.default_chunk_config()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            row = conn.execute(
                "SELECT chunk_size, chunk_overlap FROM knowledge_bases WHERE name = ?",
                ("新知识库",),
            ).fetchone()
        self.assertEqual(row[0], expected.chunk_size)
        self.assertEqual(row[1], expected.chunk_overlap)
        config = knowledge_bases.get_kb_chunk_config("新知识库")
        self.assertEqual(config["chunk_size"], expected.chunk_size)
        self.assertEqual(config["chunk_overlap"], expected.chunk_overlap)

    def test_chunk_config_endpoint_reports_zero_overlap_as_saved(self):
        app_storage.knowledge_base_settings.set_chunk_config("研发部", 600, 0)

        config = knowledge_bases.get_kb_chunk_config("研发部")

        self.assertEqual(config["chunk_size"], 600)
        self.assertEqual(config["chunk_overlap"], 0)

    def test_chunk_config_marks_missing_index_for_rebuild(self):
        index = get_chunk_index()
        with patch.object(index, "available_routes", return_value=set()):
            legacy_config = knowledge_bases.get_kb_chunk_config("研发部")
        with patch.object(index, "available_routes", return_value={"fixed"}):
            current_config = knowledge_bases.get_kb_chunk_config("研发部")

        self.assertTrue(legacy_config["index_needs_rebuild"])
        self.assertFalse(current_config["index_needs_rebuild"])


if __name__ == "__main__":
    unittest.main()
