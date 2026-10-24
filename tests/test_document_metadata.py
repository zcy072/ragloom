import sqlite3
import unittest

try:
    import app_storage
    from services.core import config as application_config
except ModuleNotFoundError:
    app_storage = None

from tests.support import IsolatedStorageTestCase


@unittest.skipUnless(app_storage, "requires the SQLite application module")
class DocumentMetadataTest(IsolatedStorageTestCase):
    def setUp(self):
        super().setUp()
        app_storage.initialize_database()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (name, created_at) VALUES (?, ?)",
                ("研发部", application_config.db_now()),
            )
            for filename in ("考勤制度.pdf", "费用报告.pdf"):
                conn.execute(
                    """
                    INSERT INTO documents (kb_name, filename, path, size, uploaded_at, status)
                    VALUES (?, ?, ?, ?, ?, 'ready')
                    """,
                    (
                        "研发部",
                        filename,
                        f"/tmp/{filename}",
                        1,
                        application_config.db_now(),
                    ),
                )

    def test_metadata_is_returned_and_filters_source_files(self):
        app_storage.documents.update_metadata(
            "研发部",
            "考勤制度.pdf",
            {
                "document_type": "制度",
                "tags": ["人事", "2026"],
                "effective_date": "2026-01-01",
            },
        )
        app_storage.documents.update_metadata(
            "研发部",
            "费用报告.pdf",
            {
                "document_type": "报告",
                "tags": ["财务"],
                "effective_date": "2025-12-01",
            },
        )

        documents = app_storage.documents.list_documents("研发部")
        policy = next(document for document in documents if document["filename"] == "考勤制度.pdf")
        self.assertEqual(policy["tags"], ["人事", "2026"])

        sources = app_storage.documents.filtered_sources(
            "研发部",
            {
                "document_type": "制度",
                "tags": ["人事"],
                "effective_date_from": "2026-01-01",
            },
        )
        self.assertEqual(sources, ["考勤制度.pdf"])
        self.assertEqual(app_storage.documents.filtered_sources("研发部", None), None)

        options = app_storage.documents.filter_options("研发部")
        self.assertCountEqual(options["document_types"], ["报告", "制度"])
        self.assertCountEqual(options["tags"], ["2026", "人事", "财务"])

    def test_query_rewrite_setting_defaults_to_enabled_and_can_be_changed(self):
        # per-KB 开关归 KnowledgeBaseSettings，不在 DocumentStore 上。
        self.assertTrue(app_storage.knowledge_base_settings.query_rewrite_enabled("研发部"))
        self.assertFalse(app_storage.knowledge_base_settings.set_query_rewrite_enabled("研发部", False))
        self.assertFalse(app_storage.knowledge_base_settings.query_rewrite_enabled("研发部"))

    def test_missing_chunk_values_are_mapped_to_named_defaults(self):
        defaults = application_config.default_chunk_config()
        with sqlite3.connect(app_storage.DB_FILE) as conn:
            conn.execute(
                "UPDATE knowledge_bases SET chunk_size = NULL, chunk_overlap = NULL "
                "WHERE name = ?",
                ("研发部",),
            )

        config = app_storage.knowledge_base_settings.chunk_config(
            "研发部",
            default_size=defaults.chunk_size,
            default_overlap=defaults.chunk_overlap,
        )

        self.assertEqual(config, defaults)
        self.assertEqual(config.chunk_size, application_config.DEFAULT_CHUNK_SIZE)

    def test_zero_chunk_overlap_is_a_real_setting_not_a_missing_value(self):
        defaults = application_config.default_chunk_config()
        app_storage.knowledge_base_settings.set_chunk_config("研发部", 600, 0)

        config = app_storage.knowledge_base_settings.chunk_config(
            "研发部",
            default_size=defaults.chunk_size,
            default_overlap=defaults.chunk_overlap,
        )

        self.assertEqual((config.chunk_size, config.chunk_overlap), (600, 0))

    def test_invalid_effective_date_is_rejected(self):
        with self.assertRaises(ValueError):
            app_storage.documents.normalize_metadata({"effective_date": "2026/01/01"})

    def test_metadata_normalization_has_a_fixed_output(self):
        metadata = app_storage.documents.normalize_metadata(
            {
                "document_type": " 制度 ",
                "tags": "人事， 2026,人事",
            }
        )

        self.assertEqual(
            metadata,
            {
                "document_type": "制度",
                "tags": ["人事", "2026"],
                "effective_date": "",
            },
        )

    def test_pdf_parser_mode_is_persisted_and_returned_with_document(self):
        self.assertEqual(
            app_storage.documents.parser_modes("研发部")["考勤制度.pdf"], "native_text"
        )
        mode = app_storage.documents.update_parser_mode(
            "研发部", "考勤制度.pdf", "paddle_ocr"
        )

        documents = app_storage.documents.list_documents("研发部")
        document = next(item for item in documents if item["filename"] == "考勤制度.pdf")

        self.assertEqual(mode, "paddle_ocr")
        self.assertEqual(document["parser_mode"], "paddle_ocr")

    def test_document_list_rows_have_fixed_fields_and_safe_defaults(self):
        document = next(
            item
            for item in app_storage.documents.list_documents("研发部")
            if item["filename"] == "考勤制度.pdf"
        )

        self.assertEqual(
            set(document),
            {
                "filename",
                "path",
                "size",
                "uploaded_at",
                "status",
                "error_message",
                "index_job_id",
                "document_type",
                "tags",
                "effective_date",
                "parser_mode",
            },
        )
        self.assertEqual(document["size"], 1)
        self.assertEqual(document["status"], "ready")
        self.assertEqual(document["error_message"], "")
        self.assertEqual(document["index_job_id"], "")
        self.assertEqual(document["tags"], [])
        self.assertEqual(document["parser_mode"], "native_text")


if __name__ == "__main__":
    unittest.main()
