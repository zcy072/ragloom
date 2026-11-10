"""Knowledge-base document metadata and filesystem persistence."""

import datetime
import json
import os
import sqlite3
import uuid
from collections.abc import Callable
from typing import cast

from services.core import sqlite_connections as application_db
from services.core.config import validate_document_id
from services.core.contracts import NormalizedRetrievalFilters, RetrievalFilters
from services.core.errors import ResourceNotFoundError
from services.documents.contracts import (
    DocumentFilterOptions,
    DocumentMetadata,
    DocumentMetadataInput,
    DocumentRecord,
)
from services.documents.parser_modes import (
    DEFAULT_PDF_PARSER_MODE,
    normalize_document_parser_mode,
)
from services.indexing.jobs import insert_job


class DocumentStore:
    """Persist document lifecycle, metadata, filters, and source allowlists."""

    def __init__(
        self,
        db_file: Callable[[], str],
        data_dir: Callable[[], str],
        now: Callable[[], str],
        validate_kb_id: Callable[[str], str],
        validate_filename: Callable[[str], str],
        *,
        max_type_length: int,
        max_tags: int,
        max_tag_length: int,
    ) -> None:
        """保存文档元数据存储所需的路径、校验函数和长度限制配置。"""
        self._db_file = db_file
        self._data_dir = data_dir
        self._now = now
        self._validate_kb_id = validate_kb_id
        self._validate_filename = validate_filename
        self._max_type_length = max_type_length
        self._max_tags = max_tags
        self._max_tag_length = max_tag_length

    def document_exists(self, kb_id: str, document_id: str) -> bool:
        with application_db.connect(self._db_file()) as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM documents WHERE kb_id = ? AND document_id = ?",
                    (self._validate_kb_id(kb_id), validate_document_id(document_id)),
                ).fetchone()
                is not None
            )

    @staticmethod
    def _require_active(conn: sqlite3.Connection, kb_id: str) -> None:
        if (
            conn.execute(
                "SELECT 1 FROM knowledge_bases WHERE kb_id = ? AND state = 'active'", (kb_id,)
            ).fetchone()
            is None
        ):
            raise ResourceNotFoundError("知识库不存在", code="knowledge_base_not_found")

    def get_document(self, kb_id: str, document_id: str) -> DocumentRecord:
        with application_db.connect(self._db_file()) as conn:
            self._require_active(conn, self._validate_kb_id(kb_id))
            return self._get_document(conn, kb_id, validate_document_id(document_id))

    def _get_document(
        self, conn: sqlite3.Connection, kb_id: str, document_id: str
    ) -> DocumentRecord:
        row = conn.execute(
            "SELECT filename, size, uploaded_at, status, error_message, index_job_id, "
            "document_type, tags, effective_date, parser_mode, document_id "
            "FROM documents WHERE kb_id = ? AND document_id = ?",
            (kb_id, document_id),
        ).fetchone()
        if row is None:
            raise ResourceNotFoundError("文档不存在", code="document_not_found")
        return self._document_from_row(tuple(row))

    def create_document_with_job(
        self,
        kb_id: str,
        document_id: str,
        filename: str,
        *,
        size: int,
        parser_mode: str,
        metadata: DocumentMetadataInput,
    ) -> str:
        """初始文档与任务共用一个短事务；调用方已发布独立原文件并持文档锁。"""
        kb_id, document_id = self._validate_kb_id(kb_id), validate_document_id(document_id)
        filename = self._validate_filename(filename)
        normalized = self.normalize_metadata(metadata)
        job_id = uuid.uuid4().hex
        with application_db.connect(self._db_file()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_active(conn, kb_id)
            conn.execute(
                "INSERT INTO documents (document_id, kb_id, filename, size, uploaded_at, "
                "status, index_job_id, document_type, tags, effective_date, parser_mode) "
                "VALUES (?, ?, ?, ?, ?, 'processing', ?, ?, ?, ?, ?)",
                (
                    document_id,
                    kb_id,
                    filename,
                    size,
                    self._now(),
                    job_id,
                    normalized["document_type"],
                    json.dumps(normalized["tags"], ensure_ascii=False),
                    normalized["effective_date"],
                    parser_mode,
                ),
            )
            insert_job(
                conn, job_id, kb_id, document_id, filename, "process", self._now(),
                parser_mode=parser_mode,
            )
        return job_id

    def schedule_reindex(
        self, kb_id: str, document_id: str, *, parser_mode: str | None = None
    ) -> str:
        kb_id, document_id = self._validate_kb_id(kb_id), validate_document_id(document_id)
        job_id = uuid.uuid4().hex
        with application_db.connect(self._db_file()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_active(conn, kb_id)
            record = self._get_document(conn, kb_id, document_id)
            mode = normalize_document_parser_mode(
                record["filename"], parser_mode or record["parser_mode"]
            )
            # 新解析方式只登记在任务上，发布成功后才写回文档。
            insert_job(
                conn, job_id, kb_id, document_id, record["filename"], "process", self._now(),
                parser_mode=mode,
            )
            # 已就绪文档标为 rebuilding：新版本发布前旧索引继续参与检索，与整库重建一致。
            # 调用方持有文档锁，rebuilding 文档不会正在发布，旧索引仍完好，保持 rebuilding；
            # 遗留的 publishing 文档旧索引不可信，标为 processing。
            conn.execute(
                "UPDATE documents SET status = CASE WHEN status IN "
                "('ready', 'rebuilding') THEN 'rebuilding' ELSE 'processing' END, "
                "error_message = '', index_job_id = ? WHERE kb_id = ? AND document_id = ?",
                (job_id, kb_id, document_id),
            )
        return job_id

    def schedule_document_delete(self, kb_id: str, document_id: str) -> str:
        kb_id, document_id = self._validate_kb_id(kb_id), validate_document_id(document_id)
        job_id = uuid.uuid4().hex
        with application_db.connect(self._db_file()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_active(conn, kb_id)
            record = self._get_document(conn, kb_id, document_id)
            insert_job(conn, job_id, kb_id, document_id, record["filename"], "delete", self._now())
            conn.execute(
                "DELETE FROM documents WHERE kb_id = ? AND document_id = ?", (kb_id, document_id)
            )
        return job_id

    def knowledge_base_exists(self, kb_id: str) -> bool:
        """校验 kb_id 后，检查该知识库是否存在且处于 active 状态。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge_bases WHERE kb_id = ? AND state = 'active'",
                (kb_id,),
            ).fetchone()
        return row is not None

    def ready_document_ids(self, kb_id: str) -> list[str]:
        """列出允许参与检索的文件：已就绪的，以及重建或发布中仍保留旧版本索引的。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT document_id FROM documents WHERE kb_id = ? "
                "AND status IN ('ready', 'rebuilding', 'publishing') ORDER BY document_id",
                (kb_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def has_documents(self, kb_id: str | None = None) -> bool:
        """判断全局或指定知识库中是否至少存在一份文档。"""
        if kb_id is not None:
            kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            if kb_id is None:
                row = conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM documents WHERE kb_id = ? LIMIT 1", (kb_id,)
                ).fetchone()
        return row is not None

    @staticmethod
    def _normalize_text(value, label: str, max_length: int) -> str:
        """校验用户输入的元数据文本，并合并多余空白字符。"""
        if value in (None, ""):
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{label}必须是文本")
        text = " ".join(value.split())
        if len(text) > max_length:
            raise ValueError(f"{label}不能超过 {max_length} 个字符")
        return text

    def _normalize_tags(self, value) -> list[str]:
        """接受逗号分隔文本或列表，返回数量受限且去重后的规范标签。"""
        if value in (None, ""):
            return []
        if isinstance(value, str):
            values = value.replace("，", ",").split(",")
        elif isinstance(value, list):
            values = value
        else:
            raise ValueError("标签必须是文本或列表")
        tags: list[str] = []
        for raw_tag in values:
            if not isinstance(raw_tag, str):
                raise ValueError("标签必须是文本")
            tag = " ".join(raw_tag.split())
            if not tag:
                continue
            if len(tag) > self._max_tag_length:
                raise ValueError(f"单个标签不能超过 {self._max_tag_length} 个字符")
            if tag not in tags:
                tags.append(tag)
        if len(tags) > self._max_tags:
            raise ValueError(f"最多可填写 {self._max_tags} 个标签")
        return tags

    def normalize_metadata(self, metadata: DocumentMetadataInput | None) -> DocumentMetadata:
        """校验可编辑元数据，并整理为文档表固定使用的字段结构。"""
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("文档元数据必须是对象")
        effective_date = self._normalize_text(metadata.get("effective_date"), "生效日期", 10)
        if effective_date:
            try:
                datetime.date.fromisoformat(effective_date)
            except ValueError as exc:
                raise ValueError("生效日期必须使用 YYYY-MM-DD 格式") from exc
        return {
            "document_type": self._normalize_text(
                metadata.get("document_type"), "文档类型", self._max_type_length
            ),
            "tags": self._normalize_tags(metadata.get("tags")),
            "effective_date": effective_date,
        }

    @staticmethod
    def _stored_tags(value) -> list[str]:
        """解析数据库中保存的 JSON 标签，异常值视为没有标签。"""
        try:
            decoded = json.loads(value) if value else []
        except (TypeError, json.JSONDecodeError):
            return []
        return [tag for tag in decoded if isinstance(tag, str) and tag]

    def update_metadata(
        self, kb_id: str, document_id: str, metadata: DocumentMetadataInput
    ) -> DocumentMetadata:
        """校验并保存文档元数据，同时返回规范化后的字段。"""
        kb_id = self._validate_kb_id(kb_id)
        document_id = validate_document_id(document_id)
        normalized = self.normalize_metadata(metadata)
        with application_db.connect(self._db_file()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_active(conn, kb_id)
            cursor = conn.execute(
                """
                UPDATE documents SET document_type = ?, tags = ?, effective_date = ?
                WHERE kb_id = ? AND document_id = ?
                """,
                (
                    normalized["document_type"],
                    json.dumps(normalized["tags"], ensure_ascii=False),
                    normalized["effective_date"],
                    kb_id,
                    document_id,
                ),
            )
        if cursor.rowcount != 1:
            raise ValueError("文档不存在")
        return normalized

    @staticmethod
    def _normalize_filter_values(value, label: str, max_length: int) -> tuple[str, ...]:
        """规范化一个单值或列表筛选字段，并去除重复选择。"""
        if value in (None, ""):
            return ()
        values = value if isinstance(value, list) else [value]
        normalized: list[str] = []
        for raw_value in values:
            text = DocumentStore._normalize_text(raw_value, label, max_length)
            if text and text not in normalized:
                normalized.append(text)
        if len(normalized) > 20:
            raise ValueError(f"{label}最多可选择 20 项")
        return tuple(normalized)

    def normalize_filters(self, filters: RetrievalFilters | None) -> NormalizedRetrievalFilters:
        """校验文档筛选条件，并转换为可安全参与查询构建的规范值。"""
        if not filters:
            return {
                "document_ids": (),
                "document_types": (),
                "tags": (),
                "effective_date_from": "",
                "effective_date_to": "",
            }
        if not isinstance(filters, dict):
            raise ValueError("检索筛选必须是对象")
        if "source" in filters or "sources" in filters:
            raise ValueError("来源文件筛选请使用 document_ids")

        date_from = self._normalize_text(filters.get("effective_date_from"), "生效开始日期", 10)
        date_to = self._normalize_text(filters.get("effective_date_to"), "生效结束日期", 10)
        for label, value in (("生效开始日期", date_from), ("生效结束日期", date_to)):
            if value:
                try:
                    datetime.date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(f"{label}必须使用 YYYY-MM-DD 格式") from exc
        if date_from and date_to and date_from > date_to:
            raise ValueError("生效开始日期不能晚于结束日期")
        nfv = self._normalize_filter_values
        return {
            "document_ids": tuple(
                validate_document_id(value)
                for value in nfv(filters.get("document_ids"), "文档 ID", 36)
            ),
            "document_types": nfv(
                filters.get("document_types", filters.get("document_type")),
                "文档类型",
                self._max_type_length,
            ),
            "tags": nfv(filters.get("tags"), "标签", self._max_tag_length),
            "effective_date_from": date_from,
            "effective_date_to": date_to,
        }

    def has_filters(self, filters: RetrievalFilters | None) -> bool:
        """判断请求是否包含至少一个实际生效的文档筛选条件。"""
        return any(self.normalize_filters(filters).values())

    def filtered_document_ids(
        self, kb_id: str, filters: RetrievalFilters | None
    ) -> list[str] | None:
        """在进入两类检索后端前，将界面元数据筛选条件解析为文档 ID 列表。

        返回 ``None`` 表示未启用筛选；返回空列表则表示筛选已启用但没有任何匹配文档。
        """
        kb_id = self._validate_kb_id(kb_id)
        normalized = self.normalize_filters(filters)
        if not any(normalized.values()):
            return None
        clauses = ["kb_id = ?"]
        params: list[str] = [kb_id]
        for key, column in (("document_ids", "document_id"), ("document_types", "document_type")):
            if normalized[key]:
                placeholders = ", ".join("?" for _ in normalized[key])
                clauses.append(f"{column} IN ({placeholders})")
                params.extend(normalized[key])
        if normalized["effective_date_from"]:
            clauses.append("effective_date >= ?")
            params.append(normalized["effective_date_from"])
        if normalized["effective_date_to"]:
            clauses.append("effective_date <= ?")
            params.append(normalized["effective_date_to"])
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                f"SELECT document_id, tags FROM documents WHERE {' AND '.join(clauses)} ORDER BY document_id",
                params,
            ).fetchall()
        # Tags live as JSON because they are edited as a small list.  Filter them after
        # the SQL-friendly fields, keeping the query parameterized and easy to inspect.
        selected_tags = set(normalized["tags"])
        return [
            document_id
            for document_id, tags in rows
            if not selected_tags or selected_tags.intersection(self._stored_tags(tags))
        ]

    def filter_options(self, kb_id: str) -> DocumentFilterOptions:
        """返回筛选界面可选的文档（ID 与显示文件名）、类型和标签去重列表。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            if not conn.execute(
                "SELECT 1 FROM knowledge_bases WHERE kb_id = ? AND state = 'active'",
                (kb_id,),
            ).fetchone():
                raise ValueError("知识库不存在")
            rows = conn.execute(
                "SELECT document_id, filename, document_type, tags FROM documents "
                "WHERE kb_id = ? ORDER BY filename",
                (kb_id,),
            ).fetchall()
        tags = sorted({tag for _, _, _, stored in rows for tag in self._stored_tags(stored)})
        return {
            "documents": [{"document_id": row[0], "filename": row[1]} for row in rows],
            "document_types": sorted({row[2] for row in rows if row[2]}),
            "tags": tags,
        }

    def knowledge_base_path(self, kb_id: str) -> str:
        """将校验后的 kb_id 映射到其原始文件存储目录；路径不含显示名称。"""
        return os.path.join(self._data_dir(), self._validate_kb_id(kb_id))

    def document_directory(self, kb_id: str, document_id: str) -> str:
        return os.path.join(self.knowledge_base_path(kb_id), validate_document_id(document_id))

    def stored_path(self, kb_id: str, document_id: str, filename: str) -> str:
        return os.path.join(
            self.document_directory(kb_id, document_id), self._validate_filename(filename)
        )

    def uploads_directory(self) -> str:
        return os.path.join(self._data_dir(), ".uploads")

    @classmethod
    def _document_from_row(cls, row: tuple[object, ...]) -> DocumentRecord:
        """Convert the document-list query projection into its typed API record."""
        return {
            "document_id": str(row[10]),
            "filename": str(row[0]),
            "size": cast(int, row[1]),
            "uploaded_at": str(row[2]),
            "status": str(row[3]),
            "error_message": str(row[4] or ""),
            "index_job_id": str(row[5] or ""),
            "document_type": str(row[6] or ""),
            "tags": cls._stored_tags(row[7]),
            "effective_date": str(row[8] or ""),
            "parser_mode": str(row[9] or DEFAULT_PDF_PARSER_MODE),
        }

    def list_documents(self, kb_id: str) -> list[DocumentRecord]:
        """返回文档管理页面所需的记录、元数据和索引状态；只读，不创建目录。"""
        kb_id = self._validate_kb_id(kb_id)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                """
                SELECT filename, size, uploaded_at, status, error_message, index_job_id,
                       document_type, tags, effective_date, parser_mode, document_id
                FROM documents WHERE kb_id = ? ORDER BY uploaded_at DESC, filename
                """,
                (kb_id,),
            ).fetchall()
        return [self._document_from_row(tuple(row)) for row in rows]

    def document_path(self, kb_id: str, document_id: str) -> str:
        record = self.get_document(kb_id, document_id)
        return self.stored_path(kb_id, document_id, record["filename"])
