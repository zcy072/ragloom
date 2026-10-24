"""Knowledge-base document metadata and filesystem persistence."""

import datetime
import json
import os
from collections.abc import Callable
from typing import cast

from services.core import sqlite_connections as application_db
from services.core.contracts import NormalizedRetrievalFilters, RetrievalFilters
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


class DocumentStore:
    """Persist document lifecycle, metadata, filters, and source allowlists."""
    def __init__(
        self,
        db_file: Callable[[], str],
        data_dir: Callable[[], str],
        now: Callable[[], str],
        validate_kb_name: Callable[[str], str],
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
        self._validate_kb_name = validate_kb_name
        self._validate_filename = validate_filename
        self._max_type_length = max_type_length
        self._max_tags = max_tags
        self._max_tag_length = max_tag_length

    def document_exists(self, kb_name: str, filename: str) -> bool:
        """判断指定知识库中是否存在该文件的元数据记录。"""
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE kb_name = ? AND filename = ?", (kb_name, filename)
            ).fetchone()
        return row is not None

    def upsert_uploaded_document(
        self,
        kb_name: str,
        filename: str,
        *,
        path: str,
        size: int,
        job_id: str,
        parser_mode: str,
        metadata: DocumentMetadataInput,
    ) -> None:
        """保存新上传文档及其规范化后的筛选元数据。"""
        kb_name = self._validate_kb_name(kb_name)
        filename = self._validate_filename(filename)
        normalized = self.normalize_metadata(metadata)
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (kb_name, filename, path, size, uploaded_at, status, index_job_id,
                     document_type, tags, effective_date, parser_mode)
                VALUES (?, ?, ?, ?, ?, 'processing', ?, ?, ?, ?, ?)
                ON CONFLICT(kb_name, filename) DO UPDATE SET
                    path = excluded.path, size = excluded.size,
                    uploaded_at = excluded.uploaded_at, status = 'processing',
                    error_message = '', index_job_id = excluded.index_job_id,
                    document_type = excluded.document_type,
                    tags = excluded.tags, effective_date = excluded.effective_date,
                    parser_mode = excluded.parser_mode
                """,
                (
                    kb_name,
                    filename,
                    path,
                    size,
                    self._now(),
                    job_id,
                    normalized["document_type"],
                    json.dumps(normalized["tags"], ensure_ascii=False),
                    normalized["effective_date"],
                    parser_mode,
                ),
            )

    def knowledge_base_exists(self, kb_name: str) -> bool:
        """校验知识库名称后，检查其是否已在数据库中创建。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            row = conn.execute("SELECT 1 FROM knowledge_bases WHERE name = ?", (kb_name,)).fetchone()
        return row is not None

    def ready_sources(self, kb_name: str) -> list[str]:
        """列出最新索引任务已成功完成、允许参与检索的文件。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT filename FROM documents "
                "WHERE kb_name = ? AND status = 'ready' ORDER BY filename",
                (kb_name,),
            ).fetchall()
        return [row[0] for row in rows]

    def disable_chroma_source(self, kb_name: str, source: str, error_message: str = "") -> None:
        """某文件的新 Chroma 向量写入失败后，将其标记为仅使用 FTS5。"""
        kb_name = self._validate_kb_name(kb_name)
        source = self._validate_filename(source)
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                """
                INSERT INTO chroma_disabled_sources (kb_name, source, error_message, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(kb_name, source) DO UPDATE SET
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (kb_name, source, str(error_message or "")[:500], self._now()),
            )

    def enable_chroma_source(self, kb_name: str, source: str) -> None:
        """Chroma 向量同步成功后，清除该文件的仅 FTS5 检索标记。"""
        kb_name = self._validate_kb_name(kb_name)
        source = self._validate_filename(source)
        with application_db.connect(self._db_file()) as conn:
            conn.execute(
                "DELETE FROM chroma_disabled_sources WHERE kb_name = ? AND source = ?",
                (kb_name, source),
            )

    def chroma_disabled_sources(self, kb_name: str) -> list[str]:
        """列出已就绪但因近期向量同步失败而被排除出 Chroma 的文件。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT source FROM chroma_disabled_sources WHERE kb_name = ? ORDER BY source",
                (kb_name,),
            ).fetchall()
        return [row[0] for row in rows]

    def retrieval_source_state(self, kb_name: str) -> tuple[list[str], list[str]]:
        """通过一次一致性查询返回可用来源及其 Chroma 降级来源。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                """
                SELECT d.filename, c.source IS NOT NULL
                FROM documents d
                LEFT JOIN chroma_disabled_sources c
                  ON c.kb_name = d.kb_name AND c.source = d.filename
                WHERE d.kb_name = ? AND d.status = 'ready'
                ORDER BY d.filename
                """,
                (kb_name,),
            ).fetchall()
        ready = [row[0] for row in rows]
        disabled = [filename for filename, is_disabled in rows if is_disabled]
        return ready, disabled

    def has_documents(self, kb_name: str | None = None) -> bool:
        """判断全局或指定知识库中是否至少存在一份文档。"""
        with application_db.connect(self._db_file()) as conn:
            if kb_name is None:
                row = conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM documents WHERE kb_name = ? LIMIT 1", (kb_name,)
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

    def normalize_metadata(
        self, metadata: DocumentMetadataInput | None
    ) -> DocumentMetadata:
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
        """安全解析数据库中保存的 JSON 标签，兼容历史或异常数据。"""
        try:
            decoded = json.loads(value) if value else []
        except (TypeError, json.JSONDecodeError):
            return []
        return [tag for tag in decoded if isinstance(tag, str) and tag]

    def parser_modes(self, kb_name: str) -> dict[str, str]:
        """读取每个文件已保存的合法解析方式，保证重建结果可复现。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                "SELECT filename, parser_mode FROM documents WHERE kb_name = ?", (kb_name,)
            ).fetchall()
        modes: dict[str, str] = {}
        for filename, stored_mode in rows:
            try:
                modes[filename] = normalize_document_parser_mode(filename, stored_mode)
            except ValueError:
                modes[filename] = DEFAULT_PDF_PARSER_MODE
        return modes

    def update_parser_mode(self, kb_name: str, filename: str, parser_mode: str | None) -> str:
        """保存校验后的解析方式，使后续处理和完整重建使用同一策略。"""
        kb_name = self._validate_kb_name(kb_name)
        filename = self._validate_filename(filename)
        normalized = normalize_document_parser_mode(filename, parser_mode)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                "UPDATE documents SET parser_mode = ? WHERE kb_name = ? AND filename = ?",
                (normalized, kb_name, filename),
            )
        if cursor.rowcount != 1:
            raise ValueError("文档不存在")
        return normalized

    def update_metadata(
        self, kb_name: str, filename: str, metadata: DocumentMetadataInput
    ) -> DocumentMetadata:
        """校验并保存文档元数据，同时返回规范化后的字段。"""
        kb_name = self._validate_kb_name(kb_name)
        filename = self._validate_filename(filename)
        normalized = self.normalize_metadata(metadata)
        with application_db.connect(self._db_file()) as conn:
            cursor = conn.execute(
                """
                UPDATE documents SET document_type = ?, tags = ?, effective_date = ?
                WHERE kb_name = ? AND filename = ?
                """,
                (
                    normalized["document_type"],
                    json.dumps(normalized["tags"], ensure_ascii=False),
                    normalized["effective_date"], kb_name, filename,
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

    def normalize_filters(
        self, filters: RetrievalFilters | None
    ) -> NormalizedRetrievalFilters:
        """校验文档筛选条件，并转换为可安全参与查询构建的规范值。"""
        if not filters:
            return {
                "sources": (), "document_types": (), "tags": (),
                "effective_date_from": "", "effective_date_to": "",
            }
        if not isinstance(filters, dict):
            raise ValueError("检索筛选必须是对象")

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
            "sources": nfv(
                filters.get("sources", filters.get("source")), "来源文件", 255
            ),
            "document_types": nfv(
                filters.get("document_types", filters.get("document_type")),
                "文档类型", self._max_type_length,
            ),
            "tags": nfv(filters.get("tags"), "标签", self._max_tag_length),
            "effective_date_from": date_from,
            "effective_date_to": date_to,
        }

    def has_filters(self, filters: RetrievalFilters | None) -> bool:
        """判断请求是否包含至少一个实际生效的文档筛选条件。"""
        return any(self.normalize_filters(filters).values())

    def filtered_sources(
        self, kb_name: str, filters: RetrievalFilters | None
    ) -> list[str] | None:
        """在进入两类检索后端前，将界面元数据筛选条件解析为文件名列表。

        返回 ``None`` 表示未启用筛选；返回空列表则表示筛选已启用但没有任何匹配文件。
        """
        kb_name = self._validate_kb_name(kb_name)
        normalized = self.normalize_filters(filters)
        if not any(normalized.values()):
            return None
        clauses = ["kb_name = ?"]
        params: list[str] = [kb_name]
        for key, column in (("sources", "filename"), ("document_types", "document_type")):
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
                f"SELECT filename, tags FROM documents WHERE {' AND '.join(clauses)} ORDER BY filename",
                params,
            ).fetchall()
        # Tags live as JSON because they are edited as a small list.  Filter them after
        # the SQL-friendly fields, keeping the query parameterized and easy to inspect.
        selected_tags = set(normalized["tags"])
        return [
            filename for filename, tags in rows
            if not selected_tags or selected_tags.intersection(self._stored_tags(tags))
        ]

    def filter_options(self, kb_name: str) -> DocumentFilterOptions:
        """返回筛选界面可选的文件名、类型和标签去重列表。"""
        kb_name = self._validate_kb_name(kb_name)
        with application_db.connect(self._db_file()) as conn:
            if not conn.execute("SELECT 1 FROM knowledge_bases WHERE name = ?", (kb_name,)).fetchone():
                raise ValueError("知识库不存在")
            rows = conn.execute(
                "SELECT filename, document_type, tags FROM documents "
                "WHERE kb_name = ? ORDER BY filename",
                (kb_name,),
            ).fetchall()
        tags = sorted({tag for _, _, stored in rows for tag in self._stored_tags(stored)})
        return {
            "sources": [row[0] for row in rows],
            "document_types": sorted({row[1] for row in rows if row[1]}),
            "tags": tags,
        }

    def knowledge_base_path(self, kb_name: str) -> str:
        """将校验后的知识库名称映射到其原始文件存储目录。"""
        return os.path.join(self._data_dir(), self._validate_kb_name(kb_name))

    def ensure_knowledge_base_dir(self, kb_name: str) -> str:
        """确保知识库的原始文件目录存在，并返回该目录路径。"""
        path = self.knowledge_base_path(kb_name)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def _document_from_row(cls, row: tuple[object, ...]) -> DocumentRecord:
        """Convert the document-list query projection into its typed API record."""
        return {
            "filename": str(row[0]),
            "path": str(row[1]),
            "size": cast(int, row[2]),
            "uploaded_at": str(row[3]),
            "status": str(row[4]),
            "error_message": str(row[5] or ""),
            "index_job_id": str(row[6] or ""),
            "document_type": str(row[7] or ""),
            "tags": cls._stored_tags(row[8]),
            "effective_date": str(row[9] or ""),
            "parser_mode": str(row[10] or DEFAULT_PDF_PARSER_MODE),
        }

    def list_documents(self, kb_name: str) -> list[DocumentRecord]:
        """返回文档管理页面所需的记录、元数据和索引状态。"""
        self.ensure_knowledge_base_dir(kb_name)
        with application_db.connect(self._db_file()) as conn:
            rows = conn.execute(
                """
                SELECT filename, path, size, uploaded_at, status, error_message, index_job_id,
                       document_type, tags, effective_date, parser_mode
                FROM documents WHERE kb_name = ? ORDER BY uploaded_at DESC, filename
                """,
                (kb_name,),
            ).fetchall()
        return [self._document_from_row(tuple(row)) for row in rows]

    def document_path(self, kb_name: str, filename: str) -> str:
        """根据校验后的知识库和文件名生成原文件的安全路径。"""
        return os.path.join(self.knowledge_base_path(kb_name), self._validate_filename(filename))

    def delete_document(self, kb_name: str, filename: str) -> None:
        """在索引清理任务已安排后，删除原文件及其元数据记录。"""
        path = self.document_path(kb_name, filename)
        if os.path.exists(path):
            os.remove(path)
        with application_db.connect(self._db_file()) as conn:
            conn.execute("DELETE FROM documents WHERE kb_name = ? AND filename = ?", (kb_name, filename))
