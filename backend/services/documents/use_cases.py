"""Application use cases for document mutations and indexing coordination."""

import os
import uuid
from dataclasses import dataclass
from typing import BinaryIO

from services.core import config as application_config
from services.core.container import get_application_container
from services.core.errors import (
    InvalidRequestError,
    OperationFailedError,
    PayloadTooLargeError,
    ResourceNotFoundError,
    UnsupportedMediaError,
)
from services.documents import mineru as mineru_client
from services.documents.contracts import DocumentMetadata, DocumentMetadataInput
from services.documents.parser_modes import (
    PDF_PARSER_MINERU_CLOUD,
    is_docx_filename,
    is_image_filename,
    is_pdf_filename,
    normalize_document_parser_mode,
)
from services.documents.types import SUPPORTED_DOCUMENT_EXTENSIONS
from services.indexing import pipeline as indexing_pipeline
from services.indexing import worker as index_worker
from services.indexing.jobs import INDEX_JOB_FAILED
from services.retrieval import retriever

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# 文件系统按字节限制单级名称（常见为 255）。上传的临时文件名会再加上 41 字节
# （前导点、32 位随机串和 .upload），MinerU 结果目录再加 7 字节，所以新上传的名称留出余量。
# validate_filename 仍只限制 255 个字符，不会让已有文档变得不可访问。
MAX_UPLOAD_FILENAME_BYTES = 200
_SUPPORTED_EXTENSIONS = set(SUPPORTED_DOCUMENT_EXTENSIONS)


@dataclass(frozen=True)
class DocumentMutationResult:
    """Public values produced by a document mutation use case."""

    job_id: str
    metadata: DocumentMetadata | None = None
    parser_mode: str | None = None


def _validated_filename(filename: str | None) -> str:
    """校验并规范化上传文件名，将底层校验错误转换为应用错误。"""
    try:
        name = application_config.validate_filename(filename or "")
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_document_filename") from exc
    if len(name.encode("utf-8")) > MAX_UPLOAD_FILENAME_BYTES:
        raise InvalidRequestError(
            f"文件名过长：不能超过 {MAX_UPLOAD_FILENAME_BYTES} 字节"
            "（英文数字约 200 个字符，每个汉字占 3 字节，约 66 个汉字）",
            code="invalid_document_filename",
        )
    return name


def _validated_parser_mode(filename: str, parser_mode: str | None) -> str:
    """确认文件可使用指定解析器，并检查云端解析所需配置。"""
    try:
        normalized = normalize_document_parser_mode(filename, parser_mode)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_document_parser") from exc
    if normalized == PDF_PARSER_MINERU_CLOUD and not mineru_client.is_configured():
        raise InvalidRequestError(
            "未配置 MINERU_API_TOKEN，无法使用 MinerU 云端解析",
            code="mineru_not_configured",
        )
    return normalized


def _enqueue_document_refresh(kb_name: str, filename: str) -> str:
    """Create a durable refresh job before waking the in-process worker."""
    index_jobs = get_application_container().index_jobs()
    job_id = index_jobs.create(kb_name, "process", filename)
    if not index_jobs.attach_document_job(kb_name, filename, job_id):
        index_jobs.finish(job_id, INDEX_JOB_FAILED, "文档不存在")
        raise ResourceNotFoundError("文档不存在", code="document_not_found")
    retriever.drop_cached_retriever(kb_name)
    index_worker.enqueue(job_id)
    return job_id


def upload_document(
    kb_name: str,
    filename: str | None,
    source: BinaryIO,
    *,
    metadata: DocumentMetadataInput | None,
    parser_mode: str | None,
) -> DocumentMutationResult:
    """Atomically store an upload and create its durable indexing job."""
    documents = get_application_container().documents
    index_jobs = get_application_container().index_jobs()
    filename = _validated_filename(filename)
    if os.path.splitext(filename)[1].lower() not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaError(
            "暂不支持该文件类型", code="unsupported_document_type"
        )
    normalized_parser = _validated_parser_mode(filename, parser_mode)
    try:
        normalized_metadata = documents.normalize_metadata(metadata)
    except ValueError as exc:
        raise InvalidRequestError(
            f"文档元数据无效：{exc}", code="invalid_document_metadata"
        ) from exc
    if not documents.knowledge_base_exists(kb_name):
        raise ResourceNotFoundError(
            "知识库不存在", code="knowledge_base_not_found"
        )

    path = documents.document_path(kb_name, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = os.path.join(
        os.path.dirname(path), f".{filename}.{uuid.uuid4().hex}.upload"
    )
    size = 0
    try:
        with open(temporary_path, "wb") as output:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise PayloadTooLargeError(
                        "单个文件不能超过 50 MB", code="document_too_large"
                    )
                output.write(chunk)
        with indexing_pipeline.document_lock(kb_name, filename):
            os.replace(temporary_path, path)
            job_id = index_jobs.create(kb_name, "process", filename)
            try:
                documents.upsert_uploaded_document(
                    kb_name,
                    filename,
                    path=path,
                    size=size,
                    job_id=job_id,
                    parser_mode=normalized_parser,
                    metadata=normalized_metadata,
                )
            except Exception as exc:
                index_jobs.finish(job_id, INDEX_JOB_FAILED, str(exc))
                raise
            index_worker.enqueue(job_id)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return DocumentMutationResult(
        job_id=job_id, metadata=normalized_metadata, parser_mode=normalized_parser
    )


def update_document_metadata(
    kb_name: str, filename: str, metadata: DocumentMetadataInput
) -> DocumentMutationResult:
    """Persist document filter metadata without rebuilding content indexes."""
    documents = get_application_container().documents
    try:
        normalized = documents.normalize_metadata(metadata)
    except ValueError as exc:
        raise InvalidRequestError(
            str(exc), code="invalid_document_metadata"
        ) from exc
    path = documents.document_path(kb_name, filename)
    with indexing_pipeline.document_lock(kb_name, filename):
        if not documents.document_exists(
            kb_name, filename
        ) or not os.path.isfile(path):
            raise ResourceNotFoundError("文档不存在", code="document_not_found")
        try:
            documents.update_metadata(kb_name, filename, normalized)
        except ValueError as exc:
            raise ResourceNotFoundError(
                str(exc), code="document_not_found"
            ) from exc
    return DocumentMutationResult(job_id="", metadata=normalized)


def update_document_parser(
    kb_name: str, filename: str, parser_mode: str
) -> DocumentMutationResult:
    """Persist a parser selection and schedule a complete document refresh."""
    documents = get_application_container().documents
    if not (is_pdf_filename(filename) or is_docx_filename(filename) or is_image_filename(filename)):
        raise InvalidRequestError(
            "只有 PDF、DOCX 或图片文档可以更换解析方式",
            code="document_parser_not_editable",
        )
    normalized_parser = _validated_parser_mode(filename, parser_mode)
    path = documents.document_path(kb_name, filename)
    with indexing_pipeline.document_lock(kb_name, filename):
        if not documents.document_exists(
            kb_name, filename
        ) or not os.path.isfile(path):
            raise ResourceNotFoundError("文档不存在", code="document_not_found")
        try:
            documents.update_parser_mode(
                kb_name, filename, normalized_parser
            )
        except ValueError as exc:
            raise ResourceNotFoundError(
                str(exc), code="document_not_found"
            ) from exc
        job_id = _enqueue_document_refresh(kb_name, filename)
    return DocumentMutationResult(job_id=job_id, parser_mode=normalized_parser)


def delete_document(kb_name: str, filename: str) -> DocumentMutationResult:
    """Delete source state and schedule durable cleanup of both indexes."""
    documents = get_application_container().documents
    index_jobs = get_application_container().index_jobs()
    with indexing_pipeline.document_lock(kb_name, filename):
        if not documents.document_exists(kb_name, filename):
            raise ResourceNotFoundError("文档不存在", code="document_not_found")
        job_id = index_jobs.create(kb_name, "delete", filename)
        try:
            mineru_client.remove_saved_result(kb_name, filename)
            documents.delete_document(kb_name, filename)
        except Exception as exc:
            index_jobs.finish(job_id, INDEX_JOB_FAILED, str(exc))
            raise OperationFailedError(
                f"文档删除失败：{exc}", code="document_delete_failed"
            ) from exc
    retriever.drop_cached_retriever(kb_name)
    index_worker.enqueue(job_id)
    return DocumentMutationResult(job_id=job_id)
