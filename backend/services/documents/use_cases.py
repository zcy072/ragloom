"""Application use cases for document mutations and indexing coordination."""

import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import BinaryIO

from services.core import config as application_config
from services.core.container import get_application_container
from services.core.errors import (
    InvalidRequestError,
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
from services.retrieval import retriever

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# 沿用 200 字节的上传名称限制，为 MinerU 结果目录等解析产物的后缀预留余量。
MAX_UPLOAD_FILENAME_BYTES = 200
_SUPPORTED_EXTENSIONS = set(SUPPORTED_DOCUMENT_EXTENSIONS)


@dataclass(frozen=True)
class DocumentMutationResult:
    """Public values produced by a document mutation use case."""

    job_id: str
    document_id: str
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


logger = logging.getLogger(__name__)


def _wake_worker(job_id: str) -> None:
    """任务已经提交；唤醒失败由周期补扫处理，不能回滚成功上传。"""
    try:
        if not index_worker.enqueue(job_id):
            logger.info("Index job %s awaits queued scan", job_id)
    except Exception:
        logger.warning("Failed to wake index worker for %s", job_id, exc_info=True)


def _remove_new_directory(path: str) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning(
            "Upload cleanup failed for %s; startup sweep will retry", path, exc_info=True
        )


def upload_document(
    kb_id: str,
    filename: str | None,
    source: BinaryIO,
    *,
    metadata: DocumentMetadataInput | None,
    parser_mode: str | None,
) -> DocumentMutationResult:
    """每次上传创建独立身份；文档和初始任务在同一事务提交。"""
    documents = get_application_container().documents
    filename = _validated_filename(filename)
    if os.path.splitext(filename)[1].lower() not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaError("暂不支持该文件类型", code="unsupported_document_type")
    normalized_parser = _validated_parser_mode(filename, parser_mode)
    try:
        normalized_metadata = documents.normalize_metadata(metadata)
    except ValueError as exc:
        raise InvalidRequestError(
            f"文档元数据无效：{exc}", code="invalid_document_metadata"
        ) from exc
    if not documents.knowledge_base_exists(kb_id):
        raise ResourceNotFoundError("知识库不存在", code="knowledge_base_not_found")
    document_id = str(uuid.uuid4())
    upload_dir = os.path.join(documents.uploads_directory(), uuid.uuid4().hex)
    os.makedirs(upload_dir, exist_ok=False)
    temporary_path = os.path.join(upload_dir, filename)
    directory = documents.document_directory(kb_id, document_id)
    size = 0
    try:
        with open(temporary_path, "wb") as output:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise PayloadTooLargeError("单个文件不能超过 50 MB", code="document_too_large")
                output.write(chunk)
        with indexing_pipeline.document_lock(kb_id, document_id):
            if not documents.knowledge_base_exists(kb_id):
                raise ResourceNotFoundError("知识库不存在", code="knowledge_base_not_found")
            # 新 ID 目录不能覆盖。建目录失败时不清理非本次创建的目录。
            os.makedirs(directory, exist_ok=False)
            try:
                os.replace(temporary_path, documents.stored_path(kb_id, document_id, filename))
                job_id = documents.create_document_with_job(
                    kb_id,
                    document_id,
                    filename,
                    size=size,
                    parser_mode=normalized_parser,
                    metadata=normalized_metadata,
                )
            except BaseException:
                _remove_new_directory(directory)
                raise
    finally:
        _remove_new_directory(upload_dir)
    retriever.drop_cached_retriever(kb_id)
    _wake_worker(job_id)
    return DocumentMutationResult(job_id, document_id, normalized_metadata, normalized_parser)


def update_document_metadata(
    kb_id: str,
    document_id: str,
    metadata: DocumentMetadataInput,
) -> DocumentMutationResult:
    documents = get_application_container().documents
    try:
        normalized = documents.normalize_metadata(metadata)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), code="invalid_document_metadata") from exc
    with indexing_pipeline.document_lock(kb_id, document_id):
        documents.get_document(kb_id, document_id)
        documents.update_metadata(kb_id, document_id, normalized)
    return DocumentMutationResult("", document_id, metadata=normalized)


def reindex_document(
    kb_id: str, document_id: str, *, parser_mode: str | None = None
) -> DocumentMutationResult:
    documents = get_application_container().documents
    with indexing_pipeline.document_lock(kb_id, document_id):
        record = documents.get_document(kb_id, document_id)
        normalized_parser = record["parser_mode"]
        if parser_mode is not None:
            filename = record["filename"]
            if not (
                is_pdf_filename(filename)
                or is_docx_filename(filename)
                or is_image_filename(filename)
            ):
                raise InvalidRequestError(
                    "只有 PDF、DOCX 或图片文档可以更换解析方式", code="document_parser_not_editable"
                )
            normalized_parser = _validated_parser_mode(filename, parser_mode)
        if not os.path.isfile(documents.document_path(kb_id, document_id)):
            raise ResourceNotFoundError("原文件不存在", code="document_not_found")
        job_id = documents.schedule_reindex(kb_id, document_id, parser_mode=normalized_parser)
    retriever.drop_cached_retriever(kb_id)
    _wake_worker(job_id)
    return DocumentMutationResult(job_id, document_id, parser_mode=normalized_parser)


def update_document_parser(
    kb_id: str, document_id: str, parser_mode: str
) -> DocumentMutationResult:
    return reindex_document(kb_id, document_id, parser_mode=parser_mode)


def delete_document(kb_id: str, document_id: str) -> DocumentMutationResult:
    documents = get_application_container().documents
    with indexing_pipeline.document_lock(kb_id, document_id):
        job_id = documents.schedule_document_delete(kb_id, document_id)
    retriever.drop_cached_retriever(kb_id)
    _wake_worker(job_id)
    return DocumentMutationResult(job_id, document_id)
