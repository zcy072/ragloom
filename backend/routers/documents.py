"""Document upload, metadata, parser selection, deletion, and preview endpoints."""

import html
import json
import mimetypes
import os
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

import app_storage
from routers import dependencies, schemas
from services.documents import loaders as document_loader
from services.documents import use_cases as document_use_cases
from services.documents.types import DocumentLoadError
from services.identity import auth as auth_service

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents/{kb_name}", response_model=list[schemas.DocumentResponse])
def list_documents(
    kb_name: dependencies.KnowledgeBasePath,
    _principal=Depends(auth_service.require_knowledge_base_reader),
):
    """返回指定知识库中当前用户可见的文档记录。"""
    return app_storage.documents.list_documents(kb_name)


@router.get("/document_parsers", response_model=schemas.DocumentParserOptionsResponse)
def list_document_parsers(_principal=Depends(auth_service.current_principal)):
    """返回后端当前支持和可用的文档解析方式。"""
    options = document_loader.list_document_parser_options()
    return {"document": options, "pdf": options}


@router.get("/document_formats", response_model=list[schemas.DocumentFormatResponse])
def list_document_formats(_principal=Depends(auth_service.current_principal)):
    """返回允许上传的文件格式及每种格式支持的解析方式。"""
    return document_loader.list_document_format_options()


@router.post("/documents/{kb_name}", response_model=schemas.DocumentUploadResponse)
def upload_document(
    kb_name: dependencies.KnowledgeBasePath,
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    parser_mode: str | None = Form(None),
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """保存上传文件和元数据，并创建异步解析与索引任务。"""
    try:
        if metadata is None:
            supplied_metadata = None
        else:
            metadata_request = schemas.DocumentMetadataRequest.model_validate(
                json.loads(metadata)
            )
            supplied_metadata = metadata_request.model_dump()
    except (TypeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(422, f"文档元数据无效：{exc}") from exc
    result = document_use_cases.upload_document(
        kb_name,
        file.filename,
        file.file,
        metadata=supplied_metadata,
        parser_mode=parser_mode,
    )
    return {
        "status": "success",
        "job_id": result.job_id,
        "metadata": result.metadata,
        "parser_mode": result.parser_mode,
    }


@router.put(
    "/documents/{kb_name}/{filename}/metadata",
    response_model=schemas.DocumentMetadataUpdateResponse,
)
def update_document_metadata(
    kb_name: dependencies.KnowledgeBasePath,
    filename: dependencies.FilenamePath,
    data: schemas.DocumentMetadataRequest,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """保存由业务数据库负责的文档筛选元数据。"""
    result = document_use_cases.update_document_metadata(
        kb_name, filename, data.model_dump()
    )
    return {
        "status": "success",
        "metadata": result.metadata,
    }


@router.put(
    "/documents/{kb_name}/{filename}/parser",
    response_model=schemas.DocumentParserUpdateResponse,
)
def update_document_parser(
    kb_name: dependencies.KnowledgeBasePath,
    filename: dependencies.FilenamePath,
    data: schemas.DocumentParserRequest,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """修改文档解析方式，并创建重新解析和索引任务。"""
    result = document_use_cases.update_document_parser(
        kb_name, filename, data.parser_mode
    )
    return {
        "status": "success",
        "job_id": result.job_id,
        "parser_mode": result.parser_mode,
    }


@router.delete("/documents/{kb_name}/{filename}", response_model=schemas.JobResponse)
def delete_document(
    kb_name: dependencies.KnowledgeBasePath,
    filename: dependencies.FilenamePath,
    _principal=Depends(auth_service.require_knowledge_base_manager),
):
    """删除文档关系数据，并创建持久化索引清理任务。"""
    result = document_use_cases.delete_document(kb_name, filename)
    return {"status": "success", "job_id": result.job_id}


@router.get("/documents/{kb_name}/{filename}")
def download_document(
    kb_name: dependencies.KnowledgeBasePath,
    filename: dependencies.FilenamePath,
    preview: bool = False,
    _principal=Depends(auth_service.require_knowledge_base_reader),
):
    """返回原始文件；预览请求则以内联且安全的方式返回支持格式。"""
    file_path = app_storage.documents.document_path(kb_name, filename)
    if not os.path.exists(file_path):
        if preview:
            safe_filename = html.escape(filename)
            page = (
                "<html><head><meta charset='utf-8'><style>"
                "body{font-family:sans-serif;padding:40px;text-align:center;color:#666;background:#f8fafc;}"
                "h2{color:#ef4444;}</style></head><body><h2>文件未找到</h2>"
                f"<p>原文档 <b>{safe_filename}</b> 已被改名或删除，无法预览。</p></body></html>"
            )
            return HTMLResponse(content=page, status_code=404)
        raise HTTPException(404, "File not found")

    if preview:
        extension = os.path.splitext(filename)[1].lower()
        try:
            if extension == ".docx":
                body = document_loader.docx_preview_html(file_path)
                page = (
                    "<html><head><meta charset='utf-8'><style>"
                    "body{font-family:sans-serif;padding:30px;line-height:1.7;color:#263538;font-size:14px;}"
                    "h1,h2,h3,h4{margin:22px 0 10px;}p{margin:0 0 12px;}"
                    "table{width:100%;border-collapse:collapse;margin:16px 0;}"
                    "td{border:1px solid #dce5e5;padding:8px;vertical-align:top;}"
                    "tr:first-child{background:#f3f7f7;font-weight:600;}"
                    f"</style></head><body><h2>{html.escape(filename)}</h2><hr/>{body}</body></html>"
                )
                return HTMLResponse(
                    content=page,
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
                )
            if extension in {".txt", ".md"}:
                content = document_loader.read_text_document(file_path, filename)
                page = (
                    "<html><head><meta charset='utf-8'><style>"
                    "body{font-family:monospace;padding:20px;white-space:pre-wrap;"
                    "color:#333;background:#f8fafc;}</style></head>"
                    f"<body>{html.escape(content)}</body></html>"
                )
                return HTMLResponse(content=page)
        except DocumentLoadError:
            pass

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("text/"):
        mime_type += "; charset=utf-8"
    return FileResponse(
        file_path,
        media_type=mime_type or "application/octet-stream",
        filename=filename,
        headers={
            "Content-Disposition": f"inline; filename*=utf-8''{quote(filename)}",
            "X-Content-Type-Options": "nosniff",
        },
    )
