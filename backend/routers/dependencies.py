"""Shared FastAPI dependencies for validated path parameters."""

from typing import Annotated

from fastapi import Depends, HTTPException

from services.core import config as application_config


def valid_kb_id(kb_id: str) -> str:
    """校验路径中的知识库 ID 并返回规范化的 UUID，失败转换为 HTTP 422 响应。"""
    try:
        return application_config.validate_kb_id(kb_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def valid_document_id(document_id: str) -> str:
    try:
        return application_config.validate_document_id(document_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


KnowledgeBaseIdPath = Annotated[str, Depends(valid_kb_id)]

DocumentIdPath = Annotated[str, Depends(valid_document_id)]
