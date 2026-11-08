"""Shared FastAPI dependencies for validated path parameters."""

from typing import Annotated

from fastapi import Depends, HTTPException

from services.core import config as application_config


def _validated_kb_name(value: str) -> str:
    """校验知识库名称，并把校验失败转换为 HTTP 422 响应。"""
    try:
        return application_config.validate_kb_name(value)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# FastAPI matches dependency inputs by path-parameter name. These wrappers intentionally
# mirror the three names used by existing routes; merging them would break parameter binding.
def valid_kb_name(kb_name: str) -> str:
    """校验名为 ``kb_name`` 的路径参数，供 FastAPI 依赖注入使用。"""
    return _validated_kb_name(kb_name)


def valid_knowledge_base_name(name: str) -> str:
    """校验名为 ``name`` 的知识库路径参数。"""
    return _validated_kb_name(name)


def valid_old_knowledge_base_name(old_name: str) -> str:
    """校验重命名接口中名为 ``old_name`` 的旧知识库名称。"""
    return _validated_kb_name(old_name)


def valid_filename(filename: str) -> str:
    """校验文件名，并把失败结果转换为 HTTP 422 响应。"""
    try:
        return application_config.validate_filename(filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


KnowledgeBasePath = Annotated[str, Depends(valid_kb_name)]
KnowledgeBaseNamePath = Annotated[str, Depends(valid_knowledge_base_name)]
OldKnowledgeBaseNamePath = Annotated[str, Depends(valid_old_knowledge_base_name)]
FilenamePath = Annotated[str, Depends(valid_filename)]
