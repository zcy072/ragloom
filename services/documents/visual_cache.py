"""Persistent cache for model-derived visual analyses."""

import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from services.documents import mineru as mineru_client
from services.documents.visual_candidates import VisualCandidate

VISUAL_CACHE_FILENAME = "pymodel_visual_index.json"
STANDALONE_IMAGE_CACHE_FILENAME = "pymodel_image_visual_index.json"
VISUAL_PROMPT_VERSION = "mineru-visual-v3"
STANDALONE_IMAGE_PROMPT_VERSION = "standalone-image-visual-v3"


def _cache_path(file_path: str) -> Path:
    """返回文档 MinerU 结果目录中的视觉分析缓存路径。"""
    return mineru_client.result_dir_for_document(file_path) / VISUAL_CACHE_FILENAME


def _standalone_image_cache_path(file_path: str) -> Path:
    """将独立图片的模型结果存入其持久化工件目录，而非知识库源目录。"""
    return mineru_client.result_dir_for_document(file_path) / STANDALONE_IMAGE_CACHE_FILENAME


def _read_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """读取有效缓存；文件缺失或损坏时按空缓存处理。"""
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
        return entries if isinstance(entries, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _write_cache(
    cache_path: Path,
    entries: dict[str, dict[str, Any]],
    prompt_version: str,
) -> None:
    """通过临时文件原子发布指定版本的视觉缓存。"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "prompt_version": prompt_version,
                    "entries": entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_visual_cache(file_path: str) -> dict[str, dict[str, Any]]:
    """读取 PDF/DOCX 视觉分析缓存。"""
    return _read_cache(_cache_path(file_path))


def read_standalone_image_cache(file_path: str) -> dict[str, dict[str, Any]]:
    """读取独立图片视觉缓存；缓存损坏不影响本次解析。"""
    return _read_cache(_standalone_image_cache_path(file_path))


def write_visual_cache(file_path: str, entries: dict[str, dict[str, Any]]) -> None:
    """原子写入 PDF/DOCX 视觉分析缓存。"""
    _write_cache(_cache_path(file_path), entries, VISUAL_PROMPT_VERSION)


def write_standalone_image_cache(
    file_path: str, entries: dict[str, dict[str, Any]]
) -> None:
    """原子写入独立图片视觉缓存，避免重建重复调用远程模型。"""
    _write_cache(
        _standalone_image_cache_path(file_path),
        entries,
        STANDALONE_IMAGE_PROMPT_VERSION,
    )


def visual_cache_key(
    candidate: VisualCandidate, image_bytes: bytes, config: dict[str, str]
) -> str:
    """根据图片内容、提示上下文、提示版本和模型标识生成缓存键。"""
    context = json.dumps(
        {
            "block_type": candidate.block_type,
            "caption": candidate.caption,
            "footnote": candidate.footnote,
            "html": candidate.html,
            "headings": candidate.headings,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    model_signature = (
        f"{config.get('base_url', '').rstrip('/')}|{config.get('model', '')}"
    )
    digest = hashlib.sha256(image_bytes).hexdigest()
    value = "\x1f".join(
        (VISUAL_PROMPT_VERSION, model_signature, digest, context)
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def standalone_image_cache_key(
    candidate: VisualCandidate, image_bytes: bytes, config: dict[str, str]
) -> str:
    """为独立图片按内容、提示版本和模型标识生成与 MinerU 隔离的缓存键。"""
    model_signature = f"{config.get('base_url', '').rstrip('/')}|{config.get('model', '')}"
    digest = hashlib.sha256(image_bytes).hexdigest()
    value = "\x1f".join(
        (STANDALONE_IMAGE_PROMPT_VERSION, model_signature, digest, candidate.block_type)
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_visual_analysis(entries: dict, key: str, analysis: Any) -> None:
    """将一条通过校验的视觉分析结果序列化到可变缓存映射。"""
    entries[key] = asdict(analysis)
