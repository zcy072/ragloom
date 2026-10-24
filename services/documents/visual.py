"""从 MinerU JSON 引用的图片工件中构建可检索的文本证据。"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from services.core import llm_config as model_config
from services.core.concurrency import resource_slot
from services.core.container import get_application_container
from services.core.settings import get_settings
from services.documents import visual_ocr, visual_vision
from services.documents.parser_modes import (
    IMAGE_PARSER_VISION_MODEL,
    PDF_PARSER_MINERU_CLOUD,
    normalize_document_parser_mode,
)
from services.documents.visual_cache import (
    cache_visual_analysis,
    read_standalone_image_cache,
    read_visual_cache,
    standalone_image_cache_key,
    visual_cache_key,
    write_standalone_image_cache,
    write_visual_cache,
)
from services.documents.visual_candidates import (
    VisualCandidate,
    discover_visual_candidates,
)
from services.documents.visual_vision import VisualAnalysis

logger = logging.getLogger(__name__)


VISUAL_KIND_LABELS = visual_vision.VISUAL_KIND_LABELS


def _vision_model_config() -> dict[str, str]:
    """合并视觉模型提供商配置和环境变量覆盖项。"""
    settings = get_settings()
    model_providers = get_application_container().model_providers
    provider = settings.vision_provider
    if provider:
        configured = model_config.get_model_config(provider, model_providers)
        if configured is None:
            raise RuntimeError(f"视觉模型提供商未配置完整：{provider}")
        config = dict(configured)
    else:
        config = dict(model_config.get_active_model_config(model_providers))
    overrides = {
        "api_key": settings.vision_api_key,
        "base_url": settings.vision_base_url,
        "model": settings.vision_model,
    }
    for key, value in overrides.items():
        if value:
            config[key] = value
    return {key: str(config.get(key, "")).strip() for key in ("api_key", "base_url", "model")}


def _analyze_with_model(
    candidate: VisualCandidate,
    image_bytes: bytes,
    config: dict[str, str],
) -> VisualAnalysis:
    """保留可替换测试入口，并委托模型传输模块完成实际请求。"""
    settings = get_settings()
    # 单文档内由线程池限流，这里再限制所有并行文档合计发往视觉模型的请求数。
    with resource_slot("vision", settings.vision_global_max_concurrency):
        return visual_vision.analyze_with_model(
            candidate,
            image_bytes,
            config,
            timeout_seconds=settings.vision_timeout_seconds,
        )


_analysis_from_payload = visual_vision.analysis_from_payload


def _analyze_with_model_retry(
    candidate: VisualCandidate,
    image_bytes: bytes,
    config: dict[str, str],
    model_unavailable: threading.Event,
) -> tuple[VisualAnalysis | None, Exception | None]:
    """带重试分析一张图片，并在同一文档内共享有限的故障熔断标记。

    提供商级故障会设置共享标记，其他并发图片任务随即跳过远程调用并回退 OCR，
    避免对同一个不可用模型集中重复请求；单张图的回答格式错误不触发熔断。
    """
    if model_unavailable.is_set():
        return None, None
    model_error: Exception | None = None
    for _attempt in range(2):
        if model_unavailable.is_set():
            break
        try:
            return _analyze_with_model(candidate, image_bytes, config), None
        except Exception as exc:
            model_error = exc
    # 只有单张图的回答格式不合约定时，模型本身仍可用，不连累同文档的其余图片。
    if model_error is not None and not isinstance(
        model_error, visual_vision.VisionResponseError
    ):
        model_unavailable.set()
    return None, model_error


def _ocr_fallback(candidate: VisualCandidate, image_bytes: bytes) -> VisualAnalysis | None:
    """保留可替换测试入口，并委托本地 OCR 回退模块。"""
    return visual_ocr.ocr_fallback(candidate, image_bytes)


def _extract_ocr_text(image_bytes: bytes) -> str:
    """保留可替换测试入口，并委托本地 OCR 模块。"""
    return visual_ocr.extract_ocr_text(image_bytes)


def _standalone_image_document(
    filename: str,
    candidate: VisualCandidate,
    analysis: VisualAnalysis | None,
    ocr_text: str,
    analysis_method: str,
) -> Document | None:
    """组合模型语义与 OCR 原文，生成独立图片的固定检索路线。"""
    fragments: list[str] = []
    metadata: dict[str, Any] = {
        "source": filename,
        "parser_mode": IMAGE_PARSER_VISION_MODEL,
        "visual_analysis": analysis_method,
    }
    if analysis is not None and analysis.indexable:
        metadata["visual_kind"] = analysis.kind
        fragments.append(f"视觉内容类型：{VISUAL_KIND_LABELS.get(analysis.kind, '图片')}")
        if analysis.title:
            fragments.append(f"主题：{analysis.title}")
            metadata["contextual_title"] = analysis.title
        if analysis.content:
            fragments.append(analysis.content)
    if ocr_text:
        fragments.append(f"图片 OCR 文字：\n{ocr_text}")
    page_content = "\n\n".join(fragment for fragment in fragments if fragment.strip()).strip()
    return Document(page_content=page_content, metadata=metadata) if page_content else None


def load_standalone_image_document(
    file_path: str,
    filename: str,
    parser_mode: str | None,
) -> Document | None:
    """视觉优先解析一张独立上传图片，并以 OCR 作为补充和故障回退。

    成功的模型分析（包括明确判定为非索引内容的结果）都会缓存。OCR 始终在本地
    执行，既保留图片中的精确名称和编号，也保证视觉模型不可用时仍可建立索引。
    """
    if normalize_document_parser_mode(filename, parser_mode) != IMAGE_PARSER_VISION_MODEL:
        return None

    settings = get_settings()
    # 完全禁用图片分析时，直接返回 None
    if settings.is_image_analysis_disabled:
        return None

    candidate = VisualCandidate(
        page=0,
        block_type="image",
        image_path=Path(file_path),
        relative_path=filename,
    )
    try:
        image_bytes = candidate.image_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"无法读取图片：{exc}") from exc

    analysis: VisualAnalysis | None = None
    analysis_method = "ocr"
    vision_enabled = settings.is_vision_enabled
    config: dict[str, str] = {}
    if vision_enabled:
        try:
            config = _vision_model_config()
            entries = read_standalone_image_cache(file_path)
            cache_key = standalone_image_cache_key(candidate, image_bytes, config)
            analysis = _analysis_from_payload(entries.get(cache_key))
            if analysis is None:
                analysis, model_error = _analyze_with_model_retry(
                    candidate, image_bytes, config, threading.Event()
                )
                if analysis is not None:
                    cache_visual_analysis(entries, cache_key, analysis)
                    try:
                        write_standalone_image_cache(file_path, entries)
                    except OSError as exc:
                        logger.warning("Standalone image visual cache write failed for %s: %s", filename, exc)
                elif model_error is not None:
                    logger.warning("Standalone image visual model unavailable for %s; using OCR fallback: %s", filename, model_error)
            if analysis is not None:
                analysis_method = "vision+ocr"
        except Exception as exc:
            logger.warning("Standalone image vision analysis unavailable for %s; using OCR fallback: %s", filename, exc)

    try:
        ocr_text = _extract_ocr_text(image_bytes)
    except Exception as exc:
        logger.warning("Standalone image OCR unavailable for %s: %s", filename, exc)
        ocr_text = ""
    return _standalone_image_document(
        filename, candidate, analysis, ocr_text, analysis_method
    )


def _contextual_title(filename: str, candidate: VisualCandidate, analysis: VisualAnalysis) -> str:
    """组合文件名、章节路径和图标题，生成用于检索展示的上下文标题。"""
    parts = [os.path.splitext(os.path.basename(filename))[0]]
    for value in (*candidate.headings, analysis.title or candidate.caption):
        normalized = " ".join(value.split())
        if normalized and normalized not in parts:
            parts.append(normalized)
    return " > ".join(parts)[:300]


def _document_from_analysis(
    filename: str,
    candidate: VisualCandidate,
    analysis: VisualAnalysis,
    analysis_method: str,
) -> Document | None:
    """将有效的视觉分析转换为带来源、页码和分析方式元数据的 Document。"""
    if not analysis.indexable:
        return None
    fragments = [f"视觉内容类型：{VISUAL_KIND_LABELS.get(analysis.kind, '图片')}"]
    if candidate.caption:
        fragments.append(f"图注：{candidate.caption}")
    if analysis.title and analysis.title != candidate.caption:
        fragments.append(f"主题：{analysis.title}")
    if analysis.content:
        fragments.append(analysis.content)
    if candidate.footnote:
        fragments.append(f"图注说明：{candidate.footnote}")
    page_content = "\n\n".join(fragment for fragment in fragments if fragment.strip()).strip()
    if not page_content:
        return None
    return Document(
        page_content=page_content,
        metadata={
            "source": filename,
            "page": candidate.page,
            "parser_mode": PDF_PARSER_MINERU_CLOUD,
            "chunk_route": "visual",
            "contextual_title": _contextual_title(filename, candidate, analysis),
            "visual_kind": analysis.kind,
            "visual_image": candidate.relative_path,
            "visual_analysis": analysis_method,
        },
    )


def load_mineru_visual_documents(
    file_path: str,
    filename: str,
    parser_mode: str | None,
) -> list[Document]:
    """分析 MinerU 筛选的视觉内容，优先复用模型缓存，失败时回退本地 OCR。"""
    if normalize_document_parser_mode(filename, parser_mode) != PDF_PARSER_MINERU_CLOUD:
        return []
    candidates = discover_visual_candidates(file_path)
    if not candidates:
        return []

    settings = get_settings()
    # 完全禁用图片分析时，直接返回空列表
    if settings.is_image_analysis_disabled:
        return []

    vision_enabled = settings.is_vision_enabled
    try:
        config = _vision_model_config() if vision_enabled else {}
    except Exception as exc:
        logger.warning(
            "MinerU visual model configuration unavailable for %s: %s", filename, exc
        )
        config = {}
        vision_enabled = False
    entries = read_visual_cache(file_path)
    cache_changed = False
    ocr_warning_reported = False
    prepared: list[tuple[VisualCandidate, bytes, str, VisualAnalysis | None]] = []
    for candidate in candidates:
        try:
            image_bytes = candidate.image_path.read_bytes()
        except OSError as exc:
            logger.warning(
                "MinerU visual artifact cannot be read [%s/%s]: %s",
                filename,
                candidate.relative_path,
                exc,
            )
            continue
        cache_key = visual_cache_key(candidate, image_bytes, config)
        cached = _analysis_from_payload(entries.get(cache_key)) if vision_enabled else None
        prepared.append((candidate, image_bytes, cache_key, cached))

    analyses: dict[int, VisualAnalysis] = {
        index: cached
        for index, (_candidate, _image_bytes, _candidate_cache_key, cached) in enumerate(prepared)
        if cached is not None
    }
    pending = [index for index in range(len(prepared)) if index not in analyses]

    # 缓存命中不占并发槽位；只有未命中的图片进入受限并发模型/OCR 路径，控制成本和限流压力。
    model_error: Exception | None = None
    if vision_enabled and pending:
        model_unavailable = threading.Event()
        worker_count = min(get_settings().vision_global_max_concurrency, len(pending))
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="vision-analysis"
        ) as executor:
            futures: dict[Future, int] = {
                executor.submit(
                    _analyze_with_model_retry,
                    prepared[index][0],
                    prepared[index][1],
                    config,
                    model_unavailable,
                ): index
                for index in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    analysis, error = future.result()
                except Exception as exc:
                    analysis, error = None, exc
                    model_unavailable.set()
                if analysis is not None:
                    analyses[index] = analysis
                    cache_visual_analysis(entries, prepared[index][2], analysis)
                    cache_changed = True
                elif model_error is None and error is not None:
                    model_error = error
        if model_error is not None:
            logger.warning(
                "MinerU visual model unavailable for some %s visuals; using OCR fallback: %s",
                filename,
                model_error,
            )

    documents: list[Document] = []
    for index, (candidate, image_bytes, _candidate_cache_key, _cached) in enumerate(prepared):
        analysis = analyses.get(index)
        analysis_method = "vision"
        if analysis is None:
            analysis_method = "ocr"
            try:
                analysis = _ocr_fallback(candidate, image_bytes)
            except Exception as exc:
                if not ocr_warning_reported:
                    logger.warning(
                        "MinerU visual OCR fallback unavailable for %s: %s", filename, exc
                    )
                    ocr_warning_reported = True
        if analysis is None:
            continue
        document = _document_from_analysis(filename, candidate, analysis, analysis_method)
        if document is not None:
            documents.append(document)

    if cache_changed:
        try:
            write_visual_cache(file_path, entries)
        except OSError as exc:
            logger.warning("MinerU visual cache write failed for %s: %s", filename, exc)
    return documents
