"""Dependency-free parser-mode contract shared by API, storage, and ingestion."""

from __future__ import annotations

import os

PDF_PARSER_NATIVE_TEXT = "native_text"
PDF_PARSER_MINERU_CLOUD = "mineru_cloud"
PDF_PARSER_PADDLE_OCR = "paddle_ocr"
DEFAULT_PDF_PARSER_MODE = PDF_PARSER_NATIVE_TEXT
IMAGE_PARSER_VISION_MODEL = "vision_model"
DEFAULT_IMAGE_PARSER_MODE = IMAGE_PARSER_VISION_MODEL

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})

PDF_PARSER_MODES = (
    PDF_PARSER_NATIVE_TEXT,
    PDF_PARSER_MINERU_CLOUD,
    PDF_PARSER_PADDLE_OCR,
)

PDF_PARSER_LABELS = {
    PDF_PARSER_NATIVE_TEXT: "本地文本解析",
    PDF_PARSER_MINERU_CLOUD: "MinerU 云端解析",
    PDF_PARSER_PADDLE_OCR: "Paddle 本地 OCR",
}

IMAGE_PARSER_LABELS = {
    IMAGE_PARSER_VISION_MODEL: "智能视觉解析（失败时 Paddle OCR）",
    PDF_PARSER_PADDLE_OCR: "Paddle 本地 OCR",
}


def is_pdf_filename(filename: str) -> bool:
    """判断文件名是否为需要 PDF 解析策略的 PDF 文件。"""
    return os.path.splitext(filename or "")[1].lower() == ".pdf"


def is_docx_filename(filename: str) -> bool:
    """判断文件名是否为需要 DOCX 解析策略的 Word 文件。"""
    return os.path.splitext(filename or "")[1].lower() == ".docx"


def is_image_filename(filename: str) -> bool:
    """判断文件名是否为支持单独选择解析方式的知识库图片。"""
    return os.path.splitext(filename or "")[1].lower() in IMAGE_EXTENSIONS


def normalize_pdf_parser_mode(value: str | None) -> str:
    """校验 PDF 解析方式；空值时默认使用本地文本解析。"""
    if value is None:
        return DEFAULT_PDF_PARSER_MODE
    if not isinstance(value, str):
        raise ValueError("文档解析方式必须是文本")
    mode = value.strip().lower()
    if not mode:
        return DEFAULT_PDF_PARSER_MODE
    if mode not in PDF_PARSER_MODES:
        raise ValueError("未知的文档解析方式")
    return mode


def normalize_document_parser_mode(filename: str, value: str | None) -> str:
    """按文件类型校验解析方式，并兼容旧图片记录的本地 OCR 默认值。"""
    if is_image_filename(filename):
        if value is None:
            return DEFAULT_IMAGE_PARSER_MODE
        if not isinstance(value, str):
            raise ValueError("文档解析方式必须是文本")
        mode = value.strip().lower()
        # 历史版本把所有非 PDF/DOCX 都记录为 native_text；保留其本地处理边界。
        if not mode or mode == PDF_PARSER_NATIVE_TEXT:
            return PDF_PARSER_PADDLE_OCR
        if mode not in IMAGE_PARSER_LABELS:
            raise ValueError("未知的图片解析方式")
        return mode
    if not (is_pdf_filename(filename) or is_docx_filename(filename)):
        return DEFAULT_PDF_PARSER_MODE
    mode = normalize_pdf_parser_mode(value)
    if is_docx_filename(filename) and mode == PDF_PARSER_PADDLE_OCR:
        return DEFAULT_PDF_PARSER_MODE
    return mode
