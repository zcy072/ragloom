"""Validation and OCR preparation for images attached to chat requests."""

import asyncio
import base64
import binascii
import logging

from services.core.errors import (
    InvalidRequestError,
    PayloadTooLargeError,
    UnsupportedMediaError,
)
from services.documents import ocr

logger = logging.getLogger(__name__)

MAX_CHAT_IMAGE_BYTES = 5 * 1024 * 1024
SUPPORTED_CHAT_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


async def prepare_chat_image(
    image_base64: str | None, image_mime: str | None
) -> tuple[str, str, str]:
    """校验聊天附图，并返回规范化图片数据和 OCR 文本。"""
    if not image_base64:
        return "", "", ""
    mime = image_mime or "image/jpeg"
    if mime not in SUPPORTED_CHAT_IMAGE_MIMES:
        raise UnsupportedMediaError(
            "仅支持 JPG、PNG、WebP 和 GIF 图片", code="unsupported_chat_image"
        )
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise InvalidRequestError(
            "图片数据格式无效", code="invalid_chat_image_data"
        ) from exc
    if len(image_bytes) > MAX_CHAT_IMAGE_BYTES:
        raise PayloadTooLargeError(
            "聊天图片不能超过 5 MB", code="chat_image_too_large"
        )
    try:
        ocr_text = await asyncio.to_thread(ocr.extract_image_text, image_bytes)
    except Exception as exc:
        logger.warning("Chat image OCR failed: %s", exc)
        ocr_text = ""
    return image_base64, mime, ocr_text
