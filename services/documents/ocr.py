"""延迟初始化 PaddleOCR，并在内存中提取图片文字。"""

import io
import logging
import threading

from PIL import Image

from services.core.concurrency import resource_slot
from services.core.settings import get_settings

_ocr_engine = None
_ocr_init_lock = threading.Lock()


def get_ocr_engine():
    """按需初始化适合 CPU 环境的中文 PaddleOCR 引擎；并发调用只初始化一次。"""
    global _ocr_engine
    if _ocr_engine is None:
        with _ocr_init_lock:
            if _ocr_engine is None:
                try:
                    from paddleocr import PaddleOCR

                    logging.getLogger("ppocr").setLevel(logging.ERROR)
                    _ocr_engine = PaddleOCR(
                        use_angle_cls=False,
                        lang="ch",
                        ocr_version="PP-OCRv4",
                        enable_mkldnn=False,
                    )
                except ImportError:
                    _ocr_engine = False
    return _ocr_engine


def _extract_ocr_text(ocr_result) -> str:
    """从 PaddleOCR 3.x 的结果对象中取出 rec_texts，按返回顺序拼成文本。"""
    texts: list[str] = []
    for item in ocr_result or []:
        if item and "rec_texts" in item:
            texts.extend(item["rec_texts"])
    return "\n".join(texts)


# 约 600 dpi 的 A4 扫描件（3500 万像素）以内；RGB 解码后最多约 120 MB。
OCR_IMAGE_MAX_PIXELS = 40_000_000


def ocr_image(image: Image.Image, ocr, *, max_dimension: int = 1280) -> str:
    """限制图片最大边长后执行 OCR，避免超大图片占用过多内存和计算时间。"""
    import numpy as np

    # 打开图片只读取头部，在解码像素前拒绝体积小但分辨率极大的图片。
    width, height = image.size
    if width * height > OCR_IMAGE_MAX_PIXELS:
        raise ValueError(f"图片分辨率过大（{width}×{height}），无法识别")
    image = image.convert("RGB")
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    # PaddleOCR 引擎是否线程安全未经验证，默认同一时刻只允许一次推理。
    with resource_slot("ocr", get_settings().ocr_max_concurrency):
        result = ocr.ocr(np.array(image))
    return _extract_ocr_text(result)


def extract_image_text(image_bytes: bytes, *, max_dimension: int = 1280) -> str:
    """对内存中的图片字节执行 OCR，无需写入临时源文件。"""
    ocr = get_ocr_engine()
    if not ocr:
        raise RuntimeError("PaddleOCR is unavailable")
    with Image.open(io.BytesIO(image_bytes)) as image:
        return ocr_image(image, ocr, max_dimension=max_dimension)
