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
    """兼容旧版与 3.x 的 OCR 返回格式，整理为阅读顺序的文本。"""
    texts = []
    if hasattr(ocr_result, "__iter__") and not isinstance(ocr_result, (list, dict)):
        try:
            items = list(ocr_result)
        except Exception:
            items = []
    else:
        items = ocr_result if isinstance(ocr_result, list) else [ocr_result]

    for item in items:
        if not item:
            continue
        try:
            if "rec_texts" in item:
                texts.extend(item["rec_texts"])
                continue
            if "rec_text" in item:
                texts.extend(item["rec_text"])
                continue
        except Exception:
            pass

        # PaddleOCR 3.x 使用 rec_texts 字典，旧版返回坐标和文字的嵌套行；在此统一兼容。
        if not isinstance(item, list):
            continue
        positioned_text = []
        for line in item:
            if not (isinstance(line, (list, tuple)) and len(line) >= 2):
                continue
            if not isinstance(line[1], (list, tuple)):
                continue
            try:
                center_y = sum(point[1] for point in line[0]) / 4
                positioned_text.append((center_y, line[1][0]))
            except Exception:
                texts.append(line[1][0])

        # 旧版结果可能乱序，按纵向中心点排序并合并相邻文本，恢复阅读行。
        positioned_text.sort(key=lambda value: value[0])
        current_line = []
        last_y = None
        for center_y, text in positioned_text:
            if last_y is None or abs(center_y - last_y) < 20:
                current_line.append(text)
            else:
                texts.append("".join(current_line))
                current_line = [text]
            last_y = center_y
        if current_line:
            texts.append("".join(current_line))
    return "\n".join(texts)


def ocr_image(image: Image.Image, ocr, *, max_dimension: int = 1280) -> str:
    """限制图片最大边长后执行 OCR，避免超大图片占用过多内存和计算时间。"""
    import numpy as np

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
