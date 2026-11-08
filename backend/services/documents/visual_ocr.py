"""Local OCR fallback used by visual document analysis."""

from services.documents import mineru as mineru_client
from services.documents.visual_candidates import VisualCandidate
from services.documents.visual_vision import VisualAnalysis

VISION_OCR_MAX_DIMENSION = 2400
VISION_TABLE_SNIPPET_MAX_CHARS = 6000


def ocr_fallback(
    candidate: VisualCandidate, image_bytes: bytes
) -> VisualAnalysis | None:
    """Convert a MinerU visual artifact to minimally useful OCR evidence."""
    from services.documents import ocr

    ocr_text = ocr.extract_image_text(
        image_bytes, max_dimension=VISION_OCR_MAX_DIMENSION
    ).strip()
    if not ocr_text:
        return None
    parts = [f"图片 OCR 文字：\n{ocr_text}"]
    if candidate.block_type == "table" and candidate.html:
        partial_table = mineru_client.content_text(candidate.html)
        if partial_table:
            parts.append(
                "MinerU 已提取的部分表格文字：\n"
                f"{partial_table[:VISION_TABLE_SNIPPET_MAX_CHARS]}"
            )
    return VisualAnalysis(
        indexable=True,
        kind="table" if candidate.block_type == "table" else "image",
        title=candidate.caption,
        content="\n\n".join(parts),
    )


def extract_ocr_text(image_bytes: bytes) -> str:
    """Extract text from an independent image for use beside vision output."""
    from services.documents import ocr

    return ocr.extract_image_text(
        image_bytes, max_dimension=VISION_OCR_MAX_DIMENSION
    ).strip()
