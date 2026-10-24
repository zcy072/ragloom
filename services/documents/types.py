"""文档入库共享的数据契约和支持的文件扩展名。"""

from dataclasses import dataclass

from langchain_core.documents import Document

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".md", ".txt", *IMAGE_EXTENSIONS)


class DocumentLoadError(RuntimeError):
    """用户选定的解析器失败时抛出，由索引任务记录并展示给用户。"""


@dataclass
class DocumentRoutes:
    """已为稳定固定路线和可选结构化路线准备好的内容集合。"""

    fixed: list[Document]
    structured: list[Document]
