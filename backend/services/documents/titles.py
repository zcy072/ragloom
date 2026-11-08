"""上下文标题：把源文件名和章节路径拼成检索时使用的标题前缀。"""

import os

CONTEXTUAL_TITLE_MAX_CHARS = 300


def contextual_title(source: str, *parts: str) -> str:
    """拼接文件名和各级标题，忽略空段与重复段，并以 " > " 连接后截断。

    文件名取去掉扩展名的部分。重复段通常来自解析器已写入的标题，去重避免
    "员工手册 > 员工手册 > 总则" 这样的输出。
    """
    title_parts: list[str] = []
    for value in (os.path.splitext(os.path.basename(source))[0], *parts):
        normalized = " ".join(value.split())
        if normalized and normalized not in title_parts:
            title_parts.append(normalized)
    return " > ".join(title_parts)[:CONTEXTUAL_TITLE_MAX_CHARS]
