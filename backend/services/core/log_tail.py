"""读取日志文件末尾若干行，不整文件载入内存。"""

import os
import re
from collections.abc import Iterator

_BLOCK_BYTES = 8192
# 带级别过滤时最多向前扫描的字节数，避免在没有匹配行的大文件上读完整个文件。
SCAN_LIMIT_BYTES = 8 * 1024 * 1024
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
# 日志行格式为 `日期 时间,毫秒 级别 名称: 消息`；堆栈等续行不匹配该前缀。
_HEADER = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,.]\d+ (\w+) ")


def _reverse_lines(path: str) -> Iterator[str]:
    """从文件尾部按块反向读取，最新的行先产出。"""
    with open(path, "rb") as handle:
        position = handle.seek(0, os.SEEK_END)
        scanned = 0
        carry = b""
        first = True
        while position > 0 and scanned < SCAN_LIMIT_BYTES:
            step = min(_BLOCK_BYTES, position)
            position -= step
            scanned += step
            handle.seek(position)
            parts = (handle.read(step) + carry).split(b"\n")
            # 第一段可能被块边界截断，留到下一轮拼接；已到文件开头则它就是完整行。
            carry = parts[0] if position > 0 else b""
            complete = parts[1:] if position > 0 else parts
            for raw in reversed(complete):
                # 文件以换行结尾时，最后一段是空串，并不是一行日志。
                if first:
                    first = False
                    if not raw:
                        continue
                yield raw.decode("utf-8", errors="replace").rstrip("\r")


def tail_lines(path: str, count: int, level: str | None = None) -> list[str]:
    """返回最后 `count` 行（旧的在前）。

    指定级别时只保留该级别的日志行，连同紧随其后的堆栈续行，行数仍以 `count` 为上限。
    """
    if count < 1:
        return []
    wanted = level.upper() if level else None
    collected: list[str] = []
    continuation: list[str] = []
    for line in _reverse_lines(path):
        if wanted is None:
            collected.append(line)
            if len(collected) >= count:
                break
            continue
        header = _HEADER.match(line)
        if header is None:
            continuation.append(line)
            continue
        if header.group(1) == wanted:
            # 反向遍历：续行先于所属的日志行进入列表，最终整体翻转后顺序正确。
            collected.extend([*continuation, line])
            if len(collected) >= count:
                break
        continuation = []
    result = collected[:count] if wanted else collected
    result.reverse()
    return result
