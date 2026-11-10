"""按来源知识库过滤聊天历史和追踪记录中的结构化证据。

会话和追踪记录的主库权限不能代替每个来源库的权限：部门库的回答可能引用通用库，
用户之后被撤权或来源库被删除时，新请求不应再取得这些库的原文。
网页来源没有 kb_id、带 url，不属于知识库来源，原样保留。
"""

from collections.abc import Callable, Iterable
from typing import Any

KbReadable = Callable[[str], bool]


def kb_reader(
    can_read: KbReadable, is_active: KbReadable
) -> KbReadable:
    """组合权限与状态检查，并缓存同一次响应内的判断结果。"""
    cache: dict[str, bool] = {}

    def readable(kb_id: str) -> bool:
        """知识库必须处于 active 且调用者有读取权限。"""
        if kb_id not in cache:
            cache[kb_id] = bool(is_active(kb_id)) and bool(can_read(kb_id))
        return cache[kb_id]

    return readable


def _visible(item: object, readable: KbReadable) -> bool:
    """判断一条来源记录能否返回；非字典结构原样保留。"""
    if not isinstance(item, dict):
        return True
    kb_id = str(item.get("kb_id") or "")
    if kb_id:
        return readable(kb_id)
    # 没有 kb_id 的记录只有网页来源（带 url）可以保留。
    return bool(item.get("url"))


def _visible_items(items: Iterable[Any], readable: KbReadable) -> list[Any]:
    """保留调用者可读的来源记录。"""
    return [item for item in items if _visible(item, readable)]


def filter_message_sources(sources: list[Any], readable: KbReadable) -> list[Any]:
    """过滤消息的顶层来源及其嵌套 evidence。"""
    result: list[Any] = []
    for source in _visible_items(sources, readable):
        if isinstance(source, dict) and isinstance(source.get("evidence"), list):
            source = {**source, "evidence": _visible_items(source["evidence"], readable)}
        result.append(source)
    return result


_TRACE_SOURCE_LISTS = ("candidates", "reranked", "web_results", "context")


def filter_trace_payload(payload: dict[str, Any], readable: KbReadable) -> dict[str, Any]:
    """过滤追踪记录中候选、重排、网页和上下文里的来源结构。"""
    filtered = dict(payload)
    for key in _TRACE_SOURCE_LISTS:
        value = filtered.get(key)
        if isinstance(value, list):
            filtered[key] = _visible_items(value, readable)
    return filtered
