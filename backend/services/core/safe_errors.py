"""Describe exceptions from model and other upstream calls without their raw text.

上游异常正文可能回显请求内容或凭证，截断不等于脱敏。日志、追踪和运行记录只使用这里
给出的允许字段：异常类别、HTTP 状态码、应用错误码和上游请求 ID。
"""

import re

from services.core.errors import ApplicationError

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def exception_status_code(exc: BaseException) -> int | None:
    """取出异常携带的 HTTP 状态码（SDK 或 httpx 异常）；没有时返回 None。"""
    for candidate in (exc, getattr(exc, "response", None)):
        status = getattr(candidate, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def safe_error_summary(exc: BaseException) -> str:
    """只用允许的字段描述异常：类别、HTTP 状态、应用错误码和上游请求 ID。

    上游异常正文可能回显请求内容或凭证，截断不等于脱敏，因此任何情况下都不拼接异常消息。
    """
    parts: list[str] = []
    if isinstance(exc, ApplicationError):
        parts.append(f"code={exc.code}")
    status = exception_status_code(exc)
    if status is not None:
        parts.append(f"HTTP {status}")
    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str) and _SAFE_REQUEST_ID.fullmatch(request_id):
        parts.append(f"request_id={request_id}")
    name = type(exc).__name__
    return f"{name} ({', '.join(parts)})" if parts else name
