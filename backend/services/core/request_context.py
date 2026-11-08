"""请求 ID：同一请求或索引任务的日志行带同一个 ID，问答的追踪 ID 也取这个值。"""

import contextvars
import logging
import uuid
from collections.abc import Callable

_NO_REQUEST = "-"
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ragloom_request_id", default=_NO_REQUEST
)


def new_request_id() -> str:
    """生成与追踪 ID 同格式的 32 位十六进制 ID。"""
    return uuid.uuid4().hex


def current_request_id() -> str | None:
    """返回当前上下文的请求 ID；不在请求或任务中时返回 None。"""
    value = _request_id.get()
    return None if value == _NO_REQUEST else value


def bind_request_id(request_id: str) -> contextvars.Token[str]:
    """把请求 ID 绑定到当前上下文；asyncio.to_thread 会带着它进入线程。"""
    return _request_id.set(request_id)


def reset_request_id(token: contextvars.Token[str]) -> None:
    """恢复绑定前的请求 ID。"""
    _request_id.reset(token)


def carry_request_id[T](function: Callable[..., T]) -> Callable[..., T]:
    """包装要交给线程池的函数，使它在工作线程中带着提交时的请求 ID 运行。

    ThreadPoolExecutor 不会自动传递 contextvars，不包装时线程里的日志没有请求 ID。
    包装结果可以被多个线程同时调用（例如 executor.map）。
    """
    request_id = _request_id.get()

    def run(*args, **kwargs) -> T:
        token = _request_id.set(request_id)
        try:
            return function(*args, **kwargs)
        finally:
            _request_id.reset(token)

    return run


def install_log_record_factory() -> None:
    """让每条日志记录都带 request_id 属性；可重复调用。"""
    previous = logging.getLogRecordFactory()
    if getattr(previous, "_ragloom_request_id", False):
        return

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.request_id = _request_id.get()
        return record

    factory._ragloom_request_id = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)
