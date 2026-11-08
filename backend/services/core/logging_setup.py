"""进程日志配置：标准输出之外，再写一份按大小轮转的文件日志。"""

import logging
import os
from logging.handlers import RotatingFileHandler

from services.core.request_context import install_log_record_factory
from services.core.settings import ApplicationSettings

LOG_FILE_NAME = "backend.log"
# 请求 ID 放在级别之后，log_tail 按“时间 级别 ”识别行首；不在请求或任务中时为 "-"。
LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
# uvicorn 的日志器不向根日志器传播，需要单独挂接文件 handler。
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
_APPLICATION_LOGGERS = ("services", "routers", "app_storage")
_FILE_HANDLER_MARKER = "_ragloom_file_handler"

logger = logging.getLogger(__name__)


def log_file_path(settings: ApplicationSettings) -> str:
    """返回文件日志的完整路径。"""
    return os.path.join(settings.log_dir, LOG_FILE_NAME)


def _file_handler(settings: ApplicationSettings) -> RotatingFileHandler | None:
    """创建轮转文件 handler；关闭或目录不可写时返回 None，不影响服务启动。"""
    if settings.log_file_backups <= 0:
        return None
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        handler = RotatingFileHandler(
            log_file_path(settings),
            maxBytes=settings.log_file_max_bytes,
            backupCount=settings.log_file_backups,
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning(
            "File logging disabled, cannot write %s: %s", settings.log_dir, exc
        )
        return None
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    setattr(handler, _FILE_HANDLER_MARKER, True)
    return handler


def _remove_file_handlers(target: logging.Logger) -> None:
    """移除先前由本模块挂接的文件 handler，使重复调用不会重复写入。"""
    for handler in list(target.handlers):
        if getattr(handler, _FILE_HANDLER_MARKER, False):
            target.removeHandler(handler)
            handler.close()


def configure_logging(settings: ApplicationSettings) -> None:
    """配置标准输出和文件日志；可重复调用，以最后一次配置为准。"""
    install_log_record_factory()
    logging.basicConfig(level=settings.log_level, format=LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    for name in _APPLICATION_LOGGERS:
        logging.getLogger(name).setLevel(settings.log_level)
    logging.getLogger("jieba").setLevel(logging.WARNING)

    attached = [root, *(logging.getLogger(name) for name in _UVICORN_LOGGERS)]
    for target in attached:
        _remove_file_handlers(target)
    handler = _file_handler(settings)
    if handler is None:
        return
    # 同一个 handler 只挂一次，轮转时多个日志器共享同一个文件句柄。
    for target in attached:
        if target is root or not target.propagate:
            target.addHandler(handler)
