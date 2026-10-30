"""依赖就绪检查：SQLite 能执行查询、索引存储可查询时才算就绪。"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from services.core import sqlite_connections as application_db

logger = logging.getLogger(__name__)

CHECK_TIMEOUT_SECONDS = 2.0


class _ReadyProbe(Protocol):
    def check_ready(self, timeout: float) -> str: ...


@dataclass(frozen=True)
class DependencyStatus:
    """一个外部依赖的检查结果；detail 可能含内部地址，只给有运维权限的人看。"""

    name: str
    ok: bool
    latency_ms: float
    detail: str


def _run(name: str, check: Callable[[], str]) -> DependencyStatus:
    """执行单项检查并计时；异常只记录类型和消息，不向上抛出。"""
    started = time.perf_counter()
    try:
        detail, ok = check(), True
    except Exception as exc:
        detail, ok = f"{type(exc).__name__}: {exc}", False
        logger.warning("依赖 %s 未就绪：%s", name, detail)
    return DependencyStatus(name, ok, round((time.perf_counter() - started) * 1000, 1), detail)


def _check_sqlite(db_file: str) -> str:
    with application_db.connect(db_file, timeout=CHECK_TIMEOUT_SECONDS) as conn:
        conn.execute("SELECT 1 FROM accounts LIMIT 1").fetchall()
    return "ok"


def check_dependencies(db_file: str, index: _ReadyProbe) -> list[DependencyStatus]:
    """依次检查 SQLite 和索引存储，每项最多等待 CHECK_TIMEOUT_SECONDS。"""
    return [
        _run("sqlite", lambda: _check_sqlite(db_file)),
        _run("milvus", lambda: index.check_ready(CHECK_TIMEOUT_SECONDS)),
    ]
