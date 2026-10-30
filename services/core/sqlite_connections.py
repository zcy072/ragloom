"""应用数据库的连接工厂。

`sqlite3.connect` 作为上下文管理器只提交或回滚事务，并不关闭连接——连接要等到
引用计数归零才释放。本模块的连接工厂保持原有提交/回滚语义，同时确保连接关闭。

关闭时机在 WAL 模式下有实际影响：SQLite 只在最后一个连接关闭时自动执行检查点，
残留的未关闭连接会让 `-wal` 一直留着旧页面内容。
"""

import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

# 请求线程与索引 worker 并发写同一个文件，VACUUM 还会短暂加排他锁。sqlite3 默认等锁
# 5 秒，这里放宽到 15 秒，让压缩期间的写入等待而不是直接失败。
LOCK_TIMEOUT_SECONDS = 15.0


@contextmanager
def _managed_connection(
    db_file: str, timeout: float
) -> Iterator[sqlite3.Connection]:
    """提供 SQLite 提交、回滚和确定性关闭的共享实现。"""
    conn = sqlite3.connect(db_file, timeout=timeout)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def connect(
    db_file: str, *, timeout: float = LOCK_TIMEOUT_SECONDS
) -> AbstractContextManager[sqlite3.Connection]:
    """连接主数据库；WAL 仍由 schema 初始化阶段单独启用。"""
    return _managed_connection(db_file, timeout)


def enable_write_ahead_logging(conn: sqlite3.Connection) -> str:
    """把主数据库切换为 WAL，使读取不再被写入阻塞。

    WAL 是持久化设置，只需在启动时执行一次。返回生效后的日志模式，便于启动日志记录
    实际结果——只读挂载和部分网络文件系统会拒绝切换并保留原模式。
    """
    row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
    return str(row[0]) if row else ""


def optional_text(value: object) -> str | None:
    """Map a nullable SQLite text column without turning NULL into a string."""
    return None if value is None else str(value)
