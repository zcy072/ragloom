"""worker 启动前清扫没有数据库归属的上传和文档目录。"""

import logging
import os
import shutil
from pathlib import Path

from services.core import sqlite_connections as application_db
from services.core.config import validate_document_id, validate_kb_id
from services.core.settings import get_settings

logger = logging.getLogger(__name__)


def _uuid_directories(root: Path, validate) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        return []
    directories = []
    for entry in os.scandir(root):
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            if validate(entry.name) == entry.name:
                directories.append(Path(entry.path))
        except ValueError:
            continue
    return directories


def _remove(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Startup cleanup failed for %s", path, exc_info=True)


def sweep_unregistered_documents(
    *, db_file: str | None = None, data_dir: str | None = None
) -> None:
    """只清本项目 UUID 文档层级；已登记文档与未完成删除任务的目标均保留。"""
    settings = get_settings()
    with application_db.connect(db_file or settings.db_file) as conn:
        retained = set(conn.execute("SELECT kb_id, document_id FROM documents"))
        retained.update(
            conn.execute(
                "SELECT kb_id, document_id FROM index_jobs WHERE operation = 'delete' "
                "AND status IN ('queued', 'running', 'failed', 'interrupted')"
            )
        )
    uploads = Path(data_dir or settings.data_dir) / ".uploads"
    if uploads.is_dir() and not uploads.is_symlink():
        for entry in os.scandir(uploads):
            # 只处理服务自己生成的十六进制上传 token 目录。
            if (
                len(entry.name) == 32
                and all(c in "0123456789abcdef" for c in entry.name)
                and entry.is_dir(follow_symlinks=False)
            ):
                _remove(Path(entry.path))
    for root in (Path(data_dir or settings.data_dir), Path(settings.mineru_results_dir)):
        for kb_dir in _uuid_directories(root, validate_kb_id):
            for document_dir in _uuid_directories(kb_dir, validate_document_id):
                if (kb_dir.name, document_dir.name) not in retained:
                    _remove(document_dir)
