"""单个索引任务的隔离工作区：源文件快照与任务私有的 MinerU 结果目录。

解析阶段不持有知识库锁，期间同名文件可能被重新上传、删除或重建。
任务因此只读取自己的快照，并把 MinerU 结果、视觉缓存写入任务私有目录；
通过归属校验后，才在持锁阶段把结果目录发布为文档的正式结果。
"""

import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from services.core.settings import get_settings
from services.documents import mineru as mineru_client
from services.documents.visual_cache import (
    STANDALONE_IMAGE_CACHE_FILENAME,
    VISUAL_CACHE_FILENAME,
)

logger = logging.getLogger(__name__)

WORKSPACE_DIRNAME = ".index_jobs"


@dataclass(frozen=True)
class JobWorkspace:
    """一个索引任务使用的快照路径与结果目录。"""

    job_id: str
    source_path: str
    formal_path: str

    @property
    def result_dir(self) -> Path:
        """任务私有的 MinerU 结果目录，由快照路径按既有规则推导。"""
        return mineru_client.result_dir_for_document(self.source_path)

    @property
    def formal_result_dir(self) -> Path:
        """文档正式的 MinerU 结果目录。"""
        return mineru_client.result_dir_for_document(self.formal_path)

    @property
    def _data_job_dir(self) -> Path:
        return Path(self.source_path).parent

    @property
    def _results_job_dir(self) -> Path:
        return self.result_dir.parent

    def publish_result_dir(self) -> None:
        """用任务结果目录替换文档的正式结果目录；调用方须持有知识库锁。

        两个目录位于同一个结果根目录下，重命名是原子的；
        替换失败时恢复旧目录，读取方不会看到只写到一半的结果。
        """
        task_dir = self.result_dir
        if not task_dir.is_dir():
            return
        formal_dir = self.formal_result_dir
        formal_dir.parent.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        try:
            if formal_dir.exists():
                backup = formal_dir.parent / f".{formal_dir.name}.{uuid.uuid4().hex}.backup"
                formal_dir.rename(backup)
            task_dir.rename(formal_dir)
        except OSError:
            if backup is not None and backup.exists() and not formal_dir.exists():
                backup.rename(formal_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)

    def cleanup(self) -> None:
        """删除任务的快照与私有结果目录；失败只记录日志，启动清理会兜底。"""
        for directory in (self._data_job_dir, self._results_job_dir):
            try:
                shutil.rmtree(directory, ignore_errors=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("索引任务工作区清理失败 %s: %s", directory, exc)


def _snapshot_file(source: str, target: str) -> None:
    """优先硬链接；上传总是原子替换文件，所以快照不会被后续写入改动。"""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def create_workspace(
    job_id: str,
    kb_dir: str,
    filename: str,
    *,
    carry_saved_results: bool = False,
) -> JobWorkspace:
    """为任务创建源文件快照，并带入已有的视觉缓存以避免重复调用视觉模型。

    ``carry_saved_results`` 用于重建：把已保存的整个 MinerU 结果目录一并带入，
    解析时直接复用，不再上传云端。调用方须持有知识库锁，保证快照与数据库中
    该文档的当前版本一致。
    """
    formal_path = os.path.join(kb_dir, filename)
    job_dir = os.path.join(kb_dir, WORKSPACE_DIRNAME, job_id)
    workspace = JobWorkspace(
        job_id=job_id,
        source_path=os.path.join(job_dir, filename),
        formal_path=formal_path,
    )
    try:
        os.makedirs(job_dir, exist_ok=True)
        _snapshot_file(formal_path, workspace.source_path)
        if carry_saved_results and workspace.formal_result_dir.is_dir():
            shutil.copytree(
                workspace.formal_result_dir, workspace.result_dir, dirs_exist_ok=True
            )
        else:
            _carry_over_visual_caches(workspace)
    except BaseException:
        workspace.cleanup()
        raise
    return workspace


def _carry_over_visual_caches(workspace: JobWorkspace) -> None:
    """复制正式目录中的视觉缓存；独立图片没有 MinerU 解析，缓存只存在于这里。"""
    formal_dir = workspace.formal_result_dir
    for cache_name in (VISUAL_CACHE_FILENAME, STANDALONE_IMAGE_CACHE_FILENAME):
        cache_file = formal_dir / cache_name
        if not cache_file.is_file():
            continue
        target = workspace.result_dir / cache_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache_file, target)


def sweep_stale_workspaces() -> None:
    """启动时清除上次进程遗留的全部任务工作区；此时没有任何任务在运行。"""
    settings = get_settings()
    for root in (settings.data_dir, settings.mineru_results_dir):
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            stale = os.path.join(entry.path, WORKSPACE_DIRNAME)
            if os.path.isdir(stale):
                shutil.rmtree(stale, ignore_errors=True)
