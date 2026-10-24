"""Local paths and persisted result artifacts produced by MinerU."""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from services.core.settings import get_settings

from .errors import MinerUError

logger = logging.getLogger(__name__)


def _data_dir() -> Path:
    """返回当前运行配置中的原始文档根目录。"""
    return Path(get_settings().data_dir)


def _results_root() -> Path:
    """返回可配置的 MinerU 持久化结果根目录。"""
    return Path(get_settings().mineru_results_dir)


def _path_component(value: str) -> str:
    """把输入限制为安全的单级路径名称，并拒绝空值和路径穿越。"""
    name = Path(value).name
    if name in {"", ".", ".."}:
        raise ValueError("无效的 MinerU 结果目录名称")
    return name


def _result_dir_for_filename(parent: Path, filename: str) -> Path:
    """返回文档独立的 MinerU 结果目录。"""
    safe_filename = _path_component(filename)
    return parent / f"{safe_filename}.mineru"


def result_dir_for_document(file_path: str) -> Path:
    """将源文档路径映射到独立的 ``.mineru`` 结果目录。"""
    source = Path(file_path).resolve()
    data_root = _data_dir().resolve()
    try:
        relative = source.relative_to(data_root)
        return _result_dir_for_filename(
            _results_root() / relative.parent, relative.name
        )
    except ValueError:
        return _result_dir_for_filename(_results_root(), source.name)


def result_dir_for_knowledge_base(kb_name: str) -> Path:
    """返回保存指定知识库全部 MinerU 结果的目录。"""
    return _results_root() / _path_component(kb_name)


def read_saved_markdown(file_path: str) -> str:
    """直接读取已持久化的完整 Markdown，不再发起云端请求。"""
    markdown_path = result_dir_for_document(file_path) / "full.md"
    try:
        return markdown_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def read_saved_content_list(file_path: str) -> list[Any]:
    """读取清单指定的结构化 JSON，同时防止路径逃逸。"""
    result_dir = result_dir_for_document(file_path)
    try:
        manifest = json.loads(
            (result_dir / "pymodel_manifest.json").read_text(encoding="utf-8")
        )
        relative_path = (
            manifest.get("content_list", "") if isinstance(manifest, dict) else ""
        )
        if not isinstance(relative_path, str) or not relative_path:
            return []
        content_path = (result_dir / relative_path).resolve()
        if os.path.commonpath((str(result_dir.resolve()), str(content_path))) != str(
            result_dir.resolve()
        ):
            return []
        content_list = json.loads(content_path.read_text(encoding="utf-8"))
        return content_list if isinstance(content_list, list) else []
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def resolve_saved_artifact(file_path: str, relative_path: str) -> Path | None:
    """仅当引用文件位于结果目录内部时返回其安全路径。"""
    if not isinstance(relative_path, str) or not relative_path.strip():
        return None
    result_dir = result_dir_for_document(file_path).resolve()
    try:
        artifact_path = (result_dir / relative_path).resolve()
        if os.path.commonpath((str(result_dir), str(artifact_path))) != str(result_dir):
            return None
        return artifact_path if artifact_path.is_file() else None
    except (OSError, ValueError):
        return None


def read_saved_page_texts(file_path: str) -> list[tuple[int, str]]:
    """重新加载缓存页文本；结果目录不完整时返回空列表。"""
    from .content import content_list_to_page_texts

    try:
        content_list = read_saved_content_list(file_path)
        return content_list_to_page_texts(content_list) if content_list else []
    except MinerUError:
        return []


def remove_saved_result(kb_name: str, filename: str) -> bool:
    """删除单个文档持久化的 MinerU 结果目录。"""
    target = _result_dir_for_filename(
        result_dir_for_knowledge_base(kb_name), filename
    )
    if not target.exists():
        return False
    try:
        shutil.rmtree(target)
        return True
    except OSError as exc:
        logger.warning("MinerU result cleanup failed for %s: %s", filename, exc)
        return False


def remove_saved_results_for_knowledge_base(kb_name: str) -> bool:
    """删除指定知识库拥有的全部 MinerU 结果。"""
    target = result_dir_for_knowledge_base(kb_name)
    if not target.exists():
        return False
    try:
        shutil.rmtree(target)
        return True
    except OSError as exc:
        logger.warning("MinerU result cleanup failed for %s: %s", kb_name, exc)
        return False


def rename_saved_results_for_knowledge_base(old_name: str, new_name: str) -> bool:
    """知识库重命名时迁移其全部持久化 MinerU 结果。"""
    source = result_dir_for_knowledge_base(old_name)
    target = result_dir_for_knowledge_base(new_name)
    if not source.exists():
        return False
    if target.exists():
        raise OSError("目标知识库已存在 MinerU 解析结果目录")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    return True
