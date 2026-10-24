"""MinerU cloud API client and result-bundle download workflow."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from services.core.concurrency import resource_slot
from services.core.settings import get_settings

from .content import content_list_to_page_texts
from .errors import MinerUError

logger = logging.getLogger(__name__)


MAX_RESULT_ZIP_BYTES = 250 * 1024 * 1024
MAX_EXTRACTED_RESULT_BYTES = 350 * 1024 * 1024
# MinerU 精准解析 API 的单文件页数上限；超出时云端会在排队解析后才拒绝。
MAX_DOCUMENT_PAGES = 200
def get_api_token() -> str:
    """读取 MinerU API 令牌；缺失时返回空字符串。"""
    return get_settings().mineru_api_token


def is_configured() -> bool:
    """判断当前环境是否已配置 MinerU API 令牌。"""
    return bool(get_api_token())


def _api_base_url() -> str:
    """读取 MinerU API 地址，并去除末尾斜杠以便拼接路径。"""
    return get_settings().mineru_api_base_url


def _read_api_data(response: requests.Response, action: str) -> dict[str, Any]:
    """校验 MinerU HTTP 响应和业务码，并返回其中的 data 对象。"""
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MinerUError(f"MinerU {action}失败：{exc}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise MinerUError(f"MinerU {action}返回了无效响应") from exc
    if not isinstance(payload, dict) or payload.get("code", 0) != 0:
        message = payload.get("msg") if isinstance(payload, dict) else "未知错误"
        raise MinerUError(f"MinerU {action}失败：{message or '未知错误'}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MinerUError(f"MinerU {action}未返回任务数据")
    return data


def _wait_for_result(
    session: requests.Session,
    headers: dict[str, str],
    batch_id: str,
    data_id: str,
) -> str:
    """轮询 MinerU 批处理任务，完成后返回结果压缩包地址。"""
    settings = get_settings()
    timeout_seconds = settings.mineru_timeout_seconds
    interval_seconds = settings.mineru_poll_interval_seconds
    deadline = time.monotonic() + timeout_seconds
    result_url = f"{_api_base_url()}/extract-results/batch/{batch_id}"

    while time.monotonic() < deadline:
        data = _read_api_data(
            session.get(result_url, headers=headers, timeout=(10, 60)), "任务状态查询"
        )
        tasks = data.get("extract_result", [])
        if not isinstance(tasks, list):
            raise MinerUError("MinerU 任务状态格式无效")
        task = next(
            (
                item
                for item in tasks
                if isinstance(item, dict) and item.get("data_id") in {None, data_id}
            ),
            None,
        )
        if task:
            state = str(task.get("state", "")).lower()
            if state == "done":
                zip_url = task.get("full_zip_url")
                if isinstance(zip_url, str) and zip_url:
                    return zip_url
                raise MinerUError("MinerU 已完成解析，但没有返回结果文件")
            if state == "failed":
                raise MinerUError(f"MinerU 解析失败：{task.get('err_msg') or '未知错误'}")
        time.sleep(interval_seconds)
    raise MinerUError("MinerU 解析超时，请稍后重试或改用本地 Paddle OCR")


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    """在大小和路径校验后解压结果包，防止压缩炸弹与 Zip Slip。"""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        total_size = sum(member.file_size for member in members)
        if total_size > MAX_EXTRACTED_RESULT_BYTES:
            raise MinerUError("MinerU 结果文件过大，已拒绝解压")
        for member in members:
            target = (destination / member.filename).resolve()
            if os.path.commonpath((str(destination_root), str(target))) != str(destination_root):
                raise MinerUError("MinerU 结果包含不安全的文件路径")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _persist_result_bundle(
    archive_path: Path,
    extracted: Path,
    result_dir: Path,
    source_name: str,
    content_list_path: Path,
) -> None:
    """原子地保存完整 MinerU 工件，避免读者看到只写到一半的新结果。"""
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = result_dir.parent / f".{result_dir.name}.{uuid.uuid4().hex}.tmp"
    backup: Path | None = None
    try:
        shutil.copytree(extracted, staging)
        shutil.copy2(archive_path, staging / "result.zip")
        manifest = {
            "source_filename": source_name,
            "parser": "mineru_cloud",
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "content_list": content_list_path.relative_to(extracted).as_posix(),
        }
        (staging / "pymodel_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result_dir.exists():
            backup = result_dir.parent / f".{result_dir.name}.{uuid.uuid4().hex}.backup"
            result_dir.rename(backup)
        staging.rename(result_dir)
        if backup is not None:
            shutil.rmtree(backup)
    except OSError:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not result_dir.exists():
            backup.rename(result_dir)
        raise


def _download_result_archive(session: requests.Session, zip_url: str, archive_path: Path) -> None:
    """下载已完成的 MinerU 结果包，对短暂网络错误进行有限次数重试。"""
    attempts = get_settings().mineru_download_attempts
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        archive_path.unlink(missing_ok=True)
        try:
            # 首次尝试复用传入的 session（性能最优）
            # 重试时创建新 session 避免复用有问题的连接（如代理干扰）
            download_session = session if attempt == 1 else requests.Session()
            with download_session.get(zip_url, stream=True, timeout=(15, 300)) as response:
                response.raise_for_status()
                with archive_path.open("wb") as output:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            downloaded += len(chunk)
                            if downloaded > MAX_RESULT_ZIP_BYTES:
                                raise MinerUError("MinerU 结果文件过大")
            logger.info(
                "MinerU 结果下载成功 (尝试 %d/%d, 大小 %.2f MB)",
                attempt, attempts, downloaded / (1024 * 1024)
            )
            return
        except MinerUError:
            archive_path.unlink(missing_ok=True)
            raise
        except requests.RequestException as exc:
            archive_path.unlink(missing_ok=True)
            last_error = exc
            is_ssl_error = "SSL" in str(exc) or "ssl" in str(type(exc).__name__).lower()

            if attempt < attempts:
                wait_seconds = min(2 ** (attempt - 1), 4)
                logger.warning(
                    "MinerU 结果下载失败 (尝试 %d/%d)%s: %s，%d 秒后重试",
                    attempt, attempts,
                    " [SSL错误]" if is_ssl_error else "",
                    exc,
                    wait_seconds
                )
                time.sleep(wait_seconds)
            else:
                logger.error(
                    "MinerU 结果下载失败，已达最大重试次数 (%d/%d): %s",
                    attempt, attempts, exc
                )

    if last_error:
        raise MinerUError(f"MinerU 结果下载失败（已重试 {attempts} 次）：{last_error}") from last_error


def check_page_limit(file_path: str) -> None:
    """PDF 超过 MinerU 页数上限时，在上传前直接报错，避免白白占用云端解析名额。

    无法读取页数（损坏、加密或非 PDF）时不拦截，交给 MinerU 给出权威结论。
    """
    if Path(file_path).suffix.lower() != ".pdf":
        return
    try:
        from pypdf import PdfReader

        page_count = len(PdfReader(file_path).pages)
    except Exception:
        return
    if page_count > MAX_DOCUMENT_PAGES:
        raise MinerUError(
            f"PDF 共 {page_count} 页，超过 MinerU 单文件 {MAX_DOCUMENT_PAGES} 页的上限，"
            "请拆分文件，或改用原生文本 / Paddle 本地 OCR 解析"
        )


def parse_document(file_path: str, result_dir: str | Path | None = None) -> list[tuple[int, str]]:
    """上传支持的文件到 MinerU，返回非空的“零基页码、文本”列表。

    所有索引任务合计最多同时占用 MINERU_MAX_CONCURRENCY 个云端解析名额，
    超出的任务在此排队，避免多文档并行时触发云端限流。
    """
    check_page_limit(file_path)
    with resource_slot("mineru", get_settings().mineru_max_concurrency):
        return _parse_document(file_path, result_dir)


def _parse_document(file_path: str, result_dir: str | Path | None) -> list[tuple[int, str]]:
    token = get_api_token()
    if not token:
        raise MinerUError("未配置 MINERU_API_TOKEN，无法使用 MinerU 云端解析")

    source = Path(file_path)
    if not source.is_file():
        raise MinerUError("待解析文档不存在")

    data_id = uuid.uuid4().hex
    headers = {"Authorization": f"Bearer {token}"}
    session = requests.Session()
    try:
        request_payload = {
            "files": [{"name": source.name, "data_id": data_id, "is_ocr": True}],
            "model_version": get_settings().mineru_model_version,
            "language": get_settings().mineru_language,
            "enable_formula": get_settings().mineru_enable_formula,
            "enable_table": get_settings().mineru_enable_table,
        }
        data = _read_api_data(
            session.post(
                f"{_api_base_url()}/file-urls/batch",
                headers=headers,
                json=request_payload,
                timeout=(10, 60),
            ),
            "上传地址申请",
        )
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls")
        if not isinstance(batch_id, str) or not batch_id or not isinstance(file_urls, list) or not file_urls:
            raise MinerUError("MinerU 未返回可用的上传地址")
        upload_url = file_urls[0].get("url") if isinstance(file_urls[0], dict) else file_urls[0]
        if not isinstance(upload_url, str) or not upload_url:
            raise MinerUError("MinerU 返回的上传地址无效")

        with source.open("rb") as upload_file:
            try:
                upload_response = session.put(upload_url, data=upload_file, timeout=(10, 300))
                upload_response.raise_for_status()
            except requests.RequestException as exc:
                raise MinerUError(f"MinerU 文件上传失败：{exc}") from exc

        zip_url = _wait_for_result(session, headers, batch_id, data_id)
        with tempfile.TemporaryDirectory(prefix="pymodel-mineru-") as temporary_directory:
            archive_path = Path(temporary_directory) / "result.zip"
            _download_result_archive(session, zip_url, archive_path)

            extracted = Path(temporary_directory) / "extracted"
            try:
                _safe_extract_zip(archive_path, extracted)
            except (OSError, zipfile.BadZipFile) as exc:
                raise MinerUError(f"MinerU 结果解压失败：{exc}") from exc
            content_list_path = next(extracted.rglob("*_content_list_v2.json"), None)
            if content_list_path is None:
                raise MinerUError("MinerU 未返回逐页结构化结果，无法保留引用页码")
            try:
                content_list = json.loads(content_list_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MinerUError("MinerU 结构化结果读取失败") from exc
            pages = content_list_to_page_texts(content_list)
            if not pages:
                raise MinerUError("MinerU 未提取到可索引文本")
            if result_dir is not None:
                try:
                    _persist_result_bundle(
                        archive_path,
                        extracted,
                        Path(result_dir),
                        source.name,
                        content_list_path,
                    )
                except OSError as exc:
                    raise MinerUError(f"MinerU 结果保存失败：{exc}") from exc
            return pages
    finally:
        session.close()
