#!/usr/bin/env python3
"""Exercise a deployed RAGLoom instance through its public HTTP API."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

# 会话 Cookie 只保存在本进程内存中，不落盘；写请求必须带固定请求头（CSRF 防护）。
WRITE_HEADERS = {"X-RAGLoom-Request": "1"}


def new_session() -> urllib.request.OpenerDirector:
    """创建一个像浏览器一样保存会话 Cookie 的内存客户端。"""
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
    )


def request_json(
    session: urllib.request.OpenerDirector,
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: object | None = None,
    body: bytes | None = None,
    content_type: str = "application/json",
) -> object:
    """Send one JSON API request and surface useful HTTP error details."""
    headers = {"Accept": "application/json"}
    if method not in {"GET", "HEAD", "OPTIONS"}:
        headers.update(WRITE_HEADERS)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode()
    if body is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with session.open(request, timeout=20) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} returned {exc.code}: {detail}") from exc
    return json.loads(raw) if raw else {}


def multipart_file(filename: str, content: bytes) -> tuple[bytes, str]:
    """Build the small multipart request used by the smoke upload."""
    boundary = f"pymodel-smoke-{uuid.uuid4().hex}"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: text/markdown; charset=utf-8\r\n\r\n"
        ).encode()
        + content
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def wait_for_job(
    session: urllib.request.OpenerDirector, base_url: str, kb_id: str, job_id: str
) -> None:
    """Wait until the deployed indexing worker completes the upload job."""
    encoded_kb = urllib.parse.quote(kb_id, safe="")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        jobs = request_json(session, base_url, f"/api/kbs/{encoded_kb}/index_jobs")
        job = next((item for item in jobs if item["id"] == job_id), None)
        if job and job["status"] == "completed":
            return
        if job and job["status"] in {"failed", "superseded", "interrupted"}:
            raise RuntimeError(
                f"index job {job_id} ended as {job['status']}: {job['error_message']}"
            )
        time.sleep(1)
    raise RuntimeError(f"index job {job_id} did not complete within 90 seconds")


def login(
    session: urllib.request.OpenerDirector, base_url: str, username: str, password: str
) -> None:
    """Authenticate through the unified plaintext-password login contract.

    成功后服务端通过 Set-Cookie 下发会话，由 session 自动保存并在后续请求携带。
    """
    response = request_json(
        session,
        base_url,
        "/api/login",
        method="POST",
        payload={
            "username": username,
            "password": password,
        },
    )
    if not isinstance(response, dict) or response.get("status") != "success":
        raise RuntimeError("login did not succeed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default=os.getenv("SMOKE_ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("SMOKE_ADMIN_PASSWORD"))
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if not args.password:
        parser.error("set SMOKE_ADMIN_PASSWORD or pass --password")

    session = new_session()
    login(session, args.base_url, args.username, args.password)
    request_json(session, args.base_url, "/api/kbs")
    request_json(session, args.base_url, "/api/document_parsers")
    if args.probe_only:
        print("Deployment probe passed")
        return 0

    suffix = uuid.uuid4().hex[:10]
    kb_name = f"交付冒烟-{suffix}"
    filename = f"smoke-{suffix}.md"
    encoded_kb = ""
    created = False
    try:
        created_kb = request_json(
            session,
            args.base_url,
            "/api/kbs",
            method="POST",
            payload={"name": kb_name},
        )
        created = True
        encoded_kb = created_kb["kb_id"]
        body, content_type = multipart_file(
            filename,
            b"# Deployment smoke test\n\nTemporary upload and indexing verification.\n",
        )
        uploaded = request_json(
            session,
            args.base_url,
            f"/api/documents/{encoded_kb}",
            method="POST",
            body=body,
            content_type=content_type,
        )
        wait_for_job(session, args.base_url, encoded_kb, uploaded["job_id"])
        documents = request_json(session, args.base_url, f"/api/documents/{encoded_kb}")
        if not any(
            item["document_id"] == uploaded["document_id"] and item["status"] == "ready"
            for item in documents
        ):
            raise RuntimeError("uploaded document did not become ready")
        request_json(session, args.base_url, f"/api/kbs/{encoded_kb}", method="DELETE")
        created = False
    finally:
        if created:
            try:
                request_json(
                    session, args.base_url, f"/api/kbs/{encoded_kb}", method="DELETE"
                )
            except Exception as exc:
                print(f"WARNING: cleanup failed for {kb_name}: {exc}")
    print("Deployment CRUD smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
