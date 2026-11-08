#!/usr/bin/env python3
"""Exercise a deployed RAGLoom instance through its public HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    token: str = "",
    payload: object | None = None,
    body: bytes | None = None,
    content_type: str = "application/json",
) -> object:
    """Send one JSON API request and surface useful HTTP error details."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
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
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} returned {exc.code}: {detail}") from exc
    return json.loads(raw) if raw else {}


def multipart_file(filename: str, content: bytes) -> tuple[bytes, str]:
    """Build the small multipart request used by the smoke upload."""
    boundary = f"pymodel-smoke-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/markdown; charset=utf-8\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def wait_for_job(base_url: str, token: str, kb_name: str, job_id: str) -> None:
    """Wait until the deployed indexing worker completes the upload job."""
    encoded_kb = urllib.parse.quote(kb_name, safe="")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        jobs = request_json(base_url, f"/api/kbs/{encoded_kb}/index_jobs", token=token)
        job = next((item for item in jobs if item["id"] == job_id), None)
        if job and job["status"] == "completed":
            return
        if job and job["status"] in {"failed", "superseded", "interrupted"}:
            raise RuntimeError(
                f"index job {job_id} ended as {job['status']}: {job['error_message']}"
            )
        time.sleep(1)
    raise RuntimeError(f"index job {job_id} did not complete within 90 seconds")


def login(base_url: str, username: str, password: str) -> str:
    """Authenticate through the unified plaintext-password login contract."""
    response = request_json(
        base_url,
        "/api/login",
        method="POST",
        payload={
            "username": username,
            "password": password,
        },
    )
    return response["access_token"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default=os.getenv("SMOKE_ADMIN_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("SMOKE_ADMIN_PASSWORD"))
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    if not args.password:
        parser.error("set SMOKE_ADMIN_PASSWORD or pass --password")

    token = login(args.base_url, args.username, args.password)
    request_json(args.base_url, "/api/kbs", token=token)
    request_json(args.base_url, "/api/document_parsers", token=token)
    if args.probe_only:
        print("Deployment probe passed")
        return 0

    suffix = uuid.uuid4().hex[:10]
    kb_name = f"交付冒烟-{suffix}"
    filename = f"smoke-{suffix}.md"
    encoded_kb = urllib.parse.quote(kb_name, safe="")
    created = False
    try:
        request_json(
            args.base_url,
            "/api/kbs",
            method="POST",
            token=token,
            payload={"name": kb_name},
        )
        created = True
        body, content_type = multipart_file(
            filename,
            b"# Deployment smoke test\n\nTemporary upload and indexing verification.\n",
        )
        uploaded = request_json(
            args.base_url,
            f"/api/documents/{encoded_kb}",
            method="POST",
            token=token,
            body=body,
            content_type=content_type,
        )
        wait_for_job(args.base_url, token, kb_name, uploaded["job_id"])
        documents = request_json(args.base_url, f"/api/documents/{encoded_kb}", token=token)
        if not any(
            item["filename"] == filename and item["status"] == "ready"
            for item in documents
        ):
            raise RuntimeError("uploaded document did not become ready")
        request_json(
            args.base_url,
            f"/api/kbs/{encoded_kb}",
            method="DELETE",
            token=token,
        )
        created = False
    finally:
        if created:
            try:
                request_json(
                    args.base_url,
                    f"/api/kbs/{encoded_kb}",
                    method="DELETE",
                    token=token,
                )
            except Exception as exc:
                print(f"WARNING: cleanup failed for {kb_name}: {exc}")
    print("Deployment CRUD smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
