#!/usr/bin/env python3
"""Generate or verify the committed OpenAPI contract snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import api  # noqa: E402

SNAPSHOT_PATH = PROJECT_DIR / "contracts" / "openapi.json"


def serialized_openapi() -> str:
    """Return the application schema in a deterministic representation."""
    return json.dumps(
        api.app.openapi(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed snapshot differs from the application schema",
    )
    args = parser.parse_args()
    generated = serialized_openapi()

    if args.check:
        if not SNAPSHOT_PATH.exists() or SNAPSHOT_PATH.read_text(encoding="utf-8") != generated:
            print(
                "OpenAPI snapshot is stale. Run: "
                "docker compose run --rm --no-deps -v \"$PWD:/app\" "
                "backend python scripts/openapi_snapshot.py"
            )
            return 1
        print(f"OpenAPI snapshot is current: {SNAPSHOT_PATH.relative_to(PROJECT_DIR)}")
        return 0

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(generated, encoding="utf-8")
    print(f"Updated {SNAPSHOT_PATH.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
