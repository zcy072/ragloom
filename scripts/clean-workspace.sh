#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Only remove dependencies, caches, and build outputs that are reproducible from tracked files.
generated_paths=(
    "$PROJECT_DIR/.venv"
    "$PROJECT_DIR/.uv-cache"
    "$PROJECT_DIR/.uv-tools"
    "$PROJECT_DIR/.ruff_cache"
    "$PROJECT_DIR/.pytest_cache"
    "$PROJECT_DIR/.mypy_cache"
    "$PROJECT_DIR/.coverage"
    "$PROJECT_DIR/coverage"
    "$PROJECT_DIR/htmlcov"
    "$PROJECT_DIR/__pycache__"
    "$PROJECT_DIR/frontend/node_modules"
    "$PROJECT_DIR/frontend/dist"
    "$PROJECT_DIR/frontend/playwright-report"
    "$PROJECT_DIR/frontend/test-results"
    "$PROJECT_DIR/playwright-report"
    "$PROJECT_DIR/test-results"
)

for path in "${generated_paths[@]}"; do
    if [[ -e "$path" ]]; then
        rm -rf -- "$path"
        printf 'Removed %s\n' "${path#"$PROJECT_DIR/"}"
    fi
done

for source_dir in routers services tests evaluation scripts; do
    path="$PROJECT_DIR/$source_dir"
    if [[ -d "$path" ]]; then
        find "$path" -type d -name __pycache__ -prune -exec rm -rf -- {} +
    fi
done

printf 'Workspace-generated dependencies, caches, and build outputs are clean.\n'
printf 'Preserved storage/, .env, source files, lock files, and Docker data.\n'
