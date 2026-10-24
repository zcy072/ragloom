#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

run_step() {
    printf '\n==> %s\n' "$1"
    shift
    "$@"
}

run_step_with_retries() {
    local label="$1"
    local attempts="$2"
    local attempt
    shift 2
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        printf '\n==> %s (attempt %d/%d)\n' "$label" "$attempt" "$attempts"
        if "$@"; then
            return 0
        fi
        if ((attempt < attempts)); then
            sleep $((attempt * 3))
        fi
    done
    return 1
}

run_step_with_retries "Docker images" 3 \
    docker compose build backend frontend backend-check frontend-check e2e

run_step "Backend static checks" docker compose run --rm --no-deps backend-check

run_step "Backend tests" docker compose run --rm --no-deps \
    -v "$PROJECT_DIR:/app" \
    -e PYTHONPYCACHEPREFIX=/tmp/pymodel-pycache \
    backend python -m unittest discover -s tests -v

run_step "OpenAPI snapshot" docker compose run --rm --no-deps \
    -v "$PROJECT_DIR:/app" \
    -e PYTHONPYCACHEPREFIX=/tmp/pymodel-pycache \
    backend \
    python scripts/openapi_snapshot.py --check

run_step "Frontend checks" docker compose run --rm --no-deps frontend-check
run_step "Browser tests" docker compose run --rm --no-deps e2e
run_step "Nginx config" docker compose run --rm --no-deps frontend nginx -t

printf '\nAll local checks passed.\n'
