#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -z "${SMOKE_ADMIN_PASSWORD:-}" ]]; then
    printf 'SMOKE_ADMIN_PASSWORD is required.\n' >&2
    exit 1
fi

curl -fsS http://127.0.0.1:8000/api/health >/dev/null
curl -fsS http://127.0.0.1:8080/ >/dev/null

run_smoke() {
    docker compose run --rm --no-deps \
        -v "$PROJECT_DIR:/app" \
        -e PYTHONPYCACHEPREFIX=/tmp/pymodel-pycache \
        -e SMOKE_ADMIN_PASSWORD \
        -e SMOKE_ADMIN_USERNAME="${SMOKE_ADMIN_USERNAME:-admin}" \
        backend python scripts/deployment_smoke.py \
        --base-url http://backend:8000 "$@"
}

run_smoke

docker compose restart milvus backend frontend
for _ in {1..90}; do
    if curl -fsS http://127.0.0.1:8000/api/health >/dev/null \
        && curl -fsS http://127.0.0.1:8080/ >/dev/null; then
        run_smoke --probe-only
        printf 'Deployment restart smoke test passed.\n'
        exit 0
    fi
    sleep 1
done

printf 'Deployment did not become healthy after restart.\n' >&2
exit 1
