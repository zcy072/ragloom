#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_DIR"

# 浏览器测试默认交给 GitHub CI；本地加 --e2e 才跑（会拉取约 3.7GB 的 Playwright 镜像）。
RUN_E2E=0
for arg in "$@"; do
    case "$arg" in
        --e2e) RUN_E2E=1 ;;
        *) echo "未知参数：$arg（可用：--e2e）" >&2; exit 2 ;;
    esac
done

IMAGES=(backend frontend backend-check frontend-check)
if ((RUN_E2E)); then
    IMAGES+=(e2e)
fi

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
    docker compose build "${IMAGES[@]}"

run_step "Backend static checks" docker compose run --rm --no-deps backend-check

run_step "Backend tests" docker compose run --rm --no-deps \
    -v "$PROJECT_DIR:/repo" -w /repo/backend \
    -e PYTHONPYCACHEPREFIX=/tmp/pymodel-pycache \
    backend python -m unittest discover -s tests -v

run_step "Frontend checks" docker compose run --rm --no-deps frontend-check
if ((RUN_E2E)); then
    run_step "Browser tests" docker compose run --rm --no-deps e2e
else
    printf '\n==> Browser tests skipped (run with --e2e, or see GitHub CI)\n'
fi
run_step "Nginx config" docker compose run --rm --no-deps frontend nginx -t

printf '\nAll local checks passed.\n'
