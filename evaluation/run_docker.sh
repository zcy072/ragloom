#!/usr/bin/env bash
# 在后端镜像里运行离线检索评测。额外参数原样传给 run_eval.py，例如 --limit 5。
# 需要先 export SILICONFLOW_API_KEY。数据和评测存储位于 EVAL_DIR（默认 /tmp/crud_eval_data）。
set -euo pipefail

if [ -z "${SILICONFLOW_API_KEY:-}" ]; then
  echo "请先执行：read -rs SILICONFLOW_API_KEY; export SILICONFLOW_API_KEY" >&2
  exit 1
fi

EVAL_DIR="${EVAL_DIR:-/tmp/crud_eval_data}"
if [ ! -d "$EVAL_DIR/set" ]; then
  echo "找不到评测数据：$EVAL_DIR/set，请先按 evaluation/README.md 第 1 步生成。" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
exec docker compose run --rm --no-deps \
  -v "$PWD:/app" \
  -v "$EVAL_DIR:/eval" \
  -e SILICONFLOW_API_KEY \
  -e PYTHONPYCACHEPREFIX=/tmp/pymodel-pycache \
  backend python evaluation/run_eval.py \
  --data /eval/set --storage /eval/storage --output /eval/report.json "$@"
