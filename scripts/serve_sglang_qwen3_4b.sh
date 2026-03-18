#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/serve_sglang_qwen3_4b.sh
# 可选环境变量：
#   MODEL_NAME_OR_PATH (默认: Qwen/Qwen3-4B-Instruct)
#   HOST               (默认: 0.0.0.0)
#   PORT               (默认: 30000)
#   TP_SIZE            (默认: 1)
#   MEM_FRACTION       (默认: 0.90)

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"

echo "[SGLang] Starting OpenAI-compatible server..."
echo "model=${MODEL_NAME_OR_PATH}, host=${HOST}, port=${PORT}, tp=${TP_SIZE}"

python -m sglang.launch_server \
  --model-path "${MODEL_NAME_OR_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION}"
