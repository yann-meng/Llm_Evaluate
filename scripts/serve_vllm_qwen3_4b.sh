#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash scripts/serve_vllm_qwen3_4b.sh
# 可选环境变量：
#   MODEL_NAME_OR_PATH (默认: Qwen/Qwen3-4B-Instruct)
#   HOST               (默认: 0.0.0.0)
#   PORT               (默认: 8000)
#   TENSOR_PARALLEL    (默认: 1)
#   GPU_MEM_UTIL       (默认: 0.90)
#   MAX_MODEL_LEN      (默认: 32768)

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

echo "[vLLM] Starting OpenAI-compatible server..."
echo "model=${MODEL_NAME_OR_PATH}, host=${HOST}, port=${PORT}, tp=${TENSOR_PARALLEL}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME_OR_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}"
