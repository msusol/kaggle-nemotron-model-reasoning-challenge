#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server inside the nemotron-vllm-gb10 container.
#
# Usage:
#   bash scripts/run_vllm.sh                                              # default model
#   bash scripts/run_vllm.sh --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
#   bash scripts/run_vllm.sh --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
#
# Server listens on host port 8000. Stop with Ctrl-C or docker stop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
MAX_MODEL_LEN=8192
PORT=8000
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
    --port)           PORT="$2"; shift 2 ;;
    *)                PASSTHROUGH+=("$1"); shift ;;
  esac
done

echo "Starting vLLM server: model=${MODEL}, port=${PORT}"

docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -p "${PORT}:${PORT}" \
  -w /workspace \
  nemotron-vllm-gb10:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --download-dir /workspace/.cache/huggingface \
    "${PASSTHROUGH[@]}"
