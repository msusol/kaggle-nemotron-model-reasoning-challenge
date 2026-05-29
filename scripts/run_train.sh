#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

# shellcheck source=scripts/load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

USE_4BIT_FLAG=""
if [[ "${USE_4BIT:-false}" == "true" ]]; then
  USE_4BIT_FLAG="--use-4bit"
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-}"
RUN_SUFFIX="${RUN_NAME:+_${RUN_NAME}}_${RUN_TS}"
LOG_FILE="${WORKSPACE}/output/train${RUN_SUFFIX}.log"
ADAPTER_DIR="${WORKSPACE}/output/adapter${RUN_SUFFIX}"
CONTAINER_ADAPTER_DIR="/workspace/output/adapter${RUN_SUFFIX}"
mkdir -p "${WORKSPACE}/output"

docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/train_lora.py \
    --model-id "${BASE_MODEL}" \
    --train-file "${TRAIN_FILE}" \
    --valid-file "${VALID_FILE}" \
    --output-dir "${CONTAINER_ADAPTER_DIR}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --learning-rate "${LEARNING_RATE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --warmup-ratio "${WARMUP_RATIO:-0.03}" \
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-0}" \
    ${USE_4BIT_FLAG} \
  2>&1 | tee "${LOG_FILE}"
echo "Log saved to ${LOG_FILE}"
echo "Adapter saved to ${ADAPTER_DIR}"
