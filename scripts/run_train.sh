#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

# shellcheck source=scripts/load_config.sh
source "${SCRIPT_DIR}/load_config.sh"

FORCE_PREPARE=false
USE_4BIT_FLAG=""
for arg in "$@"; do
  [[ "$arg" == "--force-prepare" ]] && FORCE_PREPARE=true
done
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

# Quantized model cache: run scripts/run_prepare.sh once to build .cache/nemotron_4bit.
# Auto-prepare is intentionally disabled here — running prepare + train in sequence
# doubles memory pressure and causes both to be killed. Run prepare separately first.
if [[ "${FORCE_PREPARE}" == "true" ]]; then
  echo "Force-prepare requested — clearing cache and re-quantizing..."
  rm -rf "${WORKSPACE}/.cache/nemotron_4bit"
  bash "${SCRIPT_DIR}/run_prepare.sh"
fi

# Pre-loading memory setup: drop page cache and raise min_free_kbytes.
#
# Root cause of loading OOM: default min_free_kbytes=45166 (44 MB) lets page
# cache grow unchecked. During safetensors mmap loading, shard pages accumulate
# in page cache while GPU tensors grow simultaneously — both in the same 128 GB
# unified pool. At 100% loaded: ~57 GB GPU + ~57 GB page cache = ~114 GB, leaving
# only ~14 GB headroom. CUDA context overhead or fragmentation crosses the limit.
#
# Fix: set min_free_kbytes=10 GB. The kernel then reclaims old shard pages as new
# GPU tensors are allocated, capping combined usage at ~117 GB (128 - 10 reserve).
# Swap is cleared separately because swapoff needs the host swap file path, which
# does not exist inside the alpine container's filesystem.
sync
docker run --rm --privileged alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches && echo 10485760 > /proc/sys/vm/min_free_kbytes' \
  2>/dev/null \
  || sudo -n sh -c \
  'echo 3 > /proc/sys/vm/drop_caches && echo 10485760 > /proc/sys/vm/min_free_kbytes' \
  2>/dev/null \
  || true
# swapoff needs the host filesystem (/swap.img) — run via host sudo, not inside container.
sudo -n sh -c 'swapoff -a && swapon -a' 2>/dev/null || true

ionice -c 2 -n 7 docker run --rm --privileged \
  --name "nemotron-trainer" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/huggingface":/home/ubuntu/.cache/huggingface \
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
  2>&1 | tee "${LOG_FILE}" || true

# Restore min_free_kbytes to system default after training.
docker run --rm --privileged alpine sh -c \
  'echo 45166 > /proc/sys/vm/min_free_kbytes' 2>/dev/null \
  || sudo -n sh -c 'echo 45166 > /proc/sys/vm/min_free_kbytes' 2>/dev/null \
  || true

echo "Log saved to ${LOG_FILE}"
echo "Adapter saved to ${ADAPTER_DIR}"
