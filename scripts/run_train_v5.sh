#!/usr/bin/env zsh
# Run train_v5_sft.py inside the nemotron-gb10 container.
# Warmstarts from huikang v27 adapter, 240 steps, short responses.
#
# Usage:
#   bash scripts/run_train_v5.sh
#   RUN_NAME=v5_test bash scripts/run_train_v5.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

RUN_NAME="${RUN_NAME:-v5_$(date +%Y%m%d_%H%M%S)}"
ADAPTER_OUT="/workspace/output/adapter_${RUN_NAME}"
LOG_FILE="${WORKSPACE}/output/train_${RUN_NAME}.log"

echo "RUN_NAME:    ${RUN_NAME}"
echo "Adapter out: ${ADAPTER_OUT}"
echo "Log:         ${LOG_FILE}"

# ── GPU pre-flight ─────────────────────────────────────────────────────────────
GPU_MB=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | \
         awk '{s+=$1} END{print s+0}')
if [[ "${GPU_MB}" -gt 1024 ]]; then
  echo "ABORT: GPU has ${GPU_MB} MiB held by active processes (> 1 GB) — stop all GPU workloads before training." >&2
  exit 1
fi
echo "GPU pre-flight OK (${GPU_MB} MiB in use)"

# ── run training ───────────────────────────────────────────────────────────────
docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/huggingface":/home/ubuntu/.cache/huggingface \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/train_v5_sft.py \
    --warmstart-dir /workspace/output/adapter_huikang_v27 \
    --train-file    /workspace/data/v0.5_train.jsonl \
    --output-dir    "${ADAPTER_OUT}" \
    --model-id      nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --max-steps     240 \
    --learning-rate 2e-4 \
    --max-seq-length 6144 \
    --batch-size    1 \
    --grad-accum    16 \
    --seed          3407 \
  2>&1 | tee "${LOG_FILE}"
