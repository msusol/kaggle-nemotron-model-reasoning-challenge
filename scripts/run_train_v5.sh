#!/usr/bin/env zsh
# Run train_v5_sft.py inside the nemotron-gb10 container.
# Warmstarts from huikang v27 adapter, 240 steps, short responses.
#
# Usage (always inside a tmux session):
#   tmux new -s train_v5
#   RUN_NAME=v5_sft bash scripts/run_train_v5.sh
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

# ── stop gnome-remote-desktop (~6 GB GPU freed) ────────────────────────────────
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true

# ── pause non-training containers ──────────────────────────────────────────────
echo "Pausing non-training containers..."
bash "${SCRIPT_DIR}/services.sh" pause || true

# ── GPU pre-flight ─────────────────────────────────────────────────────────────
GPU_MB=$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
         awk '{sum+=$1} END{print sum+0}')
echo "GPU compute-apps usage: ${GPU_MB} MiB"
if [[ "${GPU_MB:-0}" -gt 1024 ]]; then
  echo "ABORT: GPU has ${GPU_MB} MiB held by active processes (> 1 GB) — stop all GPU workloads before training." >&2
  bash "${SCRIPT_DIR}/services.sh" resume || true
  exit 1
fi

# ── drop page cache + VM tuning ────────────────────────────────────────────────
echo "Dropping page cache and tuning VM..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── run training ───────────────────────────────────────────────────────────────
set +e
docker run --rm --privileged \
  --name "nemotron-trainer-v5" \
  --oom-score-adj 300 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
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
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

# ── restore VM defaults ────────────────────────────────────────────────────────
docker run --rm --privileged alpine sh -c \
  'echo 45166 > /proc/sys/vm/min_free_kbytes && echo 100 > /proc/sys/vm/vfs_cache_pressure' \
  2>/dev/null || true

# ── resume services ────────────────────────────────────────────────────────────
echo "Resuming paused containers..."
bash "${SCRIPT_DIR}/services.sh" resume || true
systemctl --user start gnome-remote-desktop.service 2>/dev/null || true

if [[ ${TRAIN_EXIT} -eq 0 ]]; then
  echo "Log saved to ${LOG_FILE}"
  echo "Adapter saved to ${WORKSPACE}/output/adapter_${RUN_NAME}"
else
  echo "Training failed (exit ${TRAIN_EXIT}) — check ${LOG_FILE}"
  exit ${TRAIN_EXIT}
fi
