#!/usr/bin/env zsh
# Run train_v9_sft.py for v0.12 inside the nemotron-gb10 container.
# Warmstarts from run13 step-1000; trains on 25,500-row augmented dataset.
#
# Usage (always inside a tmux session):
#   tmux new -s train_v12
#   RUN_NAME=v12_spark \
#   TRAIN_FILE=data/v0.12_train.jsonl \
#   WARMSTART_ADAPTER=output/adapter_v9_run13_ckpt \
#   MAX_SEQ_LENGTH=8192 \
#   MAX_STEPS=600 \
#   LEARNING_RATE=1e-4 \
#   bash scripts/run_train_v12.sh
#
# The project directory is bind-mounted as /workspace inside the container.
# All relative paths (output/, data/, scripts/) resolve to /workspace/<path>.
# Checkpoints land in output/adapter_<RUN_NAME>/ on the host automatically.
#
# MAX_SEQ_LENGTH=4096 (default) — run13 used 2048; v0.4 ran 8192 with 27M params but
# with 878M trainable params the backward pass needs ~4× more activation memory at 8192.
# Start at 4096; if no OOM within the first 10 steps, can try 8192 next run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

RUN_NAME="${RUN_NAME:-v12_$(date +%Y%m%d_%H%M%S)}"
TRAIN_FILE="${TRAIN_FILE:-data/v0.12_train.jsonl}"
WARMSTART_ADAPTER="${WARMSTART_ADAPTER:-output/adapter_v9_run13_ckpt}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MIN_SEQ_LENGTH="${MIN_SEQ_LENGTH:-0}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:-600}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ADAPTER_OUT="/workspace/output/adapter_${RUN_NAME}"
LOG_FILE="${WORKSPACE}/output/train_${RUN_NAME}.log"
mkdir -p "${WORKSPACE}/output"

echo "RUN_NAME:         ${RUN_NAME}"
echo "TRAIN_FILE:       ${TRAIN_FILE}"
echo "WARMSTART:        ${WARMSTART_ADAPTER}"
echo "RESUME_CKPT:      ${RESUME_FROM_CHECKPOINT:-none}"
echo "MIN_SEQ_LENGTH:   ${MIN_SEQ_LENGTH}"
echo "MAX_SEQ_LENGTH:   ${MAX_SEQ_LENGTH}"
echo "MAX_STEPS:        ${MAX_STEPS}"
echo "LEARNING_RATE:    ${LEARNING_RATE}"
echo "Adapter out:      ${ADAPTER_OUT}"
echo "Log:              ${LOG_FILE}"

# ── stop gnome-remote-desktop (~6 GB GPU freed) ────────────────────────────────
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true

# ── pause non-training containers ──────────────────────────────────────────────
echo "Pausing non-training containers..."
bash "${SCRIPT_DIR}/services.sh" pause || true

# ── GPU pre-flight stage 1: compute-apps check ────────────────────────────────
GPU_MB=$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
         awk '{sum+=$1} END{print sum+0}')
echo "GPU compute-apps usage: ${GPU_MB} MiB"
if [[ "${GPU_MB:-0}" -gt 1024 ]]; then
  echo "ABORT: GPU has ${GPU_MB} MiB held by active processes (> 1 GB) — stop all GPU workloads before training." >&2
  bash "${SCRIPT_DIR}/services.sh" resume || true
  exit 1
fi

# ── remove stale container from previous killed run ───────────────────────────
docker rm -f nemotron-trainer-v12 2>/dev/null || true

# ── drop page cache + VM tuning (pass 1 — before preflight) ───────────────────
echo "Dropping page cache and tuning VM (pass 1)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── GPU pre-flight stage 2: torch memory check ────────────────────────────────
_run_preflight() {
  docker run --rm --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    nemotron-gb10:latest \
    python -c "
import torch
torch.cuda.init()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
used = total - free
print(f'GPU free={free/1e9:.1f}GB total={total/1e9:.1f}GB used={used/1e9:.1f}GB')
if free < 70e9:
    print(f'PREFLIGHT_FAIL only {free/1e9:.1f}GB free — model load (60 GB) will OOM')
elif used > 20e9:
    print(f'PREFLIGHT_FAIL {used/1e9:.1f}GB stale GPU allocations — nvidia_uvm reload may help')
elif free < 90e9:
    print(f'WARNING: {free/1e9:.1f}GB free — some stale allocs present but training should fit')
" 2>&1
}

echo "GPU pre-flight (torch)..."
_PREFLIGHT=$(_run_preflight)
echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"

if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
  echo "Stale GPU allocations detected — reloading nvidia_uvm module..."
  docker run --rm --privileged alpine sh -c \
    'rmmod nvidia_uvm 2>/dev/null && modprobe nvidia_uvm 2>/dev/null && echo "nvidia_uvm reloaded" || echo "nvidia_uvm reload failed"'

  _PREFLIGHT=$(_run_preflight)
  echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"
  if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
    echo "GPU pre-flight failed after driver reset — aborting. Try a full reboot."
    bash "${SCRIPT_DIR}/services.sh" resume || true
    exit 1
  fi
fi

# ── drop page cache + VM tuning (pass 2 — immediately before training) ────────
echo "Dropping page cache and tuning VM (pass 2)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── page-cache dropper during training ────────────────────────────────────────
(while true; do
  docker run --rm --privileged -v /:/host alpine sh -c \
    'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 30
done) &
_dropper_pid=$!

# ── run training ───────────────────────────────────────────────────────────────
set +e
ionice -c 2 -n 7 docker run --privileged \
  --name "nemotron-trainer-v12" \
  --oom-score-adj -300 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/huggingface":/home/ubuntu/.cache/huggingface \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/train_v9_sft.py \
    --train-file     "${TRAIN_FILE}" \
    --output-dir     "${ADAPTER_OUT}" \
    --model-id       nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --max-steps      "${MAX_STEPS}" \
    --learning-rate  "${LEARNING_RATE}" \
    --min-seq-length "${MIN_SEQ_LENGTH}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --batch-size     1 \
    --grad-accum     16 \
    --lora-r         32 \
    --lora-alpha     32 \
    --seed           3407 \
    --ckpt-every     100 \
    ${WARMSTART_ADAPTER:+--warmstart-adapter "${WARMSTART_ADAPTER}"} \
    ${RESUME_FROM_CHECKPOINT:+--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}"} \
  2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
kill "${_dropper_pid}" 2>/dev/null || true

# ── restore VM defaults ────────────────────────────────────────────────────────
docker run --rm --privileged alpine sh -c \
  'echo 45166 > /proc/sys/vm/min_free_kbytes && echo 100 > /proc/sys/vm/vfs_cache_pressure' \
  2>/dev/null || true

if [[ ${TRAIN_EXIT} -eq 0 ]]; then
  echo "Resuming paused containers..."
  bash "${SCRIPT_DIR}/services.sh" resume || true
  systemctl --user start gnome-remote-desktop.service 2>/dev/null || true
  echo "Log saved to ${LOG_FILE}"
  echo "Adapter saved to ${WORKSPACE}/output/adapter_${RUN_NAME}"
else
  echo "Training failed (exit ${TRAIN_EXIT}) — containers left paused to preserve memory for debugging."
  echo "Log saved to ${LOG_FILE}"
  echo "To restore services: bash scripts/services.sh resume"
  exit ${TRAIN_EXIT}
fi
