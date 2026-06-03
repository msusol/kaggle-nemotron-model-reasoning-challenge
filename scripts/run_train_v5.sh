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
mkdir -p "${WORKSPACE}/output"

echo "RUN_NAME:    ${RUN_NAME}"
echo "Adapter out: ${ADAPTER_OUT}"
echo "Log:         ${LOG_FILE}"

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
# A SIGKILL'd container can hold 6-8 GB of driver-level allocations; removing it
# here lets the torch preflight detect any residual orphaned allocations.
docker rm -f nemotron-trainer-v5 2>/dev/null || true

# ── drop page cache + VM tuning (pass 1 — before preflight) ───────────────────
# Clears NIM/stopped-container page-cache residue so the torch preflight threshold
# check passes.
echo "Dropping page cache and tuning VM (pass 1)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── GPU pre-flight stage 2: torch memory check (12 GB threshold) ──────────────
# Detects orphaned CUDA driver allocations that nvidia-smi --query-compute-apps
# cannot see. Threshold: 12 GB = ~8 GB new CUDA context overhead + 4 GB headroom.
# Model BF16 (~60 GB) + safetensors page cache during load (~60 GB) + baseline
# must fit in 128 GB. A dirty baseline > 12 GB risks OOM at ~73% weight loading.
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
if used > 12e9:
    print(f'PREFLIGHT_FAIL {used/1e9:.1f}GB stale GPU allocations after cache flush')
elif free < 60e9:
    print('WARNING: less than 60 GB free — stale CUDA allocations may be present')
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
# Clears any cache added by the preflight container. This is the definitive state
# the training container will see at model load time.
echo "Dropping page cache and tuning VM (pass 2)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── run training ───────────────────────────────────────────────────────────────
set +e
ionice -c 2 -n 7 docker run --privileged \
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
  -e HF_DATASETS_OFFLINE=1 \
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
