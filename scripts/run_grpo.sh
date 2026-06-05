#!/usr/bin/env zsh
# Run train_grpo.py inside nemotron-gb10 container.
# Init from v0.5-sft-unsloth adapter (0.60), self-improve via GRPO.
#
# Usage (always inside tmux):
#   tmux new -s grpo_v6
#   RUN_NAME=grpo_v6 bash scripts/run_grpo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

RUN_NAME="${RUN_NAME:-grpo_v6_$(date +%Y%m%d_%H%M%S)}"
ADAPTER_OUT="/workspace/output/adapter_${RUN_NAME}"
LOG_FILE="${WORKSPACE}/output/train_${RUN_NAME}.log"
mkdir -p "${WORKSPACE}/output"

echo "RUN_NAME:    ${RUN_NAME}"
echo "Adapter out: ${ADAPTER_OUT}"
echo "Log:         ${LOG_FILE}"

# ── memory cleanup ─────────────────────────────────────────────────────────
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true
echo "Pausing non-training containers..."
bash "${SCRIPT_DIR}/services.sh" pause || true

GPU_MB=$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
         awk '{sum+=$1} END{print sum+0}')
echo "GPU compute-apps usage: ${GPU_MB} MiB"
if [[ "${GPU_MB:-0}" -gt 1024 ]]; then
  echo "ABORT: GPU has ${GPU_MB} MiB — stop all GPU workloads first." >&2
  bash "${SCRIPT_DIR}/services.sh" resume || true
  exit 1
fi

docker rm -f "nemotron-grpo-${RUN_NAME}" 2>/dev/null || true

echo "Dropping page cache (pass 1)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

_run_preflight() {
  docker run --rm --privileged -e NVIDIA_VISIBLE_DEVICES=all nemotron-gb10:latest \
    python3 -c "
import torch
torch.cuda.init(); torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
used = total - free
print(f'GPU free={free/1e9:.1f}GB total={total/1e9:.1f}GB used={used/1e9:.1f}GB')
if free < 70e9:
    print(f'PREFLIGHT_FAIL only {free/1e9:.1f}GB free')
elif used > 20e9:
    print(f'PREFLIGHT_FAIL {used/1e9:.1f}GB stale allocs')
elif free < 90e9:
    print(f'WARNING: {free/1e9:.1f}GB free')
" 2>&1
}

echo "GPU pre-flight..."
_PREFLIGHT=$(_run_preflight)
echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"

if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
  docker run --rm --privileged alpine sh -c \
    'rmmod nvidia_uvm 2>/dev/null && modprobe nvidia_uvm 2>/dev/null || true'
  _PREFLIGHT=$(_run_preflight)
  echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"
  if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
    echo "Pre-flight failed — aborting."
    bash "${SCRIPT_DIR}/services.sh" resume || true
    exit 1
  fi
fi

echo "Dropping page cache (pass 2)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure' \
  2>/dev/null || true

# ── run GRPO ────────────────────────────────────────────────────────────────
set +e
ionice -c 2 -n 7 docker run --privileged \
  --name "nemotron-grpo-${RUN_NAME}" \
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
  python scripts/train_grpo.py \
    --adapter-dir     /workspace/output/adapter_v5_sft_unsloth \
    --train-file      /workspace/data/train.csv \
    --output-dir      "${ADAPTER_OUT}" \
    --model-id        nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --num-steps       500 \
    --num-generations 4 \
    --learning-rate   1e-6 \
    --max-new-tokens  512 \
    --kl-coeff        0.04 \
    --batch-size      1 \
    --seed            3407 \
  2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

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
  echo "GRPO failed (exit ${TRAIN_EXIT}) — containers left paused for debugging."
  echo "Log: ${LOG_FILE}"
  echo "To restore: bash scripts/services.sh resume"
  exit ${TRAIN_EXIT}
fi
