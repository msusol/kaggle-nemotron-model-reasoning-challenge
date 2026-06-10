#!/usr/bin/env bash
# Pre-seed the HF->Megatron iter_0000000 checkpoint in isolation on DGX Spark.
#
# Runs scripts/convert_hf_to_megatron_spark.py inside nemo-rl-spark:latest with
# the SAME memory mitigations as run_nemo_grpo_spark.sh (extra swap, VM tuning,
# host-side page-cache dropper) -- but with NO Ray, optimizer, reference model,
# or KV cache, so the whole machine is available for the one ~54 GB save.
#
# After this succeeds once, every training launch SKIPS the import (the worker
# detects the complete iter_0000000 and takes the cheap load_checkpoint path).
#
# Usage (always inside tmux to survive SSH drops):
#   tmux new -s convert
#   bash scripts/run_convert_spark.sh
#
#   # force re-conversion (delete existing checkpoint first):
#   FORCE=1 bash scripts/run_convert_spark.sh
#
# Env vars:
#   FORCE        0 -- set to 1 to delete and rebuild an existing iter_0000000
#   SWAP_GB      60 -- extra swap added for the CPU copy during save
#   CONFIG_PATH  /workspace/docs/plans/lora-grpo-spark-plan/04-lora-grpo-config.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

FORCE="${FORCE:-0}"
SWAP_GB="${SWAP_GB:-60}"
CONFIG_PATH="${CONFIG_PATH:-/workspace/docs/plans/lora-grpo-spark-plan/04-lora-grpo-config.yaml}"
LOG_FILE="${WORKSPACE}/output/convert_$(date +%Y%m%d_%H%M%S).log"
RUN_NAME="convert_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${WORKSPACE}/output"

_FORCE_FLAG=""
if [[ "${FORCE}" == "1" ]]; then
  _FORCE_FLAG="--force"
fi

echo "CONFIG_PATH: ${CONFIG_PATH}"
echo "FORCE:       ${FORCE}"
echo "SWAP_GB:     ${SWAP_GB}"
echo "Log:         ${LOG_FILE}"

# ── Verify image ─────────────────────────────────────────────────────────────
if ! docker image inspect nemo-rl-spark:latest >/dev/null 2>&1; then
  echo "ERROR: nemo-rl-spark:latest not found. Build it first:" >&2
  echo "  docker build -f docs/plans/lora-grpo-spark-plan/Dockerfile.spark \\" >&2
  echo "    --build-arg MAX_JOBS=8 -t nemo-rl-spark:latest ." >&2
  exit 1
fi

# ── Cleanup on exit ──────────────────────────────────────────────────────────
_cleanup() {
  local exit_code=$?
  docker stop nemo-convert-page-dropper 2>/dev/null || true
  kill "${_dropper_pid:-}" 2>/dev/null || true

  # Remove extra swap (swapon/swapoff require root -> privileged container)
  docker run --rm --privileged -v /tmp:/tmp alpine sh -c \
    'swapoff /tmp/convert_extra_swap 2>/dev/null; rm -f /tmp/convert_extra_swap; true' \
    2>/dev/null || rm -f /tmp/convert_extra_swap

  # Restore VM tuning to defaults
  docker run --rm --privileged alpine sh -c \
    'echo 45166 > /proc/sys/vm/min_free_kbytes \
     && echo 100 > /proc/sys/vm/vfs_cache_pressure \
     && echo 60 > /proc/sys/vm/swappiness' \
    2>/dev/null || true

  bash "${SCRIPT_DIR}/services.sh" resume 2>/dev/null || true
  systemctl --user start gnome-remote-desktop.service 2>/dev/null || true

  if [[ ${exit_code} -eq 0 ]]; then
    echo ""
    echo "Conversion complete. iter_0000000 pre-seeded."
    echo "Next: launch training normally -- it will SKIP the HF->mcore import."
    echo "  SMOKE_TEST=1 RUN_NAME=grpo_v11_smoke bash scripts/run_nemo_grpo_spark.sh"
  else
    echo ""
    echo "Conversion exited (code ${exit_code}). Log: ${LOG_FILE}"
  fi
}
trap _cleanup EXIT

# ── Free the GPU and pause non-essential services ────────────────────────────
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true
echo "Pausing non-training containers..."
bash "${SCRIPT_DIR}/services.sh" pause 2>/dev/null || true

GPU_MB=$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
         awk '{sum+=$1} END{print sum+0}')
echo "GPU compute-apps usage: ${GPU_MB} MiB"
if [[ "${GPU_MB:-0}" -gt 1024 ]]; then
  echo "ABORT: GPU has ${GPU_MB} MiB in use -- stop all GPU workloads first." >&2
  exit 1
fi

docker rm -f "nemo-convert-${RUN_NAME}" 2>/dev/null || true

# ── Drop page cache + VM tuning ──────────────────────────────────────────────
echo "Dropping page cache and tuning VM..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && echo 1 > /proc/sys/vm/dirty_background_ratio \
   && echo 2 > /proc/sys/vm/dirty_ratio \
   && echo 100 > /proc/sys/vm/dirty_writeback_centisecs \
   && echo 100 > /proc/sys/vm/dirty_expire_centisecs \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── GPU preflight (need >= 65 GB free to load + convert the 54 GB model) ─────
echo "GPU pre-flight..."
_PREFLIGHT=$(docker run --rm --privileged -e NVIDIA_VISIBLE_DEVICES=all nemo-rl-spark:latest \
  python3 -c "
import torch
torch.cuda.init(); torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f'GPU free={free/1e9:.1f}GB total={total/1e9:.1f}GB used={(total-free)/1e9:.1f}GB')
if free < 65e9:
    print(f'PREFLIGHT_FAIL only {free/1e9:.1f}GB free (need >=65 GB)')
" 2>&1)
echo "${_PREFLIGHT}" | grep -E "^GPU|^PREFLIGHT"
if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
  echo "Stale GPU allocations -- reloading nvidia_uvm..."
  docker run --rm --privileged alpine sh -c \
    'rmmod nvidia_uvm 2>/dev/null && modprobe nvidia_uvm 2>/dev/null && echo reloaded || echo "reload failed"'
fi

# ── Extra swap for the ~54 GB CPU copy during save_megatron_model ────────────
echo "Creating ${SWAP_GB} GB extra swap for the checkpoint save..."
rm -f /tmp/convert_extra_swap
docker run --rm --privileged -v /tmp:/tmp alpine sh -c \
  "fallocate -l ${SWAP_GB}G /tmp/convert_extra_swap \
   && chmod 0600 /tmp/convert_extra_swap \
   && mkswap /tmp/convert_extra_swap \
   && swapon /tmp/convert_extra_swap \
   && echo 100 > /proc/sys/vm/swappiness \
   && echo 'Swap enabled'"
echo "Swap ready: $(free -h | grep -i swap)"

# ── Host-side page-cache dropper during conversion ───────────────────────────
docker rm -f nemo-convert-page-dropper 2>/dev/null || true
docker run --rm --privileged --name nemo-convert-page-dropper alpine sh -c \
  'while true; do echo 3 > /proc/sys/vm/drop_caches; sleep 10; done' &
_dropper_pid=$!

# ── Run the conversion ───────────────────────────────────────────────────────
echo ""
echo "Starting standalone HF->Megatron conversion..."
set +e
docker run \
  --name "nemo-convert-${RUN_NAME}" \
  --privileged \
  --oom-score-adj -300 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e TORCH_CUDA_ARCH_LIST="12.1" \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_HUB_CACHE=/workspace/.cache/huggingface \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/huggingface":/workspace/.cache/huggingface \
  -w /opt/nemo-rl \
  nemo-rl-spark:latest \
  python3 /workspace/scripts/convert_hf_to_megatron_spark.py \
    --config "${CONFIG_PATH}" ${_FORCE_FLAG} \
  2>&1 | tee -a "${LOG_FILE}"
CONVERT_EXIT=$(docker inspect "nemo-convert-${RUN_NAME}" --format='{{.State.ExitCode}}' 2>/dev/null || echo 1)
docker rm "nemo-convert-${RUN_NAME}" 2>/dev/null || true
set -e

kill "${_dropper_pid:-}" 2>/dev/null || true
_dropper_pid=""

if [[ ${CONVERT_EXIT} -ne 0 ]]; then
  echo "" >&2
  echo "Conversion failed (exit ${CONVERT_EXIT}). Last 20 log lines:" >&2
  tail -20 "${LOG_FILE}" >&2 || true
  exit "${CONVERT_EXIT}"
fi
