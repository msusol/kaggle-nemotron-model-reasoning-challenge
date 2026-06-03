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

# Stop gnome-remote-desktop: it holds a CUDA context in GPU HBM (~6 GB) that
# competes with model loading. Stopping it before training reliably frees those
# allocations. It is restarted by the resume step after training completes.
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true

# Pause non-training containers to free system RAM before loading the model.
echo "Pausing non-training containers..."
bash "${SCRIPT_DIR}/services.sh" pause || true

# GPU pre-flight — two-stage check:
#
# Stage 1 (host, via nvidia-smi): sum of GPU memory held by active compute processes
# must be < 1 GB. On a clean system this is ~176 MiB (gnome-remote-desktop). Anything
# above 1 GB means a zombie training container or other GPU-heavy process is still
# running and must be stopped before training starts.
#
# Stage 2 (inside container, via torch): force-initialize a new CUDA context to trigger
# driver GC of orphaned allocations from any prior SIGKILL'd container. After init +
# empty_cache the reading must be < 12 GB.
#
# Threshold rationale (12 GB):
#   The pre-flight container itself adds ~8 GB of CUDA context overhead.
#   A clean system (after GRD stop + docker rm -f) reads ~8 GB inside the container.
#   Any reading > 12 GB means ~4+ GB of orphaned driver allocations survive.
#   Memory budget: model BF16 (~60 GB) + safetensors page cache during load (~60 GB)
#   + baseline = peak. With a dirty 11.7 GB baseline: 11.7+60+60 = 131.7 GB > 130.7 GB
#   → OOM at ~76% loading. The old 25 GB threshold allowed this through. [cite:ADR-0002]
echo "GPU pre-flight..."

GPU_APPS_MIB=$(nvidia-smi --query-compute-apps=used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk '{sum+=$1} END{print sum+0}')
echo "GPU compute-apps usage: ${GPU_APPS_MIB} MiB"
if [[ "${GPU_APPS_MIB:-0}" -gt 1024 ]]; then
  echo "ABORT: GPU has ${GPU_APPS_MIB} MiB held by active processes (> 1 GB) — stop all GPU workloads before training."
  bash "${SCRIPT_DIR}/services.sh" resume || true
  exit 1
fi

# Remove any stale container from a previous killed run BEFORE the torch pre-flight
# so that the pre-flight sees the post-cleanup GPU state, not the pre-cleanup state.
# A SIGKILL'd container can hold 6-8 GB of driver-level allocations; removing it here
# lets the pre-flight detect any residual orphaned allocations that docker rm can't clear.
docker rm -f nemotron-trainer 2>/dev/null || true

# Drop page cache, clear swap, and tune VM BEFORE the torch pre-flight.
#
# On the GB10 unified-memory system, stopped NIM containers (general_chat_llm,
# fast_python_coder) leave their model-file pages in the Linux page cache, which
# torch.cuda.mem_get_info() counts as "used" because the OS and CUDA draw from the
# same LPDDR5X pool. Running drop_caches here clears those file-backed pages before
# the pre-flight threshold check, preventing spurious PREFLIGHT_FAIL when NIM
# containers were running prior to this script.
#
# nvidia_uvm module reload (the fallback for actual CUDA driver orphans) fails in
# CUDA Forward Compatibility mode, so clearing page cache is the primary remedy.
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# Python must exit cleanly (no sys.exit(1)) so the CUDA context is torn down
# properly on container exit. Aborting from within Python via sys.exit(1) causes
# incomplete CUDA teardown, which leaves orphaned allocations — the opposite of
# what we want. Print a marker line instead and let bash decide.
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

_PREFLIGHT=$(_run_preflight)
echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"

if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
  # Reload the nvidia_uvm kernel module to force-release orphaned driver allocations.
  # A --privileged container has CAP_SYS_MODULE and can rmmod/modprobe host modules
  # without interactive sudo. Safe here because stage-1 confirmed no active compute
  # apps are running.
  echo "Stale GPU allocations detected — reloading nvidia_uvm module..."
  docker run --rm --privileged alpine sh -c \
    'rmmod nvidia_uvm 2>/dev/null && modprobe nvidia_uvm 2>/dev/null && echo "nvidia_uvm reloaded" || echo "nvidia_uvm reload failed"'

  # Re-check after the module reload.
  _PREFLIGHT=$(_run_preflight)
  echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"
  if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
    echo "GPU pre-flight failed after driver reset — aborting. Try a full reboot."
    bash "${SCRIPT_DIR}/services.sh" resume || true
    exit 1
  fi
fi

# Second drop_caches + VM tuning immediately before training.
# The first pass (before the pre-flight) cleared NIM page-cache residue so the
# threshold check passes. This second pass catches any cache added by the pre-flight
# container and sets the min_free_kbytes / vfs_cache_pressure tuning for loading.
#
# min_free_kbytes=1 GB: nudges the kernel to reclaim file-backed pages before they
# crowd out CUDA allocations (40 GB was too aggressive — stalled loading at ~57%).
# vfs_cache_pressure=500 makes file-backed page eviction 5× more aggressive.
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# Temporarily disable exit-on-error so we can capture the docker exit code even on
# OOM kill (exit 137) or Python crash, without the script aborting at the pipe.
set +e
ionice -c 2 -n 7 docker run --privileged \
  --name "nemotron-trainer" \
  --oom-score-adj 300 \
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
  2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
echo  # blank line — tqdm leaves no trailing newline on exit

# Restore VM defaults regardless of training outcome (kernel tuning only).
docker run --rm --privileged alpine sh -c \
  'echo 45166 > /proc/sys/vm/min_free_kbytes && echo 100 > /proc/sys/vm/vfs_cache_pressure' \
  2>/dev/null || true

if [[ ${TRAIN_EXIT} -eq 0 ]]; then
  # Resume containers and report success only when training completed normally.
  echo "Resuming paused containers..."
  bash "${SCRIPT_DIR}/services.sh" resume || true
  systemctl --user start gnome-remote-desktop.service 2>/dev/null || true
  echo "Log saved to ${LOG_FILE}"
  echo "Adapter saved to ${ADAPTER_DIR}"
else
  echo "Training failed (exit ${TRAIN_EXIT}) — containers left paused to preserve memory for debugging."
  echo "Log saved to ${LOG_FILE}"
  echo "To restore services: bash scripts/services.sh resume"
  exit ${TRAIN_EXIT}
fi
