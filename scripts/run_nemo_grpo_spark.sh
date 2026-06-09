#!/usr/bin/env zsh
# Run v0.11 LoRA GRPO training via NeMo RL (nano-v3 branch) on DGX Spark.
#
# Single-container architecture (no vLLM sidecar):
#   nemo-rl-spark:latest  --privileged  (trains + generates via Megatron backend)
#
# HBM budget (steady state, estimated):
#   Base weights (BF16, frozen)  ~60 GB
#   LoRA adapters (r=32)         ~0.5 GB
#   Adam optimizer (LoRA only)   ~1–2 GB
#   Activations + ckpt           ~10–15 GB
#   KV cache (Megatron gen)      ~12 GB
#   Total                        ~85 GB  (~45 GB headroom on 130.7 GB HBM)
#
# IMPORTANT: Always run inside tmux to survive SSH disconnects.
#
# Usage:
#   tmux new -s grpo_v11
#   RUN_NAME=grpo_v11 bash scripts/run_nemo_grpo_spark.sh
#
#   # 3-step smoke test:
#   SMOKE_TEST=1 RUN_NAME=grpo_v11_smoke bash scripts/run_nemo_grpo_spark.sh
#
#   # Resume from last checkpoint:
#   RESUME=1 RUN_NAME=grpo_v11 bash scripts/run_nemo_grpo_spark.sh
#
# Env vars:
#   RUN_NAME        grpo_v11_<timestamp>  (default)
#   SMOKE_TEST      0  — set to 1 for 3-step validation run (overrides NUM_STEPS)
#   NUM_STEPS       300  — total GRPO training steps
#   RESUME          0  — set to 1 to resume from last checkpoint
#   TRAIN_FILE      /workspace/data/v0.9_grpo_train.jsonl  (default)
#   VAL_FILE        /workspace/data/v0.9_grpo_valid.jsonl   (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

# ── Config ──────────────────────────────────────────────────────────────────
SMOKE_TEST="${SMOKE_TEST:-0}"
NUM_STEPS="${NUM_STEPS:-300}"
RESUME="${RESUME:-0}"
TRAIN_FILE="${TRAIN_FILE:-/workspace/data/v0.9_grpo_train.jsonl}"
VAL_FILE="${VAL_FILE:-/workspace/data/v0.9_grpo_valid.jsonl}"
RUN_NAME="${RUN_NAME:-grpo_v11_$(date +%Y%m%d_%H%M%S)}"

# 3-step smoke test overrides NUM_STEPS
if [[ "${SMOKE_TEST}" == "1" ]]; then
  NUM_STEPS=3
fi

CONFIG_PATH="/workspace/docs/plans/lora-grpo-spark-plan/04-lora-grpo-config.yaml"
EXP_NAME="$(date +%Y%m%d)/lora-grpo-spark/${RUN_NAME}"
RESULTS_DIR="${WORKSPACE}/results/${EXP_NAME}"
LOG_FILE="${WORKSPACE}/output/train_${RUN_NAME}.log"

mkdir -p "${WORKSPACE}/output" "${RESULTS_DIR}/logs"

echo "RUN_NAME:    ${RUN_NAME}"
echo "NUM_STEPS:   ${NUM_STEPS}"
echo "SMOKE_TEST:  ${SMOKE_TEST}"
echo "RESUME:      ${RESUME}"
echo "TRAIN_FILE:  ${TRAIN_FILE}"
echo "VAL_FILE:    ${VAL_FILE}"
echo "Results dir: ${RESULTS_DIR}"
echo "Log:         ${LOG_FILE}"

# ── Verify nemo-rl-spark:latest image exists ────────────────────────────────
if ! docker image inspect nemo-rl-spark:latest >/dev/null 2>&1; then
  echo ""
  echo "ERROR: nemo-rl-spark:latest not found. Build it first:" >&2
  echo "  docker build \\"
  echo "    -f docs/plans/lora-grpo-spark-plan/Dockerfile.spark \\"
  echo "    --build-arg MAX_JOBS=8 \\"
  echo "    -t nemo-rl-spark:latest \\"
  echo "    ." >&2
  exit 1
fi

# ── Verify GRPO data files exist (host paths mirror /workspace) ─────────────
_TRAIN_HOST="${WORKSPACE}/${TRAIN_FILE#/workspace/}"
_VAL_HOST="${WORKSPACE}/${VAL_FILE#/workspace/}"
for _f in "${_TRAIN_HOST}" "${_VAL_HOST}"; do
  if [[ ! -f "${_f}" ]]; then
    echo ""
    echo "ERROR: ${_f} not found." >&2
    echo "Run: python scripts/prepare_v11_data.py" >&2
    exit 1
  fi
done

# ── Cleanup on exit ──────────────────────────────────────────────────────────
_cleanup() {
  local exit_code=$?
  kill "${_dropper_pid:-}" 2>/dev/null || true

  # Restore VM tuning
  docker run --rm --privileged alpine sh -c \
    'echo 45166 > /proc/sys/vm/min_free_kbytes && echo 100 > /proc/sys/vm/vfs_cache_pressure' \
    2>/dev/null || true

  bash "${SCRIPT_DIR}/services.sh" resume || true
  systemctl --user start gnome-remote-desktop.service 2>/dev/null || true

  if [[ ${exit_code} -eq 0 ]]; then
    echo ""
    echo "GRPO v0.11 training complete."
    echo "Log:         ${LOG_FILE}"
    echo "Results dir: ${RESULTS_DIR}"
  else
    echo ""
    echo "Training exited (code ${exit_code})."
    echo "Log: ${LOG_FILE}"
    echo "To resume: RESUME=1 RUN_NAME=${RUN_NAME} bash scripts/run_nemo_grpo_spark.sh"
  fi
}
trap _cleanup EXIT

# ── Memory cleanup ───────────────────────────────────────────────────────────
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true

echo "Pausing non-training containers..."
bash "${SCRIPT_DIR}/services.sh" pause || true

GPU_MB=$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
         awk '{sum+=$1} END{print sum+0}')
echo "GPU compute-apps usage: ${GPU_MB} MiB"
if [[ "${GPU_MB:-0}" -gt 1024 ]]; then
  echo "ABORT: GPU has ${GPU_MB} MiB — stop all GPU workloads first." >&2
  exit 1
fi

docker rm -f "nemo-rl-spark-${RUN_NAME}" 2>/dev/null || true

# ── Drop page cache + VM tuning (pass 1) ────────────────────────────────────
echo "Dropping page cache and tuning VM (pass 1)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── GPU preflight ─────────────────────────────────────────────────────────────
_run_preflight() {
  docker run --rm --privileged -e NVIDIA_VISIBLE_DEVICES=all nemo-rl-spark:latest \
    python3 -c "
import torch
torch.cuda.init()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
used = total - free
print(f'GPU free={free/1e9:.1f}GB total={total/1e9:.1f}GB used={used/1e9:.1f}GB')
if free < 65e9:
    print(f'PREFLIGHT_FAIL only {free/1e9:.1f}GB free (need >=65 GB for model load)')
elif used > 20e9:
    print(f'PREFLIGHT_FAIL {used/1e9:.1f}GB stale GPU allocations — nvidia_uvm reload may help')
elif free < 90e9:
    print(f'WARNING: {free/1e9:.1f}GB free — stale allocs present but training should fit')
" 2>&1
}

echo "GPU pre-flight..."
_PREFLIGHT=$(_run_preflight)
echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"

if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
  echo "Stale GPU allocations — reloading nvidia_uvm..."
  docker run --rm --privileged alpine sh -c \
    'rmmod nvidia_uvm 2>/dev/null && modprobe nvidia_uvm 2>/dev/null && echo "nvidia_uvm reloaded" || echo "nvidia_uvm reload failed"'

  _PREFLIGHT=$(_run_preflight)
  echo "${_PREFLIGHT}" | grep -E "^GPU|^WARNING|^PREFLIGHT"
  if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
    echo "GPU pre-flight failed after driver reset — aborting. Try a full reboot." >&2
    exit 1
  fi
fi

# ── Drop page cache + VM tuning (pass 2, immediately before training) ────────
echo "Dropping page cache and tuning VM (pass 2)..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true' \
  2>/dev/null || true

# ── Page-cache dropper during training (every 30s) ───────────────────────────
(while true; do
  docker run --rm --privileged -v /:/host alpine sh -c \
    'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  sleep 30
done) &
_dropper_pid=$!

# ── Build Hydra config overrides ─────────────────────────────────────────────
_OVERRIDES=(
  "data.train_jsonl_fpath=${TRAIN_FILE}"
  "data.validation_jsonl_fpath=${VAL_FILE}"
  "checkpointing.checkpoint_dir=/workspace/results/${EXP_NAME}"
  "logger.log_dir=/workspace/results/${EXP_NAME}/logs"
  "grpo.max_num_steps=${NUM_STEPS}"
)
if [[ "${RESUME}" == "1" ]]; then
  _OVERRIDES+=("checkpointing.resume_from_checkpoint=true")
fi

echo ""
echo "Starting NeMo RL GRPO training (${NUM_STEPS} steps)..."
echo "Config overrides:"
for _o in "${_OVERRIDES[@]}"; do
  echo "  ${_o}"
done
echo ""

# ── Run training (foreground — tmux provides disconnect protection) ───────────
set +e
docker run \
  --name "nemo-rl-spark-${RUN_NAME}" \
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
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e TORCH_CUDA_ARCH_LIST="12.1" \
  -e TOKENIZERS_PARALLELISM=false \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/huggingface":/workspace/.cache/huggingface \
  -v "${WORKSPACE}/results":/workspace/results \
  -w /opt/nemo-rl \
  nemo-rl-spark:latest \
  bash -c "
    uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
      --config ${CONFIG_PATH} \
      ${_OVERRIDES[*]}
  " \
  2>&1 | tee "${LOG_FILE}"
TRAIN_EXIT=${pipestatus[1]}
set -e

kill "${_dropper_pid:-}" 2>/dev/null || true
_dropper_pid=""

if [[ ${TRAIN_EXIT} -ne 0 ]]; then
  echo "" >&2
  echo "Training failed (exit ${TRAIN_EXIT})." >&2
  echo "Last 20 log lines:" >&2
  tail -20 "${LOG_FILE}" >&2 || true
  exit ${TRAIN_EXIT}
fi
