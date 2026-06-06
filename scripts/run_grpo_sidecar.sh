#!/usr/bin/env zsh
# Run GRPO v0.10 training with vLLM FP8/NVFP4 sidecar for fast rollout generation.
#
# CRITICAL LOAD ORDER — both models share the 130.7 GB unified memory pool:
#
#   Step 1: Start training container FIRST.
#           Loading BF16 peaks at ~126 GB (model alloc + page cache fight).
#           The training container's dropper thread manages this safely.
#           Steady state after load: ~83 GB committed, ~47 GB free.
#
#   Step 2: Training container writes .trainer_model_ready flag when loaded.
#           We detect it here and start the vLLM NVFP4 sidecar (default).
#           A 3s Alpine dropper keeps page cache low during vLLM loading.
#           NVFP4 load peak: 83 + ~12 (page cache, aggressively dropped) + 20 (CUDA) ~= 115 GB.
#           Steady state: ~108 GB (83 training + 25 vLLM). Set SIDECAR_MODEL=FP8 for ~120 GB.
#
#   Step 3: After vLLM is healthy and initial LoRA is loaded, write .vllm_sidecar_ready.
#           Training container detects it and starts the GRPO loop.
#
# Architecture (NVFP4 default):
#   nemotron-gb10       (BF16, --network=host) ~83 GB  train_grpo_sidecar.py
#   nemotron-vllm-gb10  (NVFP4, --network=host) ~25 GB  vllm.entrypoints.openai.api_server
#   Total: ~108 GB (22 GB headroom)
#
# Usage (always inside tmux):
#   tmux new -s grpo_v10
#   RUN_NAME=grpo_v10 bash scripts/run_grpo_sidecar.sh
#
# Env vars:
#   SIDECAR_MODEL      NVFP4 (default, ~25 GB) or FP8 (~37 GB)
#   SIDECAR_PORT       8000 (default)
#   ROLLOUT_SYNC_STEPS 50 (default)
#   RUN_NAME           grpo_v10_<timestamp> (default)
#   VLLM_ALREADY_UP    0 (default) — set to 1 to skip vLLM startup on trainer retry
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

# ── Config ──────────────────────────────────────────────────────────────────
SIDECAR_MODEL="${SIDECAR_MODEL:-NVFP4}"     # FP8 ~37 GB; NVFP4 ~25 GB (needs cutlass-dsl smoke test)
SIDECAR_PORT="${SIDECAR_PORT:-8000}"
ROLLOUT_SYNC_STEPS="${ROLLOUT_SYNC_STEPS:-50}"
NUM_STEPS="${NUM_STEPS:-500}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
VLLM_ALREADY_UP="${VLLM_ALREADY_UP:-0}"  # set to 1 to reuse a running vLLM on trainer retry
RUN_NAME="${RUN_NAME:-grpo_v10_$(date +%Y%m%d_%H%M%S)}"
ADAPTER_OUT="/workspace/output/adapter_${RUN_NAME}"
OUTPUT_DIR="${WORKSPACE}/output/adapter_${RUN_NAME}"
LOG_FILE="${WORKSPACE}/output/train_${RUN_NAME}.log"
mkdir -p "${WORKSPACE}/output" "${OUTPUT_DIR}"

TRAINER_READY_FLAG="${OUTPUT_DIR}/.trainer_model_ready"
VLLM_READY_FLAG="${OUTPUT_DIR}/.vllm_sidecar_ready"
rm -f "${TRAINER_READY_FLAG}" "${VLLM_READY_FLAG}"

case "${SIDECAR_MODEL}" in
  NVFP4)
    VLLM_MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    VLLM_GPU_MEM_UTIL="0.25"
    NVFP4_ENV=(-e VLLM_USE_FLASHINFER_MOE_FP4=1 -e VLLM_FLASHINFER_MOE_BACKEND=throughput)
    ;;
  FP8|*)
    VLLM_MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    # Trainer steady-state ≈ 71 GB → ~50 GB free. 0.35×121.69=42.6 GB (model 32 GB + 10.6 GB KV cache).
    VLLM_GPU_MEM_UTIL="0.35"
    NVFP4_ENV=()
    ;;
esac

echo "RUN_NAME:        ${RUN_NAME}"
echo "SIDECAR_MODEL:   ${SIDECAR_MODEL} (${VLLM_MODEL_ID})"
echo "SIDECAR_PORT:    ${SIDECAR_PORT}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Log:             ${LOG_FILE}"

# ── Cleanup on exit ──────────────────────────────────────────────────────────
# vLLM takes 10-15 min to start. On trainer OOM/failure, leave it running so
# the next retry can skip the startup cost via VLLM_ALREADY_UP=1.
_cleanup() {
  local exit_code=$?
  bash "${SCRIPT_DIR}/services.sh" resume || true
  systemctl --user start gnome-remote-desktop.service 2>/dev/null || true
  if [[ ${exit_code} -eq 0 ]]; then
    echo "Training complete — stopping vLLM sidecar..."
    docker stop "nemotron-vllm-sidecar" 2>/dev/null || true
    docker rm -f "nemotron-vllm-sidecar" 2>/dev/null || true
  else
    echo ""
    echo "Trainer exited (code ${exit_code}) — vLLM sidecar LEFT RUNNING (saves 10-15 min restart)."
    echo "To retry trainer only:"
    echo "  VLLM_ALREADY_UP=1 RUN_NAME=grpo_v10_retry bash scripts/run_grpo_sidecar.sh"
    echo "To stop vLLM manually:"
    echo "  docker stop nemotron-vllm-sidecar && docker rm -f nemotron-vllm-sidecar"
  fi
}
trap _cleanup EXIT

# ── Memory cleanup ───────────────────────────────────────────────────────────
systemctl --user stop gnome-remote-desktop.service 2>/dev/null || true
echo "Pausing non-training containers..."
bash "${SCRIPT_DIR}/services.sh" pause || true

if [[ "${VLLM_ALREADY_UP}" != "1" ]]; then
  GPU_MB=$(nvidia-smi --query-compute-apps=used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | \
           awk '{sum+=$1} END{print sum+0}')
  echo "GPU compute-apps usage: ${GPU_MB} MiB"
  if [[ "${GPU_MB:-0}" -gt 1024 ]]; then
    echo "ABORT: GPU has ${GPU_MB} MiB — stop all GPU workloads first." >&2
    exit 1
  fi
  docker rm -f "nemotron-vllm-sidecar" 2>/dev/null || true
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
" 2>&1
}

echo "GPU pre-flight..."
_PREFLIGHT=$(_run_preflight)
echo "${_PREFLIGHT}" | grep -E "^GPU|^PREFLIGHT"

if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
  docker run --rm --privileged alpine sh -c \
    'rmmod nvidia_uvm 2>/dev/null && modprobe nvidia_uvm 2>/dev/null || true'
  _PREFLIGHT=$(_run_preflight)
  echo "${_PREFLIGHT}" | grep -E "^GPU|^PREFLIGHT"
  if echo "${_PREFLIGHT}" | grep -q "^PREFLIGHT_FAIL"; then
    echo "Pre-flight failed — aborting."
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

# ── Remap SFT adapter to GRPO key format (one-time, ~10s) ────────────────────
# SFT has 12,008 Unsloth per-expert keys (backbone.layers, no .default.).
# Unsloth creates only 232 fused-MoE keys in GRPO mode (model.layers, .default.).
# The remap drops the 11,776 per-expert keys and renames the 232 remainder so they
# exactly match the GRPO model's parameter names for warmstart injection.
GRPO_WARMSTART_HOST="${WORKSPACE}/output/adapter_v5_sft_grpo_warmstart"
if [[ ! -f "${GRPO_WARMSTART_HOST}/adapter_model.safetensors" ]]; then
  echo "Remapping SFT adapter to GRPO format..."
  docker run --rm \
    --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v "${WORKSPACE}":/workspace \
    -w /workspace \
    nemotron-gb10:latest \
    python scripts/remap_sft_adapter_for_grpo.py \
      --src /workspace/output/adapter_v5_sft_unsloth \
      --dst /workspace/output/adapter_v5_sft_grpo_warmstart
else
  echo "GRPO warmstart adapter already exists: ${GRPO_WARMSTART_HOST}"
fi

# ── STEP 1: Start training container first ───────────────────────────────────
# The BF16 training model load peaks at ~126 GB (model + page cache sharing the
# same unified pool). The container's dropper thread manages this safely.
# We start it detached so we can poll the ready flag and launch vLLM in parallel.
echo ""
echo "Starting training container (BF16 model load — this takes ~5-10 min)..."
docker run --detach \
  --name "nemotron-grpo-${RUN_NAME}" \
  --privileged \
  --oom-score-adj 500 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
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
  python scripts/train_grpo_sidecar.py \
    --adapter-dir        /workspace/output/adapter_v5_sft_unsloth \
    --grpo-warmstart     /workspace/output/adapter_v5_sft_grpo_warmstart \
    --train-file         /workspace/data/v0.9_train.jsonl \
    --output-dir         "${ADAPTER_OUT}" \
    --vllm-server-url    "http://localhost:${SIDECAR_PORT}" \
    --rollout-sync-steps "${ROLLOUT_SYNC_STEPS}" \
    --num-steps          "${NUM_STEPS}" \
    --num-generations    "${NUM_GENERATIONS}" \
    --max-new-tokens     "${MAX_NEW_TOKENS}" \
    --kl-coeff           0.04 \
    --batch-size         1 \
    --seed               3407 \
  > "${LOG_FILE}" 2>&1 &
TRAINER_BG_PID=$!

echo "Training container started (BG pid ${TRAINER_BG_PID})."
echo "Tailing log in background — watch: tail -f ${LOG_FILE}"

# ── STEP 2: Wait for TRAINER_MODEL_READY flag ────────────────────────────────
echo "Waiting for training model to load (flag: ${TRAINER_READY_FLAG})..."
_TRAINER_TIMEOUT=1800  # 30 minutes
_ELAPSED=0
while [[ ! -f "${TRAINER_READY_FLAG}" ]]; do
  sleep 15
  _ELAPSED=$(( _ELAPSED + 15 ))
  # Show recent log progress every 60s
  if (( _ELAPSED % 60 == 0 )); then
    echo "  [${_ELAPSED}s] trainer still loading..."
    tail -3 "${LOG_FILE}" 2>/dev/null | sed 's/^/  /'
  fi
  if [[ ${_ELAPSED} -ge ${_TRAINER_TIMEOUT} ]]; then
    echo "ERROR: Training model did not finish loading within ${_TRAINER_TIMEOUT}s" >&2
    docker logs --tail 30 "nemotron-grpo-${RUN_NAME}" >&2 || true
    exit 1
  fi
done
echo "TRAINER_MODEL_READY — $(cat "${TRAINER_READY_FLAG}")"

# ── VLLM_ALREADY_UP: skip vLLM startup if sidecar is already running ─────────
if [[ "${VLLM_ALREADY_UP}" == "1" ]]; then
  echo "VLLM_ALREADY_UP=1 — checking existing sidecar health..."
  if curl -sf "http://localhost:${SIDECAR_PORT}/health" >/dev/null 2>&1; then
    echo "vLLM sidecar healthy — skipping startup sequence."
    echo "vllm_ready=true sidecar_model=${SIDECAR_MODEL} (reused)" > "${VLLM_READY_FLAG}"
  else
    echo "WARNING: sidecar not responding — falling back to full startup."
    VLLM_ALREADY_UP=0
  fi
fi

if [[ "${VLLM_ALREADY_UP}" != "1" ]]; then

# ── STEP 3: Drop page cache before vLLM load ─────────────────────────────────
echo "Dropping page cache before vLLM load..."
sync
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes' \
  2>/dev/null || true

# ── STEP 4: Start vLLM sidecar with aggressive page-cache dropper ─────────────
# NVFP4 load peak: 71 GB (trainer stable) + ~12 GB (page cache, 3s dropper) + 25 GB (model) ≈ 108 GB
# A background Alpine dropper every 3s keeps page cache from accumulating.
echo ""
echo "Starting vLLM sidecar (${SIDECAR_MODEL} → ${VLLM_MODEL_ID}, gpu_util=${VLLM_GPU_MEM_UTIL})..."
docker run --detach \
  --name "nemotron-vllm-sidecar" \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --oom-score-adj -300 \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/workspace/.cache/huggingface \
  "${NVFP4_ENV[@]}" \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -w /workspace \
  nemotron-vllm-gb10:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_ID}" \
    --dtype auto \
    --enable-lora \
    --max-lora-rank 32 \
    --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}" \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --port "${SIDECAR_PORT}" \
    --host 0.0.0.0 \
    --download-dir /workspace/.cache/huggingface

# Aggressive page-cache dropper while vLLM loads (background, stops when health OK)
echo "Starting 3s page-cache dropper during vLLM load..."
_dropper_pid=""
(
  while true; do
    docker run --rm --privileged -v /:/host alpine sh -c \
      'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    sleep 3
  done
) &
_dropper_pid=$!

# ── STEP 5: Wait for vLLM health (live log stream) ───────────────────────────
_VLLM_HEALTH_URL="http://localhost:${SIDECAR_PORT}/health"
_VLLM_TIMEOUT=900   # 15 minutes
echo "Streaming vLLM sidecar logs (waiting for health at ${_VLLM_HEALTH_URL})..."
docker logs -f nemotron-vllm-sidecar 2>&1 &
_vllm_log_pid=$!

_ELAPSED=0
until curl -sf "${_VLLM_HEALTH_URL}" >/dev/null 2>&1; do
  if [[ ${_ELAPSED} -ge ${_VLLM_TIMEOUT} ]]; then
    kill "${_vllm_log_pid}" 2>/dev/null || true
    kill "${_dropper_pid}" 2>/dev/null || true
    echo "ERROR: vLLM sidecar did not become healthy within ${_VLLM_TIMEOUT}s" >&2
    docker logs --tail 50 nemotron-vllm-sidecar >&2 || true
    exit 1
  fi
  sleep 15
  _ELAPSED=$(( _ELAPSED + 15 ))
done
kill "${_vllm_log_pid}" 2>/dev/null || true
kill "${_dropper_pid}" 2>/dev/null || true
echo "vLLM sidecar healthy (${_ELAPSED}s)"

# ── STEP 6: Load initial LoRA adapter ────────────────────────────────────────
INITIAL_ADAPTER="/workspace/output/adapter_v5_sft_unsloth"
echo "Loading initial LoRA adapter: ${INITIAL_ADAPTER}..."
_LORA_RESP=$(curl -s -w "\n%{http_code}" \
  -X POST "http://localhost:${SIDECAR_PORT}/v1/load_lora_adapter" \
  -H "Content-Type: application/json" \
  -d "{\"lora_name\":\"nemotron-policy\",\"lora_path\":\"${INITIAL_ADAPTER}\"}")
_LORA_HTTP=$(echo "${_LORA_RESP}" | tail -1)
echo "LoRA load response: HTTP ${_LORA_HTTP}"
if [[ "${_LORA_HTTP}" != "200" ]]; then
  echo "WARNING: LoRA load returned HTTP ${_LORA_HTTP} — training will run without LoRA until first sync"
fi

# ── STEP 7: Signal trainer that vLLM is ready ────────────────────────────────
echo "vllm_ready=true sidecar_model=${SIDECAR_MODEL}" > "${VLLM_READY_FLAG}"
echo "VLLM_READY flag written — trainer will now start the GRPO loop."

fi  # end VLLM_ALREADY_UP startup block

# ── STEP 8: Follow trainer logs ──────────────────────────────────────────────
echo ""
echo "Following training log (Ctrl-C stops this script but containers keep running):"
echo "  docker logs -f nemotron-grpo-${RUN_NAME}"
echo ""
tail -f "${LOG_FILE}" &
_tail_pid=$!

# Wait for training container to exit
wait "${TRAINER_BG_PID}" 2>/dev/null || true
TRAIN_EXIT=$(docker inspect --format='{{.State.ExitCode}}' "nemotron-grpo-${RUN_NAME}" 2>/dev/null || echo "1")
kill "${_tail_pid}" 2>/dev/null || true

# Reset VM tuning (trap EXIT handles container + service cleanup)
docker run --rm --privileged alpine sh -c \
  'echo 45166 > /proc/sys/vm/min_free_kbytes && echo 100 > /proc/sys/vm/vfs_cache_pressure' \
  2>/dev/null || true

if [[ "${TRAIN_EXIT}" == "0" ]]; then
  echo ""
  echo "GRPO v0.10 training complete."
  echo "Log:     ${LOG_FILE}"
  echo "Adapter: ${OUTPUT_DIR}"
else
  echo "GRPO failed (exit ${TRAIN_EXIT}) — containers left running for debugging."
  echo "  docker logs nemotron-vllm-sidecar | tail -50"
  echo "  docker logs nemotron-grpo-${RUN_NAME} | tail -50"
  echo "  To restore services: bash scripts/services.sh resume"
  trap - EXIT
  exit "${TRAIN_EXIT}"
fi
