#!/usr/bin/env zsh
# Run train_grpo.py inside the nemotron-gb10 container.
#
# Usage (inside a tmux session):
#   tmux new -s grpo
#   RUN_NAME=grpo_v5 bash scripts/run_grpo.sh
#   # Ctrl+B D to detach
#
# Pass --test-steps N to run a smoke test (N steps only):
#   RUN_NAME=grpo_test bash scripts/run_grpo.sh --test-steps 50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

RUN_NAME="${RUN_NAME:-grpo_v5}"
PASSTHROUGH=("$@")

LOG_FILE="${WORKSPACE}/output/grpo_${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "RUN_NAME:   ${RUN_NAME}"
echo "Log:        ${LOG_FILE}"
echo "Config:     configs/nemotron_grpo.yaml"

# ── GPU pre-flight ────────────────────────────────────────────────────────────
GPU_USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader \
  | awk '{sum+=$1} END {print sum+0}')
if (( GPU_USED > 1024 )); then
  echo "ABORT: GPU has ${GPU_USED} MiB held by active processes (> 1 GB)." \
       "Stop all GPU workloads before training." >&2
  exit 1
fi
echo "GPU pre-flight OK (${GPU_USED} MiB in use)"

# ── pause non-training containers ─────────────────────────────────────────────
bash "${SCRIPT_DIR}/services.sh" pause

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
  python scripts/train_grpo.py \
    --config configs/nemotron_grpo.yaml \
    --run-name "${RUN_NAME}" \
    "${PASSTHROUGH[@]}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE="${PIPESTATUS[0]}"

bash "${SCRIPT_DIR}/services.sh" resume

if [[ "${EXIT_CODE}" -eq 0 ]]; then
  echo ""
  echo "GRPO training complete. Adapter saved to:"
  ls -d "${WORKSPACE}/output/adapter_${RUN_NAME}_"* 2>/dev/null | sort | tail -1
else
  echo "Training exited with code ${EXIT_CODE}." >&2
fi
