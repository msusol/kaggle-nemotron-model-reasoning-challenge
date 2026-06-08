#!/usr/bin/env zsh
# Pre-warm the FlashInfer + Triton kernel cache for the vLLM sidecar.
#
# FlashInfer JIT-compiles GB10-specific GEMM kernels (gemm_sm120) via ninja
# on first use. These kernels are NOT included in the pre-built wheel. When the
# trainer is running (~61 GB), only ~28 GB is free and the ninja parallel build
# gets OOM-killed.
#
# Run this script ONCE with a clean GPU (no trainer running). It starts vLLM
# standalone with 80% GPU utilisation so all kernels compile with plenty of
# headroom. Compiled artefacts are saved to persistent cache directories:
#   .cache/flashinfer  → /root/.cache/flashinfer  (FlashInfer GEMM kernels)
#   .cache/vllm_compile→ /root/.cache/vllm        (vLLM torch-compile cache)
#   .cache/triton      → TRITON_CACHE_DIR          (Triton PTX cache)
#
# After this script exits, run_grpo_sidecar.sh mounts the same cache dirs so
# subsequent runs skip compilation entirely.
#
# Usage (once, on a clean GPU):
#   tmux new -s vllm_prewarm
#   bash scripts/prewarm_vllm_sidecar.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

SIDECAR_MODEL="${SIDECAR_MODEL:-FP8}"
SIDECAR_PORT="${SIDECAR_PORT:-8001}"   # non-default port so it doesn't clash with a running sidecar

case "${SIDECAR_MODEL}" in
  NVFP4)
    VLLM_MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    NVFP4_VLLM_ARGS=(--moe-backend flashinfer_cutedsl)
    ;;
  FP8|*)
    VLLM_MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    NVFP4_VLLM_ARGS=()
    ;;
esac

mkdir -p \
  "${WORKSPACE}/.cache/flashinfer" \
  "${WORKSPACE}/.cache/vllm_compile" \
  "${WORKSPACE}/.cache/triton"

echo "=== vLLM kernel pre-warm ==="
echo "Model:    ${VLLM_MODEL_ID}"
echo "Port:     ${SIDECAR_PORT}"
echo "Caches:"
echo "  FlashInfer → ${WORKSPACE}/.cache/flashinfer"
echo "  vLLM       → ${WORKSPACE}/.cache/vllm_compile"
echo "  Triton     → ${WORKSPACE}/.cache/triton"
echo ""
echo "Starting vLLM (no trainer — full GPU available for compilation)..."
docker rm -f nemotron-vllm-prewarm 2>/dev/null || true
docker run --detach \
  --name "nemotron-vllm-prewarm" \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e VLLM_TORCH_COMPILE_WORKERS=1 \
  -e MAX_JOBS=4 \
  -e TRITON_CACHE_DIR=/workspace/.cache/triton \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/vllm_compile":/root/.cache/vllm \
  -v "${WORKSPACE}/.cache/flashinfer":/root/.cache/flashinfer \
  -w /workspace \
  nemotron-vllm-gb10:latest \
  python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_ID}" \
    --dtype auto \
    --enable-lora \
    --max-lora-rank 32 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --port "${SIDECAR_PORT}" \
    --host 0.0.0.0 \
    --download-dir /workspace/.cache/huggingface \
    --enforce-eager \
    "${NVFP4_VLLM_ARGS[@]}"

echo "Streaming logs (waiting for healthy — compilation may take 15-30 min)..."
docker logs -f nemotron-vllm-prewarm 2>&1 &
_log_pid=$!

_TIMEOUT=3600  # 60 min max
_ELAPSED=0
_HEALTH_URL="http://localhost:${SIDECAR_PORT}/health"
until curl -sf "${_HEALTH_URL}" >/dev/null 2>&1; do
  if [[ ${_ELAPSED} -ge ${_TIMEOUT} ]]; then
    kill "${_log_pid}" 2>/dev/null || true
    echo "ERROR: pre-warm did not complete within ${_TIMEOUT}s" >&2
    docker logs --tail 30 nemotron-vllm-prewarm >&2 || true
    docker rm -f nemotron-vllm-prewarm 2>/dev/null || true
    exit 1
  fi
  sleep 15
  _ELAPSED=$(( _ELAPSED + 15 ))
done
kill "${_log_pid}" 2>/dev/null || true

echo ""
echo "vLLM healthy after ${_ELAPSED}s — kernel cache is warm."
echo "Stopping pre-warm container..."
docker rm -f nemotron-vllm-prewarm 2>/dev/null || true

echo ""
echo "Cache directories populated:"
du -sh "${WORKSPACE}/.cache/flashinfer" "${WORKSPACE}/.cache/vllm_compile" "${WORKSPACE}/.cache/triton" 2>/dev/null || true
echo ""
echo "You can now run the smoke test — kernels will load from cache, no recompilation:"
echo "  NUM_STEPS=10 RUN_NAME=grpo_v10_smoke bash scripts/run_grpo_sidecar.sh"
