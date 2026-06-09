#!/usr/bin/env zsh
# Download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (~60 GB) to the
# Spark workspace HuggingFace cache for use by NeMo RL training.
#
# The model is downloaded in standard HF hub cache format so NeMo RL can
# resolve it by model ID (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).
#
# Requires: HF_TOKEN env var set (model requires license acceptance at
# https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
#
# IMPORTANT: Always run inside tmux — download takes ~30–60 min.
#
# Usage:
#   tmux new -s download_model
#   HF_TOKEN=hf_... bash scripts/download_model_spark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Export it or add to .env:" >&2
  echo "  export HF_TOKEN=hf_..." >&2
  exit 1
fi

MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CACHE_DIR="${WORKSPACE}/.cache/huggingface"
mkdir -p "${CACHE_DIR}"

echo "Downloading ${MODEL_ID} → ${CACHE_DIR}"
echo "This takes ~30–60 min depending on network speed (~60 GB)."
echo ""

docker run --rm \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "${WORKSPACE}":/workspace \
  nemo-rl-spark:latest \
  huggingface-cli download \
    "${MODEL_ID}" \
    --local-dir-use-symlinks False

# Verify
echo ""
echo "Verifying download..."
docker run --rm \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_HOME=/workspace/.cache/huggingface \
  -v "${WORKSPACE}":/workspace \
  nemo-rl-spark:latest \
  python3 -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_HOME']
model_dir = os.path.join(cache, 'hub', 'models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
snapshots = os.path.join(model_dir, 'snapshots')
if not os.path.exists(snapshots):
    raise RuntimeError(f'Model not found at {model_dir}')
snap = sorted(os.listdir(snapshots))[-1]
snap_path = os.path.join(snapshots, snap)
safetensors = [f for f in os.listdir(snap_path) if f.endswith('.safetensors')]
print(f'Snapshot:     {snap}')
print(f'Safetensors:  {len(safetensors)} files')
assert len(safetensors) == 13, f'Expected 13 safetensors, got {len(safetensors)}'
print('Model download verified OK.')
"
