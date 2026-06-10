#!/usr/bin/env zsh
# Download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (~60 GB) to the
# Spark workspace HuggingFace cache for use by NeMo RL training.
#
# The model is downloaded in standard HF hub cache format so NeMo RL can
# resolve it by model ID (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).
#
# Uses huggingface-cli login (interactive) on first run; token is cached at
# .cache/huggingface/token and reused on subsequent runs.
#
# IMPORTANT: Always run inside tmux — download takes ~30–60 min.
#
# Usage:
#   tmux new -s download_model
#   bash scripts/download_model_spark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

MODEL_ID="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
CACHE_DIR="${WORKSPACE}/.cache/huggingface"
MODEL_DIR="${CACHE_DIR}/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
TOKEN_FILE="${CACHE_DIR}/token"
mkdir -p "${CACHE_DIR}"

# If model snapshots already exist, skip straight to verify
if [[ -d "${MODEL_DIR}/snapshots" ]]; then
  echo "Model already present at ${MODEL_DIR} — skipping download."
else
  echo "Downloading ${MODEL_ID} → ${CACHE_DIR}"
  echo "This takes ~30–60 min depending on network speed (~60 GB)."
  echo ""

  if [[ -f "${TOKEN_FILE}" ]]; then
    echo "HuggingFace token cached — skipping login."
    docker run --rm \
      --privileged \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e HF_HOME=/workspace/.cache/huggingface \
      -e HF_HUB_CACHE=/workspace/.cache/huggingface \
      -v "${WORKSPACE}":/workspace \
      nemo-rl-spark:latest \
      hf download "${MODEL_ID}"   else
    echo "No cached token found — prompting for HuggingFace login."
    docker run --rm -it \
      --privileged \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -e HF_HOME=/workspace/.cache/huggingface \
      -e HF_HUB_CACHE=/workspace/.cache/huggingface \
      -v "${WORKSPACE}":/workspace \
      nemo-rl-spark:latest \
      bash -c "hf auth login && hf download ${MODEL_ID}"
  fi
fi

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
model_dir = os.path.join(cache, 'models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16')
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
