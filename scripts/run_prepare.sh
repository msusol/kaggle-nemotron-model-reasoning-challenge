#!/usr/bin/env bash
# Run once to quantize and cache the 4-bit model. See scripts/prepare_quantized_model.py.
#
# WARNING — broken for Nemotron-H (30B-A3B MoE):
#   bitsandbytes load_in_4bit only quantizes standard nn.Linear layers. The
#   MoE expert weight tensors in Nemotron-H are stored as batched 3-D tensors
#   (not wrapped in nn.Linear), so they stay BF16. Result: the "quantized"
#   cache is ~57 GB instead of the expected ~15 GB — no useful memory saving.
#
#   Running this script will waste ~10 min and 57 GB of disk space building a
#   cache that is effectively the same size as the original BF16 model. The
#   two large output shards (46 GB + 11 GB) also cause worse mmap page-cache
#   pressure during training than the original 13 × 5 GB HF shards.
#
#   DO NOT RUN until the prepare script is fixed to quantize expert layers.
#   See docs/investigate/ for the root-cause analysis.
echo "ERROR: run_prepare.sh is currently disabled — the quantized cache it"
echo "       produces is 57 GB (broken, MoE experts not quantized) and causes"
echo "       worse OOM pressure than loading from the original HF model."
echo "       Fix prepare_quantized_model.py to handle expert layers first."
exit 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

source "${SCRIPT_DIR}/load_config.sh"

sync && sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

ionice -c 2 -n 7 docker run --rm --privileged \
  --name "nemotron-prepare" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/prepare_quantized_model.py
