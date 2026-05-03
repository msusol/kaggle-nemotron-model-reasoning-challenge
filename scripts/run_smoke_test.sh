#!/usr/bin/env bash
# Usage: bash scripts/run_smoke_test.sh       # test nemotron-gb10:latest (26.01 primary)
#        bash scripts/run_smoke_test.sh 25    # test nemotron-gb10-25:latest (25.12 archive)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

VARIANT="${1:-}"
if [[ "$VARIANT" == "25" ]]; then
    IMAGE="nemotron-gb10-25:latest"
else
    IMAGE="nemotron-gb10:latest"
fi

docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e TRITON_PRINT_AUTOTUNING=1 \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -w /workspace \
  "$IMAGE" \
  python scripts/smoke_test_nemotron.py
