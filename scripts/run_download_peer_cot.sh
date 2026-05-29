#!/usr/bin/env bash
# Download and convert the peer CoT dataset inside the nemotron-gb10 container.
#
# Dataset: kienngx/nemotron-30b-competition-trainingdata-cot-labels
#
# Usage:
#   bash scripts/run_download_peer_cot.sh               # download + convert
#   bash scripts/run_download_peer_cot.sh --download-only
#
# Kaggle credentials (in priority order):
#   1. KAGGLE_USERNAME + KAGGLE_KEY in .env
#   2. KAGGLE_API_TOKEN in .env (mapped to KAGGLE_KEY inside container)
#   3. ~/.kaggle/kaggle.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

# kagglehub uses KAGGLE_KEY; map KAGGLE_API_TOKEN if KAGGLE_KEY not set
if [[ -z "${KAGGLE_KEY:-}" && -n "${KAGGLE_API_TOKEN:-}" ]]; then
  KAGGLE_KEY="${KAGGLE_API_TOKEN}"
fi

if [[ -z "${KAGGLE_KEY:-}" ]]; then
  echo "ERROR: KAGGLE_KEY (or KAGGLE_API_TOKEN) is not set." >&2
  exit 1
fi

IMAGE="nemotron-gb10:latest"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -e KAGGLE_USERNAME="${KAGGLE_USERNAME:-}" \
  -e KAGGLE_KEY="${KAGGLE_KEY}" \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  "$IMAGE" \
  python scripts/download_peer_cot.py \
    --out-dir /workspace/data \
    "$@"
