#!/usr/bin/env bash
# Run download_data.py inside the nemotron-gb10 container.
#
# Usage:
#   bash scripts/run_download.sh               # download + convert
#   bash scripts/run_download.sh --download-only  # inventory only, skip conversion
#
# Kaggle credentials (in priority order):
#   1. KAGGLE_API_TOKEN env var or .env
#   2. ~/.kaggle/access_token file
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

# Fall back to ~/.kaggle/access_token if KAGGLE_API_TOKEN not already set.
if [[ -z "${KAGGLE_API_TOKEN:-}" && -f "${HOME}/.kaggle/access_token" ]]; then
  KAGGLE_API_TOKEN="$(cat "${HOME}/.kaggle/access_token")"
fi

if [[ -z "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "ERROR: KAGGLE_API_TOKEN is not set and ~/.kaggle/access_token not found." >&2
  exit 1
fi

IMAGE="nemotron-gb10:latest"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -e KAGGLE_USERNAME="${KAGGLE_USERNAME:-}" \
  -e KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN}" \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  "$IMAGE" \
  python scripts/download_data.py \
    --out-dir /workspace/data \
    "$@"
