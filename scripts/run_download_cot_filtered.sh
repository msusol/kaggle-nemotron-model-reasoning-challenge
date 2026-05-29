#!/usr/bin/env zsh
# Download and convert kishanvavdara/nemotron-reasoning-traj (correctness-filtered)
# inside the nemotron-gb10 container.
#
# Usage:
#   zsh scripts/run_download_cot_filtered.sh               # download + convert
#   zsh scripts/run_download_cot_filtered.sh --download-only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

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
  python scripts/download_cot_filtered.py \
    --out-dir /workspace/data \
    "$@"
