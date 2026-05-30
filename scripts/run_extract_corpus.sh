#!/usr/bin/env bash
# Decode the pre-tokenized huikang corpus to JSONL inside the nemotron-gb10 container.
#
# Usage:
#   bash scripts/run_extract_corpus.sh
#   bash scripts/run_extract_corpus.sh --valid-split 0.03
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

ZIP="/workspace/.cache/huikang-artifacts/huikang-nemotron-artifacts.zip"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/extract_huikang_corpus.py \
    --zip        "${ZIP}" \
    --out-train  /workspace/data/v0.4_train.jsonl \
    --out-valid  /workspace/data/v0.4_valid.jsonl \
    "$@"
