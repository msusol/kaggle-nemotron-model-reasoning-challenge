#!/usr/bin/env zsh
# Generate v0.5 SFT training data (competition CSV + synthetic) inside the
# nemotron-gb10 container.
#
# Output: data/v0.5_train.jsonl (~21,500 records, messages format, short responses)
#
# Usage:
#   bash scripts/run_prepare_v5_sft_data.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/prepare_v5_sft_data.py \
    --train-csv /workspace/data/train.csv \
    --out       /workspace/data/v0.5_train.jsonl
