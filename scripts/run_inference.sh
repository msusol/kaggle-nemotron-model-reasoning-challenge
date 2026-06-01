#!/usr/bin/env bash
# Run infer_lora.py inside the nemotron-gb10 container.
#
# Usage:
#   bash scripts/run_inference.sh                        # v0.4_valid.jsonl → output/predictions.jsonl
#   bash scripts/run_inference.sh --batch-size 4         # pass extra args to infer_lora.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

ADAPTER_DIR=""
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapter-dir) ADAPTER_DIR="$2"; shift 2 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ -z "${ADAPTER_DIR}" ]]; then
  ADAPTER_DIR="$(ls -d "${WORKSPACE}"/output/adapter_* 2>/dev/null | sort | tail -1)"
  if [[ -z "${ADAPTER_DIR}" ]]; then
    echo "ERROR: No adapter_* directory found in output/. Pass --adapter-dir or run training first." >&2
    exit 1
  fi
fi
echo "Using adapter: ${ADAPTER_DIR}"

docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/triton":/home/ubuntu/.triton \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/infer_lora.py \
    --adapter-dir "/workspace/output/$(basename "${ADAPTER_DIR}")" \
    --data-file   /workspace/data/v0.4_valid.jsonl \
    --output-file "/workspace/output/predictions_$(basename "${ADAPTER_DIR}" | sed 's/adapter_//')".jsonl \
    "${PASSTHROUGH[@]}"
