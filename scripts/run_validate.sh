#!/usr/bin/env bash
# Run validate_metric.py inside the nemotron-gb10 container.
#
# Usage:
#   bash scripts/run_validate.sh                         # score output/predictions.jsonl
#   bash scripts/run_validate.sh --rel-tol 1e-3          # pass extra args to validate_metric.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

PREDICTIONS="/workspace/output/predictions.jsonl"
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions) PREDICTIONS="$2"; shift 2 ;;
    *) PASSTHROUGH+=("$1"); shift ;;
  esac
done

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/validate_metric.py \
    --predictions "${PREDICTIONS}" \
    --labels      /workspace/data/v0.3_valid_labels.jsonl \
    "${PASSTHROUGH[@]}"
