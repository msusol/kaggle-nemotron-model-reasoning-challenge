#!/usr/bin/env zsh
# Convert the huikang corpus to NeMo pre-tokenized SFT format.
#
# Runs on host Python (no container needed — no tokenizer load required).
#
# Usage:
#   bash scripts/run_prepare_nemo_dataset.sh
#   bash scripts/run_prepare_nemo_dataset.sh --max-seq-length 8192
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

ZIP="${WORKSPACE}/.cache/huikang-artifacts/huikang-nemotron-artifacts.zip"

python "${SCRIPT_DIR}/prepare_nemo_dataset.py" \
  --zip        "${ZIP}" \
  --out-train  "${WORKSPACE}/data/nemo_train.jsonl" \
  --out-valid  "${WORKSPACE}/data/nemo_valid.jsonl" \
  "$@"

echo ""
echo "Spot-check (first train example):"
head -1 "${WORKSPACE}/data/nemo_train.jsonl" | python -c "
import json, sys
d = json.loads(sys.stdin.read())
n_masked = d['labels'].count(-100)
n_total  = len(d['input_ids'])
n_resp   = n_total - n_masked
print(f'  seq_len={n_total}  masked(prompt)={n_masked}  unmasked(response)={n_resp}')
assert n_total <= 8192,  f'seq_len {n_total} exceeds 8192'
assert n_masked > 0,     'no masked tokens — prompt region missing'
assert n_resp   > 0,     'no unmasked tokens — response region missing'
assert len(d['labels']) == n_total, 'labels length mismatch'
print('  OK')
"
