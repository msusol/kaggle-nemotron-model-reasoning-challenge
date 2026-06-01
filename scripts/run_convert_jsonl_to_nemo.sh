#!/usr/bin/env zsh
# Convert v0.4 text JSONL to NeMo pre-tokenized SFT format using the
# canonical SYSTEM_PROMPT from train_lora.py (v0.4b and later).
#
# Requires the Nemotron tokenizer — runs inside the nemotron-gb10 container.
#
# Usage:
#   bash scripts/run_convert_jsonl_to_nemo.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

if [[ -f "${WORKSPACE}/.env" ]]; then
  set -a && source "${WORKSPACE}/.env" && set +a
fi

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set}" \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "${WORKSPACE}":/workspace \
  -v "${WORKSPACE}/.cache/huggingface":/home/ubuntu/.cache/huggingface \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/convert_jsonl_to_nemo.py \
    --train          /workspace/data/v0.4_train.jsonl \
    --valid          /workspace/data/v0.4_valid.jsonl \
    --out-train      /workspace/data/nemo_dataset/nemo_train.jsonl \
    --out-valid      /workspace/data/nemo_dataset/nemo_valid.jsonl \
    --model-id       "${BASE_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}" \
    --max-seq-length 8192

echo ""
echo "Spot-check (first train example):"
head -1 "${WORKSPACE}/data/nemo_dataset/nemo_train.jsonl" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
n_masked = d['labels'].count(-100)
n_total  = len(d['input_ids'])
n_resp   = n_total - n_masked
print(f'  seq_len={n_total}  masked(prompt)={n_masked}  unmasked(response)={n_resp}')
assert n_total <= 8192, f'seq_len {n_total} exceeds 8192'
assert n_masked > 0,    'no masked tokens — prompt region missing'
assert n_resp   > 0,    'no unmasked tokens — response region missing'
assert len(d['labels']) == n_total, 'labels length mismatch'
print('  OK')
"
