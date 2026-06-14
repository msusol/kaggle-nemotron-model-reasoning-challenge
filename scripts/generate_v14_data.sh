#!/usr/bin/env zsh
# Build the v0.14 training dataset.
#
# v0.14 adds short synthetic traces for the 3 categories absent from v0.13:
#   spelling, equation_numeric_deduce, equation_numeric_guess
# All huikang traces for these categories exceed 4,096 tokens (medians 4,884 / 5,874 / 6,148),
# so they were silently dropped from v0.13 by the token filter.
#
# Pipeline:
#   1. Generate 500 synthetic examples per missing category (+ equation_symbolic refresh)
#   2. Merge with v0.12 base (25,500 rows)
#   3. Token-filter (>4096 tokens dropped) then balance → data/v0.14_train.jsonl
#
# Requires:
#   - data/v0.12_train.jsonl                         (the v0.12 base corpus)
#   - nemotron-gb10:latest Docker image              (for tokenizer in balance step)
#   - .cache/huggingface/                            (Nemotron tokenizer cached locally)
#
# Run from repo root on GB10 (NOT inside an active training container):
#   zsh scripts/generate_v14_data.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== v0.14 data build ==="
echo "Working directory: $REPO_ROOT"
echo ""

# ── Step 1: Generate synthetic examples ──────────────────────────────────────

echo "[1/4] Generating spelling examples..."
python3 scripts/generate_spelling.py \
    --n 500 \
    --seed 42 \
    --out data/spelling_synthetic.jsonl

echo "[1/4] Generating equation_numeric_deduce examples..."
python3 scripts/generate_equation_numeric_deduce.py \
    --n 500 \
    --seed 42 \
    --out data/eq_num_deduce_synthetic.jsonl

echo "[1/4] Generating equation_numeric_guess examples..."
python3 scripts/generate_equation_numeric_guess.py \
    --n 500 \
    --seed 42 \
    --out data/eq_num_guess_synthetic.jsonl

echo "[1/4] Generating equation_symbolic examples..."
python3 scripts/generate_equation_symbolic.py \
    --n 500 \
    --seed 42 \
    --out data/equation_symbolic_synthetic.jsonl

echo ""

# ── Step 2: Merge ────────────────────────────────────────────────────────────

echo "[2/4] Merging into data/v0.14_merged.jsonl..."
cat \
    data/v0.12_train.jsonl \
    data/equation_symbolic_synthetic.jsonl \
    data/spelling_synthetic.jsonl \
    data/eq_num_deduce_synthetic.jsonl \
    data/eq_num_guess_synthetic.jsonl \
    > data/v0.14_merged.jsonl

TOTAL=$(wc -l < data/v0.14_merged.jsonl)
echo "  Merged: ${TOTAL} rows"
echo ""

# ── Step 3: Token-filter then balance (requires Docker) ───────────────────────

echo "[3/4] Token-filtering and balancing (inside Docker)..."
docker run \
    --rm \
    --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:/workspace" \
    -v "${REPO_ROOT}/.cache/huggingface:/home/ubuntu/.cache/huggingface" \
    -w /workspace \
    nemotron-gb10:latest \
    python3 scripts/balance_dataset.py \
        --input  data/v0.14_merged.jsonl \
        --output data/v0.14_train.jsonl \
        --max-tokens 4096 \
        --max-per-category 1500 \
        --min-per-category 300 \
        --seed 42

echo ""

# ── Step 4: Summary ──────────────────────────────────────────────────────────

FINAL=$(wc -l < data/v0.14_train.jsonl)
echo "[4/4] Done. data/v0.14_train.jsonl: ${FINAL} rows"
echo ""
echo "Next: update run_train_v14.sh to point at data/v0.14_train.jsonl"
echo "      and start training with warmstart from the best run16 checkpoint."
