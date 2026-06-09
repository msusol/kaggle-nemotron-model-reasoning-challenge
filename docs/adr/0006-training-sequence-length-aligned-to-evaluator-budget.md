# ADR-0006 — Align MAX_SEQ_LENGTH to Evaluator Budget (4096 tokens)

**Status:** Accepted

## Context

The competition evaluator notebook (`metric/nvidia-nemotron-metric`) was read directly via the
Kaggle CLI (`kaggle kernels pull metric/nvidia-nemotron-metric`). This revealed the exact
inference environment used to score every submission.

### Evaluator facts

| Parameter | Value | Source |
|---|---|---|
| Inference engine | **vLLM** (`LLM(..., enable_lora=True)`) | `generate_predictions()` |
| `max_model_len` | **4096** tokens (prompt + output combined) | `score()` default |
| `max_tokens` | **3584** tokens (max output) | `score()` default |
| `enable_thinking` | **True** (passed to `apply_chat_template`) | `generate_predictions()` |
| `temperature` | 1.0 (sampling, not greedy) | `score()` default |
| `max_lora_rank` | 32 (matches our r=32) | `score()` default |
| Adapter loader | `LoRARequest('adapter', 1, lora_path)` — not PEFT | `generate_predictions()` |

The evaluator is **not** HuggingFace PEFT. It is vLLM's built-in LoRA loader, which reads
`adapter_config.json` and `adapter_model.safetensors` directly. This clarifies why
Unsloth-specific fields in `adapter_config.json` caused ERROR status (see ADR-0005).

### Training / evaluation alignment gap

Runs 1–5 used `MAX_SEQ_LENGTH` values of 2048 or 7680. Neither is aligned to the evaluator:

- **2048** — excludes 76% of training examples; only trains on the easiest problems.
- **7680** — includes examples longer than the evaluator's 4096-token context. At inference,
  the evaluator cannot fully render a problem+answer that required 7680 tokens during training.
  The model trained on signals it will never encounter in evaluation.

Additionally, with `enable_thinking=True` the model generates a `<think>…</think>` chain
before the answer. That chain + the `\boxed{answer}` must fit within `max_tokens=3584`. Models
conditioned on very long sequences may produce lengthy thinking chains that get truncated before
`\boxed{}`, scoring 0 on those problems.

### Training data analysis

Dataset: `data/v0.9_train.jsonl` — 13,730 examples across 14 categories.

**Token length distribution** (approximation: characters ÷ 3.5; measured after
`apply_chat_template` with `enable_thinking=True` on the Kaggle tokenizer):

```
Token range    Count    Pct   Distribution
-----------  -------  -----  ------------------------------------------------
      0–512    2,630  19.2%  ████████████████████
    512–1024   1,465  10.7%  ███████████
  1024–2048    5,216  38.0%  ████████████████████████████████████████  ← bulk
  2048–3072    2,993  21.8%  ██████████████████████
  3072–4096      854   6.2%  ██████
  ─────────────────────────── 4096 token evaluator budget ────────────
  4096–5120      282   2.1%  ██
  5120–6144      122   0.9%
  6144–7680      151   1.1%  █
      7680+       17   0.1%

Within 4096-token budget:  13,158 / 13,730 (95.8%)
Excluded above budget:        572 / 13,730  (4.2%)
Median:   ~1,393 tokens
p90:      ~3,089 tokens
p95:      ~3,774 tokens
Max:      ~8,731 tokens
```

**Answer format check:** 100% of assistant responses contain a non-empty `\boxed{answer}` in
the final 20% of the response. No filtering on answer format is required.

**Chat template alignment:**
- No system prompt in training data ✓ (matches evaluator — evaluator sends bare user message)
- `\boxed{}` instruction in user content ✓ (both training data and evaluator append this)
- `enable_thinking=True` in `apply_chat_template` during training ✓ (matches evaluator)

## Decision

From run6 onwards, set `MAX_SEQ_LENGTH = 4096` and `MIN_SEQ_LENGTH = 0`.

This retains 13,158 examples (95.8% of the dataset) — every example whose full formatted
sequence fits within the evaluator's context window. Examples exceeding 4096 tokens are
**dropped, not truncated**, preserving `\boxed{}` answer integrity.

The filter in `cell-data` tokenizes each example via `apply_chat_template(enable_thinking=True)`
before comparing against `MAX_SEQ_LENGTH`, so the count already includes chat-template overhead.

## Consequences

- Run6 trains on ~822 steps (13,158 examples ÷ 16 effective batch size) vs run5's 593 steps.
  Expected ~110–140 min on RTX Pro 6000.
- Every training example is representable by the evaluator — no train/eval context mismatch.
- The model sees the full dataset distribution (short and long problems alike) in one pass,
  unlike alternating short/long curriculum of runs 1–5.
- Score variance from `temperature=1.0` means ±0.01 differences between runs may be noise.
  A threshold of >0.57 (vs plateau at 0.56) is a meaningful signal.
- The 572 dropped examples (>4096 tokens) are the hardest problems in the dataset. They are
  lost from training. If those problems appear in the test set, the model has no SFT signal
  for them beyond its base-model capability.

## Related

- ADR-0005 — adapter key filtering (Unsloth fused MoE vs standard PEFT/vLLM)
- `notebook/v09_train_kaggle.ipynb` — `cell-config` (run6 settings), `cell-data` (filter logic)
- `metric/nvidia-nemotron-metric` on Kaggle — source of evaluator facts above
