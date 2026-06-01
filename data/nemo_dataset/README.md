# Huikang Nemotron — NeMo SFT Pre-tokenized Corpus (r=32)

Pre-tokenized supervised fine-tuning dataset for **NVIDIA Nemotron-3 Nano 30B (A3B)**,
derived from
[samvalladares/huikang-nemotron-artifacts](https://www.kaggle.com/datasets/samvalladares/huikang-nemotron-artifacts)
and formatted for the
[NVIDIA NeMo](https://github.com/NVIDIA/NeMo) SFT training pipeline.

Created for the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
competition.

---

## What this dataset contains

| File | Examples | Description |
|---|---|---|
| `nemo_train.jsonl` | 15,159 | Training split (95%) |
| `nemo_valid.jsonl` | 820 | Validation split (5%) |

### Why 95/5 and not the conventional 80/20

The 5% validation split is intentional. For this specific dataset and training regime,
a larger held-out set would waste training coverage without improving evaluation quality:

- **820 examples is sufficient for the purpose.** The validation set monitors that
  training loss is decreasing — it is a sanity-check signal, not a model selection
  gate. 820 examples does this reliably.
- **1-epoch SFT cannot overfit.** Each example is seen exactly once. A large validation
  set has nothing additional to detect.
- **The competition leaderboard is the real evaluation.** Local validation accuracy is
  a proxy; the final metric is the Kaggle public score. Over-investing in held-out size
  reduces training coverage without improving the score that matters.
- **Coverage over rare categories.** The smallest category (`cryptarithm_deduce`) has
  125 examples total. At 80/20 only ~100 would train; at 95/5 ~119 train. Every
  example pulled from training on rare categories directly hurts coverage of those
  test types.
- **Consistent with the 0.85 reference.** Tong Hui Kang's reference pipeline trained
  on the full corpus with a small held-out set at 1 epoch. This split matches that pattern.

Each line is a JSON object with two fields:

```json
{
  "input_ids": [10, 25708, 1010, 11, ...],
  "labels":    [-100, -100, -100, -100, ...]
}
```

- **`input_ids`** — full token ID sequence using the Nemotron tokenizer
  (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`). Covers the complete chat turn:
  system prompt + user question + assistant reasoning trace + final boxed answer.
- **`labels`** — same length as `input_ids`. The system+user region is masked with
  `-100` (no loss computed); the assistant response region contains the real token IDs
  (loss computed here). Standard SFT masking convention.

---

## Sequence structure

Each sequence follows Nemotron's chat template:

```
[ system tokens ] [ user tokens ] [ <|im_start|>assistant\n<think>\n ]
^─────────────────────────────────────────────────────────────────────^
                    labels = -100  (no loss)

[ CoT reasoning ] [ \n</think>\n\boxed{answer} ] [ <|im_end|> ]
^────────────────────────────────────────────────────────────────^
                    labels = real token IDs  (loss computed here)
```

The split point between masked and unmasked is where the assistant begins generating
its visible reasoning. The `<think>` token itself is in the masked region because the
original corpus training did not compute loss on it.

---

## Sequence length statistics

Computed over the training split (15,159 examples):

| Statistic | Tokens |
|---|---|
| Min | 202 |
| p25 | 1,063 |
| Median (p50) | 3,284 |
| p75 | 4,556 |
| p90 | 6,671 |
| p99 | 7,532 |
| Max | 7,999 |
| Avg prompt length | ~599 |
| Avg response length | ~2,608 |

All 15,979 examples fit within 8,192 tokens. None were filtered.

---

## Problem categories

14 reasoning categories covering the full competition test set:

| Category | Count | % |
|---|---|---|
| matching | 4,515 | 28.3% |
| bit_manipulation | 2,059 | 12.9% |
| cipher | 1,756 | 11.0% |
| splitting | 1,500 | 9.4% |
| concatenation | 1,500 | 9.4% |
| unit_conversion | 987 | 6.2% |
| gravity | 975 | 6.1% |
| spelling | 648 | 4.1% |
| equation_numeric_deduce | 635 | 4.0% |
| numeral | 624 | 3.9% |
| lstrip | 300 | 1.9% |
| cryptarithm_guess | 183 | 1.1% |
| equation_numeric_guess | 172 | 1.1% |
| cryptarithm_deduce | 125 | 0.8% |

---

## Concrete example

A short example from `nemo_train.jsonl` (349 tokens total):

```json
{
  "input_ids": [10, 25708, 1010, 11, 1010, 10, 3263, 1010, ...],
  "labels":    [-100, -100, -100, -100, -100, -100, -100, -100, ...]
}
```

Structural breakdown:
- `len(input_ids)` = 349
- `labels.count(-100)` = 269  →  prompt region (system + user + `<think>`)
- `len - masked` = 80   →  response region (CoT + `</think>` + `\boxed{answer}`)
- Boundary: `labels[268]=-100`, `labels[269]=1048` (first generated token)

To decode with the Nemotron tokenizer:
```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

import json
with open("nemo_train.jsonl") as f:
    ex = json.loads(f.readline())

# Full sequence
print(tok.decode(ex["input_ids"]))

# Response only (unmasked region)
resp_ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
print(tok.decode(resp_ids))
```

---

## Using with NVIDIA NeMo

Configure `nemotron_sft_config.yaml` to point at these files:

```yaml
model:
  pre_tokenized_dataset: True
  data:
    train_ds:
      file_names: ["/data/nemo_train.jsonl"]
      max_seq_length: 8192
    validation_ds:
      file_names: ["/data/nemo_valid.jsonl"]
      max_seq_length: 8192
  peft:
    peft_scheme: "lora"
    lora_tuning:
      r: 32
      adapter_alpha: 64
      lora_dropout: 0.05
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]
```

NeMo reads the integer arrays directly without re-tokenizing — no tokenizer load needed
at training time.

---

## Provenance and attribution

This dataset is a format conversion only — all reasoning content originates from the
work of two contributors credited below. No new traces, problems, or token content
were added.

---

### Tong Hui Kang (`huikang`)

**Original corpus author.** Achieved a **0.85 public leaderboard score** with the
reference notebook — the benchmark this dataset is designed to replicate under NeMo.

His contributions underlying every example in this dataset:

- **Deterministic solver framework** (`reasoners/` + `augmenters/`): one Python solver
  per problem category that exhaustively searches the rule space and emits its search
  log as CoT text. The CoT is guaranteed correct by construction — the solver only
  records a trace when `status == 'rule_found'`. This is why the corpus achieves a
  starting training loss of **0.386** (vs ~1.5–23 for LLM-generated CoT alternatives).

- **Test-set category discovery**: Tong Hui Kang's pipeline covers all 14 competition
  categories, including 8 test-only types (`matching`, `splitting`, `concatenation`,
  `spelling`, `equation_numeric_deduce`, `lstrip`, `cryptarithm_guess`,
  `cryptarithm_deduce`) absent from the public `train.csv`. This is the primary reason
  the 0.85 reference outperforms approaches trained on competition data alone.

- **Tinker training framework**: custom PyTorch training library with
  `StepLinearDecayLRSchedule` (linear LR decay to 0), stratified batching across 14
  categories, and per-epoch logprob collection for iterative self-improvement. The
  published artifacts represent **version 26** of this iterative loop.

- **Reference training config** (`nemotron-base-model-generation/config.json`):
  ```json
  {
    "lr_schedule": {"learning_rate": 0.0002, "class_name": "StepLinearDecayLRSchedule"},
    "lora_rank": 32,  "max_length": 8192,
    "train_mlp": true, "train_attn": true, "train_unembed": true
  }
  ```
  `train_mlp=true` = MoE expert layers (`up_proj`/`down_proj`) included in LoRA.
  `train_attn=true` = attention (`q/k/v/o_proj`). `train_unembed=true` = `lm_head`.

- Competition discussion: [#687961 — MoE expert layer memory analysis](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/687961)

---

### Sam Valladares (`samvalladares`)

Published Tong Hui Kang's pre-tokenized artifacts as a public Kaggle dataset under
CC0-1.0, making this work accessible and compliant with competition Rule 6.

- Source dataset: [samvalladares/huikang-nemotron-artifacts](https://www.kaggle.com/datasets/samvalladares/huikang-nemotron-artifacts) (CC0-1.0)

---

### This dataset

| Detail | Value |
|---|---|
| What was added | NeMo SFT format conversion (`input_ids` / `labels` integer arrays) |
| Conversion script | `scripts/prepare_nemo_dataset.py` in [msusol/kaggle-nemotron-model-reasoning-challenge](https://github.com/msusol/kaggle-nemotron-model-reasoning-challenge) |
| Kaggle slug | `gdataranger/huikang-nemotron-nemo-sft-r32` |
| Tokenizer | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Train/valid split | Deterministic 95/5 MD5 hash on `problem_id` — identical to the text-format JSONL split |
| License | CC0-1.0 (inherited from source) |
| Competition | [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) |
