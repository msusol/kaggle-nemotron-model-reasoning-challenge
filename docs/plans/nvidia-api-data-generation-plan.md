# NVIDIA API Data Generation Plan

**Status:** Deferred — investigate after v0.5 SFT scores and v0.6 GRPO run  
**Priority:** Medium — reduces dependency on borrowed adapters for future training rounds

---

## Motivation

Throughout v0.4 and v0.5 we have relied on two things we don't own:

1. **huikang's pre-built adapter** (`huikang/nemotron-adapter` v27) as a warmstart for v0.5 SFT
2. **samvalladares/huikang-nemotron-artifacts** — huikang's Tinker-generated CoT corpus for v0.4 training data

Both are publicly available Kaggle resources, but they introduce risks:
- Format quirks we had to work around (`\boxed{–}` placeholder, empty system prompts)
- Dependency on a competitor's published work staying available after the competition ends
- Limited to what huikang chose to generate; no control over coverage or quality

**build.nvidia.com** provides a free inference API that could generate our own training data
from scratch, eliminating both dependencies.

---

## What build.nvidia.com is

Hosted inference API — call NVIDIA-hosted models and get completions back.
OpenAI-compatible REST API, free tier with rate limits, no GPU required.

Key models available:
- `nvidia/llama-3.1-nemotron-70b-instruct` — capable reasoning model, same class as huikang used for Tinker
- `nvidia/nemotron-3-nano-30b-a3b-instruct` — the base of the competition model itself

Used purely at **data generation time** — produces training data that is then used to fine-tune locally on the GB10.

---

## API setup

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],  # free key from build.nvidia.com
)

response = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.6,
    max_tokens=512,
)
answer = response.choices[0].message.content
```

API key registration: https://build.nvidia.com/ → "Get API Key"  
No credit card required for free tier.

---

## Use cases in this project

### Use case 1 — Replace rule-based synthetic generators in `prepare_v5_sft_data.py`

Currently `prepare_v5_sft_data.py` generates 12,000 synthetic examples using simple
deterministic generators (`gen_bit`, `gen_cipher`, `gen_unit`, `gen_numeral`, `gen_eq`).
These are ported directly from kuangyicheng's notebook and produce correct but formulaic
training examples.

**With the NVIDIA API:** call `nvidia/llama-3.1-nemotron-70b-instruct` on each generated
problem to produce a more varied one-sentence trace:

```python
PROMPT = """Solve this pattern recognition problem in ONE sentence ending with 
'Final answer: \\boxed{{answer}}'.

{problem}"""
```

This would increase diversity in the assistant responses beyond the single fixed template
kuangyicheng used, potentially improving generalisation.

### Use case 2 — Regenerate huikang-style CoT corpus with clean format

Run each of the 9,500 competition problems through the API with a long-CoT prompt to
produce fresh reasoning traces without Tinker's format quirks (`\boxed{–}`, empty system
fields). This would be a clean replacement for `samvalladares/huikang-nemotron-artifacts`.

**Caveat:** Long-CoT responses still hit Kaggle's `max_new_tokens` limit at eval time.
Only useful if the intent is to test whether the 0.85 score can be improved further with
cleaner data — not directly helpful for matching 0.87+.

### Use case 3 — Generate independent warmstart data (reduce adapter borrowing)

Instead of warmstarting from huikang's v27 adapter, use the API to generate a full
clean corpus and train from scratch. This eliminates the dependency on borrowed adapters
entirely — important if:
- The competition ends and huikang's public Kaggle datasets are removed
- We want to publish our own adapter lineage without relying on a competitor's weights

Requires more training steps than the 240-step v0.5 warmstart approach.

### Use case 4 — GRPO reward signal augmentation

For v0.6 GRPO, the reward function currently checks `\boxed{}` extraction against
ground truth. The NVIDIA API could be used to generate **additional problem variants**
with known answers, expanding the GRPO training distribution beyond the 9,500
competition problems.

---

## Rate limits and volume estimates

| Free tier | Value |
|---|---|
| Requests per minute | ~60 RPM |
| Tokens per minute | ~40,000 TPM |
| Daily limit | ~1,000 requests (varies by model) |

For 9,500 competition problems at ~60 RPM: ~2.6 hours of API calls.  
For 21,500 v0.5 examples: ~6 hours. Feasible to batch overnight.

A `scripts/generate_api_data.py` script with retry/backoff + rate limiting would handle this.

---

## Implementation sketch

```python
# scripts/generate_api_data.py (not yet written)
#
# Reads train.csv + synthetic problems, calls NVIDIA API for each,
# writes output to data/v_api_train.jsonl in the same format as
# data/v0.5_train.jsonl (messages + bucket fields).
#
# Usage:
#   NVIDIA_API_KEY=nvapi-... python scripts/generate_api_data.py \
#       --input data/train.csv \
#       --output data/v_api_train.jsonl \
#       --model nvidia/llama-3.1-nemotron-70b-instruct \
#       --response-style short   # or: long-cot
```

---

## Relationship to existing plans

| Plan | Status | Dependency on borrowed assets |
|---|---|---|
| v0.5 SFT (`v0.5-sft-kuangyicheng-plan.md`) | In progress | Warmstart: huikang v27 adapter |
| v0.6 GRPO (`v0.6-grpo-plan.md`) | Pending v0.5 | Init: v0.5 adapter (our own) |
| v0.5 NeMo (`v0.5-nemo-framework-plan.md`) | Deferred | Dataset: v0.5 data (our own) |
| **This plan** | Deferred | None — generates everything independently |

By v0.6 GRPO we are already using our own adapter (init from v0.5). The main remaining
dependency is the v27 warmstart used in v0.5 SFT. If we need a v0.7+ SFT run after the
competition, this plan provides a path to generate fresh data without relying on
kuangyicheng or huikang's published resources.

---

## Decision criteria — when to pursue

- v0.5 scores **below 0.80**: consider API-generated short-response data to improve SFT quality before GRPO
- Competition ends and huikang/kuangyicheng datasets become unavailable: Use case 3 (train from scratch)
- v0.6 GRPO plateaus: Use case 4 (expand GRPO problem distribution via API)
- Otherwise: defer; v0.5 + v0.6 path is sufficient for competition timeline
