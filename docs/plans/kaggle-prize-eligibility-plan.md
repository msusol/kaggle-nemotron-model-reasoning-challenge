# Kaggle Prize Eligibility Plan

**Status:** In progress — prize notebook exists but needs update to reflect v0.5 Unsloth approach  
**Dependency:** v0.5-sft-unsloth adapter score from GB10 training run (currently running)

---

## What prize eligibility actually requires

Based on competition rules [cite:1] and our existing prize notebook
(`gdataranger/nemotron-3-nano-30b-lora-reasoning-challenge`):

1. **Public notebook on Kaggle** — documents approach, shows inference ✅ (exists, needs update)
2. **Training methodology documented** — describe what was done, config, data ✅
3. **Adapter demonstrated in inference** — load adapter on Kaggle, run predictions ✅
4. **Write-up** — approach explanation ⬜ pending

**Training does NOT need to run on Kaggle.** The existing prize notebook explicitly
states training ran on GB10 via Docker. This is standard practice — document the
external training, demonstrate the result on Kaggle. The competition rules as documented
do not require training to happen on Kaggle's compute.

---

## What the current prize notebook shows (outdated)

`gdataranger/nemotron-3-nano-30b-lora-reasoning-challenge` currently documents:
- v0.2-cot approach (Gemini CoT traces, `kienngx` dataset)
- Training via `train_lora.py` on GB10
- Pre-computed outputs from v0.2-cot adapter (score 0.54)

**This needs updating** to reflect the v0.5 Unsloth approach and final adapter.

---

## Required notebook updates (after v0.5-sft-unsloth scores)

| Section | Current state | Required update |
|---|---|---|
| Approach description | v0.2-cot, Gemini traces | v0.5 SFT, kuangyicheng short-response approach |
| Dataset | `kienngx` CoT dataset | `train.csv` (9,500) + synthetic (12,000) |
| Training config | `train_lora.py`, seq=2048 | `train_v5_sft.py`, Unsloth, seq=6144, 240 steps |
| Adapter reference | `output/adapter_20260528_211916` | `output/adapter_v5_sft_unsloth` |
| Score | 0.54 (v0.2-cot) | v0.5-sft-unsloth score (pending) |
| Pre-computed outputs | v0.2-cot inference | v0.5-unsloth inference |
| Section 5 (results) | Empty | Fill with final scores table |

---

## The Kaggle CPU training question

Tong Hui Kang noted in competition discussion #687961 that people have trained the
30B model on Kaggle's CPU with 96 GB RAM. This is **not required for prize eligibility**
but is valuable for two reasons:

### 1. Reproducibility demonstration
Showing that someone with only Kaggle's free compute can reproduce the training approach
(even if slower) is good for competition transparency and the community. It answers:
"Can I reproduce this without a DGX Spark?"

### 2. Prize eligibility insurance
If the competition judges interpret prize eligibility as requiring the approach to be
runnable on Kaggle's infrastructure, CPU training provides that guarantee. Better to
have it than to find out later it was required.

---

## Kaggle CPU training feasibility

| Parameter | Value | Notes |
|---|---|---|
| Kaggle competition RAM | 96 GB | Confirmed by Tong, fits 60 GB BF16 model |
| Kaggle notebook time limit | 9 hours | Standard competition limit |
| Estimated step time (CPU, standard PEFT) | 10–30 min/step | Very rough; depends on CPU cores |
| Steps feasible in 9 hours | ~20–50 | Not 240 — need to test |
| Expected score | ~0.62 | kuangyicheng's Kaggle notebook result |

**Key unknown**: actual per-step time on Kaggle's competition CPU. Must measure
empirically. Run `max_steps=3` first to get a real timing, then calculate feasibility.

### Modifications needed to run on Kaggle CPU

The kuangyicheng notebook needs these changes:
```python
# 1. Use competition model path (already correct in their notebook)
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"

# 2. Reduce max_steps based on measured step time
training_args = SFTConfig(
    max_steps=STEPS_THAT_FIT_IN_9_HOURS,  # measure first with max_steps=3
    ...
)

# 3. No other changes needed for CPU — model loads fine in 96 GB RAM
```

The notebook as-is will fall back to CPU automatically when no GPU is available.
`FastLanguageModel.from_pretrained` handles CPU loading.

---

## Recommended execution order

### Step 1 — Wait for GB10 Unsloth run to complete (happening now, ~6h)

Training `v5_sft_unsloth` on GB10. Expected completion ~02:00 AM.

### Step 2 — Submit GB10 adapter and record Kaggle score

Package `output/adapter_v5_sft_unsloth` and submit.
This is the primary score — target ≥ 0.85.

### Step 3 — Update prize eligibility notebook

Update `gdataranger/nemotron-3-nano-30b-lora-reasoning-challenge` to reflect:
- v0.5 Unsloth approach
- Final adapter
- Actual Kaggle score
- Updated training config and data description

### Step 4 (optional) — Kaggle CPU training notebook

Fork `kuangyicheng/nemotron-087-training` with modifications:
- Add timing cell: `max_steps=3` to measure actual CPU step time
- Based on result: set `max_steps` to fit within 9 hours
- Run, submit resulting adapter, record score

This provides a Kaggle-native training demonstration at no cost.

---

## What does NOT need to happen

- Training does not need to happen on Kaggle for prize eligibility
- The CPU training notebook is a bonus, not a blocker
- The prize notebook inference section works as-is (just needs adapter path update)
- No need to install Unsloth in the Kaggle notebook — standard PEFT inference is fine
  for the prize notebook (loading + inference only, no training)
