# Community Insights — Kaggle Nemotron Discussion Threads

Aggregated findings from competition discussion threads. Each section covers one thread or poster; insights are cross-referenced where they overlap.

| Section | Thread | Poster | Key topic |
|---|---|---|---|
| 1 | [#707987](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/707987) | Buzz shocker | SFTConfig hparams, RTX Pro 6000 OOM fix |
| 2 | (same thread follow-up) | Q3 poster | Category weight distribution, 0.74 LB via symbolic solvers |
| 3 | (cross-thread) | DaoHe Liu | Unsloth MoE fused keys vs per-expert LoRA; `gate_proj` SiLU root cause |
| 4 | [#684251](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684251) | TBD | TBD — pending content |

---

## 1. Discussion #707987 — Buzz shocker: SFTConfig hparams + RTX Pro 6000 OOM fix

**Date:** 2026-06-13  
**Source:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/707987  
**Symptom / Trigger:** Community post claims `attn_implementation="eager"` + specific SFTConfig params "finally allowed" full-context (8192 tok) training on RTX Pro 6000 in 7h.

---

## 1. Applicability of posted config to our RTX Pro 6000 runs

### Context

Poster "Buzz shocker" shared a working SFTConfig for RTX Pro 6000 at `MAX_SEQ_LEN=8192`. Key params:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    attn_implementation = "eager",
    ...
)
target_modules = ["q_proj","k_proj","v_proj","o_proj","in_proj","out_proj","up_proj","down_proj"]
model = FastLanguageModel.get_peft_model(model, r=LORA_RANK, lora_alpha=LORA_ALPHA, ...)

SFTConfig(
    per_device_train_batch_size   = 1,
    gradient_accumulation_steps   = 32,   # effective batch = 32
    learning_rate                 = 5e-5, # lowered "to stabilize the high Alpha 128"
    lr_scheduler_type             = "linear",
    warmup_steps                  = 10,
    max_grad_norm                 = 1.0,
    save_strategy                 = "no",
    bf16                          = True,
    gradient_checkpointing        = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    packing                       = False,
)
```

Implied: `lora_alpha=128` (stated in comment), `r` unspecified but likely 16 or 32.

**`FastLanguageModel.from_pretrained` comparison:**

| Parameter | Poster | Our RTX Pro 6000 | Our DGX Spark |
|---|---|---|---|
| `dtype` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
| `max_seq_length` | **passed (8192)** | **not passed** (post-load override) | **not passed** (post-load override) |
| `load_in_4bit` | False | False | False |
| `load_in_8bit` | False | False | False |
| `full_finetuning` | False | False | False |
| `trust_remote_code` | **True** | **omitted (False)** | **omitted (False)** |
| `unsloth_force_compile` | False | False | False |
| `attn_implementation` | eager | eager | eager |

`trust_remote_code=True` (poster): loads old `modeling_nemotron_h.py` from the model repo, which has broken KV cache at inference (~20x generation slowdown). We deliberately omit it so transformers ≥5.3.0 uses its native NemotronH implementation. See discussion #690161.

**Platform difference — `trust_remote_code` had opposite requirements on each platform:**

| | DGX Spark (GB10) | RTX Pro 6000 (Kaggle) |
|---|---|---|
| Environment | Docker image, `transformers 5.5.3` installed fresh | Kaggle base env, older transformers pre-5.3.0 |
| Omit `trust_remote_code` | ✅ Works immediately — native `NemotronHForCausalLM` | ❌ Hangs — transformers shows interactive "Do you wish to run custom code?" prompt |
| `trust_remote_code=True` | ✅ Also works (but loads buggy file) | ✅ Required to avoid hang — loads buggy `modeling_nemotron_h.py` |
| KV cache bug impact | N/A during SFT (no generation loop) | N/A during SFT (no generation loop) |

**Why the KV cache bug doesn't affect SFT training:** The broken cache in `modeling_nemotron_h.py` (name mismatch: `past_key_values` vs `cache_params`) only fires during autoregressive generation — each token recomputes the full sequence instead of using cached states. SFT uses teacher-forced forward passes with no generation loop, so `trust_remote_code=True` is harmless for training loss but would be catastrophic for GRPO rollouts and submission inference.

**Why Kaggle notebooks temporarily required `trust_remote_code=True`:** Two failed attempts to avoid it:
1. Pin `transformers==5.5.3` → pip conflicts with Kaggle base env; even when install succeeded, the already-loaded module wasn't replaced without a kernel restart
2. Omit without upgrading → interactive prompt hangs the notebook indefinitely

`trust_remote_code=True` was restored in Kaggle notebooks (`dade990`, Jun 3) and permanently removed (`7f44792`, Jun 7) once Kaggle's base environment shipped `transformers ≥ 5.3.0`.

`max_seq_length` not passed (ours): Unsloth caps `model.max_seq_length` to 2048 on NemotronH regardless of what is passed here. We skip it and apply a post-load override (`model.max_seq_length = MAX_SEQ_LENGTH`) which is the only path that actually sticks. Passing it in `from_pretrained` is a no-op for this model class.

**`SFTConfig` comparison:**

Side-by-side comparison of poster's config vs our two platforms:

| Parameter | Poster (discussion #707987) | Our RTX Pro 6000 (`v09_train_kaggle.ipynb`) | Our DGX Spark (`train_v9_sft.py` run13/14) |
|---|---|---|---|
| `max_seq_length` | 8192 | 7680 (run7), 4096 (run6) | 2048 (run13), 4096 (run14) |
| `batch_size` | 1 | 1 | 1 |
| `grad_accum` | 32 | 16 | 16 |
| `effective_batch` | 32 | 16 | 16 |
| `learning_rate` | 5e-5 | 2e-4 (fresh), 1e-4 (warmstart) | 2e-4 (run13), 1e-4 (run14) |
| `lora_r` | unknown | 32 | 32 |
| `lora_alpha` | 128 | 32 | 32 |
| `effective LoRA scale` | ~2e-4 (if r=32) | 2e-4 | 2e-4 |
| `warmup_steps` | 10 | dynamic `min(50, n_steps//10)` | dynamic `min(50, n_steps//10)` |
| `adam_beta1` | 0.9 | 0.9 | 0.9 |
| `adam_beta2` | 0.95 | 0.95 | 0.95 |
| `adam_epsilon` | 1e-8 | 1e-8 | 1e-8 |
| `weight_decay` | 0.0 | 0.0 | 0.0 |
| `max_grad_norm` | **1.0** | **1e9** | **1e9** |
| `lr_scheduler_type` | linear | linear | linear |
| `attn_implementation` | eager | eager | eager |
| `gradient_checkpointing` | True | True | True (via `_set_gradient_checkpointing()`) |
| `use_reentrant` | False | False | False |
| `packing` | False | False | False |
| `bf16` | True | True | True |
| `save_strategy` | no | no (rolling ckpt callback) | no (rolling ckpt callback) |
| `out_proj` in targets | yes | **no** (dead path) | **no** (dead path) |
| `gate_proj` in targets | no | yes (no-op) | yes (no-op) |
| per-expert MoE LoRA | no | no (Unsloth fused, dropped) | **yes** (11,776 keys, 878M params) |
| trainable params | ~27M (est.) | ~27M | **878M** |

### Investigation Checklist

- [x] Do we already train successfully at or near 8192 seq length? **Yes — 7680 since run7**
- [x] Is our OOM situation the same as the poster's? **No — we solved it via NemotronH native GC bypass + expandable allocator**
- [x] Is `attn_implementation="eager"` novel vs our setup? **No — already set (train_v9_sft.py:169)**
- [x] Is `out_proj` inclusion actionable? **No — confirmed zero-gradient dead path (see [[mamba-out-proj-lora-dead-path]])**
- [x] Is `max_grad_norm=1.0` vs our `1e9` a meaningful difference? **Yes — see Findings**
- [x] Is `grad_accum=32` vs our 16 worth switching? **No — halves throughput in 9h Kaggle window**
- [x] Is alpha=128+lr=5e-5 vs our alpha=32+lr=2e-4 meaningfully different? **Comparable effective scale**

### Findings

**`max_grad_norm = 1.0` vs `1e9`**: This is the most actionable difference. Our `1e9` is effectively no clipping. Standard clipping at `1.0` prevents gradient spikes — potentially relevant for NemotronH's hybrid MoE+Mamba architecture where Mamba SSM layers can produce large gradient norms. Our run13 loss reached 0.1220 at step 1000 without instability, so `1e9` has not caused obvious damage yet. However if future runs (larger seq, warmstart instability) show loss spikes, lowering to `1.0` is the first lever.

**`lora_alpha=128` + `lr=5e-5` vs our `alpha=32` + `lr=2e-4`**: Effective LoRA update magnitude ∝ `lr × alpha/r`. If poster uses `r=32`: `5e-5 × 4 = 2e-4` — identical to our `2e-4 × 1`. If `r=16`: `5e-5 × 8 = 4e-4` — 2× larger. No strong reason to switch; same family of solutions.

**`grad_accum=32`**: Smoother gradients but halves our step throughput on the 9h Kaggle wall-clock limit. At run7 pace (~46s/step), 32 accum would cut ~363 steps to ~181 steps — worse coverage. Not worth it.

**`warmup_steps=10`**: Very aggressive. Our dynamic warmup (up to 50 steps) is more conservative and appropriate for warmstart runs where optimizer state is cold.

**The poster's OOM fix**: They describe `attn_implementation="eager"` + their SFTConfig as what "finally allowed" 8192 training. We solved the same problem differently: Unsloth's NemotronH native GC bypass (`_set_gradient_checkpointing()`) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. These are equivalent memory solutions; their approach is not novel for us.

**`out_proj` in target_modules**: Poster includes it. We deliberately exclude it — Unsloth's Mamba fast-path `UnslothCheckpointFunction` guard fires because the SSM scan output does not carry `requires_grad=True`, producing zero gradient for `out_proj` throughout training. See [[mamba-out-proj-lora-dead-path]] for full root cause.

**`gate_proj` in target_modules**: We include it; poster omits it. Confirmed no-op — `adapter_model.safetensors` from both `adapter_v12_spark_ckpt` and `adapter_v9_run13_ckpt` contain zero `gate_proj` keys, and `expert_lora_weights.pt` contains only `lora_A_up`/`lora_B_up`/`lora_A_down`/`lora_B_down` keys (92 total, zero gate entries). **Root cause**: NemotronH MoE experts use **SiLU activation** (not SwiGLU) — there is no gate projection in the architecture. Unsloth's official docs list `gate_proj` as a NemotronH target module, but it is architecturally absent. PEFT silently skips unmatched target strings. Safe to remove from `target_modules` with no effect on adapter size or score.

### Actions Taken

None — investigation only. No config changes made.

### Resolution

**Status: Resolved (no action required)**

The poster's config solves an OOM problem we already solved differently. Only one parameter differs meaningfully (`max_grad_norm=1.0` vs our `1e9`). Deferred as a stability lever for future runs if loss instability appears.

### Follow-ups

- **`max_grad_norm=1.0` is irrelevant for run15**: Run14 peak grad_norm was **0.1759** across all 300 steps — clipping at 1.0 would never have triggered. The plateau at loss 0.285 is a data/optimizer convergence issue, not a gradient explosion issue. Do not change `max_grad_norm` for run15.

- **Adam params identical — no action needed**: `beta1=0.9`, `beta2=0.95`, `epsilon=1e-8`, `weight_decay=0.0` all match the poster exactly. No changes warranted.

- **Run15 planning**: The run14 plateau (loss 0.285 from step 80, vs run13's 0.122 at step 1000) likely reflects LR=1e-4 being too conservative for warmstart on new data. Run13→run14 warmstart on v0.12 data hit a local minimum immediately. Candidate directions once run14 score is known:
  - **If run14 scores ≥ 0.58**: continue v0.12 approach at higher LR (2e-4 warmstart or even fresh start on v0.12)
  - **If run14 scores < 0.58**: the v0.12 augmented data hurts; revert to v0.9 data and extend training further (run15 = longer run13 continuation)
  - Either way: `max_grad_norm=1.0` is not a lever here.

- **`lora_alpha` ablation**: Low priority. Effective LoRA scale matches poster when `r=32`. Not a factor in the run14 plateau.

---

## 3. Follow-up: Community discussion — category weight distribution (2026-06-13)

### Context

A second community member (Q3) reported **0.74 LB** using a symbolic-solver-generated 8,700-row CoT corpus, stating:

> *"COT formatting and possible Category weight distribution are the real challenges."*

This prompted investigation of our v0.12 training data distribution.

### Finding: severe category imbalance in v0.12_train.jsonl

```
python3 -c "
import json
from collections import Counter
cats = [json.loads(l).get('category','?') for l in open('data/v0.12_train.jsonl')]
for c,n in sorted(Counter(cats).items(), key=lambda x:-x[1]):
    print(f'{n:5d}  {c}')
"
```

| Category | Count | Share |
|---|---|---|
| `matching` | 4,515 | 17.7% |
| `concatenation` | 2,851 | 11.2% |
| `splitting` | 2,850 | 11.2% |
| `bit_manipulation` | 2,771 | 10.9% |
| `cipher` | 2,731 | 10.7% |
| `gravity` | 2,340 | 9.2% |
| `unit_conversion` | 2,257 | 8.9% |
| `numeral` | 2,008 | 7.9% |
| `spelling` | 1,219 | 4.8% |
| `equation_numeric_deduce` | 961 | 3.8% |
| `lstrip` | 300 | 1.2% |
| `equation_numeric_guess` | 180 | 0.7% |
| `cryptarithm_guess` | 178 | 0.7% |
| `cryptarithm_deduce` | 170 | 0.7% |
| `equation_numeric` | 168 | 0.7% |
| **`equation_symbolic`** | **1** | **0.004%** |
| **Total** | **25,500** | |

**Critical issue**: `equation_symbolic` has **1 training example**. The model has essentially no gradient signal for this category. The top 3 categories (`matching`, `concatenation`, `splitting`) account for 40% of all data — stratified batching distributes them evenly across batches but does not change the total training frequency ratio.

**Our `StratifiedSFTTrainer`** addresses batch-level distribution (every gradient accumulation window sees a category mix) but does not equalize total example counts per category. The model still trains on `matching` 4,515× and `equation_symbolic` 1×.

### Why the 0.74 poster likely scores higher

They used **their own symbolic solvers** to generate an 8,700-row corpus — the key word being *their own*, with presumably balanced coverage across all categories.

We also use programmatic/algorithmic generators — the huikang repo's `augmenters/` directory (`augmenters/matching.py`, `augmenters/splitting.py`, `augmenters/concatenation.py`, etc.) are exactly this: symbolic problem generators with known answers. The method is the same class of tool. The difference is **coverage**:

- Huikang augmenters exist for `matching`, `concatenation`, `splitting`, `bit_manipulation`, `cipher`, `gravity`, `unit_conversion`, `numeral` — these are the categories with thousands of examples
- `equation_symbolic` has **no huikang augmenter** — the v0.12 plan explicitly notes `| equation_symbolic | 1 | skip (no augmenter) |`
- `lstrip`, `equation_numeric`, `cryptarithm_*` have limited or no augmenters

Our v0.9 data used `cap 1500/cat` and was far more balanced. The v0.12 augmentation skewed heavily toward categories where the huikang augmenters had rich coverage, while the underrepresented categories were left at their original v0.9 counts or lower.

### Actions for run16

- **Rebalance v0.12 data**: cap or downsample over-represented categories (≥2,000 → cap at ~1,500) and oversample or generate more examples for under-represented categories (`equation_symbolic`, `equation_numeric`, `cryptarithm_*`, `lstrip`)
- **`equation_symbolic` is a data bug**: 1 example is insufficient. Generate synthetic examples via symbolic solver or extend from huikang augmenters before run16
- **Target distribution**: ≥500 examples per category minimum; cap at 1,500–2,000 per category maximum
- Run15 (currently training) uses the imbalanced v0.12 data — score will reflect this; run16 should address distribution before training

---

## 2. Applicability of posted config to DGX Spark (GB10) runs

### Context

Our DGX Spark runs (`v09_train_spark_gb10.ipynb`) use `train_v9_sft.py` with `seq=2048` (run13) and `seq=4096` (run14). The GB10 has ~400 GB unified memory.

### Findings

Not applicable. The poster's config addresses VRAM pressure on a 128 GB GPU. The GB10 has no memory constraint at our seq lengths. `max_grad_norm=1.0` is equally relevant here as a stability lever (same code path), but run13 completed at 0.1220 loss without gradient issues.

### Resolution

**Status: Deferred**

No config changes warranted for Spark runs based on this discussion.

---

## 4. Community insight — Unsloth MoE fused keys vs per-expert LoRA (DaoHe Liu)

**Date:** 2026-06-13  
**Source:** Kaggle competition discussion (DaoHe Liu)

### Context

A community member reported confusion: they used the standard Unsloth `get_peft_model` call with `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj"]` and saw:

> *"Unsloth: Detected MoE model with num_experts = 128 and target_modules = [...]. Enabling LoRA on MoE parameters: ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']"*

They observed "very few trainable parameters" because Unsloth was only applying LoRA to the fused MoE expert matrices rather than the individual expert projections they specified.

### Findings

**This is the exact fused-key problem we solved in run13.** Three layers of insight:

**1. Unsloth fuses MoE expert projections**

When Unsloth detects a MoE model, it replaces individual `gate_proj`/`up_proj`/`down_proj` targets with fused expert tensors (`mlp.experts.gate_up_proj`, `mlp.experts.down_proj`). The resulting LoRA keys are in Unsloth's internal format — not compatible with Kaggle's vLLM evaluator (`LoRARequest` via standard PEFT). This is why runs 1–12 had expert LoRA trained locally but silently dropped at Kaggle inference.

**2. NemotronH MoE uses SiLU, not SwiGLU — no `gate_proj` exists**

For standard SwiGLU MoE models, Unsloth fuses `gate_proj` + `up_proj` → `gate_up_proj`. For NemotronH, the MoE experts use **SiLU activation** with only `up_proj` and `down_proj` — no gate projection in the architecture at all. Confirmed by inspecting `expert_lora_weights.pt` from run13:

```
Total expert LoRA keys: 92
Keys: lora_A_up, lora_B_up, lora_A_down, lora_B_down  (23 layers × 4)
Gate-related keys: 0
```

Unsloth's official documentation lists `gate_proj` as a NemotronH target module — this is incorrect for the MoE layers. `gate_proj` is architecturally absent from NemotronH's routed experts.

**3. Our per-expert injection bypasses the fused path entirely**

Our solution (`train_v9_sft.py` MoE LoRA injection) injects LoRA directly into each of the 128 routed experts per MoE layer, producing PEFT-compatible keys:
```
mixer.experts.{j}.up_proj.lora_A.weight   (128 experts × 23 layers × 2 = 5,888 keys)
mixer.experts.{j}.down_proj.lora_A.weight (128 experts × 23 layers × 2 = 5,888 keys)
Total: 11,776 per-expert PEFT keys
```
These are saved to `expert_lora_weights.pt` and converted by `package_submission.sh` before submission. The Kaggle vLLM evaluator loads them correctly — confirmed by run13-step500 scoring 0.58 and run14-step300 scoring 0.64.

### Resolution

**Status: Resolved (our implementation is correct)**

The community member's confusion mirrors our own early runs (1–12). Our per-expert LoRA injection + `package_submission.sh` conversion is the correct solution. The "very few trainable parameters" symptom they describe is the fused Unsloth format — submitting those keys to Kaggle either errors or silently discards the expert LoRA, leaving only the attention adapter (~27M params).

### Actions

None — this validates our existing approach. Confirms `gate_proj` removal from `target_modules` is correct (SiLU architecture, no gate projection).
