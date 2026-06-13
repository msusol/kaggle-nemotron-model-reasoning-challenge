# Community hyperparameter config — discussion #707987 (Buzz shocker)

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

**`gate_proj` in target_modules**: We include it; poster omits it. Confirmed no-op — `adapter_model.safetensors` from both `adapter_v12_spark_ckpt` and `adapter_v9_run13_ckpt` contain zero `gate_proj` keys. NemotronH has no linear named `gate_proj` (neither in shared experts nor routed experts); PEFT silently skips unmatched target strings. Safe to remove from `target_modules` in run15 with no effect on adapter size or score.

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

## 2. Applicability of posted config to DGX Spark (GB10) runs

### Context

Our DGX Spark runs (`v09_train_spark_gb10.ipynb`) use `train_v9_sft.py` with `seq=2048` (run13) and `seq=4096` (run14). The GB10 has ~400 GB unified memory.

### Findings

Not applicable. The poster's config addresses VRAM pressure on a 128 GB GPU. The GB10 has no memory constraint at our seq lengths. `max_grad_norm=1.0` is equally relevant here as a stability lever (same code path), but run13 completed at 0.1220 loss without gradient issues.

### Resolution

**Status: Deferred**

No config changes warranted for Spark runs based on this discussion.
