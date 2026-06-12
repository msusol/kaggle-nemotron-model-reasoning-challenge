# Routed Expert LoRA: target_parameters not captured in PEFT adapter save

**Date:** 2026-06-10
**Symptom:** Run6 adapter_config.json lists `target_parameters` for routed experts, but adapter_model.safetensors contains only 278 standard LoRA keys — no routed expert weights.
**Root cause:** Unsloth's `target_parameters` trains routed-expert tensors as direct parameter updates (not LoRA A/B matrices); PEFT adapter save only captures `target_modules` LoRA weights.
**Status:** Partially resolved — run6 submission initially ERRORed (46 fused MoE expert keys present); filtered 232-key submission created and resubmitted.

---

## 1. Background: two kinds of Unsloth LoRA targets

Unsloth exposes two config fields for trainable parameters:

| Field | Meaning | Saved as LoRA? |
|---|---|---|
| `target_modules` | Standard `nn.Linear` layers wrapped with LoRA A/B matrices | ✅ Yes — `lora_A.weight` / `lora_B.weight` in safetensors |
| `target_parameters` | Raw `torch.Parameter` tensors (fused MoE expert weights) trained directly | ❌ No — direct weight deltas; not separable as LoRA adapter |

For run1–6 our `target_modules` covered:
```
["q_proj", "k_proj", "v_proj", "o_proj",
 "gate_proj", "up_proj", "down_proj", "in_proj", "out_proj"]
```

Unsloth also added to `target_parameters`:
```
["mlp.experts.gate_up_proj", "experts.down_proj"]
```

These are the **128 routed expert** fused tensors per MoE layer (shape `[128, d_ffn, d_model]`
— a single batched parameter, not 128 separate `nn.Linear` modules).

---

## 2. Investigation

### What's in the run6 safetensors

```
Total keys: 278
Key breakdown (confirmed via safe_open inspection):
  23  mixer.experts.lora_A.weight       shape=[4096, 2688]  ~21 MB each  ← fused MoE (INCOMPATIBLE)
  23  mixer.experts.lora_B.weight       shape=[1856, 4096]  ~15 MB each  ← fused MoE (INCOMPATIBLE)
  23  mixer.in_proj.lora_A.weight       shape=[32, 2688]                  ← Mamba SSM ✓
  23  mixer.in_proj.lora_B.weight       shape=[10304, 32]                 ← Mamba SSM ✓
  23  mixer.out_proj.lora_A.weight      shape=[32, 10304]                 ← Mamba SSM ✓
  23  mixer.out_proj.lora_B.weight      shape=[2688, 32]                  ← Mamba SSM ✓
  12  mixer.{q,k}_proj.lora_{A,B}       r=32 shape                        ← attention ✓
  12  mixer.{v,o}_proj.lora_{A,B}       r=32 shape                        ← attention ✓
  46  mixer.shared_experts.{up,down}_proj.lora_{A,B}                      ← shared MoE ✓
```

**Correction to prior investigation note:** Initial analysis incorrectly concluded "zero expert keys."
The safetensors DOES contain 46 `mixer.experts.lora_A/B.weight` keys (the fused MoE keys).
These large-shape tensors (e.g. [4096, 2688]) are NOT r=32 LoRA matrices — they are Unsloth's
fused expert weight format. They caused all 3 run6 submission attempts to ERROR.

### Why the routed expert updates are not saved

When Unsloth calls `model.save_pretrained()` for a PEFT adapter:
1. PEFT's save logic iterates `target_modules` and writes lora_A/lora_B tensors.
2. `target_parameters` are native `torch.Parameter` objects on the model. Their
   delta from the original base-model weight is not tracked separately — Unsloth
   trains them in-place. There is no lora_A/lora_B decomposition to save.
3. Capturing these updates requires saving the **full merged model** (or at minimum
   the updated fused parameter tensors as `modules_to_save`).

### Comparison with huikang approach

Huikang's notebook uses **per-expert LoRA** — each of the 128 experts is a separate
`nn.Linear` (or Unsloth-wrapped equivalent), so keys like:

```
base_model.model.backbone.layers.8.mixer.experts.97.down_proj.lora_A.weight
base_model.model.backbone.layers.8.mixer.experts.98.down_proj.lora_A.weight
```

appear in the adapter safetensors. This results in a much larger adapter (128 experts
× N MoE layers × 2 directions × lora_A + lora_B).

Our approach uses Unsloth's fused batched form — one tensor covers all 128 experts,
which is more memory-efficient during training but cannot be split back into per-expert
LoRA keys for a standard PEFT adapter save.

---

## 3. Findings

- The 46 `mixer.experts.lora_A/B.weight` keys are Unsloth's fused representation of the
  MoE expert LoRA weights (batched into shape [4096, 2688] per key). The standard NemotronH
  model has no `mixer.experts` module as a standalone `nn.Linear` — PEFT cannot attach
  these tensors and ERRORs. Same root cause as run1's initial submission (ADR-0005).
- The routed expert weight updates from run6 training **were discarded at save time**.
  The submitted adapter reflects only attention, shared-expert, and Mamba SSM LoRA.
- The 1.5 GB dataset size comes largely from the 46 fused expert keys (~36 MB × 2 each).
  After filtering those out the submission safetensors is ~111 MB.
- The `target_parameters` field in `adapter_config.json` is a config-level artifact.
  Whether PEFT's LoraConfig silently ignores it is version-dependent; `package_submission.sh`
  now strips it (and `use_qalora`, `qalora_group_size`) to be safe.

---

## 4. Actions Taken

- Re-inspected run6 safetensors — found 46 fused MoE expert keys (earlier check missed them).
- Created filtered submission with 232 keys: 48 attn + 92 shared experts + 92 Mamba SSM.
  Expert keys removed; adapter_config.json rebuilt with standard PEFT fields only.
- Updated `scripts/package_submission.sh` to rebuild config from scratch (not just patch 2 fields).
- Updated `notebook/v09_train_kaggle.ipynb` cell-lora warmstart path resolution to handle
  Kaggle's new dataset mount layout (`/kaggle/input/datasets/gdataranger/<slug>/`).
- Resubmitted filtered 232-key run6 adapter.

---

## 5. Resolution

**Status: Partially resolved (as of run6)**

Three run6 submission attempts ERRORed due to fused MoE expert keys. A filtered 232-key
submission was created (expert keys stripped) and resubmitted. Score pending.
The routed expert training gap (weights trained via `target_parameters` but not captured
in PEFT adapter save) is a separate known limitation — deferred per Option C.

---

## 6. Follow-ups

### Option A: Per-expert LoRA (matches huikang)  ✅ IMPLEMENTED — run12 (2026-06-11)
Replace `target_parameters` with explicit per-expert `target_modules`. Requires the
routed expert tensors to be exposed as individual `nn.Linear` modules before calling
`get_peft_model`. Unsloth may or may not support this on GB10/Blackwell — needs testing.
Adapter size would be ~10–20× larger.

### Option B: Save full merged model, then extract delta
After training, compute `trained_weight - base_weight` for each fused expert tensor and
package as `modules_to_save` or a custom delta format. Not compatible with standard
PEFT `LoRARequest` unless the evaluator supports `modules_to_save`.

### Option C: Accept the gap
Continue training only `target_modules` LoRA. The Mamba SSM layers (run6 addition) are
new coverage that may improve scores without routed expert LoRA. Revisit if run6 and
run7 scores plateau before the huikang-tier ceiling.

**Priority**: Wait for run6 score. If score is competitive (≥0.60), Option C may be
sufficient. If gap persists, investigate Option A for a future run.

---

## 7. Option A Implementation — Per-Expert LoRA via nn.Parameter injection (run12)

**Date:** 2026-06-11  
**Status: CONFIRMED WORKING**

### Approach

`NemotronHExperts` holds fused expert tensors as 3D parameters (`[E, out, in]`), not as
`nn.Linear` modules. PEFT cannot target them with `target_modules`. Instead, we inject
LoRA A/B matrices as raw `nn.Parameter` objects directly onto each `NemotronHExperts`
instance before calling `get_peft_model`, then patch the class-level `forward` to add
the LoRA contribution per-expert via index slicing.

**Parameters added per NemotronHExperts module:**

| Name | Shape | Init |
|---|---|---|
| `lora_A_up` | `[E, r, in_features]` | kaiming_uniform |
| `lora_B_up` | `[E, out_features, r]` | zeros |
| `lora_A_down` | `[E, r, in_features]` | kaiming_uniform |
| `lora_B_down` | `[E, out_features, r]` | zeros |

E=128 experts, r=32, in=2688 (hidden), out=1856 (moe_intermediate_size).

### Root Cause of Freezing (runs 9–11)

`FastBaseModel.get_peft_model` → `post_patch_model` → `prepare_model_for_training`
(`/usr/local/lib/python3.12/dist-packages/unsloth_zoo/training_utils.py` line 147):

```python
if ".lora_A." in name or ".lora_B." in name or ".lora_magnitude_vector" in name:
    requires_grad = True
else:
    requires_grad = False
```

The check requires `.lora_A.` with **dots on both sides**. Our params are named
`lora_A_up`, `lora_B_up`, etc. — underscore suffix, no trailing dot. They were silently
frozen every run (runs 9–11 all showed 27.7M trainable instead of 884M).

### Fix

After `get_peft_model` returns, explicitly re-enable `requires_grad` for all four
expert LoRA param basenames (`scripts/train_v9_sft.py` lines ~363–374):

```python
_expert_lora_param_names = {"lora_A_up", "lora_B_up", "lora_A_down", "lora_B_down"}
for _pn, _pp in model.named_parameters():
    if _pn.rsplit(".", 1)[-1] in _expert_lora_param_names:
        _pp.requires_grad_(True)
        _regrad_count += 1
```

### Proof — run12 log (2026-06-11)

```
[moe-lora] Injected per-expert LoRA into 23 NemotronHExperts modules, r=32, scaling=1.000
[moe-lora] Re-enabled requires_grad on 92 expert LoRA params
[moe-lora] first expert LoRA at 'base_model.model.model.layers.1.mixer.experts'
           A_up:[128, 32, 2688]  B_up:[128, 1856, 32]
           A_down:[128, 32, 1856]  B_down:[128, 2688, 32]
Trainable: 883,873,792 / Total: 32,461,811,136
Unsloth: Trainable parameters = 883,873,792 of 32,461,811,136 (2.72% trained)
```

92 params = 23 MoE blocks × 4 tensors. 883,873,792 trainable vs 27,711,488 in runs 9–11.

### Adapter Save

Expert LoRA weights are saved separately alongside the PEFT adapter as
`expert_lora_weights.pt` via `_save_expert_lora(output_dir)`. The Kaggle inference
notebook must load this file and apply the same per-expert forward patch at inference time.

---

## 8. AhHa — Kaggle's NemotronH is Unfused: expert LoRA IS submittable as standard PEFT

**Date:** 2026-06-11  
**Status: RESOLVED — `package_submission.sh` updated (commit a6627c3)**

### Discovery

A Kaggle discussion thread shared this starter notebook snippet:

```python
lora_config = LoraConfig(
    r=32,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    ...
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 880,138,240 || all params: 32,458,075,584 || trainable%: 2.71
```

**Standard PEFT with a regex gets 880M trainable params** — almost exactly the same as
our per-expert LoRA injection (878–883M). This is only possible if the NemotronH model
in Kaggle's evaluation environment **exposes individual `nn.Linear` modules per routed
expert** (`mixer.experts.{j}.up_proj`, `mixer.experts.{j}.down_proj`) accessible by
PEFT's named-module traversal.

Our DGX Spark training environment uses Unsloth's **fused** path: `NemotronHExperts`
stores all 128 expert weights as a single 3D tensor (`[E, out, in]`) — no individual
`nn.Linear` per expert. This is why we needed custom `nn.Parameter` injection in run12+.
The Kaggle evaluator's NemotronH is **unfused**.

### The Missing Link

Runs 9–13 all trained 856M+ expert LoRA params, but **every submission carried zero
expert LoRA at inference**:

| Submitted file | Content | Expert LoRA present? |
|---|---|---|
| `adapter_model.safetensors` | Standard PEFT keys for attention, shared-expert, Mamba in_proj | ❌ No routed-expert keys |
| `expert_lora_weights.pt` | Fused tensors `[128, r, dim]` per layer | Not in submission zip |

The `expert_lora_weights.pt` was never included in `submission.zip`. Even if it were,
the standard PEFT evaluator would not know to load it.

### Parameter Count Verification

Why 880M ≈ our 878M:

```
23 MoE layers × 128 experts × (
    up_proj lora_A  [32, 2688] + lora_B  [1856, 32] +    # 145,408 params
    down_proj lora_A [32, 1856] + lora_B [2688, 32]       # 145,408 params
) = 23 × 128 × 290,816 = 855,760,896

+ 23 × shared_expert (up+down lora_A+B) ≈ 13.4M
+ 23 × in_proj (Mamba SSM)              ≈  6.5M
+ out_proj / attention                  ≈  4.4M
                                        ≈ 880M  ✓
```

The Kaggle thread's 880M confirms the routed expert modules exist in the evaluator's
model and PEFT successfully wraps them individually.

### Fix: Convert Fused → Per-Expert PEFT Keys

`scripts/package_submission.sh` now loads `expert_lora_weights.pt` after filtering
and expands each `[128, r, dim]` tensor into 128 individual PEFT-format keys:

```
Input  (fused):   base_model.model.model.layers.{i}.mixer.experts.lora_A_up
                  shape [128, 32, 2688]

Output (per-expert, 128 keys per layer per tensor):
  base_model.model.model.layers.{i}.mixer.experts.0.up_proj.lora_A.weight  [32, 2688]
  base_model.model.model.layers.{i}.mixer.experts.1.up_proj.lora_A.weight  [32, 2688]
  ...
  base_model.model.model.layers.{i}.mixer.experts.127.up_proj.lora_A.weight [32, 2688]
```

**23 layers × 128 experts × 4 tensors = 11,776 keys added** to `adapter_model.safetensors`.
Total submission safetensors: 186 existing keys + 11,776 expert keys = **11,962 keys**.

### Scaling Compatibility

Both our custom training code and standard PEFT use `scaling = lora_alpha / r`.
With `lora_alpha=32, r=32` → `scaling=1.0` in both cases. The tensors are compatible
as-is; no pre-scaling needed before injection.

### Prior Runs Affected

| Run | Expert LoRA trained? | Expert LoRA submitted? | Impact |
|---|---|---|---|
| run9–11 | ❌ Frozen (requires_grad bug) | ❌ | 0 expert LoRA at training and inference |
| run12 | ✅ 883M trainable (fix applied) | ❌ Not converted | 856M expert params unused at Kaggle inference |
| run13 | ✅ 878M trainable (fix + out_proj removed) | ✅ Conversion in place | First run to fully use expert LoRA end-to-end |

### Evidence: Run12 Expert LoRA Was Training (Not Frozen)

Step-330 `expert_lora_weights.pt` inspection confirmed all `lora_B` tensors non-zero:

```
lora_B_up  (23 layers): max=0.034912  min=0.014404  all_zero=False
lora_B_down (23 layers): max=0.018921  min=0.014343  all_zero=False
```

The expert LoRA **was training** in run12 — the gap was only in the submission packaging,
not in the training code. Run13 is the first submission to close this gap.
