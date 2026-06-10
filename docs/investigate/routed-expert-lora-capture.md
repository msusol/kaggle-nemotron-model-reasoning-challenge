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

**Status: Partially resolved**

Three run6 submission attempts ERRORed due to fused MoE expert keys. A filtered 232-key
submission was created (expert keys stripped) and resubmitted. Score pending.
The routed expert training gap (weights trained via `target_parameters` but not captured
in PEFT adapter save) is a separate known limitation — deferred per Option C.

---

## 6. Follow-ups

### Option A: Per-expert LoRA (matches huikang)
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
