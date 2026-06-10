# Routed Expert LoRA: target_parameters not captured in PEFT adapter save

**Date:** 2026-06-10
**Symptom:** Run6 adapter_config.json lists `target_parameters` for routed experts, but adapter_model.safetensors contains only 278 standard LoRA keys — no routed expert weights.
**Root cause:** Unsloth's `target_parameters` trains routed-expert tensors as direct parameter updates (not LoRA A/B matrices); PEFT adapter save only captures `target_modules` LoRA weights.
**Status:** Deferred — no evaluator compatibility issue for run6; routed expert capture requires a different save strategy.

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
All 278 are standard lora_A / lora_B keys:
  base_model.model.model.layers.0.mixer.in_proj.lora_A.weight
  base_model.model.model.layers.0.mixer.in_proj.lora_B.weight
  base_model.model.model.layers.0.mixer.out_proj.lora_A.weight
  ...
```

No `experts`, `gate_up_proj`, or `experts.down_proj` keys present.

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

- Run6 adapter is **evaluator-safe**: all 278 keys are standard PEFT LoRA format.
  The `target_parameters` field in `adapter_config.json` is a config-level artifact;
  standard PEFT / vLLM `LoRARequest` will ignore the unknown field and load normally.
- The routed expert weight updates from run6 training **were discarded at save time**.
  The submitted adapter reflects only attention, shared-expert, and Mamba SSM LoRA.
- This is the likely explanation for why our adapter (~100 MB for attention-only runs,
  ~1.7 GB for run6 with Mamba SSM) is still much smaller than a full per-expert adapter.
- The 1.7 GB size comes from the Mamba SSM `in_proj`/`out_proj` LoRA matrices and
  shared-expert projections at r=32, not from routed expert weights.

---

## 4. Actions Taken

- Verified run6 safetensors via `safe_open` — confirmed 278 keys, all `lora_A/lora_B`,
  zero expert keys.
- Confirmed `adapter_config.json` patched correctly (base_model_name_or_path, auto_mapping=null)
  before submission; `target_parameters` field left as-is (benign for evaluator).
- Submitted run6 adapter — `SubmissionStatus.PENDING` as of 2026-06-10 13:15 UTC.

---

## 5. Resolution

**Status: Deferred**

Run6 submission is clean and evaluator-compatible. The routed expert training gap is a
known limitation of the PEFT adapter save path when using Unsloth's `target_parameters`.
Score from run6 will quantify how much the Mamba SSM LoRA (new in run6) contributes
independent of the routed expert coverage gap.

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
