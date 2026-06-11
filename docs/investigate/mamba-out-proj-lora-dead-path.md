# Mamba out_proj LoRA dead path — zero gradients via fast-path bypass

**Date:** 2026-06-11  
**Symptom:** All 23 `mixer.out_proj.lora_B.weight` tensors are exactly zero at step 150 of run12, despite `out_proj` being in `_LORA_TARGETS` and the PEFT adapter containing 46 out_proj lora_A/lora_B keys.

---

## 1. out_proj lora_B always zero at step 150

### Context

run12 (`train_v9_sft.py`) trains NemotronH 30B with:
- Standard PEFT LoRA on `_LORA_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","in_proj","out_proj"]`
- Per-expert LoRA injected on 23 `NemotronHExperts` modules (lora_A_up/B_up/A_down/B_down)
- Unsloth Mamba fast-path enabled via `_patch_mamba_fastpath(model)`

Step-150 checkpoint (`adapter_v9_run12_ckpt/adapter_model.safetensors`) inspected with `safetensors.torch.load_file`.

### Investigation Checklist

- [x] Are `out_proj.lora_B` keys present in the adapter safetensors? **Yes — 23 keys**
- [x] Are `out_proj.lora_A` keys present? **Yes — 23 keys, max≈0.0156 (kaiming init, unchanged)**
- [x] Are all `out_proj.lora_B` exactly zero? **Yes — all 23, confirmed `max=0.000000`**
- [x] Are other Mamba `in_proj.lora_B` non-zero? **Yes — all 23, max 0.010–0.054**
- [x] Are attention / shared-expert lora_B non-zero? **Yes — all active**
- [x] Is `out_proj` in `requires_grad`? **Yes — PEFT sets it; not touched by our re-grad fix**
- [x] Is the `out_proj.lora_A` value static (unchanged from init)? **Yes — max=0.01562 exactly, same across all 23 layers = kaiming init, never updated**

### Findings

- All 23 `mixer.out_proj.lora_B.weight` are **exactly zero** at step 150. The corresponding `lora_A` values are static at their kaiming initialization value (max=0.01562 identical across layers), confirming **zero gradient** — not slow learning.
- `out_proj.lora_A` and `lora_B` are in the computational graph (PEFT wraps them), but receive no gradient signal.
- Root cause: Unsloth's Mamba fast-path kernel (`_patch_mamba_fastpath`) calls the underlying weight matrix of `out_proj` directly via a fused CUDA kernel, bypassing the PEFT `LoraLinear` forward. The LoRA delta `B @ A @ x` is never computed → autograd never builds edges to lora_A/lora_B → zero gradient.
- This was true in runs 9–11 as well (same fast-path, same targets) — not introduced by the per-expert LoRA changes.
- Impact: 23 dead lora_A/lora_B pairs × 2 × r × dim parameters ≈ **3.2M wasted trainable params** that contribute nothing to training or inference. The LoRA delta at inference time is also zero, so out_proj behaves as frozen base weights.

### Actions Taken

None taken mid-run12. Documented for future runs.

### Resolution

**Status: Deferred**

Root cause confirmed: Unsloth Mamba fast-path bypasses PEFT wrapper for `out_proj`. Fix deferred to next run — removing `out_proj` from `_LORA_TARGETS` eliminates the dead params cleanly.

### Follow-ups

- **Remove `out_proj` from `_LORA_TARGETS`** for run13+. No training value; saves 3.2M params and reduces adapter size slightly.
- **Verify `in_proj` LoRA is computed through PEFT wrapper** (not also fast-pathed). Evidence: `in_proj.lora_B` is non-zero, so gradients flow — confirmed OK.
- **Consider patching `_patch_mamba_fastpath`** to route through the PEFT wrapper for `out_proj` if training that path becomes important. Likely not worth the effort given `in_proj` adaptation is sufficient.
- **Check whether `out_proj` LoRA keys should be stripped from the submission adapter** (they add filesize with zero contribution). `package_submission.sh` should filter them out or `_LORA_TARGETS` change eliminates them at source.
