# Mamba out_proj LoRA dead path — zero gradients via UnslothCheckpointFunction guard

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

- All 23 `mixer.out_proj.lora_B.weight` are **exactly zero** at step 150 and step 300. The corresponding `lora_A` values are static at their kaiming initialization value (max=0.01562, identical across all 23 layers), confirming **zero gradient** — not slow learning.
- `out_proj.lora_A` and `lora_B` are present in the PEFT adapter and have `requires_grad=True`, but receive no gradient signal.

**Refined root cause (cross-referenced with [unsloth/unsloth#5039](https://github.com/unslothai/unsloth/issues/5039)):**

The PEFT wrapper IS called for `out_proj` — the LoRA forward path exists in the graph. However, Unsloth's `UnslothCheckpointFunction.forward()` contains a guard:

```python
# Only enables gradient computation if the first argument has requires_grad=True
if args[0].requires_grad:
    ...
else:
    return None  # backward returns None → zero gradient for all params in this block
```

The Mamba fast-path fused CUDA kernel computes the SSM scan and produces the scan output tensor. That output tensor (which becomes `args[0]` — the input to `out_proj`) does **not** carry `requires_grad=True` after passing through the fused kernel. The checkpoint function therefore skips the backward pass for the entire `out_proj` block, silencing all gradients to `lora_A` and `lora_B`.

Contrast with `in_proj`: its input is the model's hidden states, which always have `requires_grad=True` (they come directly from the embedding or previous layer's residual stream) → gradients flow normally.

The same root cause was confirmed for vision LoRA in Gemma 4 (unsloth#5039): `requires_grad_for_gradient_checkpointing()` hooks on the wrong parameter, causing the backward guard to trigger on tensors without `requires_grad`.

- This was true in runs 9–11 as well (same fast-path, same targets) — not introduced by the per-expert LoRA changes.
- Impact: 23 dead lora_A/lora_B pairs ≈ **3.2M wasted trainable params**. The LoRA delta at inference time is also zero; `out_proj` behaves as frozen base weights throughout training and inference.

### Actions Taken

None taken mid-run12. Documented for future runs.

### Actions Taken

- `out_proj` removed from `_LORA_TARGETS` in `scripts/train_v9_sft.py` (committed 2026-06-11, run12 unaffected).
- Two remediation paths identified (see Follow-ups):
  - **Path A (Purge)**: strip dead `out_proj` keys from run12 adapter before submission.
  - **Path B (Monkey-patch)**: patch `mixer.forward` to manually re-inject the LoRA delta for future runs if training `out_proj` becomes a priority.

### Resolution

**Status: Partially resolved**

Root cause confirmed: `UnslothCheckpointFunction` gradient guard fires because the Mamba scan output tensor does not carry `requires_grad=True`. Same mechanism as unsloth#5039 (vision LoRA on Gemma 4). Workaround applied: `out_proj` removed from target modules for future runs. Dead keys in run12 adapter to be stripped at packaging time (Path A).

### Follow-ups

- **Path A — Purge dead keys from run12 submission adapter**: strip all 46 `out_proj` lora_A/lora_B keys from `adapter_model.safetensors` and remove `out_proj` from `adapter_config.json` target_modules before `kaggle kernels push`. Reduces adapter size and eliminates zero-weight keys that mislead the evaluator.

  ```python
  # In package_submission.sh or a prep script:
  clean_weights = {k: v for k, v in weights.items() if "out_proj" not in k}
  config["target_modules"].remove("out_proj")
  ```

- **Path B — Monkey-patch mixer.forward** to restore LoRA autograd for future runs:

  ```python
  def apply_fused_mamba_lora_patch(model):
      def make_fused_lora_forward(original_forward, peft_out_proj):
          def fused_lora_forward(hidden_states, *args, **kwargs):
              output = original_forward(hidden_states, *args, **kwargs)
              lora_A = peft_out_proj.lora_A['default']
              lora_B = peft_out_proj.lora_B['default']
              scaling = peft_out_proj.scaling['default']
              dropout = peft_out_proj.lora_dropout['default']
              lora_delta = lora_B(lora_A(dropout(hidden_states))) * scaling
              return output + lora_delta
          return fused_lora_forward
      patched = 0
      for name, module in model.named_modules():
          if name.endswith("mixer") and hasattr(module, "out_proj"):
              module.forward = make_fused_lora_forward(module.forward, module.out_proj)
              patched += 1
      print(f"Patched {patched} Mamba mixer blocks")
      return model
  ```

  **Caution**: `hidden_states` here is the mixer INPUT (shape `[B, L, hidden_size=2688]`), not the SSM scan output that `out_proj` normally receives. If `out_proj` input dim ≠ `hidden_size`, this will raise a shape error. Verify `d_inner == hidden_size` for NemotronH before using. Also adds one extra `[B, L, r]` intermediate tensor per mixer block — measure VRAM impact before committing.

- **Upstream fix**: file against unsloth — `requires_grad_for_gradient_checkpointing()` should hook on the first parameter whose INPUT tensor has `requires_grad=True`, covering Mamba scan outputs. Reference unsloth#5039 as the precedent.
