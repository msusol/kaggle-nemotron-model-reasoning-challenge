# ADR-0004 — Enable Gradient Checkpointing via NemotronH's Native `_set_gradient_checkpointing`

**Status:** Accepted

## Context

After the loading-phase OOM fix (ADR-0003), the model loads successfully into 64.4 GB
of CUDA memory. On a 130.7 GB GB10 system, this leaves only ~4.1 GB immediately
available to CUDA for the training forward and backward passes. At `seq_len=8192`,
activations for a 30B model with LoRA on all target modules require an estimated 20–40
GB, causing an OOM kill at training step 37 (2026-05-31 run).

### Prior state

`train_lora.py` had `gradient_checkpointing=False` in `SFTConfig`, with the comment:

```
# NemotronHForCausalLM does not implement gradient_checkpointing_enable();
# SFTTrainer raises ValueError if gradient_checkpointing=True.
```

This was accurate: setting `gradient_checkpointing=True` in SFTConfig causes SFTTrainer
to call `model.gradient_checkpointing_enable()` which, in both the transformers base
class and in NemotronH's own override, checks `supports_gradient_checkpointing` first
and raises `ValueError` because the class attribute is `False`.

### What transformers 5.5.3 actually provides

Inspection of `transformers.models.nemotron_h.modeling_nemotron_h` revealed:

- **`GradientCheckpointingLayer`**: a custom base class implementing GC in `__call__`.
  When `self.gradient_checkpointing=True` and `self.training=True`, it calls
  `self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)`
  instead of the normal forward. The `kwargs` binding captures `use_cache`,
  `past_key_values`, etc., which are automatically zeroed for GC compatibility.

- **`NemotronHBlock`** inherits from `GradientCheckpointingLayer`:
  `[NemotronHBlock, GradientCheckpointingLayer, Module, object]`

- **`NemotronHModel._set_gradient_checkpointing(enable, gradient_checkpointing_func)`**
  walks all modules with a `gradient_checkpointing` attribute and sets the flag + func.
  This method has **no** `supports_gradient_checkpointing` guard.

The infrastructure for gradient checkpointing is fully present. The only blocker is
the `supports_gradient_checkpointing=False` class attribute, which gates the
`gradient_checkpointing_enable()` public API but not `_set_gradient_checkpointing()`.

## Decision

Call `_set_gradient_checkpointing()` directly on the inner `NemotronHForCausalLM` model
after `get_peft_model`, bypassing the flag guard:

```python
_gc_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
try:
    model.base_model.model._set_gradient_checkpointing(
        enable=True, gradient_checkpointing_func=_gc_func
    )
    model.enable_input_require_grads()
    print("Gradient checkpointing enabled (NemotronH GradientCheckpointingLayer, use_reentrant=False)")
except Exception as _e:
    print(f"Warning: gradient checkpointing unavailable: {_e}")
```

`SFTConfig.gradient_checkpointing` remains `False` so SFTTrainer does not call
`gradient_checkpointing_enable()` a second time.

### Why `use_reentrant=False`

- `use_reentrant=True` (the older API) requires all inputs needing gradients to be
  passed as positional args, not keyword args. `GradientCheckpointingLayer.__call__`
  is designed for this (positional `hidden_states` + keyword everything else), so
  `use_reentrant=True` would also work. However, `use_reentrant=False` avoids the
  restriction, is the current recommended default, and avoids a known memory leak in
  the reentrant implementation.

- The `GradientCheckpointingLayer` docstring explicitly documents this distinction and
  confirms the `partial(**kwargs)` pattern is correct for `use_reentrant=False`.

### Why call through `model.base_model.model`

After `get_peft_model`, the model hierarchy is:

```
model                        → PeftModelForCausalLM  (PEFT outer wrapper)
model.base_model             → LoraModel             (PEFT LoRA adapter)
model.base_model.model       → NemotronHForCausalLM  (original model)
model.base_model.model.model → NemotronHModel        (transformer body)
```

`_set_gradient_checkpointing` is defined on `NemotronHModel` (and inherited by
`NemotronHForCausalLM`). Calling it on `model.base_model.model` reaches the
NemotronH implementation directly. Calling it on the PEFT outer wrapper would route
through PEFT's own method, which delegates to `base_model.gradient_checkpointing_enable()`
and would hit the ValueError guard again.

### Why `enable_input_require_grads()`

PEFT with a frozen base model: only LoRA adapter parameters have `requires_grad=True`.
The frozen base model's output tensors do not have gradients by default. Gradient
checkpointing recomputes from the *input* of each checkpointed block, so the block's
input must carry `requires_grad=True` to allow the recomputed activations to
participate in autograd. `enable_input_require_grads()` hooks the model's input
embedding to ensure gradients propagate through frozen layers.

## Alternatives Considered

### Set `NemotronHForCausalLM.supports_gradient_checkpointing = True` — rejected

Patching the class attribute would make `gradient_checkpointing_enable()` callable
normally. This would allow setting `gradient_checkpointing=True` in SFTConfig.
Rejected because it modifies a class-level attribute globally (affecting any other
code that instantiates NemotronH in the same process) and implicitly couples our
training to the public API behaviour of a method that may change.

### Reduce `max_seq_length` to 4096 — rejected as primary fix, deferred as fallback

Reducing seq_len halves activation memory and is guaranteed to fit. Rejected as primary
fix because the huikang corpus p90=6,676 tokens — truncating at 4096 would truncate
~40% of examples, losing the `</think>\n\boxed{answer}` tail for the longest ones.
Deferred: if gradient checkpointing does not resolve the OOM, reduce to 4096.

### Keep `gradient_checkpointing=False` and reduce batch size — not applicable

`batch_size=1` is already the minimum. Reducing grad_accum would slow training without
helping memory; activations scale with seq_len, not batch_size at batch=1.

## Consequences

- **Memory**: activation memory drops from ~20–40 GB to ~1–5 GB, fitting the ~4–66 GB
  headroom available after model loading.
- **Speed**: recomputing each block's forward during backward increases step time by
  approximately 1.5–2×. At ~43 s/step without GC, expect ~65–86 s/step with GC.
  Total training time estimate: 948 steps × 75 s/step ≈ **~20 h** (up from ~11 h).
- **Correctness**: `use_reentrant=False` with `partial(**kwargs)` correctly propagates
  gradients through both frozen Mamba and attention blocks to the LoRA adapters.
- **Fragility**: `_set_gradient_checkpointing` is a semi-private API. If transformers
  renames or removes it, the `try/except` fallback will print a warning and training
  will continue without GC — OOMing as before. This is an acceptable degradation path.
- **SFTConfig remains clean**: `gradient_checkpointing=False` is an honest value;
  SFTTrainer does not touch GC. The GC is an out-of-band setup step before training.
