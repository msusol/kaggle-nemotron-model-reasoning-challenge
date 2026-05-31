# HybridMambaAttentionDynamicCache Inference Fix

## Goal

Pre-initialize the Mamba SSM state cache before every `model.generate()` call to enable
the fast path on Nemotron-3-Nano-30B's hybrid Mamba-Attention architecture. Closes v0.4
Open Question #2.

## Context

`scripts/infer_lora.py` and the `generate_answer` function in
`notebook/kaggle_prize_eligibility_outline.ipynb` (cell 7) both call `model.generate()`
without passing `past_key_values`. For a hybrid Mamba-Attention model, this means the
Mamba SSM state is not explicitly pre-allocated, which may prevent the fast path from
activating.

The `HybridMambaAttentionDynamicCache` class is not exported at the top-level
`transformers` namespace — it must be retrieved via
`sys.modules[base_model.__class__.__module__]`. This was identified from the public
Kaggle adapter:
`https://www.kaggle.com/datasets/uditjain13/nemotron-30b-multi-domain-merged-peft`

## Tasks

- [ ] Add `_get_mamba_cache_cls(model)` helper to `scripts/infer_lora.py`
- [ ] Build fresh cache per batch in the generate loop in `scripts/infer_lora.py`
- [ ] Update `generate_answer` in `notebook/kaggle_prize_eligibility_outline.ipynb` cell 7
- [ ] Mark v0.4 Open Question #2 resolved in `docs/plans/v0.4-blended-plan.md`

## Implementation

### Helper (add to `scripts/infer_lora.py`)

```python
import sys

def _get_mamba_cache_cls(model):
    mod = sys.modules.get(model.__class__.__module__)
    if mod is None:
        return None
    return getattr(mod, "HybridMambaAttentionDynamicCache", None)
```

### Per-batch cache in generate loop (`scripts/infer_lora.py`)

The cache is stateful — must be re-created per generate call, not reused across rows.

```python
cache_cls = _get_mamba_cache_cls(model)
# ... inside batch loop, before model.generate():
pkv = None
if cache_cls is not None:
    try:
        pkv = cache_cls(
            model.config,
            batch_size=len(batch),
            dtype=torch.bfloat16,
            device=model.device,
        )
    except Exception:
        pkv = None
outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, past_key_values=pkv, ...)
```

### `generate_answer` update (`notebook/kaggle_prize_eligibility_outline.ipynb` cell 7)

Same `sys.modules` pattern, always `batch_size=1`.

## Verification

1. Run `scripts/infer_lora.py` on `data/v0.3_valid.jsonl` inside the Docker container.
2. Confirm no exception and generation completes normally.
3. Compare output quality against a run without the cache (optional).