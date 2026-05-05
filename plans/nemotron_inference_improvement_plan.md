# Nemotron Inference Improvement Plan

## Objective
Restore KV-cache-aware generation for the Nemotron-3-Nano-30B-A3B model so that `model.generate()` runs at practical speed for both inference and GRPO training.

## Root Cause (Confirmed)

`modeling_nemotron_h.py` in the HuggingFace model repo has a parameter name mismatch between two functions:

- `prepare_inputs_for_generation()` — creates and forwards the cache as `past_key_values`
- `forward()` — expects the cache under `cache_params`

Since the names don't match, the cache object lands in `**kwargs` and is silently discarded. The model recomputes the full context from scratch on every generation step. No error, no warning.

**Impact by training mode:**

| Mode | Generation needed? | Cache needed? | Affected? |
|---|---|---|---|
| SFT | No (parallel forward pass) | No | No |
| Inference | Yes (token-by-token) | Yes | Yes — ~2 tok/sec |
| GRPO | Yes (4+ completions per prompt) | Yes | Yes — ~2 tok/sec |

**Numbers:** ~2 tok/sec without cache; ~38 tok/sec with cache (~20× speedup). [cite:140]

## Fix Applied

`transformers >= 5.3.0` ships native NemotronH support with the correct `past_key_values` → `cache_params` mapping. Using the library's built-in implementation instead of the model repo's `modeling_nemotron_h.py` resolves the mismatch entirely. [cite:140]

Three changes required (all already applied):

### 1. Upgrade transformers
`Dockerfile.gb10`: `transformers==4.57.3` → `transformers==5.5.3`

```dockerfile
"transformers==5.5.3" \
"huggingface_hub>=1.5.0,<2.0" \   # transformers 5.5.3 requires >=1.5.0
```

### 2. Drop `trust_remote_code=True` from all model and tokenizer loading

```python
# BEFORE — pulls in the old buggy modeling_nemotron_h.py from HF model cache
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, ...)

# AFTER — uses the fixed built-in transformers implementation
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, ...)
```

Applied to: `scripts/infer_lora.py`, `scripts/train_lora.py`, `scripts/smoke_test_nemotron.py`

### 3. Disable gradient checkpointing

`NemotronHForCausalLM` in the native implementation does not declare support for gradient checkpointing; enabling it raises `ValueError`. Set explicitly in `SFTConfig`:

```python
gradient_checkpointing=False,
```

Applied to: `scripts/train_lora.py`

## Validation Steps

- [ ] Rebuild Docker image — confirm `transformers==5.5.3` installs without dependency conflicts
- [ ] Run `scripts/smoke_test_nemotron.py` — confirm model loads and generates output without `trust_remote_code`
- [ ] Benchmark `scripts/infer_lora.py` before/after — confirm throughput ~38 tok/sec

## Why This Is Better Than the Previous vLLM Path

The original plan proposed merging the LoRA adapter and serving via vLLM to work around the broken HF cache. That path is still valid as a future performance option, but the transformers fix:

- Requires no new scripts or serving infrastructure
- Works directly with the existing PEFT adapter (no merge step needed)
- Unlocks GRPO training at practical speeds as a future training option
- Can be applied on Kaggle (no internet) by uploading the `transformers-5.5.3-py3-none-any.whl` wheel as a dataset and installing with `--no-deps --force-reinstall`

## Optional Future Path: vLLM

vLLM remains available if throughput beyond ~38 tok/sec is needed for final submission inference. The approach:

1. Merge adapter: `scripts/merge_lora.py` — call `merge_and_unload()`, save to `output/merged_<timestamp>/`
2. Serve: `scripts/serve_vllm.sh` — using NVIDIA's documented vLLM recipe for this model [cite:142][cite:143]
3. Infer: `scripts/infer_vllm.py` — thin client writing competition-format predictions

Treat this as a Phase 4 option, not a blocker for current submissions.

## References
[cite:139] [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 - Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)

[cite:140] [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 · doesn't do kv cache when using Transformers · Discussion #14 - Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/14)

[cite:141] [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 · Bug Report: `model.generate()` does not use cache_params · Discussion #2 - Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/discussions/2)

[cite:142] [NVIDIA Nemotron-3-Nano-30B-A3B User Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html)

[cite:143] [Deploying NVIDIA Nemotron-3-Nano with vLLM - GitHub](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb)

[cite:144] [NemotronH - Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/nemotron_h)
