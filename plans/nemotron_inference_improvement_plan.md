# Nemotron Inference Improvement Plan

## Objective
Improve inference throughput for the LoRA-adapted Nemotron-3-Nano-30B-A3B model by moving performance-critical runs away from raw Hugging Face `model.generate()` and toward a cache-aware serving path suitable for Kaggle time limits.[1][2]

## Current State
The current inference script uses plain `transformers` generation with a PEFT adapter attached to the base model.[1] This path is functionally correct for validation, but the current Hugging Face NemotronH integration has a known cache-plumbing issue: `generate()` follows the standard `past_key_values` protocol, while the model expects its hybrid cache under `cache_params`, so KV or hybrid cache is not actually used during decoding.[3][2]

## Problem Summary
Because the hybrid cache is not threaded through correctly, generation repeatedly recomputes context instead of benefiting from incremental cached decoding.[3][2] NVIDIA’s own guidance says Hugging Face is mainly for prototyping here and points users to optimized inference engines such as vLLM, TRT-LLM, SGLang, and llama.cpp for KV-cache-aware deployment.[2][1]

## Target Architecture
The preferred production path is:

1. Load the fine-tuned PEFT or LoRA adapter onto the Nemotron base model.
2. Merge the adapter into the base weights with `merge_and_unload()`.
3. Save the merged checkpoint as a standalone model artifact.
4. Run inference through vLLM using the merged model.
5. Keep the existing Hugging Face script only for smoke tests and small validation runs.[4][5][2]

## Why vLLM
vLLM has a documented deployment path for NVIDIA Nemotron-3-Nano-30B-A3B and is part of NVIDIA’s recommended inference stack for this model family.[4][5][1] This avoids the current Hugging Face cache mismatch and gives a better chance of meeting Kaggle runtime limits.[2]

## Implementation Steps

### 1. Freeze the baseline
- Keep the current `scripts/infer_lora.py` path unchanged for reproducibility and baseline comparison.[1]
- Record current runtime metrics: prompt length, `max_new_tokens`, tokens per second, total job duration, and hardware profile.

### 2. Add a merge script
- Create a script such as `scripts/merge_lora.py`.
- Load base model + tokenizer + adapter.
- Call `merge_and_unload()` on the PEFT model.
- Save merged model and tokenizer to a new output directory such as `output/merged_nemotron_model/`.
- Validate that the merged checkpoint loads without PEFT dependencies in the inference environment.

### 3. Validate output parity
- Run a small fixed eval set through both paths: current HF+adapter flow and merged-model flow.
- Compare decoded outputs, task metrics, and any competition-specific scoring proxies.
- Accept minor formatting drift, but investigate large semantic divergence.

### 4. Stand up vLLM inference
- Start with NVIDIA’s documented vLLM recipe for Nemotron-3-Nano-30B-A3B, including `--trust-remote-code` and model-specific parser flags where needed.[4][1]
- Point vLLM at the merged model directory instead of the base-plus-adapter pair.
- Build a thin client script that submits prompts and writes predictions in the format required by the competition.

### 5. Benchmark on target hardware
- Measure end-to-end throughput on the same class of hardware expected for Kaggle submission or local dry runs.
- Benchmark several realistic settings: short prompts, long prompts, and the competition’s typical `max_new_tokens` budget.
- Record tokens per second, total runtime, memory footprint, and failure modes.[4]

### 6. Integrate into submission flow
- Add a clear switch in the pipeline such as `--backend hf` vs `--backend vllm`.
- Default local validation to Hugging Face only if simplicity matters more than speed.
- Default submission generation to vLLM once parity and stability are confirmed.

## Fallback Option
If vLLM integration becomes blocked, the secondary path is a custom generation loop that calls `model.forward()` directly and passes the initialized Nemotron hybrid cache through `cache_params` on every step.[3][6] This should restore cache-aware decoding in principle, but it is more brittle and requires maintaining custom decoding logic, stopping criteria, and sampling behavior.[3]

## Non-Goals
- Do not refactor the existing baseline path until the merged-model path is validated.
- Do not rely on raw Hugging Face `model.generate()` performance improvements landing in time for the current competition run.[2]
- Do not introduce multiple inference backends at once beyond a simple baseline-versus-vLLM split.

## Risks
| Risk | Impact | Mitigation |
|---|---|---|
| Merged model outputs differ from adapter-attached outputs | Medium | Run parity tests on a fixed sample set before switching default inference |
| vLLM environment differs from Kaggle runtime | High | Dry-run on the closest available hardware and container stack before final submission |
| Nemotron-specific flags or parser settings are incomplete | Medium | Start from NVIDIA and vLLM published examples rather than a generic vLLM command.[4][1] |
| Time spent on custom HF patching delays delivery | Medium | Treat HF monkey-patching as a last resort, not the main path.[2] |

## Recommended Deliverables
- `scripts/merge_lora.py`
- `scripts/serve_vllm.sh`
- `scripts/infer_vllm.py`
- `docs/nemotron_inference_notes.md`
- `output/merged_nemotron_model/` directory contract

## Acceptance Criteria
The migration is successful when all of the following are true:

- The merged checkpoint loads cleanly in vLLM.[4][5]
- Output quality is materially unchanged on a representative validation slice.
- End-to-end inference runtime is significantly lower than the current raw Hugging Face path.
- The submission pipeline can run without manual intervention.
- The old Hugging Face path remains available for debugging and quick validation.

## Suggested Sequence
1. Benchmark the current script.
2. Implement and test adapter merge.
3. Validate output parity.
4. Bring up vLLM with the merged model.
5. Benchmark throughput.
6. Switch submission inference to vLLM.
7. Keep custom HF cache work only as an optional experimental branch.[4][2]

## References
[1] [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 - Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)

[2] [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 · doesn't do kv cache when using Transformers · Discussion #14 - Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/discussions/14)

[3] [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 · Bug Report: `model.generate()` does not use cache_params · Discussion #2 - Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/discussions/2)

[4] [NVIDIA Nemotron-3-Nano-30B-A3B User Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html)

[5] [Deploying NVIDIA Nemotron-3-Nano with vLLM - GitHub](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb)

[6] [NemotronH - Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/nemotron_h)
