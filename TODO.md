# TODO

## Implementation Plan

### Phase 1 — Smoke test the model stack
- [x] Run `scripts/smoke_test_nemotron.py` — verify tokenizer load, model load, generation, and adapter save
- [x] Confirm saved adapter directory contains `adapter_config.json`
- [ ] Evaluate NVIDIA PyTorch container migration `25.12-py3` → `26.01-py3` per `plans/pytorch-container-migration-plan.md`

### Phase 2 — Adapt the fine-tuning objective
- [ ] Populate `data/train.jsonl` with reasoning tasks ending in `Final answer: \boxed{...}` (currently 3 toy examples)
- [ ] Verify prompt templates enforce `Final answer: \boxed{...}` as the final assistant line
- [ ] Verify `scripts/validate_metric.py` parses boxed answers and applies Kaggle-style correctness comparison
- [ ] Audit `max_new_tokens` budget — smoke test shows model gets cut off mid-answer at 64 tokens; competition outputs must reach `Final answer: \boxed{...}` before the limit; set a budget that covers full reasoning chain + closing line

### Phase 3 — Keep DSPy offline
- [ ] Build DSPy offline pipeline to generate synthetic reasoning traces
- [ ] Transfer DSPy-optimized outputs into `data/train.jsonl` as training examples

### Phase 4 — Package the submission
- [ ] Run full training with `scripts/train_lora.py` against populated dataset
- [ ] Run `scripts/validate_metric.py` against `data/valid.jsonl` to confirm boxed-answer accuracy
- [ ] Run `scripts/package_submission.sh` to produce `submission.zip`
- [ ] Verify `submission.zip` structure: contains `adapter_config.json` and adapter weights, nothing else

## DSPy + PEFT Migration Plan
- [ ] Confirm LoRA target module names for Nemotron-3-Nano-30B before launching long runs
- [ ] Verify `r=32` or below is enforced in `train_lora.py` LoRA config
- [ ] Reformat all training targets to reasoning-centric outputs with `\boxed{}` final answer
- [ ] Replace generic response quality checks with metric-aware answer extraction in validation
- [ ] Confirm adapter export via `peft.save_pretrained` produces both `adapter_config.json` and weights

## Dockerfile GB10 Adaptation
- [x] Upgrade `transformers` to ≥4.57.3 — pinned at `4.57.3` in `Dockerfile.gb10`
- [x] Add env vars `BASE_MODEL_ID`, `ADAPTER_OUTPUT_DIR`, `SUBMISSION_DIR`
- [x] Update `bitsandbytes` compute capabilities to include `sm_120`, `sm_121`
- [x] Add `scripts/smoke_test_nemotron.py`
- [x] Add `scripts/validate_metric.py`
- [x] Add `scripts/package_submission.sh`
- [x] Create `configs/` directory or remove `COPY configs` from `Dockerfile.gb10` — directory missing, image build will fail
- [x] Build image — `nemotron-gb10:latest` builds successfully
- [x] Run smoke test inside container to confirm stack is functional end-to-end

## Submission Checklist

### Adapter checks
- [ ] Base model used for training is Nemotron-3-Nano-30B
- [ ] LoRA rank is 32 or lower
- [ ] Adapter directory contains `adapter_config.json`
- [ ] Adapter loads successfully against Nemotron base model in local testing

### Behavior checks
- [ ] Model produces final answer in `\boxed{}` format consistently
- [ ] Local validation script extracts final answer from representative outputs
- [ ] Zero-temperature generation does not break answer formatting

### Packaging checks
- [ ] `submission.zip` contains only required competition assets
- [ ] Archive opens cleanly and preserves adapter file structure

### Competition checks
- [ ] Submission deadline and team-merger deadline reviewed
- [ ] Public notebook and write-up plan exist for prize eligibility

## Next steps

### Implementation Plan
1. Run `python scripts/smoke_test_nemotron.py` — first blocking step before any training
2. Populate `data/train.jsonl` with full reasoning dataset
3. Run `python scripts/train_lora.py` with populated data
4. Run `python scripts/validate_metric.py` and verify accuracy before packaging

### DSPy + PEFT Migration Plan

1. Confirm Nemotron LoRA target module names (inspect model architecture)
2. Design DSPy offline data generation pipeline for synthetic reasoning traces

### Dockerfile GB10 Adaptation

1. Run smoke test inside container to confirm stack is functional end-to-end
