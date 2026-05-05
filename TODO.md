# TODO

## Implementation Plan

### Phase 1 — Smoke test the model stack
- [x] Run `scripts/smoke_test_nemotron.py` — verify tokenizer load, model load, generation, and adapter save
- [x] Confirm saved adapter directory contains `adapter_config.json`
- [x] Evaluate NVIDIA PyTorch container migration `25.12-py3` → `26.01-py3` per `plans/pytorch-container-migration-plan.md`
- [x] Evaluate NVIDIA PyTorch container migration `26.01-py3` → `26.04-py3` — revalidate causal-conv1d, mamba-ssm, bitsandbytes, and smoke test on GB10
- [x] Update `torchao` pin in `Dockerfile.gb10` — bumped `0.16.0` → `0.17.0` for PyTorch 2.12 (pytorch/ao#2919)

### Phase 2 — Adapt the fine-tuning objective
- [x] Populate `data/train.jsonl` with reasoning tasks ending in `Final answer: \boxed{...}` — 8,550 train + 950 valid from competition train.csv (pattern recognition: bit manipulation, ciphers, unit conversion)
- [x] Verify prompt templates enforce `Final answer: \boxed{...}` as the final assistant line — `train_lora.py:format_example` uses chat template with `response` field; system prompt enforces the format
- [x] Verify `scripts/validate_metric.py` parses boxed answers and applies Kaggle-style correctness comparison — extracts last `\boxed{...}`, falls back to last number/word; `is_correct` uses exact string match + float tolerance; `valid_labels.jsonl` generated with raw answers
- [x] Audit `max_new_tokens` budget — bumped `64 → 512` in `smoke_test_nemotron.py`; inference script will need `512–2048` for full reasoning chains (Phase 3)
- [x] Run baseline training with `scripts/train_lora.py` — 1 epoch, r=32, bf16, no DSPy — complete (v0.1-baseline, train_loss=5.905, val_loss=0.663, val_acc=80.6%)
- [x] Run `scripts/validate_metric.py` against `data/valid.jsonl` to confirm boxed-answer accuracy on baseline adapter — 43.5% (413/950)
- [x] Run `scripts/package_submission.sh` on baseline adapter — `output/submission/submission.zip` (1.7GB, 6 files: adapter_config.json, adapter_model.safetensors, tokenizer files)
- [ ] Submit `output/submission/submission.zip` to Kaggle to establish a public leaderboard score

### Phase 3 — Improve with DSPy
- [ ] Build DSPy offline pipeline to generate synthetic reasoning traces
- [ ] Transfer DSPy-optimized outputs into `data/train.jsonl` as training examples
- [ ] Re-run training on DSPy-enriched data (v1.0) and compare against v0.1-baseline in leaderboard

### Phase 4 — Package the submission
- [ ] Run `scripts/package_submission.sh` on best adapter to produce `submission.zip`
- [ ] Verify `submission.zip` structure: contains `adapter_config.json` and adapter weights, nothing else

## Inference Improvements (post-baseline)

See `plans/nemotron_inference_improvement_plan.md` for full details. [cite:139][cite:140][cite:141][cite:142][cite:143][cite:144]

KV cache is broken in HF Transformers < 5.3.0 for NemotronH — `prepare_inputs_for_generation()` sends cache as `past_key_values` but `forward()` expects `cache_params`; cache silently dropped, generation runs at ~2 tok/sec [cite:140]. Fix: `transformers >= 5.3.0` has native NemotronH support with correct parameter mapping; drop `trust_remote_code=True` so the library's implementation is used instead of the old buggy `modeling_nemotron_h.py` from the model repo. Expected throughput: ~38 tok/sec (~20× speedup).

- [x] Bump `transformers` to `5.5.3` in `Dockerfile.gb10` — native NemotronH cache fix [cite:140]
- [x] Drop `trust_remote_code=True` from all model/tokenizer loading in `infer_lora.py`, `train_lora.py`, `smoke_test_nemotron.py`
- [x] Set `gradient_checkpointing=False` explicitly in `SFTConfig` — NemotronHForCausalLM doesn't declare support; throws ValueError otherwise
- [ ] Rebuild Docker image and run smoke test to confirm new transformers version loads cleanly
- [ ] Benchmark `infer_lora.py` throughput before/after — confirm ~38 tok/sec vs prior ~2 tok/sec

## Training Improvements (post-baseline)
- [ ] Switch to pre-tokenized dataset + `dataset_text_field=None` / `max_seq_length=None` in `SFTConfig` — bypasses TRL's ChatML conversion step, removes risk of double chat-template application on future runs
- [ ] Add early stopping for multi-epoch runs — `EarlyStoppingCallback(patience=3)`, `eval_strategy="steps"`, `eval_steps=100`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"` in `SFTConfig`
- [x] Set `gradient_checkpointing=False` explicitly — NemotronHForCausalLM throws ValueError if enabled with native transformers implementation

## DSPy + PEFT Migration Plan
- [ ] Confirm LoRA target module names for Nemotron-3-Nano-30B before launching long runs
- [ ] Verify `r=32` or below is enforced in `train_lora.py` LoRA config
- [ ] Reformat all training targets to reasoning-centric outputs with `\boxed{}` final answer
- [ ] Replace generic response quality checks with metric-aware answer extraction in validation
- [ ] Confirm adapter export via `peft.save_pretrained` produces both `adapter_config.json` and weights

## Dockerfile GB10 Adaptation
- [x] Upgrade `transformers` to ≥5.3.0 — pinned at `5.5.3` in `Dockerfile.gb10` (KV cache fix for NemotronH)
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
- [ ] Upload final adapter to HF Hub or Kaggle dataset so notebook can load it (needed for Section 3 of `notebook/kaggle_prize_eligibility_outline.ipynb`)
- [ ] Update notebook Section 6 training config to match actual run — `SFTConfig`, `dtype`, `get_peft_model`, no deprecated kwargs
- [ ] Fill in notebook Section 8 (Results) with validation accuracy and Kaggle leaderboard score
- [ ] Fill in notebook Section 9 (Reproducibility) with adapter repo ID, Docker image tag, training script path
- [ ] Make notebook public on Kaggle and confirm it runs end-to-end
