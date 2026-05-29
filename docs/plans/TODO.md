# TODO

## Phase 1 — Smoke test the model stack ✓
- [x] Run `scripts/smoke_test_nemotron.py` — verify tokenizer load, model load, generation, adapter save
- [x] Confirm saved adapter directory contains `adapter_config.json`
- [x] Evaluate container migration 25.12 → 26.01 → 26.04; settle on 26.01 for training
- [x] Update `torchao` pin in `Dockerfile.gb10` — bumped `0.16.0` → `0.17.0`

## Phase 2 — Baseline training ✓
- [x] Populate `data/train.jsonl` — 8,550 train + 950 valid from competition `train.csv`
- [x] Verify prompt template enforces `Final answer: \boxed{...}`
- [x] Verify `validate_metric.py` parses boxed answers correctly
- [x] Run v0.1-baseline — 1 epoch, r=32, bf16 — train_loss=5.705, val_loss=0.663, val_acc=43.5%
- [x] Package and submit — Kaggle score **0.57**

## Phase 3 — CoT data training (in progress)
- [x] Download peer CoT dataset (kienngx, Gemini-2.0-flash) — 8,358 train + 929 valid
- [x] Run v0.2-cot training — 1 epoch, r=32, bf16 — train_loss=1.546, eval_loss=1.305 (~1h55m)
- [x] Package and submit v0.2-cot — Kaggle score **0.54** (regression vs 0.57 baseline)
- [x] Run `validate_metric.py` — val acc **30.7% (285/929)** (regression vs 43.5% baseline)
- [x] Update leaderboard — noisy Gemini CoT traces hurt; many degenerate outputs in dataset

## Phase 4 — v0.3 (correctness-filtered CoT + fixed hyperparameters)

See `docs/plans/v0.3-cot-filtered-plan.md` for full details.

Root causes of v0.2 regression identified:
- Unfiltered noisy CoT traces (many degenerate Gemini outputs)
- `lr=2e-4` too high for long CoT targets → gradient spikes (visible in training plot)

Key changes for v0.3:
- Dataset: `kishanvavdara/nemotron-reasoning-traj` — correctness-filtered (~2,789 samples)
- `lr=1e-5` (20× lower), 2 epochs, `max_seq_length=4096`
- Response format: `{cot}\n</think>\n\boxed{answer}` (aligns with Nemotron pre-training)
- `target_modules`: specific regex `in_proj|out_proj|up_proj|down_proj` not `all-linear`

- [ ] Download `kishanvavdara/nemotron-reasoning-traj`; write `scripts/download_cot_filtered.py`
- [ ] Update `configs/nemotron.yaml` — lr=1e-5, epochs=2, seq_len=4096, lora_alpha=32
- [ ] Update `train_lora.py` — target_modules regex + `</think>` response format
- [ ] Run `RUN_NAME=cot_v3 bash scripts/run_train.sh`
- [ ] Validate, package, submit; record in leaderboard
- [ ] Publish best adapter to HF Hub (`marksusol/nemotron-nano-30b-lora-reasoning`)
- [ ] Update notebook Sections 8 + 9 with final results

## Kaggle Notebooks
- [x] Prize eligibility notebook public — `gdataranger/nemotron-3-nano-30b-lora-reasoning-challenge`
- [x] Submission demo published — `gdataranger/nemotron-lora-submission-demo`
- [x] Both notebooks run end-to-end (FULL_DEMO=False, trl wrapped, trl import guarded)
- [x] `ryanholbrook/nvidia-utility-script` added as input manually via Kaggle UI
- [ ] Fill in Section 8 Results once v0.2-cot Kaggle score known
- [ ] Confirm both notebooks show under competition Code tab

## Infrastructure (complete)
- [x] `Dockerfile.gb10-26-01` — validated, used for all training runs
- [x] `transformers==5.5.3` — native NemotronH KV cache fix, no `trust_remote_code`
- [x] `gradient_checkpointing=False` — NemotronH doesn't declare support
- [x] `RUN_NAME` support in `run_train.sh` — named log + adapter dirs
- [x] `scripts/download_peer_cot.py` + `run_download_peer_cot.sh`
- [x] HF Hub adapter published — `marksusol/nemotron-nano-30b-lora-reasoning` (v0.1-baseline placeholder)
