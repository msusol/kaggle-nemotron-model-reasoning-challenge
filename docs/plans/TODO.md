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
- [x] Package and submit v0.2-cot — submitted 2026-05-29, Kaggle score **TBD**
- [ ] Inference running — `bash scripts/run_inference.sh` → `output/predictions_20260528_211916.jsonl`
- [ ] Run `validate_metric.py` on v0.2-cot predictions to get val accuracy
- [ ] Update leaderboard with val acc + Kaggle score once known

## Phase 4 — Next iteration
- [ ] Review v0.2-cot Kaggle score; decide whether to iterate
- [ ] Publish best adapter to HF Hub (update `marksusol/nemotron-nano-30b-lora-reasoning`)
- [ ] Update notebook Section 8 (Results) with final val acc and Kaggle score
- [ ] Update notebook Section 9 (Reproducibility) with final adapter repo and Docker image tag

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
