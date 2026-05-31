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
- [x] Remove `trust_remote_code=True` from prize eligibility notebook cell 5 — prevents cached buggy `modeling_nemotron_h.py` overriding transformers 5.5.3 fix [cite:147]
- [ ] Fill in Section 8 Results once v0.2-cot Kaggle score known
- [ ] Confirm both notebooks show under competition Code tab

## Inference fix — HybridMambaAttentionDynamicCache

See `docs/plans/hybrid-mamba-cache-inference.md` for full details.

Pre-allocate Mamba SSM state cache before `model.generate()` via `sys.modules` pattern.
Closes v0.4 Open Question #2 (Mamba fast path). Affects `infer_lora.py` and notebook cell 7.

- [x] Add `_get_mamba_cache_cls` helper + per-batch cache init to `scripts/infer_lora.py`
- [x] Update `generate_answer` in `notebook/kaggle_prize_eligibility_outline.ipynb` cell 7
- [x] Mark v0.4 Open Question #2 resolved in `docs/plans/v0.4-blended-plan.md` — closed via `is_fast_path_available=True` patch [cite:148]

## Phase 5 — v0.4 (huikang corpus SFT)

See `docs/plans/v0.4-blended-plan.md` for full details.

v0.4a/b (synthetic CoT generation) superseded by `samvalladares/huikang-nemotron-artifacts`:
- 15,979 problems (training + test set), exhaustive algorithmic CoT (~3,292 tok median)
- Achieves 0.85 in reference notebook — already downloaded at `.cache/huikang-artifacts/`

Key changes vs v0.3: `max_seq_length=8192`, expanded `target_modules` (+ q/k/v/o/lm_head),
`num_epochs=1`, data from huikang corpus. Config updated in `nemotron.yaml` + `train_lora.py`.

- [x] Write `scripts/extract_huikang_corpus.py` — decode pre-tokenized corpus → JSONL
- [ ] Run extraction → `data/v0.4_train.jsonl` + `data/v0.4_valid.jsonl`; verify type distribution
- [ ] Run `RUN_NAME=huikang_v4 bash scripts/run_train.sh`
- [ ] Validate, package, submit; record in leaderboard
- [ ] Link `samvalladares/huikang-nemotron-artifacts` from prize eligibility notebook (Rule 6)

## Phase 6 — v0.5 (GRPO self-improvement)

See `docs/plans/v0.5-grpo-plan.md` for full details.

GRPO lets the model generate its own reasoning and rewards correctness against ground truth —
no external CoT needed. Requires only competition problems + answers (9,500 examples we already have).
TRL 0.15.2 has `GRPOTrainer` but needs `mergekit` installed.

Sequencing options (see plan for detail):
- Option 1 (preferred): init from best v0.4 adapter — warm start on all 6 types
- Option 2 (parallel): run v0.5 from v0.3 concurrently with v0.4; take best Kaggle score
- Option 3 (fast): skip v0.4, go straight from v0.3 if deadline is tight

- [ ] Add `mergekit` to `Dockerfile.gb10`; rebuild `nemotron-gb10:latest`
- [ ] Confirm `from trl import GRPOTrainer` works in rebuilt image
- [ ] Write `scripts/train_grpo.py` with reward function + GRPOConfig
- [ ] Write `configs/nemotron_grpo.yaml`
- [ ] Write `scripts/run_grpo.sh`
- [ ] Test run: 50 steps on 100 problems — verify reward signal is non-zero
- [ ] Full run: `RUN_NAME=grpo_v5 bash scripts/run_grpo.sh`
- [ ] Validate, package, submit; record in leaderboard
- [ ] Confirm v0.4 dataset published (Rule 6 lineage) before any v0.5 submission

## Infrastructure (complete)
- [x] `Dockerfile.gb10` (26.04) — current primary image for all training runs
- [x] `transformers==5.5.3` — native NemotronH KV cache fix, no `trust_remote_code`
- [x] `gradient_checkpointing=True, use_reentrant=False` — PyTorch generic checkpointing; reduces activation memory without native NemotronH declaration
- [x] `RUN_NAME` support in `run_train.sh` — named log + adapter dirs
- [x] `scripts/download_peer_cot.py` + `run_download_peer_cot.sh`
- [x] HF Hub adapter published — `marksusol/nemotron-nano-30b-lora-reasoning` (v0.1-baseline placeholder)
- [x] `is_fast_path_available=True` patch in `train_lora.py` — forces Mamba CUDA kernels via `sys.modules` scan; `--use-4bit` blocked (QLoRA broken for Nemotron-H) [cite:148]
