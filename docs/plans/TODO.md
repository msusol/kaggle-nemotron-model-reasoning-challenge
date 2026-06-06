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
- [ ] Confirm both notebooks show under competition Code tab
- [ ] **DEADLINE BLOCKER** — Update prize eligibility notebook for v0.5 Unsloth approach:
  - [ ] Update approach description: v0.5 SFT, kuangyicheng short-response, Unsloth joint training
  - [ ] Update dataset section: `train.csv` (9,500) + synthetic (12,000), NOT kienngx CoT
  - [ ] Update training config: `train_v5_sft.py`, `FastLanguageModel`, 240 steps, seq=6144
  - [ ] Update adapter reference to `output/adapter_v5_sft_unsloth` once scored
  - [ ] Fill Section 8 Results with final score table (all versions v0.1 → v0.5)
  - [ ] Update Section 5 write-up with Unsloth/MoE key findings
  - [ ] Remove/update stale v0.2-cot pre-computed outputs
- [ ] Link `samvalladares/huikang-nemotron-artifacts` from prize eligibility notebook (Rule 6)
- [ ] (Optional) Kaggle CPU training notebook — fork kuangyicheng, `max_steps=3` timing test first
  - See `docs/plans/kaggle-prize-eligibility-plan.md` for full plan

## ⚠️  COMPETITION DEADLINE
- **June 8, 2026** — Entry deadline + Team merger deadline ✅ rules accepted
- **June 15, 2026 at 11:59 PM UTC** — Final submission deadline
- Today: June 3, 2026 — **12 days remaining**

All items marked DEADLINE BLOCKER must complete before June 15.

## Inference fix — HybridMambaAttentionDynamicCache

See `docs/plans/hybrid-mamba-cache-inference.md` for full details.

Pre-allocate Mamba SSM state cache before `model.generate()` via `sys.modules` pattern.
Closes v0.4 Open Question #2 (Mamba fast path). Affects `infer_lora.py` and notebook cell 7.

- [x] Add `_get_mamba_cache_cls` helper + per-batch cache init to `scripts/infer_lora.py`
- [x] Update `generate_answer` in `notebook/nemotron_v05_sft_unsloth.ipynb` cell 7
- [x] Mark v0.4 Open Question #2 resolved in `docs/plans/v0.4-blended-plan.md` — closed via `is_fast_path_available=True` patch [cite:148]

## MoE layer name audit ✓

From Tong Hui Kang's memory analysis (discussion #687961), MoE expert layers may be
named `fc1`/`fc2` rather than `up_proj`/`down_proj`. If so, our current target_modules
regex misses 856M LoRA params — the bulk of the 877M "typical" count.

**Confirmed via meta-device layer name dump (2026-05-31):**

```
conv1d, down_proj, embeddings, gate, in_proj, k_proj, lm_head,
norm, norm_f, o_proj, out_proj, q_proj, up_proj, v_proj
```

MoE expert FFN layers are `up_proj`/`down_proj` — **not** `fc1`/`fc2`. Our regex
`q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|lm_head` covers all
trainable linear layers. `conv1d` (Mamba conv), `gate` (MoE router), and `embeddings`
are frozen in the 0.85 reference notebook and remain excluded.

- [x] Run meta-device layer name dump; paste output to confirm `fc1`/`fc2` vs `up_proj`/`down_proj`
- [x] Update `target_modules` — no change needed, names confirmed correct

## Phase 5 — v0.4 (huikang corpus SFT)

See `docs/plans/v0.4-blended-plan.md` for full details.

v0.4a/b (synthetic CoT generation) superseded by `samvalladares/huikang-nemotron-artifacts`:
- 15,979 problems (training + test set), exhaustive algorithmic CoT (~3,292 tok median)
- Achieves 0.85 in reference notebook — already downloaded at `.cache/huikang-artifacts/`

Key changes vs v0.3: `max_seq_length=8192`, expanded `target_modules` (+ q/k/v/o/lm_head),
`num_epochs=1`, data from huikang corpus. Config updated in `nemotron.yaml` + `train_lora.py`.

- [x] Write `scripts/extract_huikang_corpus.py` — decode pre-tokenized corpus → JSONL
- [x] Run extraction → `data/v0.4_train.jsonl` + `data/v0.4_valid.jsonl`; verify type distribution
## OOM fix — CUDA allocator cache during loading ✓

`from_pretrained` accumulates ~41 GB of freed-but-cached CUDA blocks (temp tensors
from dtype conversion); dropper's `drop_caches` had no effect on these. Fixed by
adding `torch.cuda.empty_cache()` to the dropper thread (2026-05-31).
See `docs/investigate/v0.4-oom-loading.md` and `docs/adr/0003-dropper-empty-cache.md`.

- [x] Add `torch.cuda.empty_cache()` to `_make_cache_dropper._loop()` in `train_lora.py`
- [x] Document finding with memory log in `docs/investigate/v0.4-oom-loading.md`
- [x] Document decision in `docs/adr/0003-dropper-empty-cache.md`

## OOM fix — training activation memory at seq_len=8192

After the loading fix, model loads to 64.4 GB with only ~4 GB CUDA free. Training
forward/backward at seq_len=8192 needs ~20–40 GB activation memory → OOM at step 37.
Fix: enable NemotronH's native gradient checkpointing via `_set_gradient_checkpointing()`
directly, bypassing the `supports_gradient_checkpointing=False` guard. Every NemotronHBlock
inherits `GradientCheckpointingLayer` which has full GC support. Blocked only by the class
flag, not by missing implementation. See `docs/investigate/v0.4-oom-training.md`.

- [x] Discover `GradientCheckpointingLayer` base class in NemotronH (transformers 5.5.3)
- [x] Confirm `_set_gradient_checkpointing()` is fully implemented on `NemotronHModel`
- [x] Enable GC via `model.base_model.model._set_gradient_checkpointing(enable=True, ...)` in `train_lora.py`
- [x] Document in `docs/investigate/v0.4-oom-training.md`

- [x] Run `RUN_NAME=huikang_v4 bash scripts/run_train.sh` — completed 2026-06-01, 948 steps / 14.4 h; train_loss=0.2217, eval_loss=0.2092, eval_token_acc=94.47%; adapter at `output/adapter_huikang_v4_20260531_191344`
- [x] Package (`submission.zip` ready at `output/submission/submission.zip`) and submit to Kaggle — score **0.49** (regression vs 0.57 baseline)
- [ ] Validate boxed-answer acc via `validate_metric.py` once inference completes (inference job running against v0.4_valid.jsonl)
- [ ] Link `samvalladares/huikang-nemotron-artifacts` from prize eligibility notebook (Rule 6)

## v0.4 regression fixes — apply before next SFT or v0.6 GRPO

See `docs/investigate/v0.4-kaggle-regression.md` for full root cause analysis.

Two confirmed bugs caused the 0.49 regression despite 94.47% token accuracy:

1. **Empty system prompt during training** — huikang `"system": ""` causes `dict.get("system", default)`
   to return `""`, so training used no system prompt. Inference injects a non-empty one → OOD.
   Fix: use `example.get("system") or "..."` in `format_example` to treat empty as missing.

2. **`\boxed{–}` placeholder in thinking chain** — huikang responses embed `\boxed{–}` inside
   `<think>` before the real answer. If Kaggle scorer takes first `\boxed{}`, every answer is `–`.
   Fix: strip placeholder in `scripts/extract_huikang_corpus.py` before writing JSONL.

- [ ] Fix `format_example` in `train_lora.py`: `example.get("system") or DEFAULT_SYSTEM`
- [ ] Fix `infer_lora.py`: use same system prompt as training
- [ ] Fix `extract_huikang_corpus.py`: strip `\boxed{–}` placeholder pattern from responses
- [ ] Re-run extraction → retrain → resubmit to confirm regression is fixed

## Phase 5 — v0.5 SFT (kuangyicheng approach)

See `docs/plans/v0.5-sft-kuangyicheng-plan.md` (plan file) and
`docs/investigate/v0.4r3-training-data-alignment.md` (root cause analysis).

Warmstart from huikang v27 adapter + 240 steps on competition CSV + synthetic short
responses. Matches the 0.87 notebook approach exactly.

- [x] Pull 0.87 notebook source: `kaggle kernels pull kuangyicheng/nemotron-087-training`
- [x] Write `scripts/prepare_v5_sft_data.py` — competition CSV + synthetic generators
- [x] Generate `data/v0.5_train.jsonl` (21,500 rows) ✓ spot-check passed
- [x] Download `huikang/nemotron-adapter/Transformers/default/27` → `output/adapter_huikang_v27/`
- [x] Patch `adapter_config.json` — set `base_model_name_or_path`
- [x] Write `scripts/train_v5_sft.py` + `scripts/run_train_v5.sh`
- [x] Run training (tmux): `RUN_NAME=v5_sft bash scripts/run_train_v5.sh` — completed 2026-06-03, 240 steps / 57 min; train_loss=0.5441, token_acc=87.2%; adapter at `output/adapter_v5_sft` — scored **0.56** (MoE keys dropped by standard PEFT)
- [x] Run training with Unsloth: `RUN_NAME=v5_sft_unsloth bash scripts/run_train_v5.sh` — completed 2026-06-04, 240 steps / 3h; train_loss=7.352, token_acc=88%; 883M params, 12008 LoRA keys; scored **0.60** (MoE LoRA started fresh — v27 key format mismatch)
- [x] Package and submit — v0.5-sft-unsloth bf16, submitted 2026-06-04
- [ ] **INVESTIGATE**: v27 MoE key format mismatch — convert `experts.w1/w2/w3` (old Unsloth) → `experts.N.up_proj/down_proj` (new Unsloth) to enable warmstart for MoE layers
- [ ] If MoE warmstart fixed → retrain → target ≥ 0.85 → proceed to v0.6 GRPO

## NVIDIA API data generation (defer — investigate if v0.5 < 0.80 or post-competition)

See `docs/plans/nvidia-api-data-generation-plan.md`.

Reduces dependency on borrowed adapters (huikang v27 warmstart, kuangyicheng approach).
Use cases: better synthetic data for SFT, clean CoT corpus without Tinker quirks,
independent warmstart training, GRPO problem distribution expansion.

- [ ] Register free API key at build.nvidia.com → set `NVIDIA_API_KEY` in `.env`
- [ ] Write `scripts/generate_api_data.py` — batched calls with retry/backoff
- [ ] Generate `data/v_api_train.jsonl` (short-response style, 21,500 rows)
- [ ] Compare Kaggle score vs v0.5 rule-based synthetic data

## NeMo dataset refresh (defer until after v0.5 SFT scores)

See `docs/plans/v0.5-nemo-framework-plan.md`.

- [ ] Update `scripts/run_convert_jsonl_to_nemo.sh` to use `data/v0.5_train.jsonl`
- [ ] Tokenize: `bash scripts/run_convert_jsonl_to_nemo.sh` → `data/nemo_dataset/nemo_v5_train.jsonl`
- [ ] Delete old dataset: `kaggle datasets delete gdataranger/huikang-nemotron-nemo-sft-r32 -y`
- [ ] Create new dataset: `gdataranger/nemotron-v5-competition-nemo-sft`

## Phase 6 — v0.9 SFT (base-model init, all 14 categories)

See `docs/plans/v0.9-plan.md`.
Base model init (no warmstart), format 4, all 14 competition categories, 1000 steps.

- [ ] Confirm `scripts/train_v9_sft.py` + `scripts/run_train_v9.sh` are ready
- [ ] Run training in tmux: `RUN_NAME=v9_sft bash scripts/run_train_v9.sh`
- [ ] Validate, package, submit; update leaderboard

## Phase 7 — v0.10 GRPO + vLLM Sidecar

See `docs/plans/v0.10-grpo-sidecar-plan.md` for architecture, memory budget, and startup sequence.
Init from v0.5-sft-unsloth (0.60 Kaggle). Two-container approach validated by mineral-hr-llm.

**Startup order is critical** — trainer must load first (peaks ~126 GB), then vLLM FP8 (~37 GB).
Flag files `.trainer_model_ready` / `.vllm_sidecar_ready` coordinate the sequence.

### Infrastructure
- [x] Rebuild `nemotron-vllm-gb10:latest` — vLLM 0.22.1, flashinfer 0.6.11.post2, peft, cutlass-dsl (2026-06-06)
- [x] Create GRPO warmstart adapter — ran `remap_sft_adapter_for_grpo.py` → `output/adapter_v5_sft_grpo_warmstart/` (232 keys, 55.5 MB)
- [x] FP8 model cached — `.cache/huggingface/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-FP8/` (38 files, 31 GB, 2026-06-06)
- [x] Fix vLLM OOM during startup — `VLLM_ENFORCE_EAGER=1` (default) skips ninja CUDA graph compilation that spiked GB10 unified memory; `VLLM_TORCH_COMPILE_WORKERS=1` as backstop; compile cache mounted at `.cache/vllm_compile`
- [x] Fix vLLM timeouts — `_VLLM_TIMEOUT` 900→2700 s (orchestration), `_vllm_wait_timeout` 1200→2700 s (trainer); FP8 cold-start takes 23+ min compiled / ~10 min eager
- [x] Reduce FP8 `VLLM_GPU_MEM_UTIL` 0.35→0.30 — leaves headroom for future compiled mode
- [x] Configurable adapter/data paths — `SFT_ADAPTER`, `GRPO_WARMSTART`, `TRAIN_FILE` env vars in `run_grpo_sidecar.sh`; defaults to v5/v0.9; override for v9 without editing script
- [ ] Cache NVFP4 model — `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` (~25 GB, needed for default NVFP4 path)

### Smoke test (run before full training)
- [ ] Smoke test NVFP4 path: verify cutlass-dsl loads, `SIDECAR_MODEL=NVFP4` reaches vLLM health 200
- [ ] Run 10-step smoke test in tmux:
  ```bash
  tmux new -s grpo_smoke
  RUN_NAME=grpo_v10_smoke \
  ROLLOUT_SYNC_STEPS=5 \
  NUM_STEPS=10 \
  NUM_GENERATIONS=4 \
  MAX_NEW_TOKENS=128 \
    bash scripts/run_grpo_sidecar.sh
  ```
- [ ] Verify startup sequence: trainer loads first, `.trainer_model_ready` written, then vLLM starts
- [ ] Verify vLLM health 200 and initial LoRA load HTTP 200
- [ ] Verify non-zero reward signal at step 5
- [ ] Verify LoRA sync at step 5 logs `vLLM LoRA sync → HTTP 200`
- [ ] Verify combined memory < 130 GB during smoke test
- [ ] Resolve open questions from plan (disable_adapter, LoRA hot-swap API, FP8+LoRA dtype)

### Full training run
- [ ] Run v9 SFT: `RUN_NAME=v9_sft bash scripts/run_train_v9.sh` → `output/adapter_v9_sft`
- [ ] Remap v9 adapter: `remap_sft_adapter_for_grpo.py` → `output/adapter_v9_sft_grpo_warmstart`
- [ ] Run full GRPO with v9 adapters:
  ```bash
  SFT_ADAPTER=/workspace/output/adapter_v9_sft \
  GRPO_WARMSTART=/workspace/output/adapter_v9_sft_grpo_warmstart \
  TRAIN_FILE=/workspace/data/v0.9_train.jsonl \
  RUN_NAME=grpo_v10 bash scripts/run_grpo_sidecar.sh
  ```
- [ ] Validate, package, submit; update leaderboard

## Next steps

### v0.9-plan.md
1. Run `train_v9_sft.py` — 1000 steps, base init, all 14 cats, format 4
2. Validate and submit; record Kaggle score in leaderboard

### v0.10-grpo-sidecar-plan.md
1. Rebuild `nemotron-vllm-gb10:latest` with peft + NVFP4 deps
2. Confirm FP8 model cached locally
3. Smoke test (10 steps, N=4, max_tokens=128) — verify load order + LoRA sync
4. Resolve open questions: `disable_adapter` on Unsloth PeftModel, vLLM LoRA hot-swap API, FP8+LoRA dtype
5. Full run: `RUN_NAME=grpo_v10 bash scripts/run_grpo_sidecar.sh` in tmux
6. Validate and submit

## Infrastructure (complete)
- [x] `Dockerfile.gb10` (26.04) — current primary image for all training runs
- [x] `transformers==5.5.3` — native NemotronH KV cache fix, no `trust_remote_code`
- [x] `gradient_checkpointing=True, use_reentrant=False` — PyTorch generic checkpointing; reduces activation memory without native NemotronH declaration
- [x] `RUN_NAME` support in `run_train.sh` — named log + adapter dirs
- [x] `scripts/download_peer_cot.py` + `run_download_peer_cot.sh`
- [x] HF Hub adapter published — `marksusol/nemotron-nano-30b-lora-reasoning` (v0.1-baseline placeholder)
- [x] `is_fast_path_available=True` patch in `train_lora.py` — forces Mamba CUDA kernels via `sys.modules` scan; `--use-4bit` blocked (QLoRA broken for Nemotron-H) [cite:148]
