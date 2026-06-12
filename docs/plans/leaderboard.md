# Model Leaderboard

Tracks every training run — config, training metrics, and validation results — so runs can be compared across phases.

## Columns

| Column | Description |
|---|---|
| Version | Short name for the run |
| Date | Run date |
| Phase | Plan phase (1–4) |
| Adapter | Path to saved adapter under `output/` |
| Epochs | Number of training epochs |
| r | LoRA rank |
| LR | Peak learning rate |
| Seq Len | `max_seq_length` |
| 4-bit | QLoRA enabled? |
| Train Loss | Final logged train loss |
| Train Acc | Final logged mean token accuracy |
| Val Loss | Eval loss at end of run (if available) |
| Val Acc (boxed) | `validate_metric.py` boxed-answer accuracy on the versioned valid set |
| Kaggle Score | Public leaderboard score from Kaggle submission |
| Notes | What changed vs prior run |

## Runs

| Version | Date | Phase | Adapter | Epochs | r | LR | Seq Len | 4-bit | Train Loss | Train Acc | Val Loss | Val Acc | Kaggle Score | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0.1-baseline | 2026-05-03 | 2 | `output/adapter_20260503_203554` | 1 | 32 | 2e-4 | 2048 | No | 5.705 | 0.796 | 0.663 | 43.5% (413/950) | 0.57 | First full run — no DSPy, raw competition data |
| v0.2-cot | 2026-05-28 | 3 | `output/adapter_20260528_211916` | 1 | 32 | 2e-4 | 2048 | No | 1.546 | — | 1.305 | 30.7% (285/929) | 0.54 | Peer CoT dataset (Gemini-2.0-flash traces) — regression vs baseline; noisy CoT traces hurt |
| v0.3-filtered | 2026-05-29 | 4 | `output/adapter_cot_v3_20260529_075411` | 2 | 32 | 1e-5 | 4096 | No | 0.49 | 83.11% | 0.634 | *pending* | 0.50 | Correctness-filtered CoT (kishanvavdara); stable training; type coverage gap in bit_manipulation/algebra caused regression |
| v0.4a-self-gen | *superseded* | 5 | — | — | — | — | — | — | — | — | — | — | — | Superseded by huikang corpus (samvalladares/huikang-nemotron-artifacts covers all 15,979 problems) |
| v0.4b-deepseek | *superseded* | 5 | — | — | — | — | — | — | — | — | — | — | — | Superseded by huikang corpus |
| v0.4-huikang (attempt 1) | 2026-05-31 | 5 | *killed* | 0 / 1 | 32 | 2e-4 | 8192 | No | 0.5568 @ step 30 | 83.81% @ step 30 | — | — | — | OOM kill at step 37/948; two bugs fixed: (1) loading OOM — CUDA allocator 41 GB freed-block cache [ADR-0003]; (2) training OOM — activation memory ~20–40 GB without GC [ADR-0004] |
| v0.4-huikang | 2026-06-01 | 5 | `output/adapter_huikang_v4_20260531_191344` | 1 | 32 | 2e-4 | 8192 | No | 0.2217 | 94.27% | 0.2092 | *pending inference* | 0.49 | First complete v0.4 run — 948 steps / 14.4 h; eval token acc 94.47%; regressed due to system prompt mismatch + \boxed{–} placeholder |
| v0.4-huikang-r2 | 2026-06-02 | 5 | `output/adapter_huikang_v4_20260601_194000` | 1 | 32 | 2e-4 | 8192 | No | 0.221 | — | 0.2072 | *pending* | 0.50 | System prompt fix +0.01 vs r1; root cause identified: system prompt contradicts no-\boxed{} in 8,046 augmenter rows (~53% of train) → those categories score 0 on Kaggle |
| v0.4-huikang-r3 | 2026-06-03 | 5 | `output/adapter_huikang_v4r3_20260602_142413` | 1 | 32 | 2e-4 | 8192 | No | 0.2218 | 94.48% | 0.2093 | *pending* | *pending* | Fix 3+4: empty system + stripped \boxed{–}; 948 steps / 14.3h; linear LR decay |
| v0.4-huikang-v26 | 2026-06-02 | 5 | `output/adapter_huikang_v26` | — | 32 | — | — | No | — | — | — | — | ❌ error | Kaggle evaluation crash — likely PEFT/all-linear incompatibility in notebook env; not investigating |
| v0.5-sft | 2026-06-03 | 5 | `output/adapter_v5_sft` | 240 steps | 32 | 2e-4 | 6144 | No | 0.5441 | 87.2% | — | — | 0.56 | Missing 232 Unsloth-only keys (MoE experts.w1/w2/w3, gate_proj, x_proj) — standard PEFT silently dropped them on v27 load; adapter only 105 MB vs v27's 1.5 GB |
| v0.5-sft-merged | 2026-06-03 | 5 | `output/adapter_v5_sft_merged` | 240 steps | 32 | 2e-4 | 6144 | No | 0.5441 | 87.2% | — | — | ❌ error | Unsloth-format keys (no .default., experts.w1/w2/w3) crash Kaggle's standard PEFT evaluator; confirmed evaluator does NOT use Unsloth |
| v0.5-sft-unsloth | 2026-06-04 | 5 | `output/adapter_v5_sft_unsloth` | 240 steps | 32 | 2e-4 | 6144 | No | 7.352 | 88.0% | — | — | 0.60 | Unsloth FastLanguageModel, 883M params, 12008 LoRA keys (full MoE+attention). MoE LoRA started fresh (v27 key format mismatch); attention warmstarted from v27. +0.04 vs standard PEFT |
| v0.5-sft-v27conv | 2026-06-04 | 5 | `output/adapter_v5_sft_v27conv` | 240 steps | 32 | 2e-4 | 6144 | No | 7.480 | 88.1% | — | — | 0.53 | v27 MoE keys converted to Unsloth format (proper warmstart). MoE started from v27 long-CoT weights → 240 steps insufficient to override long-CoT bias → regression vs zero-init (0.60) |
| v27-direct | 2026-06-04 | — | `output/adapter_huikang_v27_unsloth` | — | 32 | — | — | No | — | — | — | — | 0.53 | v27 weights converted to Unsloth backbone.layers format, no additional training. Long-CoT behavior → truncated before \boxed{} (same as v0.4). Confirms format loads correctly but 0.87 requires 240-step short-response SFT on top |
| v0.6-grpo | 2026-06-05 | 6 | `output/adapter_grpo_v6/checkpoint-300` | 330 steps (stopped) | 32 | 1e-6 | 4096 | No | avg reward 0.175 (peak 0.6) | — | — | — | 0.57 | GRPO from v0.5-sft-unsloth warm-start. **Critical bug**: only 232/12,008 LoRA keys loaded (fused-MoE path); MoE expert layers trained from random init. Reward trend declining (avg9: 0.278→0.078). Stopped step 330. Regressed vs v0.5 (0.60) — no MoE expert warm-start |
| v0.9-kaggle-run1 | 2026-06-08 | 6 | `adapters/adapter_v9_kaggle_run1` | 200 steps | 32 | 2e-4 | 2048 | No | — | — | — | — | 0.54 | Kaggle RTX Pro 6000 (kernel v61); 1000-sample subset, fresh base-model init; 48.2 min; submitted attn-only adapter (48 keys) — Unsloth fused MoE keys incompatible with evaluator |
| v0.9-kaggle-run2 | 2026-06-09 | 6 | `adapters/adapter_v9_kaggle_run2` | 1 epoch (~244 steps) | 32 | 2e-4 | 2048 | No | — | — | — | — | 0.54 | Kaggle RTX Pro 6000 (kernel v79); full dataset, warmstart from run1; 27.6 min; 140 keys (48 attn + 92 shared_experts); no fused MoE keys |
| v0.9-kaggle-run3 | 2026-06-09 | 6 | `adapters/adapter_v9_kaggle_run3` | 1 epoch (593 steps) | 32 | 2e-4 | 7680 | No | — | — | — | — | 0.56 | Kaggle RTX Pro 6000 (kernel v82); 9502 long examples (MIN_SEQ_LENGTH=2048); 135.3 min, 13.7 s/step; warmstart from run1 (rglob bug — run1 picked alphabetically over run2); 140 keys (48 attn + 92 shared_experts) |
| v0.9-kaggle-run4 | 2026-06-09 | 6 | `adapters/adapter_v9_kaggle_run4` | 1 epoch (~244 steps) | 32 | 2e-4 | 2048 | No | — | — | — | — | 0.56 | Kaggle RTX Pro 6000 (kernel v85); full short-example epoch, warmstart run3; rglob name-match fix applied; 140 keys (48 attn + 92 shared_experts) |
| v0.9-kaggle-run5 | 2026-06-09 | 6 | `adapters/adapter_v9_kaggle_run5` | 1 epoch (593 steps) | 32 | 2e-4 | 7680 | No | — | — | — | — | 0.56 | Kaggle RTX Pro 6000 (kernel v87); 9502 long examples (MIN_SEQ_LENGTH=2048), warmstart run4; 140 keys (48 attn + 92 shared_experts); matches run3/run4 — confirms Mamba SSM (not fresh-start) caused run6 regression |
| v0.9-run6 | 2026-06-10 | 6 | `adapters/adapter_v9_run6` | 1 epoch (~495 steps) | 32 | 2e-4 | 4096 | No | — | — | — | — | 0.53 | Kaggle RTX Pro 6000; fresh start (no warmstart); 9 LoRA targets incl. in_proj/out_proj for 23 Mamba SSM layers; 232 keys after filtering (48 attn + 92 shared_experts + 92 Mamba SSM); initial 3 submissions ERRORed (fused MoE keys + Unsloth config fields — see ADR-0005); regression vs run3/run4 (0.56) — likely fresh start + Mamba SSM noise |
| v0.9-kaggle-run7 | 2026-06-10 | 6 | `adapters/adapter_v9_kaggle_run7` | 1 epoch (339 steps) | 32 | 2e-4 | 7680 | No | — | — | — | — | *pending* | Kaggle RTX Pro 6000; long examples only (MIN_SEQ_LENGTH=2048), 5,428 examples; warmstart run6; 160.3 min, 28.4 s/step; 232 keys after fused-MoE drop (48 attn + 92 shared_experts + 92 Mamba SSM); PeriodicAdapterSave every 50 steps |
| v0.9-run12-moe | 2026-06-11 | 6 | `output/adapter_v9_run12_ckpt` | 330 steps (stopped) | 32 | 2e-4 | 2048 | No | 0.1948 @ step 150 | — | — | — | *pending* | DGX Spark GB10; first run with per-expert MoE LoRA working — 883,873,792 trainable (2.72%); out_proj dead (stripped at package); expert_lora_weights.pt NOT converted to PEFT keys → 856M expert params unused at Kaggle inference |
| v0.9-run13-step100 | 2026-06-11 | 6 | `output/adapter_v9_run13_ckpt` | 100 steps (of 1000) | 32 | 2e-4 | 2048 | No | 0.2211 @ step 100 | — | — | — | *pending* | DGX Spark GB10; fresh start (no warmstart); 878,880,768 trainable (2.71%); **first submission with 11,776 per-expert PEFT keys** (23 layers × 128 experts × 4 tensors) — expert LoRA converted from fused [128,r,dim] to per-expert format; canary test for Kaggle unfused-expert path |
| v0.9-run13-step500 | 2026-06-12 | 6 | `output/adapter_v9_run13/checkpoint-500` | 500 steps (of 1000) | 32 | 2e-4 | 2048 | No | 0.1864 @ step 500 (epoch boundary) / 0.1575 @ ~508 | — | — | — | *pending* | DGX Spark GB10; 878M trainable; 11,776 per-expert PEFT keys; epoch 2 complete; observed min 0.1575 intra-step; competition deadline 2026-06-15 |
| v0.10-grpo | *pending* | 7 | *pending* | 500 steps | 32 | 1e-6 | — | No | — | — | — | — | — | GRPO + vLLM FP8 sidecar; init from v0.5-sft-unsloth; standalone GRPO loop; `train_grpo_sidecar.py` |
