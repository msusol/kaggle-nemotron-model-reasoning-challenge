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
| v0.6-grpo | *pending* | 6 | *pending* | 1–2 | 32 | 1e-6 | 4096 | No | — | — | — | — | — | GRPO init from v0.5 adapter; 9,500 competition problems |
