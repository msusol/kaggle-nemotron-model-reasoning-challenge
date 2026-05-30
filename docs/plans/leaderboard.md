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
| v0.4a-self-gen | *pending* | 5 | *pending* | 2 | 32 | 1e-5 | 4096 | No | — | — | — | — | — | SFT; Nemotron self-gen CoT for 6,990 gap problems; blend with v0.3; no new infra |
| v0.4b-deepseek | *pending* | 5 | *pending* | 2 | 32 | 1e-5 | 4096 | No | — | — | — | — | — | SFT; DeepSeek-R1-32B via vLLM for gap CoT; run if v0.4a hard-type pass rate < 25% |
| v0.5-grpo | *pending* | 6 | *pending* | 1–2 | 32 | 1e-6 | 4096 | No | — | — | — | — | — | GRPO; self-improvement RL on all 9,500 competition problems; init from best v0.4 adapter |
