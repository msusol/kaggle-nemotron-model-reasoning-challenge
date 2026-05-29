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
| Val Acc (boxed) | `validate_metric.py` boxed-answer accuracy on `data/valid.jsonl` |
| Kaggle Score | Public leaderboard score from Kaggle submission |
| Notes | What changed vs prior run |

## Runs

| Version | Date | Phase | Adapter | Epochs | r | LR | Seq Len | 4-bit | Train Loss | Train Acc | Val Loss | Val Acc | Kaggle Score | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0.1-baseline | 2026-05-03 | 2 | `output/adapter_20260503_203554` | 1 | 32 | 2e-4 | 2048 | No | 5.705 | 0.796 | 0.663 | 43.5% (413/950) | 0.57 | First full run — no DSPy, raw competition data |
