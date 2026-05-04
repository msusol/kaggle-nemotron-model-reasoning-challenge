---
paths:
  - "output/**"
  - "scripts/train_lora.py"
  - "scripts/validate_metric.py"
  - "configs/nemotron.yaml"
---

# Leaderboard synchronization

Update `plans/leaderboard.md` whenever any of the following occur:

| Trigger | Action |
|---|---|
| A training run completes | Add a new row with the final train loss and accuracy from the log; set Val Loss / Val Acc to TBD |
| `validate_metric.py` is run against a completed adapter | Fill in Val Acc (boxed) for the matching run row |
| `configs/nemotron.yaml` is changed before a run | Ensure the new row reflects the updated config values, not the old ones |
| A run is abandoned or errors out | Do not add a row — only completed runs belong in the leaderboard |

## Row fields

- **Version**: increment the patch number (v0.1, v0.2, …) for config tweaks within a phase; bump the minor number (v0.1 → v1.0) when moving to a new phase or major change (e.g. DSPy data, QLoRA, multi-epoch).
- **Train Loss / Train Acc**: take the *last* logged values from the training log, not the best.
- **Val Loss**: reported by the Trainer at end of epoch if `valid_file` is set; pull from the training log.
- **Val Acc (boxed)**: the `accuracy` field output by `scripts/validate_metric.py`.
- **Notes**: one sentence — what changed vs the previous run.

## What not to add

- Do not add rows for smoke test runs or ephemeral experiments that don't produce a saved adapter.
- Do not update historical rows retroactively unless correcting a factual error.
