# Kaggle Run — Post-Completion Steps

Repeatable runbook for everything that must happen after a Kaggle training kernel
finishes (COMPLETE status). Covers adapter download, submission, warmstart dataset
creation, and next-run notebook configuration.

---

## 1. Confirm kernel is COMPLETE

```zsh
kaggle kernels status gdataranger/nemotron-v0-9-sft-training-rtx-pro-6000
# → KernelWorkerStatus.COMPLETE
```

If ERROR: pull the log, diagnose, fix notebook, push new version, re-run.

```zsh
kaggle kernels output gdataranger/nemotron-v0-9-sft-training-rtx-pro-6000 \
  -p /tmp/run-output
grep -i "error\|traceback\|exception" /tmp/run-output/*.log | tail -30
```

---

## 2. Download kernel output

```zsh
RUN_NAME="v9_run6"   # change per run
OUT=/tmp/kaggle-output-${RUN_NAME}
mkdir -p "$OUT"
kaggle kernels output gdataranger/nemotron-v0-9-sft-training-rtx-pro-6000 -p "$OUT"
ls "$OUT"
```

The adapter zip is written to `/kaggle/working/adapter_<RUN_NAME>.zip` by the
notebook's final cell. It will appear in `$OUT` after download.

---

## 3. Extract and verify the adapter

```zsh
ADAPTER_ZIP="$OUT/adapter_${RUN_NAME}.zip"
ADAPTER_DIR="/tmp/adapter_${RUN_NAME}"
mkdir -p "$ADAPTER_DIR"
unzip -o "$ADAPTER_ZIP" -d "$ADAPTER_DIR"

# If Kaggle packaged with an internal directory prefix, flatten it:
INNER=$(find "$ADAPTER_DIR" -name adapter_config.json | head -1 | xargs dirname)
[[ "$INNER" != "$ADAPTER_DIR" ]] && mv "$INNER"/* "$ADAPTER_DIR/" && rmdir "$INNER"

# Verify key count
python3 -c "
import safetensors.torch, glob
files = glob.glob('$ADAPTER_DIR/**/*.safetensors', recursive=True)
keys = []
for f in files:
    keys += list(safetensors.torch.load_file(f).keys())
print(f'Keys: {len(keys)}')
for k in sorted(keys)[:5]: print(' ', k)
"
# run6 expects ~510 keys (418 attn/MoE + 92 Mamba SSM in_proj/out_proj)
# run7+ expects same ~510 (same LoRA targets, warmstart adds no new keys)
```

---

## 4. Patch and package for submission

`scripts/package_submission.sh` handles both the `adapter_config.json` patch and
the zip in one step.

### Why the patch is needed

Unsloth writes two fields that break the vLLM evaluator:

| Field | Unsloth writes | Must be |
|---|---|---|
| `base_model_name_or_path` | `/kaggle/input/nemotron-3-nano-30b-a3b-bf16/...` (local path) | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| `auto_mapping` | Custom class dict for `modeling_nemotron_h` | `null` |

The evaluator's `LoRARequest` reads `base_model_name_or_path` to locate the base model;
a Kaggle-internal path doesn't exist in the eval environment. `auto_mapping` with a
custom class causes a load error because `modeling_nemotron_h` is not in the eval env.

```zsh
SUBMISSION_DIR=/tmp/submission_${RUN_NAME}
mkdir -p "$SUBMISSION_DIR"
bash scripts/package_submission.sh "$ADAPTER_DIR" "$SUBMISSION_DIR"
# → /tmp/submission_<RUN_NAME>/submission.zip
```

---

## 5. Submit to competition

```zsh
kaggle competitions submit nvidia-nemotron-model-reasoning-challenge \
  -f "$SUBMISSION_DIR/submission.zip" \
  -m "v0.9 ${RUN_NAME}: <short description>"
```

Check daily limit (7/day). If blocked, wait until Pacific midnight (~07:00 UTC).

```zsh
kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge | head -5
```

---

## 6. Create warmstart dataset for next run

Package the adapter directory (not the submission zip) as a Kaggle dataset so the
next run can load it via `WARMSTART_ADAPTER`.

```zsh
DATASET_NAME="nemotron-v9-${RUN_NAME}"   # e.g. nemotron-v9-run6
DATASET_DIR="/tmp/dataset_${RUN_NAME}"
mkdir -p "$DATASET_DIR/adapter_${RUN_NAME}"
cp "$ADAPTER_DIR"/adapter_config.json \
   "$ADAPTER_DIR"/adapter_model.safetensors \
   "$DATASET_DIR/adapter_${RUN_NAME}/"

# Create dataset metadata
cat > "$DATASET_DIR/dataset-metadata.json" <<JSON
{
  "title": "Nemotron v9 adapter ${RUN_NAME}",
  "id": "gdataranger/${DATASET_NAME}",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

kaggle datasets create -p "$DATASET_DIR" --dir-mode zip
# → published as gdataranger/<DATASET_NAME>
```

Confirm it's available before configuring the next run:

```zsh
kaggle datasets status gdataranger/${DATASET_NAME}
```

---

## 7. Configure and push notebook for next run

Update `notebook/v09_train_kaggle.ipynb` (`cell-config`):

| Parameter | run6 → run7 example |
|---|---|
| `RUN_NAME` | `"v9_run7"` |
| `WARMSTART_ADAPTER` | `"/kaggle/input/nemotron-v9-run6/adapter_v9_run6"` |
| `MAX_SEQ_LENGTH` | `7680` |
| `MIN_SEQ_LENGTH` | `4096` |

Update `notebook/v09-train-kaggle-kernel-metadata.json` — add the new dataset:

```json
"dataset_sources": [
  "gdataranger/nemotron-v09-training-data",
  "gdataranger/nemotron-v9-run6"
]
```

Commit, then push:

```zsh
git add notebook/v09_train_kaggle.ipynb notebook/v09-train-kaggle-kernel-metadata.json
git commit -m "feat(runN): configure notebook for runN — ..."
git push

mkdir -p /tmp/nemotron-v09-kernel
cp notebook/v09_train_kaggle.ipynb /tmp/nemotron-v09-kernel/
cp notebook/v09-train-kaggle-kernel-metadata.json /tmp/nemotron-v09-kernel/kernel-metadata.json
kaggle kernels push -p /tmp/nemotron-v09-kernel
```

Then open Kaggle UI → select RTX Pro 6000 → **Save Version → Save & Run All**.

---

## 8. Update leaderboard

Fill in `docs/plans/leaderboard.md` with the completed run's score once the
Kaggle submission finishes scoring (usually a few minutes after submit).

```zsh
kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge | head -3
```
