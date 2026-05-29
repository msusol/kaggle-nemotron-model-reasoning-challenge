# Peer CoT Dataset Training Plan

Use a competition peer's Kaggle dataset — Gemini-2.0-flash-generated chain-of-thought traces with
labels — as training data instead of the raw competition prompts. This gives the model full
reasoning chains to imitate, which should improve over the v0.1-baseline (Kaggle score 0.57) where
`response` was just `Final answer: \boxed{...}` with no intermediate reasoning.

**Dataset:** `kienngx/nemotron-30b-competition-trainingdata-cot-labels`
**URL:** https://www.kaggle.com/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels

---

## Step 1 — Inspect the dataset schema

Before writing conversion code, discover the exact column names.

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -e KAGGLE_USERNAME="${KAGGLE_USERNAME}" \
  -e KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN}" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  nemotron-gb10:latest \
  python - <<'EOF'
import kagglehub, pathlib, json, csv

path = kagglehub.dataset_download("kienngx/nemotron-30b-competition-trainingdata-cot-labels")
print("Downloaded to:", path)
for f in sorted(pathlib.Path(path).rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(path)}  ({f.stat().st_size:,} bytes)")
        if f.suffix == ".csv":
            with open(f, newline="") as fh:
                cols = next(csv.reader(fh))
            print(f"    columns: {cols}")
            with open(f, newline="") as fh:
                row = next(csv.DictReader(fh))
            print(f"    first row keys: {list(row.keys())}")
            for k, v in row.items():
                print(f"      {k}: {repr(v[:120])}")
        elif f.suffix == ".jsonl":
            with open(f) as fh:
                row = json.loads(fh.readline())
            print(f"    keys: {list(row.keys())}")
            for k, v in row.items():
                print(f"      {k}: {repr(str(v)[:120])}")
EOF
```

**Expected columns (likely, based on dataset description):**

| Column | Role |
|---|---|
| `prompt` or `problem` | The competition problem text |
| `cot` or `reasoning` or `chain_of_thought` | Gemini-generated reasoning trace |
| `answer` or `label` | Final answer (may or may not include `\boxed{}`) |

**Action:** Update column name constants in `scripts/download_peer_cot.py` (Step 2) to match
actual column names found here.

---

## Step 2 — Write conversion script `scripts/download_peer_cot.py`

Create a new script alongside `download_data.py`. Key differences from the original:

- Uses `kagglehub.dataset_download("kienngx/nemotron-30b-competition-trainingdata-cot-labels")`
  instead of `kagglehub.competition_download(...)`.
- The `response` field concatenates the CoT trace and the boxed final answer:
  ```
  <cot reasoning>\nFinal answer: \boxed{<answer>}
  ```
- Same output format (`train.jsonl`, `valid.jsonl`, `valid_labels.jsonl`) so the rest of the
  pipeline (`train_lora.py`, `validate_metric.py`, `package_submission.sh`) is unchanged.

**Schema mapping to determine during Step 1:**

```
response = f"{row[COT_COL].strip()}\nFinal answer: \\boxed{{{answer}}}"
```

Where `COT_COL` is whatever column holds the reasoning trace (e.g. `cot`, `reasoning`,
`chain_of_thought`) and `answer` is the raw label stripped of any existing `\boxed{}` wrapper to
avoid double-wrapping.

**Split:** 90/10 train/valid, `random.seed(42)` — same as `download_data.py`.

**Output files** (overwrite `data/` in-place, or use `--out-dir data/cot_v1` to keep separate):

```
data/
  train.jsonl          ← {"prompt", "response", "system", "id"}
  valid.jsonl          ← same, validation split
  valid_labels.jsonl   ← {"id", "answer"} for validate_metric.py
```

---

## Step 3 — Run the download + conversion

Add a convenience runner `scripts/run_download_peer_cot.sh` mirroring `run_download.sh` but
calling `download_peer_cot.py`:

```bash
bash scripts/run_download_peer_cot.sh
```

This mounts the workspace into the container, passes Kaggle credentials, and writes converted
files to `/workspace/data/`.

---

## Step 4 — Sanity-check the converted data

Before training, verify the conversion produced the expected format:

```bash
# Count examples
wc -l data/train.jsonl data/valid.jsonl data/valid_labels.jsonl

# Inspect a few rows — confirm response contains CoT + \boxed{}
python3 - <<'EOF'
import json
with open("data/train.jsonl") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        print(f"=== example {i} ===")
        print("prompt:", row["prompt"][:80])
        print("response:", row["response"][-120:])   # tail — should end with \boxed{...}
        if i >= 2:
            break
EOF
```

Expected: `response` tail looks like `...\nFinal answer: \boxed{42}` (or similar answer).

**Also check `max_seq_length` fit.** CoT responses are longer than bare labels. If median
tokenized length exceeds 2048, bump `max_seq_length` in `configs/nemotron.yaml` to 4096 before
training.

```bash
python3 - <<'EOF'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
lengths = []
with open("data/train.jsonl") as f:
    for line in f:
        r = json.loads(line)
        lengths.append(len(tok.encode(r["prompt"] + r["response"])))
lengths.sort()
n = len(lengths)
print(f"p50={lengths[n//2]}  p90={lengths[int(n*0.9)]}  p99={lengths[int(n*0.99)]}  max={lengths[-1]}")
EOF
```

---

## Step 5 — Run training (same as Phase 2)

No changes to `train_lora.py` or `run_train.sh`. Config in `configs/nemotron.yaml`:

```yaml
train_file: data/train.jsonl
valid_file: data/valid.jsonl
max_seq_length: 2048   # bump to 4096 if Step 4 shows p90 > 1800 tokens
num_epochs: 1
lora_r: 32
learning_rate: 2e-4
```

```bash
bash scripts/run_train.sh 2>&1 | tee output/train_cot_v1.log
```

Record the adapter directory printed at the end (e.g. `output/adapter_YYYYMMDD_HHMMSS`).

---

## Step 6 — Validate

```bash
bash scripts/run_validate.sh
```

Or explicitly:

```bash
docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_TOKEN="${HF_TOKEN}" \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)":/workspace \
  -w /workspace \
  nemotron-gb10:latest \
  python scripts/infer_lora.py \
    --adapter-dir output/adapter_YYYYMMDD_HHMMSS \
    --input-file data/valid.jsonl \
    --output-file output/predictions_cot_v1.jsonl

python scripts/validate_metric.py \
  --predictions output/predictions_cot_v1.jsonl \
  --labels data/valid_labels.jsonl
```

Compare val accuracy against v0.1-baseline (43.5%).

---

## Step 7 — Package and submit

```bash
bash scripts/package_submission.sh output/adapter_YYYYMMDD_HHMMSS
```

Upload `output/submission/submission.zip` to Kaggle. Record the public leaderboard score and add a
new row to `plans/leaderboard.md`:

| Version | Notes |
|---|---|
| v0.2-cot | Peer CoT dataset (Gemini-2.0-flash traces), same LoRA config as v0.1-baseline |

---

## Checklist

- [ ] Step 1: Inspect dataset schema — confirm column names for prompt, CoT, answer
- [ ] Step 2: Write `scripts/download_peer_cot.py` with correct column mapping
- [ ] Step 3: Write `scripts/run_download_peer_cot.sh` and run it
- [ ] Step 4: Sanity-check converted JSONL; check token lengths; adjust `max_seq_length` if needed
- [ ] Step 5: Run `bash scripts/run_train.sh` — record adapter dir and train/val loss
- [ ] Step 6: Run inference + `validate_metric.py` — compare val acc vs 43.5% baseline
- [ ] Step 7: Package with `package_submission.sh`; submit to Kaggle; record score in leaderboard

---

## Expected outcome

The v0.1-baseline used responses containing only the final answer (`Final answer: \boxed{42}`).
With full CoT traces in `response`, the model learns to reason step-by-step before committing to
an answer. This should improve both val accuracy and the Kaggle leaderboard score above 0.57.
