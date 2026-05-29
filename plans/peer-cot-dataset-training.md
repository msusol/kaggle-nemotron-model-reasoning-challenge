# Peer CoT Dataset Training Plan

Use a competition peer's Kaggle dataset — Gemini-2.0-flash-generated chain-of-thought traces with
labels — as training data instead of the raw competition prompts. This gives the model full
reasoning chains to imitate, which should improve over the v0.1-baseline (Kaggle score 0.57) where
`response` was just `Final answer: \boxed{...}` with no intermediate reasoning.

**Dataset:** `kienngx/nemotron-30b-competition-trainingdata-cot-labels`
**URL:** https://www.kaggle.com/datasets/kienngx/nemotron-30b-competition-trainingdata-cot-labels

---

## Step 1 — Inspect the dataset schema ✓

**Completed 2026-05-28.** File: `final_Nemotron_training_data.csv` (13.3 MB, 9,500 rows).

**Actual schema:**

| Column | Role |
|---|---|
| `id` | Example ID |
| `prompt` | Competition problem text |
| `answer` | Raw final answer — **no** `\boxed{}` wrapper |
| `generated_cot` | Gemini-2.0-flash reasoning trace |
| `label` | Category label (not used in training) |

**Key findings:**
- 9,500 rows total → 8,550 train / 950 valid at 90/10 split
- `answer` has no `\boxed{}` — added during conversion
- `generated_cot` char lengths: min=1, median=641, p90=2094, p99=7266, max=29022
- Rows with `generated_cot` < 20 chars are skipped (failed Gemini generations)
- Label distribution: textual cipher (1853), unit conversion (1502), bitwise (1489), physics (1477), numerical representation (1461), symbolic/algebraic (1022), UNKNOWN (696)

**kagglehub credential note:** uses `KAGGLE_KEY` env var (not `KAGGLE_API_TOKEN`);
`run_download_peer_cot.sh` maps `KAGGLE_API_TOKEN → KAGGLE_KEY` automatically.

---

## Step 2 — Write conversion script `scripts/download_peer_cot.py` ✓

**Completed 2026-05-28.** Script written at `scripts/download_peer_cot.py`.

Key implementation details:
- `kagglehub.dataset_download("kienngx/nemotron-30b-competition-trainingdata-cot-labels")`
- Skips rows with `generated_cot` < 20 chars (failed Gemini generations)
- `response = f"{generated_cot}\nFinal answer: \\boxed{{{answer}}}"`
- 90/10 train/valid split, `random.seed(42)`
- Outputs `train.jsonl`, `valid.jsonl`, `valid_labels.jsonl` — same format as `download_data.py`

---

## Step 3 — Run the download + conversion ✓

**Completed 2026-05-28.** Runner written at `scripts/run_download_peer_cot.sh`.

Maps `KAGGLE_API_TOKEN → KAGGLE_KEY` for kagglehub compatibility. Run with:

```bash
bash scripts/run_download_peer_cot.sh
```

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

- [x] Step 1: Inspect dataset schema — `id`, `prompt`, `answer`, `generated_cot`, `label`; 9,500 rows
- [x] Step 2: Write `scripts/download_peer_cot.py` with correct column mapping
- [x] Step 3: Write `scripts/run_download_peer_cot.sh`
- [ ] Step 4: Sanity-check converted JSONL; check token lengths; adjust `max_seq_length` if needed
- [ ] Step 5: Run `bash scripts/run_train.sh` — record adapter dir and train/val loss
- [ ] Step 6: Run inference + `validate_metric.py` — compare val acc vs 43.5% baseline
- [ ] Step 7: Package with `package_submission.sh`; submit to Kaggle; record score in leaderboard

---

## Expected outcome

The v0.1-baseline used responses containing only the final answer (`Final answer: \boxed{42}`).
With full CoT traces in `response`, the model learns to reason step-by-step before committing to
an answer. This should improve both val accuracy and the Kaggle leaderboard score above 0.57.
