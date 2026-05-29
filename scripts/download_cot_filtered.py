"""
Download kishanvavdara/nemotron-reasoning-traj and convert to the training JSONL
format used by train_lora.py, keeping only correctness-verified CoT samples.

Dataset schema:
  id             — example ID
  prompt         — competition problem text
  generated      — CoT reasoning trace (may include a \boxed{} at the end)
  correct answer — ground-truth answer
  generated answer — LLM extracted answer (used for correctness check)
  correctness    — "true" / "false" / "partial" — rule-based match
  problem type   — category (bit_manipulation, cipher, numeral, unit_conversion,
                             gravity, equation_symbolic, equation_numeric)

Response format (aligns with Nemotron pre-training):
  {cot_cleaned}\\n</think>\\n\\boxed{answer}

The chat template prepends <think>\\n for the assistant turn, so the CoT starts
directly. Any trailing \\boxed{} in the CoT is stripped to avoid duplication.

Usage (inside the container via scripts/run_download_cot_filtered.sh):
  python scripts/download_cot_filtered.py [--out-dir /workspace/data]
"""
import argparse
import csv
import json
import pathlib
import random
import re
import sys

DATASET = "kishanvavdara/nemotron-reasoning-traj"

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

SYSTEM_PROMPT = (
    "You are a careful reasoning model. "
    "Solve the problem step by step and end with Final answer: \\boxed{...}."
)

# Type-balanced sampling caps (reference notebook values).
# "all" means use every correct sample available for that type.
TYPE_SAMPLES = {
    "numeral":            700,
    "gravity":            600,
    "unit_conversion":    700,
    "cipher":             "all",
    "bit_manipulation":   "all",
    "equation_symbolic":  "all",
    "equation_numeric":   "all",
}

BOXED_RE = re.compile(r"\\boxed\{[^}]*\}")


def download() -> pathlib.Path:
    import kagglehub

    print(f"Downloading dataset: {DATASET} …")
    path = pathlib.Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to: {path}")
    return path


def convert(src: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = src / "nemotron_traj.csv"
    if not csv_path.exists():
        candidates = list(src.rglob("*.csv"))
        if not candidates:
            print(f"ERROR: no CSV found under {src}", file=sys.stderr)
            sys.exit(1)
        csv_path = candidates[0]
        print(f"Using CSV: {csv_path}")

    # Load and filter to correctness == "true"
    by_type: dict[str, list] = {}
    total_raw = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total_raw += 1
            if row["correctness"] != "true":
                continue
            ptype = row["problem type"].strip()
            by_type.setdefault(ptype, []).append(row)

    total_correct = sum(len(v) for v in by_type.values())
    print(f"Total rows: {total_raw} | Correct: {total_correct}")

    # Type-balanced sampling
    rows = []
    random.seed(42)
    for ptype, cap in TYPE_SAMPLES.items():
        pool = by_type.get(ptype, [])
        if cap == "all" or cap >= len(pool):
            sampled = pool
        else:
            sampled = random.sample(pool, cap)
        print(f"  {ptype}: {len(pool)} correct → {len(sampled)} sampled")
        rows.extend(sampled)

    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    train_rows, valid_rows = rows[:split], rows[split:]

    def make_record(row):
        cot = row["generated"].strip()
        cot_cleaned = BOXED_RE.sub("", cot).rstrip()
        answer = row["correct answer"].strip()
        boxed_answer = answer if answer.startswith("\\boxed{") else f"\\boxed{{{answer}}}"
        return {
            "id":     row["id"],
            "prompt": row["prompt"].strip() + PROMPT_SUFFIX,
            "response": f"{cot_cleaned}\n</think>\n{boxed_answer}",
            "system": SYSTEM_PROMPT,
            "_answer": answer,
        }

    for name, data in [("train.jsonl", train_rows), ("valid.jsonl", valid_rows)]:
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as fh:
            for row in data:
                rec = make_record(row)
                fh.write(json.dumps(
                    {k: v for k, v in rec.items() if k != "_answer"},
                    ensure_ascii=False) + "\n")
        print(f"  {path}  ({len(data):,} examples)")

    labels_path = out_dir / "valid_labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as fh:
        for row in valid_rows:
            rec = make_record(row)
            fh.write(json.dumps({"id": rec["id"], "answer": rec["_answer"]},
                                ensure_ascii=False) + "\n")
    print(f"  {labels_path}  ({len(valid_rows):,} labels)")

    print(f"\nDone: {len(train_rows):,} train + {len(valid_rows):,} valid examples.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/data")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    src = download()
    if args.download_only:
        print("--download-only set; skipping conversion.")
        return
    convert(src, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
