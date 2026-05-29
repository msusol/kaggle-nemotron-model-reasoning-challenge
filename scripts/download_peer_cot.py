"""
Download the peer CoT dataset from Kaggle and convert to the training JSONL format
used by train_lora.py:

  {"id": "...", "prompt": "<problem>", "response": "<cot>\\nFinal answer: \\boxed{<answer>}",
   "system": "You are a careful reasoning model..."}

Dataset: kienngx/nemotron-30b-competition-trainingdata-cot-labels
Source:  Gemini-2.0-flash generated chain-of-thought traces + labels for competition prompts

Schema:
  id            — example ID
  prompt        — competition problem text
  answer        — raw final answer (no \\boxed{} wrapper)
  generated_cot — Gemini-generated reasoning trace
  label         — category (not used in training)

Usage (inside the container via scripts/run_download_peer_cot.sh):
  python scripts/download_peer_cot.py [--out-dir /workspace/data]
"""
import argparse
import json
import pathlib
import random
import sys

DATASET = "kienngx/nemotron-30b-competition-trainingdata-cot-labels"
MIN_COT_CHARS = 20  # skip rows with near-empty/failed Gemini generations

SYSTEM_PROMPT = (
    "You are a careful reasoning model. "
    "Solve the problem step by step and end with Final answer: \\boxed{...}."
)


def download() -> pathlib.Path:
    import kagglehub

    print(f"Downloading dataset: {DATASET} …")
    path = pathlib.Path(kagglehub.dataset_download(DATASET))
    print(f"Downloaded to: {path}")
    return path


def convert(src: pathlib.Path, out_dir: pathlib.Path) -> None:
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = src / "final_Nemotron_training_data.csv"
    if not csv_path.exists():
        # fallback: find any CSV under src
        candidates = list(src.rglob("*.csv"))
        if not candidates:
            print(f"ERROR: no CSV found under {src}", file=sys.stderr)
            sys.exit(1)
        csv_path = candidates[0]
        print(f"Using CSV: {csv_path}")

    rows = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cot = row["generated_cot"].strip()
            if len(cot) < MIN_COT_CHARS:
                skipped += 1
                continue

            answer = row["answer"].strip()
            boxed = answer if answer.startswith("\\boxed{") else f"\\boxed{{{answer}}}"
            rows.append({
                "id": row["id"],
                "prompt": row["prompt"].strip(),
                "response": f"{cot}\nFinal answer: {boxed}",
                "system": SYSTEM_PROMPT,
                "_answer": answer,
            })

    if skipped:
        print(f"Skipped {skipped} rows with generated_cot < {MIN_COT_CHARS} chars")

    random.seed(42)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    train_rows, valid_rows = rows[:split], rows[split:]

    for name, data in [("train.jsonl", train_rows), ("valid.jsonl", valid_rows)]:
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps({k: v for k, v in row.items() if k != "_answer"},
                                    ensure_ascii=False) + "\n")
        print(f"  {path}  ({len(data):,} examples)")

    labels_path = out_dir / "valid_labels.jsonl"
    with open(labels_path, "w", encoding="utf-8") as fh:
        for row in valid_rows:
            fh.write(json.dumps({"id": row["id"], "answer": row["_answer"]},
                                ensure_ascii=False) + "\n")
    print(f"  {labels_path}  ({len(valid_rows):,} labels)")

    print(f"\nDone: {len(train_rows):,} train + {len(valid_rows):,} valid examples.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/workspace/data",
                        help="destination for train.jsonl, valid.jsonl, valid_labels.jsonl")
    parser.add_argument("--download-only", action="store_true",
                        help="download only; skip conversion")
    args = parser.parse_args()

    src = download()

    if args.download_only:
        print("--download-only set; skipping conversion.")
        return

    convert(src, pathlib.Path(args.out_dir))


if __name__ == "__main__":
    main()
