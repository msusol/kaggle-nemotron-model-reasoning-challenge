"""
Download competition data from Kaggle and convert to the training JSONL format
used by train_lora.py:

  {"prompt": "<problem>", "response": "<reasoning>\\nFinal answer: \\boxed{<answer>}",
   "system": "You are a careful reasoning model..."}

Usage (inside the container via scripts/run_download.sh):
  python scripts/download_data.py [--out-dir /workspace/data]
"""
import argparse
import json
import os
import pathlib
import sys

COMPETITION = "nvidia-nemotron-model-reasoning-challenge"

SYSTEM_PROMPT = (
    "You are a careful reasoning model. "
    "Solve the problem step by step and end with Final answer: \\boxed{...}."
)


def download(out_dir: pathlib.Path) -> None:
    import kagglehub

    print(f"Downloading competition data: {COMPETITION} …")
    path = kagglehub.competition_download(COMPETITION)
    print(f"Downloaded to: {path}")

    # Inventory what was downloaded so we know what to parse.
    print("\n--- downloaded files ---")
    for f in sorted(pathlib.Path(path).rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(path)}  ({f.stat().st_size:,} bytes)")
            if f.suffix == ".csv":
                import csv
                with open(f, newline="") as fh:
                    cols = next(csv.reader(fh))
                print(f"    columns: {cols}")
    print("------------------------\n")

    return pathlib.Path(path)


def convert(src: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Convert train.csv → train.jsonl + valid.jsonl (90/10 split)."""
    import csv
    import random

    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = src / "train.csv"
    if not train_csv.exists():
        print("ERROR: train.csv not found under", src)
        sys.exit(1)

    rows = []
    with open(train_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            answer = row["answer"].strip()
            boxed = answer if answer.startswith("\\boxed{") else f"\\boxed{{{answer}}}"
            rows.append({
                "id": row["id"],
                "prompt": row["prompt"].strip(),
                "response": f"Final answer: {boxed}",
                "system": SYSTEM_PROMPT,
                "_answer": answer,  # raw answer for labels file
            })

    random.seed(42)
    random.shuffle(rows)
    split = int(len(rows) * 0.9)
    train_rows, valid_rows = rows[:split], rows[split:]

    # train.jsonl / valid.jsonl — training format (no _answer field)
    for name, data in [("train.jsonl", train_rows), ("valid.jsonl", valid_rows)]:
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps({k: v for k, v in row.items() if k != "_answer"},
                                    ensure_ascii=False) + "\n")
        print(f"  {path}  ({len(data):,} examples)")

    # valid_labels.jsonl — {"id": ..., "answer": ...} for validate_metric.py
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
                        help="destination for train.jsonl and valid.jsonl")
    parser.add_argument("--download-only", action="store_true",
                        help="download and print file inventory, skip conversion")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    src = download(out_dir)

    if args.download_only:
        print("--download-only set; skipping conversion.")
        return

    convert(src, out_dir)


if __name__ == "__main__":
    main()
