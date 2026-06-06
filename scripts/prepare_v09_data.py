#!/usr/bin/env python3
"""Build data/v0.9_train.jsonl and data/v0.9_valid.jsonl.

Combines two sources into Format 4 — the only training format where every
required element is present:

  assistant content = {trace}\n</think>\n\boxed{answer}
  (chat template auto-prepends <think>\n at the assistant turn)

Sources:
  huikang  (data/v0.4_train.jsonl) — 15,159 rows, 14 categories (all test cats)
  kishanvavdara (kagglehub)        — 4,423 correct rows, 7 train categories

Huikang responses already end with </think> (sometimes + \boxed{}).
For 5 think-only categories (matching/splitting/concatenation/lstrip/spelling),
the answer is extracted from the last response line and appended as \boxed{}.

Usage:
    python scripts/prepare_v09_data.py \\
        --huikang   data/v0.4_train.jsonl \\
        --kv-csv    /path/to/nemotron_traj.csv \\
        --out-train data/v0.9_train.jsonl \\
        --out-valid data/v0.9_valid.jsonl
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

PROMPT_SUFFIX = "\nPlease put your final answer inside \\boxed{}."

TYPE_TO_BUCKET = {
    "bit_manipulation":        "bit_like",
    "cipher":                  "cipher_like",
    "concatenation":           "other",
    "cryptarithm_deduce":      "equation_like",
    "cryptarithm_guess":       "equation_like",
    "equation_numeric_deduce": "equation_like",
    "equation_numeric_guess":  "equation_like",
    "equation_numeric":        "equation_like",
    "equation_symbolic":       "equation_like",
    "gravity":                 "other",
    "lstrip":                  "other",
    "matching":                "other",
    "numeral":                 "numeral_like",
    "spelling":                "cipher_like",
    "splitting":               "other",
    "unit_conversion":         "unit_like",
}

THINK_ONLY_CATS = {"concatenation", "lstrip", "matching", "spelling", "splitting"}


def extract_think_only_answer(response: str, category: str) -> str:
    resp = response.rstrip()
    if resp.endswith("</think>"):
        resp = resp[: -len("</think>")].rstrip()
    lines = [l.strip() for l in resp.split("\n") if l.strip()]
    if not lines:
        return ""
    if category == "matching":
        bests = re.findall(r"Best:\s*(.+?):\s*\d+", resp)
        return bests[-1].strip() if bests else lines[-1]
    last = lines[-1]
    if " -> " in last:
        return last.split(" -> ", 1)[1].strip()
    return last


def build_huikang_example(row: dict) -> dict:
    cat  = row.get("category", "?")
    resp = row["response"].strip()
    think_pos = resp.rfind("</think>")
    has_boxed_after = bool(
        think_pos >= 0 and re.search(r"\\boxed\{", resp[think_pos:])
    )
    if not has_boxed_after and cat in THINK_ONLY_CATS:
        answer = extract_think_only_answer(resp, cat)
        if resp.endswith("</think>"):
            resp = resp + f"\n\\boxed{{{answer}}}"
    return {
        "messages": [
            {"role": "user",      "content": row["prompt"] + PROMPT_SUFFIX},
            {"role": "assistant", "content": resp},
        ],
        "bucket":   TYPE_TO_BUCKET.get(cat, "other"),
        "category": cat,
        "source":   "huikang",
        "id":       row.get("id", ""),
    }


def build_kishanvavdara_example(row: dict) -> dict:
    cat       = row.get("problem type", "?")
    generated = row["generated"].strip()
    answer    = row["correct answer"].strip()
    asst = f"{generated}\n</think>\n\\boxed{{{answer}}}"
    return {
        "messages": [
            {"role": "user",      "content": row["prompt"].strip() + PROMPT_SUFFIX},
            {"role": "assistant", "content": asst},
        ],
        "bucket":   TYPE_TO_BUCKET.get(cat, "other"),
        "category": cat,
        "source":   "kishanvavdara",
        "id":       row.get("id", ""),
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--huikang",   default="data/v0.4_train.jsonl")
    ap.add_argument("--kv-csv",    default=None,
                    help="Path to nemotron_traj.csv. Auto-downloaded if omitted.")
    ap.add_argument("--out-train", default="data/v0.9_train.jsonl")
    ap.add_argument("--out-valid", default="data/v0.9_valid.jsonl")
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--seed",       type=int,   default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    print(f"Loading huikang: {args.huikang}")
    with open(args.huikang, encoding="utf-8") as f:
        huikang_rows = [json.loads(l) for l in f]
    huikang_examples = [build_huikang_example(r) for r in huikang_rows]
    print(f"  {len(huikang_examples):,} examples built")

    if args.kv_csv is None:
        import kagglehub
        print("Downloading kishanvavdara/nemotron-reasoning-traj …")
        path = kagglehub.dataset_download("kishanvavdara/nemotron-reasoning-traj")
        csv_path = next(Path(path).glob("**/*.csv"), None)
        if csv_path is None:
            raise FileNotFoundError(f"No CSV found in {path}")
    else:
        csv_path = Path(args.kv_csv)

    print(f"Loading kishanvavdara: {csv_path}")
    with open(csv_path, encoding="utf-8") as f:
        kv_rows = [r for r in csv.DictReader(f) if r.get("correctness", "").lower() == "true"]
    kv_examples = [build_kishanvavdara_example(r) for r in kv_rows]
    print(f"  {len(kv_examples):,} examples built (correctness==true)")

    all_examples = huikang_examples + kv_examples
    print(f"Total: {len(all_examples):,}")

    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    n_valid   = max(1, int(len(all_examples) * args.valid_frac))
    valid_set = all_examples[:n_valid]
    train_set = all_examples[n_valid:]

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.out_valid, "w", encoding="utf-8") as f:
        for ex in valid_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_set):,} train → {args.out_train}")
    print(f"Wrote {len(valid_set):,} valid → {args.out_valid}")


if __name__ == "__main__":
    main()
