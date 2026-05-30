#!/usr/bin/env python3
"""Blend v0.1 competition data with v0.3 filtered CoT data.

Strategy:
  - For problems in v0.1 that have CoT in v0.3: use CoT version
  - For problems in v0.1 with no CoT: use original format (plain answer)
  - Result: mixed-format dataset with more diversity than v0.3 alone

Usage:
  python scripts/blend_datasets.py \
    --v01-train data/v0.1_train.jsonl \
    --v03-train data/v0.3_train.jsonl \
    --out data/train_blended.jsonl \
    [--show-stats]
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict


def normalize_prompt(p: str) -> str:
    """Normalize prompt for matching (strip whitespace)."""
    return p.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v01-train", required=True, help="Path to v0.1 train.jsonl")
    ap.add_argument("--v03-train", required=True, help="Path to v0.3 train.jsonl")
    ap.add_argument("--out", required=True, help="Output blended train.jsonl")
    ap.add_argument("--show-stats", action="store_true", help="Print statistics")
    args = ap.parse_args()

    print("Loading v0.1 data...", file=sys.stderr)
    v01_data = {}
    v01_count = 0
    with open(args.v01_train, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt_key = normalize_prompt(row["prompt"])
            v01_data[prompt_key] = row
            v01_count += 1
    print(f"  {v01_count} examples", file=sys.stderr)

    print("Loading v0.3 data...", file=sys.stderr)
    v03_data = {}
    v03_count = 0
    with open(args.v03_train, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt_key = normalize_prompt(row["prompt"])
            v03_data[prompt_key] = row
            v03_count += 1
    print(f"  {v03_count} examples", file=sys.stderr)

    # Blend: v0.3 preferred where it exists, fall back to v0.1
    blended = {}
    matched = 0
    for prompt_key, v01_row in v01_data.items():
        if prompt_key in v03_data:
            blended[prompt_key] = v03_data[prompt_key]
            matched += 1
        else:
            blended[prompt_key] = v01_row

    print(f"\nBlending results:", file=sys.stderr)
    print(f"  v0.1 examples: {v01_count}", file=sys.stderr)
    print(f"  v0.3 examples: {v03_count}", file=sys.stderr)
    print(f"  Matched (v0.1 → v0.3): {matched}", file=sys.stderr)
    print(f"  v0.1 only: {v01_count - matched}", file=sys.stderr)
    print(f"  v0.3 only (not in v0.1): {v03_count - len([1 for p in v03_data if p in v01_data])}", file=sys.stderr)
    print(f"  Blended total: {len(blended)}", file=sys.stderr)

    # Write blended
    print(f"\nWriting to {args.out}...", file=sys.stderr)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_fh:
        for prompt_key in sorted(blended.keys()):
            out_fh.write(json.dumps(blended[prompt_key], ensure_ascii=False) + "\n")

    print(f"Wrote {len(blended)} blended examples", file=sys.stderr)

    # Format detection for stats
    if args.show_stats:
        response_types = defaultdict(int)
        for row in blended.values():
            resp = row.get("response", "")
            has_cot = "</think>" in resp
            has_final = "Final answer:" in resp
            response_types["has_cot"] += int(has_cot)
            response_types["no_cot"] += int(not has_cot)

        print("\nResponse format breakdown:", file=sys.stderr)
        for fmt, count in sorted(response_types.items()):
            pct = 100 * count / len(blended)
            print(f"  {fmt}: {count} ({pct:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
