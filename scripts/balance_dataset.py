#!/usr/bin/env python3
"""Balance a JSONL training dataset by capping over- and repeating under-represented categories.

Reads category from the "category" field, falling back to "bucket".

Cap:    categories above --max are randomly downsampled to --max.
Repeat: categories below --min are cycled (round-robin) up to --min.
Output is shuffled with a fixed --seed for reproducibility.

Usage:
    python scripts/balance_dataset.py \\
        --input  data/v0.12_train.jsonl \\
        --output data/v0.13_train.jsonl \\
        --max-per-category 1500 \\
        --min-per-category 300 \\
        --seed 42

    # Dry run — print distribution only, write nothing:
    python scripts/balance_dataset.py \\
        --input data/v0.12_train.jsonl \\
        --dry-run
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: line {i} parse error: {e}", file=sys.stderr)
    return records


def get_category(record: dict) -> str:
    return record.get("category") or record.get("bucket") or "other"


def print_distribution(label: str, buckets: dict[str, list]) -> None:
    total = sum(len(v) for v in buckets.values())
    print(f"\n{label} ({total:,} total):")
    for cat in sorted(buckets, key=lambda c: -len(buckets[c])):
        print(f"  {cat:<35} {len(buckets[cat]):>5}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input",  required=True, help="Input JSONL file")
    ap.add_argument("--output", help="Output JSONL file (omit with --dry-run)")
    ap.add_argument("--max-per-category", type=int, default=None,
                    help="Cap categories above this count (default: no cap)")
    ap.add_argument("--min-per-category", type=int, default=None,
                    help="Repeat categories below this count (default: no repeat)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print distribution only, do not write output")
    args = ap.parse_args()

    if not args.dry_run and not args.output:
        ap.error("--output is required unless --dry-run is set")

    rng = random.Random(args.seed)

    print(f"Loading: {args.input}", flush=True)
    records = load_jsonl(Path(args.input))
    print(f"  {len(records):,} records loaded", flush=True)

    # Bucket by category
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        buckets[get_category(r)].append(r)

    print_distribution("Before balancing", buckets)

    # Cap over-represented
    if args.max_per_category is not None:
        for cat in list(buckets):
            if len(buckets[cat]) > args.max_per_category:
                buckets[cat] = rng.sample(buckets[cat], args.max_per_category)

    # Repeat under-represented
    if args.min_per_category is not None:
        for cat in list(buckets):
            n = len(buckets[cat])
            if 0 < n < args.min_per_category:
                needed = args.min_per_category - n
                source = buckets[cat][:]
                extra = []
                while len(extra) < needed:
                    batch = source[:]
                    rng.shuffle(batch)
                    extra.extend(batch)
                buckets[cat].extend(extra[:needed])

    print_distribution("After balancing", buckets)

    if args.dry_run:
        print("\n(dry-run — no file written)")
        return

    # Flatten and shuffle
    out_records = [r for recs in buckets.values() for r in recs]
    rng.shuffle(out_records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(out_records):,} records → {out_path}", flush=True)


if __name__ == "__main__":
    main()
