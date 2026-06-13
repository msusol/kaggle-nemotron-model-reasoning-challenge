#!/usr/bin/env python3
"""Balance a JSONL training dataset by capping over- and repeating under-represented categories.

Reads category from the "category" field, falling back to "bucket".

Filter: if --max-tokens is set, examples whose chat-template rendering exceeds that
        token count are dropped *before* balancing (requires transformers).
Cap:    categories above --max are randomly downsampled to --max.
Repeat: categories below --min are cycled (round-robin) up to --min.
Output is shuffled with a fixed --seed for reproducibility.

Usage:
    python scripts/balance_dataset.py \\
        --input  data/v0.13_merged.jsonl \\
        --output data/v0.13_train.jsonl \\
        --max-tokens 4096 \\
        --tokenizer-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --max-per-category 1500 \\
        --min-per-category 300 \\
        --seed 42

    # Dry run — print distribution only, write nothing:
    python scripts/balance_dataset.py \\
        --input data/v0.13_merged.jsonl \\
        --max-tokens 4096 \\
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


def filter_by_tokens(
    records: list[dict],
    max_tokens: int,
    tokenizer_id: str,
) -> list[dict]:
    """Drop records whose chat-template rendering exceeds max_tokens tokens."""
    print(f"\nToken filter: loading tokenizer '{tokenizer_id}'...", flush=True)
    from transformers import AutoTokenizer  # noqa: PLC0415  (lazy import)

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  Tokenizer loaded. Filtering > {max_tokens} tokens...", flush=True)

    kept, dropped = [], 0
    drop_by_cat: dict[str, int] = defaultdict(int)
    for i, r in enumerate(records):
        try:
            text = tok.apply_chat_template(
                r["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
        except TypeError:
            text = tok.apply_chat_template(
                r["messages"], tokenize=False, add_generation_prompt=False
            )
        n_tok = len(tok(text, truncation=False, add_special_tokens=False)["input_ids"])
        if n_tok > max_tokens:
            dropped += 1
            drop_by_cat[get_category(r)] += 1
        else:
            kept.append(r)
        if (i + 1) % 5000 == 0:
            print(f"  ... {i+1:,} processed", flush=True)

    print(f"  Kept {len(kept):,} / dropped {dropped:,} (>{max_tokens} tok)", flush=True)
    if drop_by_cat:
        print("  Dropped per category:")
        for cat, n in sorted(drop_by_cat.items(), key=lambda x: -x[1]):
            print(f"    {cat:<35} {n:>5}")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input",  required=True, help="Input JSONL file")
    ap.add_argument("--output", help="Output JSONL file (omit with --dry-run)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Drop examples with more than this many tokens before balancing "
                         "(requires transformers + --tokenizer-id)")
    ap.add_argument("--tokenizer-id",
                    default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                    help="HuggingFace model ID for tokenizer used with --max-tokens "
                         "(default: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)")
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

    # Token-length filter (before balancing so caps/repeats reflect what training sees)
    if args.max_tokens is not None:
        records = filter_by_tokens(records, args.max_tokens, args.tokenizer_id)

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
