#!/usr/bin/env python3
"""Convert the pre-tokenized huikang corpus to NeMo SFT format.

Reads corpus/corpus/{problem_id}/synthetic.jsonl entries from the zip and writes
NeMo-compatible pre-tokenized JSONL where each line is:

    {"input_ids": [int, ...], "labels": [int, ...]}

  input_ids  — full sequence: masked (system+user+<think>) + unmasked (CoT+answer)
  labels     — -100 for the masked region (no loss), actual token IDs for unmasked

Sequences longer than --max-seq-length are filtered out.

No tokenizer needed — token IDs are already in the zip.
Runs on host Python without a Docker container.

Usage:
    python scripts/prepare_nemo_dataset.py \\
        --zip .cache/huikang-artifacts/huikang-nemotron-artifacts.zip \\
        --out-train data/nemo_train.jsonl \\
        --out-valid data/nemo_valid.jsonl

Or via runner script:
    bash scripts/run_prepare_nemo_dataset.sh
"""

import argparse
import hashlib
import json
import pathlib
import sys
import zipfile
from collections import Counter

_IGNORE_INDEX = -100


def is_valid(problem_id: str, valid_frac: float) -> bool:
    """Deterministic 95/5 split via MD5 hash — same split as extract_huikang_corpus.py."""
    h = int(hashlib.md5(problem_id.encode()).hexdigest(), 16)
    return (h % 10000) < int(valid_frac * 10000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="Path to huikang-nemotron-artifacts.zip")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-valid", required=True)
    ap.add_argument("--valid-split", type=float, default=0.05)
    ap.add_argument("--max-seq-length", type=int, default=8192,
                    help="Filter out sequences longer than this (default 8192)")
    args = ap.parse_args()

    zip_path = pathlib.Path(args.zip)
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found", file=sys.stderr)
        sys.exit(1)

    out_train = pathlib.Path(args.out_train)
    out_valid = pathlib.Path(args.out_valid)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_valid.parent.mkdir(parents=True, exist_ok=True)

    print("Reading corpus index...", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as zf:
        corpus_meta: dict[str, dict] = {}
        with zf.open("corpus.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r.get("included", True):
                    corpus_meta[r["problem_id"]] = r
        print(f"  {len(corpus_meta)} included problems", file=sys.stderr)

        n_train = n_valid = n_skip = n_filtered = 0
        category_counts: Counter = Counter()

        with open(out_train, "w", encoding="utf-8") as f_tr, \
             open(out_valid, "w", encoding="utf-8") as f_va:

            for i, (pid, meta) in enumerate(corpus_meta.items()):
                if i % 1000 == 0:
                    print(f"  {i}/{len(corpus_meta)} ...", file=sys.stderr, flush=True)

                seg_path = f"corpus/corpus/{pid}/synthetic.jsonl"
                try:
                    with zf.open(seg_path) as f:
                        raw = f.read().decode("utf-8", errors="replace")
                    segments = [json.loads(l) for l in raw.strip().splitlines() if l.strip()]
                except KeyError:
                    n_skip += 1
                    continue

                masked_tokens: list[int] = []
                unmasked_tokens: list[int] = []
                for seg in segments:
                    toks = seg.get("tokens", [])
                    if seg.get("type") == "masked":
                        masked_tokens.extend(toks)
                    else:
                        unmasked_tokens.extend(toks)

                if not masked_tokens or not unmasked_tokens:
                    n_skip += 1
                    continue

                input_ids = masked_tokens + unmasked_tokens

                # Filter sequences that exceed the context window.
                if len(input_ids) > args.max_seq_length:
                    n_filtered += 1
                    continue

                # Labels: -100 for the masked (prompt) region, real IDs for the
                # unmasked (response) region. Loss is only computed on the response.
                labels = [_IGNORE_INDEX] * len(masked_tokens) + unmasked_tokens

                record = {
                    "input_ids": input_ids,
                    "labels":    labels,
                }
                line_out = json.dumps(record, ensure_ascii=False) + "\n"

                if is_valid(pid, args.valid_split):
                    f_va.write(line_out)
                    n_valid += 1
                else:
                    f_tr.write(line_out)
                    n_train += 1

                category_counts[meta.get("category", "unknown")] += 1

    print(f"\nConversion complete:", file=sys.stderr)
    print(f"  train    : {n_train}", file=sys.stderr)
    print(f"  valid    : {n_valid}", file=sys.stderr)
    print(f"  filtered : {n_filtered}  (seq_len > {args.max_seq_length})", file=sys.stderr)
    print(f"  skipped  : {n_skip}  (missing/empty segments)", file=sys.stderr)
    print(f"  → {out_train}", file=sys.stderr)
    print(f"  → {out_valid}", file=sys.stderr)

    print("\nCategory breakdown (train+valid):", file=sys.stderr)
    total = n_train + n_valid
    if total:
        for cat, n in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat:<30} {n:>6}  ({100*n/total:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
