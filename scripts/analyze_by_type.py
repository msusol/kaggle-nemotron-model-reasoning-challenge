#!/usr/bin/env python3
"""Analyze validation accuracy broken down by problem type from kishanvavdara/nemotron-reasoning-traj."""

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict


def get_type_mapping() -> dict[str, str]:
    """Load problem type from the original CSV, keyed by example ID."""
    import kagglehub

    dataset = "kishanvavdara/nemotron-reasoning-traj"
    cache_dir = pathlib.Path.home() / ".cache" / "kagglehub" / "datasets" / dataset.replace("/", "_")

    # Try cached first
    csv_files = list(cache_dir.glob("*.csv")) if cache_dir.exists() else []
    if not csv_files:
        print(f"Downloading {dataset} to find problem types...", file=sys.stderr)
        src = pathlib.Path(kagglehub.dataset_download(dataset))
        csv_files = list(src.rglob("*.csv"))

    if not csv_files:
        print(f"ERROR: no CSV found in {dataset}", file=sys.stderr)
        return {}

    csv_path = csv_files[0]
    print(f"Using CSV: {csv_path}", file=sys.stderr)

    type_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            type_map[row["id"]] = row["problem type"].strip()

    print(f"Loaded {len(type_map)} problem type mappings", file=sys.stderr)
    return type_map


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scored",
        required=True,
        help="Path to .scored.jsonl output from validate_metric.py",
    )
    ap.add_argument("--out", help="Output report path (default: stdout)")
    args = ap.parse_args()

    scored_path = pathlib.Path(args.scored)
    if not scored_path.exists():
        print(f"ERROR: {scored_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Loading problem type mapping...", file=sys.stderr)
    type_map = get_type_mapping()

    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_samples = 0

    with open(scored_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sample_id = row["id"]
            is_correct = row["correct"]
            ptype = type_map.get(sample_id, "unknown")

            by_type[ptype]["correct"] += int(is_correct)
            by_type[ptype]["total"] += 1
            total_correct += int(is_correct)
            total_samples += 1

    # Format report
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("v0.3 Accuracy by Problem Type")
    report_lines.append("=" * 60)
    report_lines.append(f"Overall: {total_correct}/{total_samples} ({100*total_correct/total_samples:.1f}%)")
    report_lines.append("")

    # Sort by count descending
    sorted_types = sorted(by_type.items(), key=lambda x: x[1]["total"], reverse=True)
    for ptype, stats in sorted_types:
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        report_lines.append(
            f"  {ptype:20s}  {stats['correct']:3d}/{stats['total']:3d}  ({acc:5.1f}%)"
        )

    report = "\n".join(report_lines)
    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
        print(f"Saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
