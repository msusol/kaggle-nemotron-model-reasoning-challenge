#!/usr/bin/env python3
"""Parse a training log file and plot loss/accuracy/grad_norm/lr over steps."""

import argparse
import ast
import re
import sys
from pathlib import Path


def parse_log(path: Path) -> list[dict]:
    pattern = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
    records = []
    with open(path) as f:
        for line in f:
            for match in pattern.finditer(line):
                try:
                    record = ast.literal_eval(match.group())
                    if isinstance(record, dict) and "loss" in record:
                        records.append(record)
                except (ValueError, SyntaxError):
                    continue
    return records


def plot(records: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    def to_float(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    steps = [r["step"] if "step" in r else (i + 1) * 10 for i, r in enumerate(records)]
    loss = [to_float(r["loss"]) for r in records]
    accuracy = [to_float(r.get("mean_token_accuracy")) for r in records]
    grad_norm = [to_float(r.get("grad_norm")) for r in records]
    lr = [to_float(r.get("learning_rate")) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(out_path.stem, fontsize=12)

    axes[0, 0].plot(steps, loss, color="tab:blue")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Step")

    axes[0, 1].plot(steps, accuracy, color="tab:green")
    axes[0, 1].set_title("Mean Token Accuracy")
    axes[0, 1].set_xlabel("Step")

    axes[1, 0].plot(steps, grad_norm, color="tab:orange")
    axes[1, 0].set_title("Grad Norm")
    axes[1, 0].set_xlabel("Step")

    axes[1, 1].plot(steps, lr, color="tab:red")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "log",
        nargs="?",
        help="Path to training log file. Defaults to latest output/train_*.log",
    )
    ap.add_argument("--out", help="Output PNG path. Defaults to same name as log.")
    args = ap.parse_args()

    if args.log:
        log_path = Path(args.log)
    else:
        logs = sorted(Path("output").glob("train_*.log"))
        if not logs:
            print("No training logs found in output/", file=sys.stderr)
            sys.exit(1)
        log_path = logs[-1]
        print(f"Using latest log: {log_path}")

    out_path = Path(args.out) if args.out else log_path.with_suffix(".png")

    records = parse_log(log_path)
    if not records:
        print("No metric records found in log.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(records)} log entries (steps {10}–{len(records)*10})")
    plot(records, out_path)


if __name__ == "__main__":
    main()
