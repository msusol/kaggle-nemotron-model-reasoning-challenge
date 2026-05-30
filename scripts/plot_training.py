#!/usr/bin/env python3
"""Parse training log file(s) and plot loss/accuracy/grad_norm/lr over steps.

Single-log mode (default):
  python scripts/plot_training.py [log_file] [--out out.png]

Comparison mode (--compare):
  python scripts/plot_training.py --compare \
      "v0.1-baseline:output/train_20260503_203554.log" \
      "v0.2-cot:output/train_20260528_211916.log" \
      "v0.3-filtered:output/train_cot_v3_20260529_075411.log" \
      --out docs/images/training_comparison_v03.png
"""

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


def parse_eval(path: Path) -> list[dict]:
    pattern = re.compile(r"\{[^{}]*'eval_loss'[^{}]*\}")
    records = []
    with open(path) as f:
        for line in f:
            for match in pattern.finditer(line):
                try:
                    record = ast.literal_eval(match.group())
                    if isinstance(record, dict) and "eval_loss" in record:
                        records.append(record)
                except (ValueError, SyntaxError):
                    continue
    return records


def to_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def plot_single(records: list[dict], out_path: Path, log_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    epochs = [to_float(r.get("epoch")) for r in records]
    loss = [to_float(r["loss"]) for r in records]
    accuracy = [to_float(r.get("mean_token_accuracy")) for r in records]
    grad_norm = [to_float(r.get("grad_norm")) for r in records]
    lr = [to_float(r.get("learning_rate")) for r in records]

    eval_recs = parse_eval(log_path) if log_path else []
    e_epochs = [to_float(r.get("epoch")) for r in eval_recs]
    e_loss   = [to_float(r.get("eval_loss")) for r in eval_recs]
    e_acc    = [to_float(r.get("eval_mean_token_accuracy")) for r in eval_recs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(out_path.stem, fontsize=12)

    axes[0, 0].plot(epochs, loss, color="tab:blue", label="train")
    if e_epochs:
        axes[0, 0].scatter(e_epochs, e_loss, color="tab:red", marker="D",
                           s=80, zorder=5, label="eval")
        axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, accuracy, color="tab:green", label="train")
    if e_epochs:
        axes[0, 1].scatter(e_epochs, e_acc, color="tab:red", marker="D",
                           s=80, zorder=5, label="eval")
        axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_title("Mean Token Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, grad_norm, color="tab:orange")
    axes[1, 0].set_title("Grad Norm")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, lr, color="tab:purple")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_compare(runs: list[tuple[str, Path]], out_path: Path) -> None:
    """Overlay multiple training runs on shared subplots (by epoch on x-axis).

    Separate subplots for loss (scales differ 10x across runs), shared subplots
    for accuracy and grad_norm. Eval checkpoints shown as scatter markers.
    """
    import matplotlib.pyplot as plt

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    EVAL_MARKERS = ["o", "s", "D", "^", "v"]

    all_data = []
    for label, path in runs:
        train_recs = parse_log(path)
        eval_recs = parse_eval(path)
        all_data.append((label, train_recs, eval_recs))

    # --- Layout: 3 rows (loss, accuracy, grad_norm), 1 col each ---------------
    # Loss gets its own subplot per run because scales differ by 10-100x.
    # Accuracy and grad_norm share a subplot across runs.
    n_runs = len(all_data)
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Training Comparison — v0.1 / v0.2 / v0.3", fontsize=13, fontweight="bold")

    # Row 1: one loss subplot per run
    loss_axes = [fig.add_subplot(3, n_runs, i + 1) for i in range(n_runs)]
    # Row 2: shared accuracy subplot (span all columns)
    ax_acc = fig.add_subplot(3, 1, 2)
    # Row 3: shared grad_norm subplot
    ax_gn = fig.add_subplot(3, 1, 3)

    for idx, (label, train_recs, eval_recs) in enumerate(all_data):
        color = COLORS[idx % len(COLORS)]
        emark = EVAL_MARKERS[idx % len(EVAL_MARKERS)]

        epochs = [to_float(r.get("epoch")) for r in train_recs]
        loss   = [to_float(r["loss"]) for r in train_recs]
        acc    = [to_float(r.get("mean_token_accuracy")) for r in train_recs]
        gn     = [to_float(r.get("grad_norm")) for r in train_recs]

        e_epochs = [to_float(r.get("epoch")) for r in eval_recs]
        e_loss   = [to_float(r.get("eval_loss")) for r in eval_recs]
        e_acc    = [to_float(r.get("eval_mean_token_accuracy")) for r in eval_recs]

        # Loss subplot (per run)
        ax = loss_axes[idx]
        ax.plot(epochs, loss, color=color, linewidth=1.5, label="train")
        if e_epochs:
            ax.scatter(e_epochs, e_loss, color=color, marker=emark,
                       s=80, zorder=5, label="eval")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if idx == 0 else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Accuracy (shared)
        ax_acc.plot(epochs, acc, color=color, linewidth=1.5, label=f"{label} train")
        if e_epochs:
            ax_acc.scatter(e_epochs, e_acc, color=color, marker=emark,
                           s=80, zorder=5, label=f"{label} eval")

        # Grad norm (shared)
        ax_gn.plot(epochs, gn, color=color, linewidth=1.5, label=label)

    ax_acc.set_title("Mean Token Accuracy (all runs)", fontsize=10)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Token Accuracy")
    ax_acc.legend(fontsize=8, ncol=2)
    ax_acc.grid(True, alpha=0.3)

    ax_gn.set_title("Grad Norm (all runs)", fontsize=10)
    ax_gn.set_xlabel("Epoch")
    ax_gn.set_ylabel("Grad Norm")
    ax_gn.legend(fontsize=8)
    ax_gn.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "log",
        nargs="?",
        help="Path to a single training log file (single-log mode).",
    )
    ap.add_argument(
        "--compare",
        nargs="+",
        metavar="LABEL:PATH",
        help="Compare multiple runs: 'label:path/to/log' pairs.",
    )
    ap.add_argument("--out", help="Output PNG path.")
    args = ap.parse_args()

    if args.compare:
        runs = []
        for spec in args.compare:
            if ":" not in spec:
                print(f"ERROR: --compare entries must be 'label:path', got: {spec}", file=sys.stderr)
                sys.exit(1)
            label, path_str = spec.split(":", 1)
            runs.append((label, Path(path_str)))
        out_path = Path(args.out) if args.out else Path("docs/images/training_comparison_v03.png")
        plot_compare(runs, out_path)
        return

    # Single-log mode
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

    print(f"Parsed {len(records)} log entries")
    plot_single(records, out_path, log_path=log_path)


if __name__ == "__main__":
    main()
