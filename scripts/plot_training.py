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

Warmstart chain mode (--compare + --split N):
  python scripts/plot_training.py --compare \
      "v0.9-run13:output/train_v9_run13.log" \
      "v0.12-run14:output/train_v12_spark.log" \
      "v0.12-run15:output/train_v12_run15.log" \
      "v0.14-run17:output/train_v14_run17.log" \
      --split 1 \
      --max-tokens 2048 4096 4096 4096 \
      --out docs/images/training_warmstart_chain.png

  --split N puts the first N runs on the left y-axis (e.g. run13 with its
  different loss scale); the remaining runs share the right y-axis.
  --max-tokens annotates each run's max sequence length in the legend.
"""

import argparse
import ast
import re
import sys
from pathlib import Path


def parse_log(path: Path) -> list[dict]:
    """Parse loss records from a training log.

    Extracts step from the tqdm progress bar (| N/MAX [...]) on the preceding
    line and attaches it as record['step']. Falls back to epoch-only if no
    tqdm line is found before a loss record.
    """
    tqdm_pat = re.compile(r"\|\s+(\d+)/\d+\s+\[")
    loss_pat = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
    records = []
    last_step = None
    with open(path) as f:
        for line in f:
            m = tqdm_pat.search(line)
            if m:
                last_step = int(m.group(1))
            for match in loss_pat.finditer(line):
                try:
                    record = ast.literal_eval(match.group())
                    if isinstance(record, dict) and "loss" in record:
                        if last_step is not None:
                            record["step"] = last_step
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


def _steps(records):
    """Return step values; fall back to index*10 if 'step' not parsed."""
    if records and "step" in records[0]:
        return [r["step"] for r in records]
    return [(i + 1) * 10 for i in range(len(records))]


def plot_single(records: list[dict], out_path: Path, log_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    xs = _steps(records)
    loss = [to_float(r["loss"]) for r in records]
    accuracy = [to_float(r.get("mean_token_accuracy")) for r in records]
    grad_norm = [to_float(r.get("grad_norm")) for r in records]
    lr = [to_float(r.get("learning_rate")) for r in records]

    eval_recs = parse_eval(log_path) if log_path else []
    e_xs   = _steps(eval_recs) if eval_recs else []
    e_loss = [to_float(r.get("eval_loss")) for r in eval_recs]
    e_acc  = [to_float(r.get("eval_mean_token_accuracy")) for r in eval_recs]

    xlabel = "Step" if (records and "step" in records[0]) else "Epoch"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(out_path.stem, fontsize=12)

    axes[0, 0].plot(xs, loss, color="tab:blue", label="train")
    if e_xs:
        axes[0, 0].scatter(e_xs, e_loss, color="tab:red", marker="D",
                           s=80, zorder=5, label="eval")
        axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(xs, accuracy, color="tab:green", label="train")
    if e_xs:
        axes[0, 1].scatter(e_xs, e_acc, color="tab:red", marker="D",
                           s=80, zorder=5, label="eval")
        axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_title("Mean Token Accuracy")
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(xs, grad_norm, color="tab:orange")
    axes[1, 0].set_title("Grad Norm")
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(xs, lr, color="tab:purple")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel(xlabel)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def plot_compare(runs: list[tuple[str, Path]], out_path: Path,
                 split: int = 0,
                 max_tokens: list[int] | None = None) -> None:
    """Overlay multiple training runs on shared subplots (step on x-axis).

    When split > 0, the first `split` runs share the LEFT y-axis (e.g. v0.9
    with a much larger loss range) and the remaining runs share the RIGHT
    y-axis — both drawn on the same loss panel so the warmstart lineage is
    visually clear.

    max_tokens: optional list of max_seq_length per run, added to legend labels.
    """
    import matplotlib.pyplot as plt

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    LEFT_STYLE  = {"linestyle": "--", "linewidth": 1.8, "alpha": 0.85}
    RIGHT_STYLE = {"linestyle": "-",  "linewidth": 1.8}

    all_data = []
    for i, (label, path) in enumerate(runs):
        tok = max_tokens[i] if (max_tokens and i < len(max_tokens)) else None
        full_label = f"{label} (seq={tok})" if tok else label
        train_recs = parse_log(path)
        all_data.append((full_label, train_recs))

    dual = split > 0
    left_data  = all_data[:split] if dual else []
    right_data = all_data[split:] if dual else all_data

    # Layout: loss (top, full width) + grad_norm + LR (bottom, side by side)
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Warmstart Training Chain — v0.9 → v0.12 → v0.14",
                 fontsize=13, fontweight="bold")

    ax_loss = fig.add_subplot(2, 1, 1)
    ax_gn   = fig.add_subplot(2, 2, 3)
    ax_lr   = fig.add_subplot(2, 2, 4)

    ax_right = ax_loss.twinx() if dual else None

    legend_handles = []

    # ── Left y-axis runs (e.g. v0.9 run13) ──────────────────────────────────
    for idx, (label, recs) in enumerate(left_data):
        xs   = _steps(recs)
        loss = [to_float(r["loss"]) for r in recs]
        gn   = [to_float(r.get("grad_norm")) for r in recs]
        lr   = [to_float(r.get("learning_rate")) for r in recs]
        color = COLORS[idx % len(COLORS)]

        line, = ax_loss.plot(xs, loss, color=color, label=label, **LEFT_STYLE)
        legend_handles.append(line)
        ax_gn.plot(xs, gn, color=color, label=label, **LEFT_STYLE)
        ax_lr.plot(xs, lr, color=color, label=label, **LEFT_STYLE)

    if dual:
        ax_loss.set_ylabel("Loss — v0.9", color=COLORS[0], fontsize=10)
        ax_loss.tick_params(axis="y", labelcolor=COLORS[0])

    # ── Right y-axis runs (v0.12 / v0.14) ────────────────────────────────────
    ax_r = ax_right if dual else ax_loss
    offset = split if dual else 0

    for idx, (label, recs) in enumerate(right_data):
        xs   = _steps(recs)
        loss = [to_float(r["loss"]) for r in recs]
        gn   = [to_float(r.get("grad_norm")) for r in recs]
        lr   = [to_float(r.get("learning_rate")) for r in recs]
        color = COLORS[(offset + idx) % len(COLORS)]

        line, = ax_r.plot(xs, loss, color=color, label=label, **RIGHT_STYLE)
        legend_handles.append(line)
        ax_gn.plot(xs, gn, color=color, label=label, **RIGHT_STYLE)
        ax_lr.plot(xs, lr, color=color, label=label, **RIGHT_STYLE)

    if dual:
        ax_right.set_ylabel("Loss — v0.12 / v0.14", fontsize=10)

    # ── Annotations ──────────────────────────────────────────────────────────
    ax_loss.set_title("Training Loss (dashed = left axis, solid = right axis)"
                      if dual else "Training Loss",
                      fontsize=10)
    ax_loss.set_xlabel("Training Step")
    ax_loss.grid(True, alpha=0.3)

    # Combined legend from both axes
    labels = [h.get_label() for h in legend_handles]
    ax_loss.legend(legend_handles, labels, fontsize=8, loc="upper right",
                   framealpha=0.85)

    ax_gn.set_title("Grad Norm", fontsize=10)
    ax_gn.set_xlabel("Training Step")
    ax_gn.set_ylabel("Grad Norm")
    ax_gn.legend(fontsize=7)
    ax_gn.grid(True, alpha=0.3)

    ax_lr.set_title("Learning Rate", fontsize=10)
    ax_lr.set_xlabel("Training Step")
    ax_lr.set_ylabel("LR")
    ax_lr.legend(fontsize=7)
    ax_lr.grid(True, alpha=0.3)

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
    ap.add_argument(
        "--split",
        type=int,
        default=0,
        metavar="N",
        help="Put first N runs on left y-axis, rest on right (dual-axis mode).",
    )
    ap.add_argument(
        "--max-tokens",
        nargs="+",
        type=int,
        metavar="N",
        help="Max sequence length per run, in same order as --compare entries.",
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
        out_path = Path(args.out) if args.out else Path("docs/images/training_warmstart_chain.png")
        plot_compare(runs, out_path, split=args.split, max_tokens=args.max_tokens)
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
