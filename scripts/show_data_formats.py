#!/usr/bin/env python3
"""
show_data_formats.py — Compare training data formats across v0.1, v0.5 SFT, and full CoT.

Shows the same problem rendered as each training format, plus the Nemotron-H chat-template
inference format, to make the mismatch between v0.1 raw answers and the expected output
format visually obvious.

Usage:
    python scripts/show_data_formats.py
    python scripts/show_data_formats.py --problem-id 001b24c4
    python scripts/show_data_formats.py --bucket numeral_like
"""

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent

# ── formatting helpers ────────────────────────────────────────────────────────

W = 90  # column width

def box(title: str, content: str, width: int = W) -> str:
    inner = width - 4
    top    = "┌─ " + title + " " + "─" * (width - len(title) - 4) + "┐"
    bottom = "└" + "─" * (width - 2) + "┘"
    lines  = []
    for raw_line in content.split("\n"):
        for wrapped in textwrap.wrap(raw_line, inner) or [""]:
            lines.append("│ " + wrapped.ljust(inner) + " │")
    return "\n".join([top] + lines + [bottom])

def hr(label: str = "", width: int = W) -> str:
    if label:
        pad = (width - len(label) - 2) // 2
        return "═" * pad + " " + label + " " + "═" * (width - pad - len(label) - 2)
    return "═" * width

# ── data loaders ─────────────────────────────────────────────────────────────

def load_train_csv(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["id"]] = {"id": r["id"], "prompt": r["prompt"], "answer": r["answer"]}
    return rows

def load_v05_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def find_v05_match(records: list[dict], prompt: str) -> dict | None:
    """Find v0.5 record whose user message prompt matches (first 80 chars)."""
    needle = prompt[:80]
    for rec in records:
        msgs = rec.get("messages", [])
        user_content = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if needle in user_content:
            return rec
    return None

def fetch_cot_dataset() -> list[dict] | None:
    """Download kishanvavdara/nemotron-reasoning-traj via kagglehub, return rows."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("kishanvavdara/nemotron-reasoning-traj")
        csv_path = next(Path(path).glob("**/*.csv"), None)
        if csv_path is None:
            print("  [kagglehub] No CSV found in downloaded dataset.")
            return None
        rows = []
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
        print(f"  [kagglehub] Loaded {len(rows)} rows from {csv_path.name}")
        return rows
    except ImportError:
        print("  [kagglehub] not installed — skipping CoT dataset. pip install kagglehub")
        return None
    except Exception as e:
        print(f"  [kagglehub] download failed: {e}")
        return None

def find_cot_match(cot_rows: list[dict], problem_id: str, prompt: str) -> dict | None:
    needle = prompt[:80]
    for r in cot_rows:
        if r.get("id") == problem_id:
            return r
        if needle in r.get("prompt", ""):
            return r
    return None

# ── inference format simulator ────────────────────────────────────────────────

def inference_format(prompt: str, answer: str) -> str:
    """Show what Nemotron-H's chat template emits + what a well-formed assistant reply looks like."""
    return (
        "<|im_start|>system\n"
        "detailed thinking on<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}\n"
        "Please put your final answer inside \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n"
        "  [reasoning trace — multiple steps, variable length]\n"
        "  ...\n"
        "</think>\n"
        f"\\boxed{{{answer}}}"
    )

def v01_training_pair(prompt: str, answer: str) -> str:
    """What v0.1 SFT actually trained on: prompt → bare answer."""
    return (
        "INPUT  (prompt):\n"
        f"  {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n"
        "\n"
        "TARGET (answer column — what the model was trained to output):\n"
        f"  {answer}\n"
        "\n"
        "⚠  No <think> block. No \\boxed{{}}. No reasoning trace.\n"
        "   Chat template prepends <think>\\n at inference — but the\n"
        "   model was never trained to write anything after it."
    )

def v05_training_pair(msgs: list[dict]) -> str:
    user    = next((m["content"] for m in msgs if m["role"] == "user"),      "")
    asst    = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return (
        "USER message (what v0.5 SFT trained on as input):\n"
        f"  {user[:200]}{'...' if len(user) > 200 else ''}\n"
        "\n"
        "ASSISTANT message (training target):\n"
        f"  {asst}\n"
        "\n"
        "⚠  Has \\boxed{} ✓   Has template reasoning ✓\n"
        "   Missing: <think>...</think> wrapper — chat template\n"
        "   prepends <think>\\n at inference, so model enters a\n"
        "   <think> block it was never trained to close with </think>."
    )

def cot_training_pair(row: dict) -> str:
    prompt     = row.get("prompt", "")
    cot        = row.get("cot", row.get("reasoning", row.get("response", "")))
    answer     = row.get("answer", "")
    correct    = row.get("correctness", row.get("correct", "?"))
    cot_preview = cot[:500] + ("..." if len(cot) > 500 else "") if cot else "[no cot column found]"
    return (
        "PROMPT:\n"
        f"  {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n"
        "\n"
        f"CORRECTNESS: {correct}   ANSWER: {answer}\n"
        "\n"
        "FULL CoT TRACE (preview):\n"
        f"  {cot_preview}\n"
        "\n"
        "✓ Contains full Nemotron-30B reasoning trace\n"
        "  When wrapped in <think>...</think>, this IS the correct\n"
        "  training format for Nemotron-H reasoning fine-tuning."
    )

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem-id", default=None,    help="Specific competition ID to display")
    ap.add_argument("--bucket",     default="numeral_like", help="Problem type to sample from")
    ap.add_argument("--n",          type=int, default=1, help="Number of examples to show")
    ap.add_argument("--skip-cot",   action="store_true",   help="Skip kagglehub CoT download")
    args = ap.parse_args()

    train_path = WORKSPACE / "data" / "train.csv"
    v05_path   = WORKSPACE / "data" / "v0.5_train.jsonl"

    print(hr("Loading data"))
    train_rows = load_train_csv(train_path)
    print(f"  train.csv:         {len(train_rows):>6,} problems  (raw competition data — v0.1 format)")
    v05_records = load_v05_jsonl(v05_path)
    print(f"  v0.5_train.jsonl:  {len(v05_records):>6,} records   (template-reasoning SFT data)")

    cot_rows = None
    if not args.skip_cot:
        print("  Fetching kishanvavdara/nemotron-reasoning-traj via kagglehub...")
        cot_rows = fetch_cot_dataset()
        if cot_rows:
            cols = list(cot_rows[0].keys())
            print(f"  nemotron_traj.csv: {len(cot_rows):>6,} rows     columns: {cols}")

    # Pick example problems
    if args.problem_id:
        ids = [args.problem_id]
    else:
        # Sample from bucket
        bucket_ids = []
        for rec in v05_records:
            if rec.get("bucket","").startswith(args.bucket.replace("_like","")):
                msgs = rec.get("messages", [])
                user_c = next((m["content"] for m in msgs if m["role"] == "user"), "")
                for pid, pdata in train_rows.items():
                    if pdata["prompt"][:60] in user_c:
                        bucket_ids.append(pid)
                        break
            if len(bucket_ids) >= args.n:
                break
        ids = bucket_ids[:args.n] if bucket_ids else list(train_rows.keys())[:args.n]

    for problem_id in ids:
        if problem_id not in train_rows:
            print(f"\n[!] ID {problem_id!r} not found in train.csv")
            continue

        raw = train_rows[problem_id]
        v05 = find_v05_match(v05_records, raw["prompt"])
        cot = find_cot_match(cot_rows, problem_id, raw["prompt"]) if cot_rows else None

        print("\n" + hr(f"PROBLEM  {problem_id}  [{args.bucket}]"))
        print(f"  Answer: {raw['answer']}")
        print()

        # ── Format 1: v0.1 raw ────────────────────────────────────────────────
        print(box("FORMAT 1 — v0.1  (raw competition data, SFT on bare answer)",
                  v01_training_pair(raw["prompt"], raw["answer"])))
        print()

        # ── Format 2: v0.5 SFT ───────────────────────────────────────────────
        if v05:
            print(box("FORMAT 2 — v0.5  (template-reasoning SFT, missing <think> wrapper)",
                      v05_training_pair(v05["messages"])))
        else:
            print(box("FORMAT 2 — v0.5  (no matching record found in v0.5_train.jsonl)", "—"))
        print()

        # ── Format 3: Full CoT ────────────────────────────────────────────────
        if cot:
            print(box("FORMAT 3 — kishanvavdara  (full Nemotron-30B CoT trace)",
                      cot_training_pair(cot)))
        elif cot_rows is not None:
            print(box("FORMAT 3 — kishanvavdara  (no match found for this problem ID)", "—"))
        else:
            print(box("FORMAT 3 — kishanvavdara  (skipped — run without --skip-cot to download)",
                      "kagglehub download: kishanvavdara/nemotron-reasoning-traj"))
        print()

        # ── Format 4: Correct inference format ───────────────────────────────
        print(box("FORMAT 4 — CORRECT target  (Nemotron-H chat template + <think> block)",
                  inference_format(raw["prompt"], raw["answer"])))
        print()

        print(hr("MISMATCH SUMMARY"))
        print(textwrap.dedent(f"""\
          v0.1  → bare answer '{raw['answer']}'   NO \\boxed{{}}   NO <think>
                  Model trained to output: '{raw['answer']}'
                  Model forced to start:   '<think>\\n...'   ← chat template
                  Result: model writes garbage reasoning then outputs '{raw['answer']}'
                  or skips </think> entirely.

          v0.5  → has \\boxed{{}}  ✓   has template reasoning ✓
                  Still missing <think>...</think> wrapper in training targets.
                  Model never sees </think> in training, so at inference it
                  enters the <think> block but may not close it cleanly.

          CoT   → full reasoning trace ✓   needs <think>...</think> wrapper ✓
                  This is what kishanvavdara's dataset provides (filtered to
                  correctness=='true').  When used as SFT targets wrapped in
                  <think>...</think>, this IS the correct training format.

          GRPO  → does not need CoT labels. Generates completions from the
                  warm-start model, rewards \\boxed{{correct}}. Requires the
                  warm-start to already know the FORMAT so reasoning fits
                  inside max_new_tokens before \\boxed{{}} appears.
        """))


if __name__ == "__main__":
    main()
