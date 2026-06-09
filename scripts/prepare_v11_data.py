#!/usr/bin/env python3
"""Convert v0.9 JSONL (messages format) to NeMo Gym GRPO format.

NeMo Gym expects {"prompt": ..., "answer": ...} JSONL pairs for GRPO training.
Our v0.9 data uses {"messages": [...], "category": ..., "id": ...} SFT format.

The assistant content in v0.9 is "{trace}\n</think>\n\\boxed{answer}".
This script extracts the last \\boxed{answer} as the ground-truth answer for
the NeMo Gym reward function.

Usage:
    python scripts/prepare_v11_data.py

    # Override paths:
    python scripts/prepare_v11_data.py \\
        --input-train  data/v0.9_train.jsonl \\
        --input-valid  data/v0.9_valid.jsonl \\
        --output-train data/v0.9_grpo_train.jsonl \\
        --output-valid data/v0.9_grpo_valid.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
PLACEHOLDER_ANSWERS = {"–", "-", "?", ""}


def extract_answer(assistant_text: str) -> str | None:
    """Return the last \\boxed{...} content, or None if absent or placeholder."""
    matches = BOXED_RE.findall(assistant_text)
    if not matches:
        return None
    answer = matches[-1].strip()
    if answer in PLACEHOLDER_ANSWERS:
        return None
    return answer


def convert_file(
    input_path: Path,
    output_path: Path,
    split: str,
) -> dict:
    no_messages = 0
    no_boxed = 0
    placeholder = 0
    written = 0
    by_category: Counter = Counter()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for lineno, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                ex = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"WARN {split}:{lineno}: {e}", file=sys.stderr)
                continue

            messages = ex.get("messages", [])
            user_msgs  = [m["content"] for m in messages if m.get("role") == "user"]
            asst_msgs  = [m["content"] for m in messages if m.get("role") == "assistant"]

            if not user_msgs or not asst_msgs:
                no_messages += 1
                continue

            answer = extract_answer(asst_msgs[-1])
            if answer is None:
                if BOXED_RE.search(asst_msgs[-1]):
                    placeholder += 1
                else:
                    no_boxed += 1
                continue

            category = ex.get("category", "unknown")
            out = {
                "prompt":   user_msgs[-1],
                "answer":   answer,
                "category": category,
                "id":       ex.get("id", ""),
                "source":   ex.get("source", ""),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            by_category[category] += 1

    return {
        "written":     written,
        "no_messages": no_messages,
        "no_boxed":    no_boxed,
        "placeholder": placeholder,
        "by_category": dict(by_category),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-train",  default="data/v0.9_train.jsonl")
    ap.add_argument("--input-valid",  default="data/v0.9_valid.jsonl")
    ap.add_argument("--output-train", default="data/v0.9_grpo_train.jsonl")
    ap.add_argument("--output-valid", default="data/v0.9_grpo_valid.jsonl")
    args = ap.parse_args()

    base = Path(__file__).parent.parent

    for split, inp_rel, out_rel in [
        ("train", args.input_train,  args.output_train),
        ("valid", args.input_valid,  args.output_valid),
    ]:
        inp = base / inp_rel
        out = base / out_rel

        if not inp.exists():
            print(f"ERROR: {inp} not found", file=sys.stderr)
            sys.exit(1)

        out.parent.mkdir(parents=True, exist_ok=True)
        stats = convert_file(inp, out, split)

        total_in = stats["written"] + stats["no_messages"] + stats["no_boxed"] + stats["placeholder"]
        print(f"\n{split}: {inp.name} → {out.name}")
        print(f"  written:              {stats['written']:>6}  / {total_in}")
        print(f"  skipped (no messages):{stats['no_messages']:>6}")
        print(f"  skipped (no \\boxed): {stats['no_boxed']:>6}")
        print(f"  skipped (placeholder):{stats['placeholder']:>6}")
        print(f"  by category:")
        for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"    {cat:<40s} {count:>5}")

    print("\nDone. Use data/v0.9_grpo_*.jsonl as input to run_nemo_grpo_spark.sh")


if __name__ == "__main__":
    main()
