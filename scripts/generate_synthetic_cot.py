#!/usr/bin/env python3
"""Generate synthetic CoT reasoning for v0.1-only competition problems.

Calls a vLLM OpenAI-compatible server (e.g. DeepSeek-R1-Distill-Qwen-32B),
verifies each generated answer against the known ground truth, and writes
verified-correct examples in Nemotron training format.

Output format (matches v0.3 training data):
    assistant content: {cot_reasoning}\n</think>\n\\boxed{answer}
The Nemotron chat template prepends <think>\n automatically.

Usage:
    # Start vLLM server first:
    bash scripts/run_vllm.sh --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

    # Generate for all v0.1-only problems:
    python scripts/generate_synthetic_cot.py \\
        --v01 data/v0.1_train.jsonl \\
        --v03 data/v0.3_train.jsonl \\
        --out data/v0.4_synthetic_cot.jsonl

    # Generate only for specific types (recommended: start with hardest):
    python scripts/generate_synthetic_cot.py \\
        --v01 data/v0.1_train.jsonl \\
        --v03 data/v0.3_train.jsonl \\
        --out data/v0.4_synthetic_cot.jsonl \\
        --types bit_manipulation algebra

    # Resume an interrupted run:
    python scripts/generate_synthetic_cot.py \\
        --v01 data/v0.1_train.jsonl \\
        --v03 data/v0.3_train.jsonl \\
        --out data/v0.4_synthetic_cot.jsonl \\
        --resume
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

# ── answer extraction (mirrors validate_metric.py) ────────────────────────────

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_answer(text: str) -> str:
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1].strip()
    words = text.strip().split()
    return words[-1].strip() if words else ""


def normalize(s: str) -> str:
    return s.strip().replace("$", "")


def answers_match(pred: str, truth: str, rel_tol: float = 1e-4) -> bool:
    p, t = normalize(pred), normalize(truth)
    if p == t:
        return True
    try:
        return math.isclose(float(p), float(t), rel_tol=rel_tol, abs_tol=0.0)
    except (ValueError, TypeError):
        return False


# ── problem type classifier ────────────────────────────────────────────────────

def classify_type(prompt: str) -> str:
    p = prompt.lower()
    if "bit manipulation" in p:       return "bit_manipulation"
    if "gravitational constant" in p: return "gravity"
    if "unit conversion" in p:        return "unit_conversion"
    if "encryption rules" in p:       return "cipher"
    if "numeral system" in p:         return "numeral"
    if "transformation rules" in p:   return "algebra"
    return "unknown"


# ── CoT extraction from DeepSeek-R1-style output ──────────────────────────────

THINK_OPEN_RE = re.compile(r"^<think>\s*", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>.*", re.DOTALL | re.IGNORECASE)


def extract_cot_content(raw_output: str) -> str:
    """Strip opening <think> tag; keep everything up to (not including) </think>.

    DeepSeek-R1 generates: <think>\nreasoning\n</think>\nanswer text
    We want:               reasoning
    The training target is then: {reasoning}\n</think>\n\\boxed{answer}
    The Nemotron chat template prepends <think>\n automatically.
    """
    text = THINK_OPEN_RE.sub("", raw_output.strip())
    # If </think> present, take everything before it
    m = THINK_CLOSE_RE.search(text)
    if m:
        text = text[: m.start()].rstrip()
    return text


# ── generation ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant. "
    "Solve the problem step by step, showing your full reasoning. "
    "End your response with the final answer inside \\boxed{} — "
    "for example: \\boxed{42}."
)


def build_user_message(row: dict) -> str:
    prompt = row["prompt"]
    # Ensure the boxed instruction suffix is present
    if "\\boxed{}" not in prompt and "boxed" not in prompt:
        prompt = prompt.rstrip() + (
            "\nPlease put your final answer inside `\\boxed{}`. "
            "For example: `\\boxed{your answer}`"
        )
    return prompt


def generate_one(client: OpenAI, row: dict, model: str, args: argparse.Namespace) -> dict | None:
    """Call the vLLM API and return a training record if the answer verifies."""
    user_msg = build_user_message(row)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"\n  API error for {row['id']}: {e}", file=sys.stderr)
        return None

    raw = response.choices[0].message.content or ""
    pred_answer = extract_answer(raw)
    truth = row.get("answer", "")

    if not answers_match(pred_answer, truth, rel_tol=args.rel_tol):
        return None  # wrong answer — discard

    cot = extract_cot_content(raw)
    assistant_content = f"{cot}\n</think>\n\\boxed{{{truth}}}"

    return {
        "id":       row["id"],
        "prompt":   user_msg,
        "response": assistant_content,
        "system":   row.get("system", SYSTEM_PROMPT),
        "source":   "synthetic",
        "type":     classify_type(row["prompt"]),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v01",     required=True,  help="Path to v0.1_train.jsonl")
    ap.add_argument("--v03",     required=True,  help="Path to v0.3_train.jsonl (already-covered IDs)")
    ap.add_argument("--out",     required=True,  help="Output JSONL for verified synthetic CoT")
    ap.add_argument("--types",   nargs="+",      help="Filter to specific types (e.g. bit_manipulation algebra)")
    ap.add_argument("--resume",  action="store_true", help="Skip IDs already in --out")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server URL")
    ap.add_argument("--model",   default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    ap.add_argument("--delay",   type=float, default=0.1, help="Seconds between requests")
    args = ap.parse_args()

    # Load v0.1 examples
    v01 = {}
    with open(args.v01) as f:
        for line in f:
            row = json.loads(line)
            v01[row["id"]] = row

    # Load v0.3 covered IDs
    v03_ids = set()
    with open(args.v03) as f:
        for line in f:
            v03_ids.add(json.loads(line)["id"])

    # Gap = v0.1 NOT in v0.3
    gap = {k: v for k, v in v01.items() if k not in v03_ids}
    print(f"v0.1 total: {len(v01)} | v0.3 covered: {len(v03_ids)} | gap: {len(gap)}", file=sys.stderr)

    # Filter by problem type if requested
    if args.types:
        gap = {k: v for k, v in gap.items() if classify_type(v["prompt"]) in args.types}
        print(f"After type filter {args.types}: {len(gap)} examples", file=sys.stderr)

    # Resume: skip already-done IDs
    done_ids = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"Resuming — skipping {len(done_ids)} already written", file=sys.stderr)
    gap = {k: v for k, v in gap.items() if k not in done_ids}

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    total, verified, failed = len(gap), 0, 0
    rows = list(gap.values())

    with open(out_path, mode) as out_fh:
        for row in tqdm(rows, desc="Generating CoT"):
            result = generate_one(client, row, args.model, args)
            if result:
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
                verified += 1
            else:
                failed += 1
            if args.delay > 0:
                time.sleep(args.delay)

    print(f"\nDone: {verified}/{total} verified ({100*verified/total:.1f}%) | {failed} failed", file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)

    # Per-type summary
    if out_path.exists():
        from collections import Counter
        type_counts: Counter = Counter()
        with open(out_path) as f:
            for line in f:
                type_counts[json.loads(line).get("type", "unknown")] += 1
        print("\nVerified examples by type:", file=sys.stderr)
        for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {t:<20} {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
