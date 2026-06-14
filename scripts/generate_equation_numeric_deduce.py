#!/usr/bin/env python3
"""Generate synthetic equation_numeric_deduce training examples (short traces, <500 tokens).

The puzzle: each operator symbol maps to a hidden arithmetic operation.
Given examples using those operators, infer the mapping, then apply it to a new pair.

Operations used (numeric only — no string concatenation):
  add      :  a + b             (result > both operands)
  abs_diff :  |a - b|           (examples always have a < b so result = b - a;
                                  this unambiguously distinguishes from add)

Why only two operations?
  add and abs_diff always produce distinct results (for a,b in 11–99 with a ≠ b):
    - add example gives a+b (≥ 22), while abs_diff gives b-a (< 99)
    - They coincide only when b-a = a+b → b=0, impossible in our range.

The huikang traces for this category exceed 6,000 tokens because they exhaustively
try dozens of operation/reversal variants. Our short CoT tries two clearly distinguished
operations and resolves unambiguously.

Usage:
    python scripts/generate_equation_numeric_deduce.py \\
        --n 500 --seed 42 \\
        --out data/eq_num_deduce_synthetic.jsonl
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROMPT_SUFFIX = "\nPlease put your final answer inside \\boxed{}."

SYMBOL_POOL = list("*$&#@!?~^%")

# Only add and abs_diff — always produces distinct results for our number range
OPERATIONS: dict[str, callable] = {
    "add":      lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
}

OP_LABEL = {
    "add":      "addition",
    "abs_diff": "absolute difference",
}


def gen_pair(rng: random.Random, op: str) -> tuple[int, int, int]:
    """Generate (a, b, result) for the given operation.

    For abs_diff, force a < b so result = b - a.
    This ensures add and abs_diff never give the same result:
      add gives a + b (e.g. 22+38=60), abs_diff gives b - a (e.g. 38-22=16).
    """
    for _ in range(200):
        a = rng.randint(11, 88)
        b = rng.randint(11, 88)
        if a == b:
            continue
        if op == "abs_diff" and a > b:
            a, b = b, a   # force a < b
        r = OPERATIONS[op](a, b)
        if r > 0:
            return a, b, r
    raise RuntimeError(f"Could not generate pair for op={op}")


def build_prompt(example_lines: list[str], test_expr: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
        "Below are a few examples:",
    ]
    lines.extend(example_lines)
    lines.append(f"\nNow, determine the result for: {test_expr}")
    return "\n".join(lines)


def cot_verify(op_name: str, pairs: list[tuple[int, int, int]]) -> tuple[bool, str]:
    """Test op_name against all pairs; return (all_match, verdict_str for first pair)."""
    all_match = all(OPERATIONS[op_name](a, b) == r for a, b, r in pairs)
    a, b, r = pairs[0]
    candidate = OPERATIONS[op_name](a, b)
    if all_match:
        verdict = "match"
    else:
        verdict = f"{candidate} ≠ {r}"
    return all_match, verdict


def build_cot(
    sym_ops: dict[str, str],
    example_pairs: dict[str, list[tuple[int, int, int]]],
    test_sym: str,
    ta: int,
    tb: int,
    tr: int,
) -> str:
    lines = [
        "I need to identify which arithmetic operation each operator symbol represents.",
        "",
    ]
    for sym, correct_op in sym_ops.items():
        pairs = example_pairs[sym]
        a, b, r = pairs[0]
        lines.append(f"Operator '{sym}' (example: {a}{sym}{b} = {r}):")
        for op_name in OPERATIONS:
            candidate = OPERATIONS[op_name](a, b)
            _, verdict = cot_verify(op_name, pairs)
            if op_name == "abs_diff":
                lines.append(
                    f"  {OP_LABEL[op_name]}: |{a} - {b}| = {candidate} → {verdict}"
                )
            else:
                lines.append(
                    f"  {OP_LABEL[op_name]}: {a} + {b} = {candidate} → {verdict}"
                )
        lines.append(f"  → '{sym}' = {OP_LABEL[correct_op]}")
        lines.append("")

    op_name = sym_ops[test_sym]
    if op_name == "abs_diff":
        lines += [
            f"Applying to test: {ta}{test_sym}{tb}",
            f"  {OP_LABEL[op_name]}: |{ta} - {tb}| = {tr}",
            f"  Result: {tr}",
        ]
    else:
        lines += [
            f"Applying to test: {ta}{test_sym}{tb}",
            f"  {OP_LABEL[op_name]}: {ta} + {tb} = {tr}",
            f"  Result: {tr}",
        ]
    return "\n".join(lines)


def fp(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n",    type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  default="data/eq_num_deduce_synthetic.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    op_names = list(OPERATIONS.keys())
    seen: set[str] = set()
    records: list[dict] = []
    attempts = 0

    while len(records) < args.n and attempts < args.n * 30:
        attempts += 1

        # Each puzzle uses both operations (one per symbol)
        syms = rng.sample(SYMBOL_POOL, 2)
        rng.shuffle(syms)
        sym_ops = {syms[0]: op_names[0], syms[1]: op_names[1]}  # add + abs_diff

        try:
            example_pairs = {
                sym: [gen_pair(rng, op) for _ in range(2)]
                for sym, op in sym_ops.items()
            }
        except RuntimeError:
            continue

        test_sym = rng.choice(syms)
        try:
            ta, tb, tr = gen_pair(rng, sym_ops[test_sym])
        except RuntimeError:
            continue

        # Interleave examples for both symbols
        ex_list = []
        for i in range(2):
            for sym in syms:
                a, b, r = example_pairs[sym][i]
                ex_list.append(f"{a}{sym}{b} = {r}")
        test_expr = f"{ta}{test_sym}{tb}"

        user_content = build_prompt(ex_list, test_expr) + PROMPT_SUFFIX
        key = fp(user_content)
        if key in seen:
            continue
        seen.add(key)

        cot = build_cot(sym_ops, example_pairs, test_sym, ta, tb, tr)
        assistant = f"{cot}\n</think>\n\\boxed{{{tr}}}"

        records.append({
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant},
            ],
            "category": "equation_numeric_deduce",
            "bucket":   "equation_like",
            "source":   "synthetic_v1",
            "id":       f"end_{fp(user_content)[:8]}",
        })

    if len(records) < args.n:
        print(
            f"WARNING: only {len(records)}/{args.n} after {attempts} attempts",
            file=sys.stderr,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records):,} equation_numeric_deduce examples → {out_path}")


if __name__ == "__main__":
    main()
