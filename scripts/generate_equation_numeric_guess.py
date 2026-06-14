#!/usr/bin/env python3
"""Generate synthetic equation_numeric_guess training examples (short traces, <500 tokens).

Same structure as equation_numeric_deduce but the operation pool includes string
concatenation.  Example: 51?71 = 5171 means '?' = concat("51","71") = 5171.

Operations:
  add      :  a + b
  abs_diff :  |a - b|   (examples always have a < b for unambiguous distinction)
  concat   :  int(str(a) + str(b))   e.g. 51 concat 71 = 5171

We always include concat as one of the two operators — that is the distinctive
feature of the "guess" category (the solver must consider the string-join hypothesis).

concat result is always different from add and abs_diff for 2-digit operands:
  - concat gives ≥ 1111 (four-digit); add gives ≤ 198; abs_diff ≤ 88
  → No ambiguity between concat and the numeric operations.

Usage:
    python scripts/generate_equation_numeric_guess.py \\
        --n 500 --seed 42 \\
        --out data/eq_num_guess_synthetic.jsonl
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROMPT_SUFFIX = "\nPlease put your final answer inside \\boxed{}."

SYMBOL_POOL = list("*$&#@!?~^%<>")

OPERATIONS: dict[str, callable] = {
    "add":      lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
    "concat":   lambda a, b: int(str(a) + str(b)),
}

OP_LABEL = {
    "add":      "addition",
    "abs_diff": "absolute difference",
    "concat":   "concatenation",
}


def gen_pair(rng: random.Random, op: str) -> tuple[int, int, int]:
    """Generate (a, b, result).

    For abs_diff, force a < b so result = b - a (distinct from add = a + b).
    For concat, use 2-digit numbers so result is always 4-digit (unambiguous).
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
    raise RuntimeError(f"Cannot generate pair for op={op}")


def build_prompt(example_lines: list[str], test_expr: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
        "Below are a few examples:",
    ]
    lines.extend(example_lines)
    lines.append(f"\nNow, determine the result for: {test_expr}")
    return "\n".join(lines)


def fmt_candidate(op_name: str, a: int, b: int) -> str:
    if op_name == "concat":
        return f"\"{a}\"+\"{b}\" = {int(str(a)+str(b))}"
    elif op_name == "abs_diff":
        return f"|{a} - {b}| = {abs(a-b)}"
    else:
        return f"{a} + {b} = {a+b}"


def build_cot(
    sym_ops: dict[str, str],
    example_pairs: dict[str, list[tuple[int, int, int]]],
    test_sym: str,
    ta: int,
    tb: int,
    tr: int,
) -> str:
    lines = [
        "I need to identify which operation each operator represents.",
        "The operation pool includes: addition, absolute difference, and string concatenation.",
        "",
    ]
    for sym, correct_op in sym_ops.items():
        pairs = example_pairs[sym]
        a, b, r = pairs[0]
        lines.append(f"Operator '{sym}' (example: {a}{sym}{b} = {r}):")
        for op_name in OPERATIONS:
            candidate = OPERATIONS[op_name](a, b)
            all_match = all(OPERATIONS[op_name](pa, pb) == pr for pa, pb, pr in pairs)
            verdict = "match" if all_match else f"{candidate} ≠ {r}"
            lines.append(f"  {OP_LABEL[op_name]}: {fmt_candidate(op_name, a, b)} → {verdict}")
        lines.append(f"  → '{sym}' = {OP_LABEL[correct_op]}")
        lines.append("")

    op_name = sym_ops[test_sym]
    lines += [
        f"Applying to test: {ta}{test_sym}{tb}",
        f"  {OP_LABEL[op_name]}: {fmt_candidate(op_name, ta, tb)}",
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
    ap.add_argument("--out",  default="data/eq_num_guess_synthetic.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # Numeric-only operations (always pair with concat)
    numeric_ops = ["add", "abs_diff"]
    seen: set[str] = set()
    records: list[dict] = []
    attempts = 0

    while len(records) < args.n and attempts < args.n * 30:
        attempts += 1

        # One symbol = concat, one = numeric op
        other_op = rng.choice(numeric_ops)
        syms = rng.sample(SYMBOL_POOL, 2)
        rng.shuffle(syms)
        sym_ops = {syms[0]: "concat", syms[1]: other_op}

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
            "category": "equation_numeric_guess",
            "bucket":   "equation_like",
            "source":   "synthetic_v1",
            "id":       f"eng_{fp(user_content)[:8]}",
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

    print(f"Generated {len(records):,} equation_numeric_guess examples → {out_path}")


if __name__ == "__main__":
    main()
