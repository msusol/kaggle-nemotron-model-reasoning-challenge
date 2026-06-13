#!/usr/bin/env python3
"""Generate synthetic equation_symbolic training examples.

equation_symbolic puzzles follow the "Alice's Wonderland" format:
  - 3 input→output examples demonstrating a hidden transformation rule
  - Ask: apply the same rule to a new input

Rules generated here (all involve symbolic/operator characters):
  delete_op   — remove all occurrences of one specific operator character
  replace_op  — replace one operator character with another throughout
  keep_op     — keep only a specific operator character, delete everything else
  swap_ops    — swap two specific characters wherever they appear

Output format matches v0.12_train.jsonl (Format 4).

Usage:
    python scripts/generate_equation_symbolic.py \\
        --n 500 \\
        --seed 42 \\
        --out data/equation_symbolic_synthetic.jsonl
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

PROMPT_SUFFIX = "\nPlease put your final answer inside \\boxed{}."

# Characters used to build symbolic expressions
# Using printable ASCII symbols that look "equation-like" / operator-like
FILL_CHARS = list("/\\#@%&|~^!?;:.,<>{}[]()$'`")
OPERATORS   = list("+-*=<>@#%^&|!/\\")


def rng_chars(rng: random.Random, n: int, pool: list[str], exclude: set[str] = frozenset()) -> str:
    eligible = [c for c in pool if c not in exclude]
    if not eligible:
        eligible = pool
    return "".join(rng.choice(eligible) for _ in range(n))


def insert_op(rng: random.Random, base: str, op: str) -> str:
    """Insert op at a random non-edge position in base."""
    if len(base) < 2:
        return op + base
    pos = rng.randint(1, len(base) - 1)
    return base[:pos] + op + base[pos:]


def gen_symbolic_expr(rng: random.Random, length: int, exclude: set[str] = frozenset()) -> str:
    pool = [c for c in FILL_CHARS if c not in exclude]
    if not pool:
        pool = FILL_CHARS
    return "".join(rng.choice(pool) for _ in range(length))


# ── Rule generators ────────────────────────────────────────────────────────────

def rule_delete_op(rng: random.Random) -> tuple[list[tuple[str,str]], str, str, str]:
    """Rule: delete all occurrences of a chosen operator character."""
    op = rng.choice(OPERATORS)
    exclude = {op}
    examples = []
    for _ in range(3):
        base = gen_symbolic_expr(rng, rng.randint(3, 5), exclude)
        inp = insert_op(rng, base, op)
        out = inp.replace(op, "")
        examples.append((inp, out))
    base = gen_symbolic_expr(rng, rng.randint(3, 5), exclude)
    test_in = insert_op(rng, base, op)
    test_out = test_in.replace(op, "")
    rule_desc = f"remove every '{op}' character from the string"
    return examples, test_in, test_out, rule_desc


def rule_replace_op(rng: random.Random) -> tuple[list[tuple[str,str]], str, str, str]:
    """Rule: replace one operator with another throughout."""
    ops = rng.sample(OPERATORS, 2)
    src, dst = ops[0], ops[1]
    exclude = {src, dst}
    examples = []
    for _ in range(3):
        base = gen_symbolic_expr(rng, rng.randint(3, 5), exclude)
        inp = insert_op(rng, base, src)
        out = inp.replace(src, dst)
        examples.append((inp, out))
    base = gen_symbolic_expr(rng, rng.randint(3, 5), exclude)
    test_in = insert_op(rng, base, src)
    test_out = test_in.replace(src, dst)
    rule_desc = f"replace every '{src}' with '{dst}'"
    return examples, test_in, test_out, rule_desc


def rule_keep_op(rng: random.Random) -> tuple[list[tuple[str,str]], str, str, str]:
    """Rule: keep only the operator character, discard everything else."""
    op = rng.choice(OPERATORS)
    exclude = {op}
    examples = []
    for _ in range(3):
        base = gen_symbolic_expr(rng, rng.randint(3, 5), exclude)
        inp = insert_op(rng, base, op)
        out = op  # keep only the op char(s)
        examples.append((inp, out))
    base = gen_symbolic_expr(rng, rng.randint(3, 5), exclude)
    test_in = insert_op(rng, base, op)
    test_out = op
    rule_desc = f"keep only the '{op}' character(s) and discard all others"
    return examples, test_in, test_out, rule_desc


def rule_swap_ops(rng: random.Random) -> tuple[list[tuple[str,str]], str, str, str]:
    """Rule: swap every occurrence of op_a with op_b and vice versa."""
    ops = rng.sample(OPERATORS, 2)
    a, b = ops[0], ops[1]
    exclude = {c for c in FILL_CHARS if c in (a, b)}

    def apply(s: str) -> str:
        return s.translate(str.maketrans(a + b, b + a))

    examples = []
    for _ in range(3):
        base = gen_symbolic_expr(rng, rng.randint(2, 4), set())
        # Insert both operators at random positions
        inp = insert_op(rng, insert_op(rng, base, a), b)
        out = apply(inp)
        examples.append((inp, out))
    base = gen_symbolic_expr(rng, rng.randint(2, 4), set())
    test_in = insert_op(rng, insert_op(rng, base, a), b)
    test_out = apply(test_in)
    rule_desc = f"swap every '{a}' with '{b}' and vice versa"
    return examples, test_in, test_out, rule_desc


RULE_GENERATORS = [rule_delete_op, rule_replace_op, rule_keep_op, rule_swap_ops]


# ── CoT trace builder ──────────────────────────────────────────────────────────

def build_cot(examples: list[tuple[str,str]], test_in: str, test_out: str, rule_desc: str) -> str:
    lines = [
        "<think>",
        "We need to infer a hidden transformation rule from the given examples.",
        "",
    ]
    for i, (inp, out) in enumerate(examples, 1):
        lines.append(f"Example {i}: {inp} → {out}")
        changed = [c for c in inp if c not in out] if len(out) < len(inp) else []
        added   = [c for c in out if c not in inp] if len(out) > len(inp) else []
        if changed:
            lines.append(f"  Characters removed: {set(changed)}")
        if added:
            lines.append(f"  Characters added: {set(added)}")
        if inp != out and not changed and not added:
            lines.append(f"  Some characters swapped or substituted.")

    lines += [
        "",
        f"Pattern: The rule is to {rule_desc}.",
        "",
        f"Applying to the test input: {test_in}",
        f"After applying the rule: {test_out}",
        "",
        f"Therefore the result is {test_out}.",
    ]
    return "\n".join(lines)


def build_prompt(examples: list[tuple[str,str]], test_in: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations.",
        "Below are a few examples:",
    ]
    for inp, out in examples:
        lines.append(f"{inp} = {out}")
    lines.append(f"\nNow, determine the result for: {test_in}")
    return "\n".join(lines)


def fp(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n",    type=int, default=500, help="Number of examples to generate (default: 500)")
    ap.add_argument("--seed", type=int, default=42,  help="Random seed (default: 42)")
    ap.add_argument("--out",  default="data/equation_symbolic_synthetic.jsonl",
                    help="Output JSONL file (default: %(default)s)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen_fps: set[str] = set()
    records: list[dict] = []
    attempts = 0
    max_attempts = args.n * 20

    rule_counts: dict[str, int] = {}

    while len(records) < args.n and attempts < max_attempts:
        attempts += 1
        gen_fn = rng.choice(RULE_GENERATORS)
        try:
            examples, test_in, test_out, rule_desc = gen_fn(rng)
        except Exception:
            continue

        if not test_out:
            continue

        prompt = build_prompt(examples, test_in)
        user_content = prompt + PROMPT_SUFFIX
        key = fp(user_content)
        if key in seen_fps:
            continue
        seen_fps.add(key)

        cot = build_cot(examples, test_in, test_out, rule_desc)
        assistant = f"{cot}\n</think>\n\\boxed{{{test_out}}}"

        rule_name = gen_fn.__name__
        rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        record_id = f"eq_sym_{fp(user_content)[:8]}"
        records.append({
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant},
            ],
            "category": "equation_symbolic",
            "bucket":   "equation_like",
            "source":   "synthetic_v1",
            "id":       record_id,
        })

    if len(records) < args.n:
        print(f"WARNING: only generated {len(records)}/{args.n} unique examples after {attempts} attempts",
              file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records):,} equation_symbolic examples → {out_path}")
    print("Rule distribution:")
    for name, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<25} {count:>5}")


if __name__ == "__main__":
    main()
