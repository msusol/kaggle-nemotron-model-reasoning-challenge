#!/usr/bin/env python3
"""Generate synthetic spelling training examples (short traces, <300 tokens).

The rule: concatenate all input words (strip spaces), then surround each character
with '–' (EN DASH, U+2013).  Example: "hello world" → "–h–e–l–l–o–w–o–r–l–d–"

Prompt mirrors the huikang Alice's Wonderland format:
  - 3 sample input→output pairs (demonstrating the rule)
  - 1 test input pair
  - Question: what is the output for the test input?

The CoT trace is intentionally short (~150 tokens) so all examples pass the
4096-token limit that dropped every huikang spelling trace.

Usage:
    python scripts/generate_spelling.py \\
        --n 500 --seed 42 \\
        --out data/spelling_synthetic.jsonl
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# EN DASH — matches the huikang spelling format exactly
DASH = "–"

PROMPT_SUFFIX = "\nPlease put your final answer inside \\boxed{}."

# Pool of short word fragments (3–8 chars) matching the style of huikang prompts
WORD_POOL = [
    "iros", "pip", "orsche", "xo", "jen", "ierter", "arman", "agation",
    "fil", "ized", "isher", "keep", "izards", "burgh", "sql", "col", "fly",
    "pos", "minton", "gerald", "business", "emet", "ilh", "odge", "idone",
    "eta", "amek", "ase", "bean", "oje", "ordan", "cha", "bow", "modules",
    "need", "centes", "zd", "max", "clip", "stone", "flow", "ridge",
    "frost", "glow", "lamp", "mark", "nest", "palm", "ring", "salt",
    "tree", "vine", "wave", "yard", "zone", "arch", "bark", "cord",
    "door", "edge", "fish", "gate", "helm", "iris", "jack", "keel",
    "lace", "mast", "noon", "opal", "pine", "quay", "rust", "silk",
    "tack", "urge", "vale", "west", "axis", "bolt", "cane", "dale",
    "fern", "gust", "hive", "isle", "jade", "knot", "loft", "mire",
    "newt", "orca", "peat", "quit", "reef", "slab", "turf", "umber",
    "veil", "wick", "yore", "zeal", "aloe", "burr", "char", "dune",
    "flax", "gorge", "hull", "inky", "joust", "kelp", "lima", "mesa",
    "nook", "oxen", "prow", "quill", "rune", "shim", "tarn", "ulna",
    "vane", "wick", "yard", "zinc", "anno", "byte", "cyan", "delta",
    "echo", "foil", "grit", "hash", "icon", "jinx", "kilo", "lore",
    "myth", "neon", "opus", "pier", "quad", "ramp", "silt", "tuft",
    "unto", "volt", "weld", "xeon", "yell", "zinc", "acme", "berg",
    "cove", "dusk", "etch", "fuse", "gale", "haze", "inti", "jolt",
    "kern", "lull", "meld", "narc", "omen", "poll", "qoph", "rill",
    "smew", "torn", "ursa", "vole", "whit", "xyst", "yawl", "zebu",
    "logging", "friends", "plaques", "drugs", "secund", "formats",
    "pulling", "fuerzas", "generate", "gossip", "proximal", "business",
    "nominal", "vibrant", "clashes", "wastes", "editor", "reversal",
    "dislike", "textual", "modules", "venues", "actress", "enrolled",
    "halfway", "chromium", "shuffled", "arrested", "concerto",
]


def apply_rule(words: list[str]) -> str:
    """Concatenate all words then surround each character with DASH."""
    concat = "".join(words)
    return DASH + DASH.join(list(concat)) + DASH


def build_prompt(examples: list[tuple[list[str], str]], test_words: list[str]) -> str:
    lines = [
        "In Alice's Wonderland, secret processing rules are used on text.",
        "",
        "This is a sample input.",
    ]
    for i, (words, _) in enumerate(examples):
        lines.append(f"{i:02d}")
        lines.append(" ".join(words))
    lines += ["", "This is a sample output."]
    for i, (words, result) in enumerate(examples):
        lines.append(f"{i:02d}")
        lines.append(f"{' '.join(words)} -> {result}")
    lines += ["", "This is your input.", "00"]
    lines.append(" ".join(test_words))
    return "\n".join(lines)


def build_cot(
    examples: list[tuple[list[str], str]],
    test_words: list[str],
    test_result: str,
) -> str:
    ex_words, ex_result = examples[0]
    ex_concat = "".join(ex_words)
    lines = [
        "Looking at the sample outputs:",
        f"  {'  '.join(ex_words)} → concat without spaces = \"{ex_concat}\"",
        f"  Each character surrounded by '{DASH}': {ex_result}",
        "",
        f"Pattern: join all words, then place '{DASH}' before, between, and after every character.",
        "",
        "Applying to the test input:",
        f"  {'  '.join(test_words)} → concat = \"{' '.join(test_words).replace(' ', '')}\"",
        f"  Result: {test_result}",
    ]
    return "\n".join(lines)


def fp(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n",    type=int, default=500, help="Number of examples to generate")
    ap.add_argument("--seed", type=int, default=42,  help="Random seed")
    ap.add_argument("--out",  default="data/spelling_synthetic.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = list(set(WORD_POOL))  # deduplicate
    seen: set[str] = set()
    records: list[dict] = []

    attempts = 0
    max_attempts = args.n * 30

    while len(records) < args.n and attempts < max_attempts:
        attempts += 1

        # Build 3 sample pairs + 1 test pair
        num_words_per = [rng.randint(2, 3) for _ in range(4)]
        groups = [
            rng.sample(pool, k)
            for k in num_words_per
        ]
        examples = [(g, apply_rule(g)) for g in groups[:3]]
        test_words = groups[3]
        test_result = apply_rule(test_words)

        user_content = build_prompt(examples, test_words) + PROMPT_SUFFIX
        key = fp(user_content)
        if key in seen:
            continue
        seen.add(key)

        cot = build_cot(examples, test_words, test_result)
        assistant = f"{cot}\n</think>\n\\boxed{{{test_result}}}"

        records.append({
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": assistant},
            ],
            "category": "spelling",
            "bucket":   "spelling",
            "source":   "synthetic_v1",
            "id":       f"spell_{fp(user_content)[:8]}",
        })

    if len(records) < args.n:
        print(
            f"WARNING: only generated {len(records)}/{args.n} unique examples "
            f"after {attempts} attempts",
            file=sys.stderr,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Generated {len(records):,} spelling examples → {out_path}")


if __name__ == "__main__":
    main()
