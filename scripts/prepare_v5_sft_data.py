#!/usr/bin/env python3
"""Prepare v0.5 SFT training data — replicates kuangyicheng/nemotron-087-training approach.

Sources:
  - data/train.csv  (9,500 competition examples, 6 known categories)
  - Synthetic generators (2,400 × 5 categories = 12,000 examples)
    Ported from kuangyicheng/nemotron-087-training notebook cell 4, SEED=3407.

Output: data/v0.5_train.jsonl (~21,500 records in messages format)

Each record:
  {
    "messages": [
      {"role": "user",      "content": "<competition_prompt>\nPlease put your final answer inside \\boxed{}."},
      {"role": "assistant", "content": "I identify one rule... Final answer: \\boxed{<answer>}."}
    ],
    "bucket": "<category_bucket>"
  }

This is the format used directly by train_v5_sft.py (TRL SFTTrainer with
apply_chat_template). No pre-tokenization. No NeMo format. No long CoT.
"""

import argparse
import csv
import json
import math
import pathlib
import random
import string
import sys

SEED = 3407
PROMPT_SUFFIX = '\nPlease put your final answer inside \\boxed{}.'
ALPHABET = string.ascii_lowercase
WORD_BANK = [
    'stone', 'cable', 'garden', 'planet', 'window', 'rocket', 'silver',
    'harbor', 'market', 'forest', 'magnet', 'bridge', 'castle', 'winter',
    'summer', 'spring', 'autumn', 'tower', 'valley', 'island',
]


# ── prompt classification ──────────────────────────────────────────────────────

def prompt_bucket(text: str) -> str:
    t = text.lower()
    if 'bit manipulation' in t or 'bit shift' in t:
        return 'bit_like'
    if 'encrypt' in t or 'decrypt' in t or 'cipher' in t:
        return 'cipher_like'
    if 'numeral' in t or 'roman' in t:
        return 'numeral_like'
    if 'unit conversion' in t or ('convert' in t and 'unit' in t):
        return 'unit_like'
    if 'equation' in t or 'algebra' in t:
        return 'equation_like'
    if 'gravity' in t or 'physics' in t:
        return 'gravity_like'
    return 'other'


# ── record builder ─────────────────────────────────────────────────────────────

def make_record(prompt: str, answer: str, bucket: str) -> dict:
    trace = (
        'I identify one rule that matches all examples, verify that the same '
        'rule is consistent across the prompt, then apply it to the target. '
        f'Final answer: \\boxed{{{answer}}}.'
    )
    return {
        'messages': [
            {'role': 'user',      'content': prompt + PROMPT_SUFFIX},
            {'role': 'assistant', 'content': trace},
        ],
        'bucket': bucket,
    }


# ── synthetic generators (ported from 0.87 notebook, SEED=3407) ───────────────

def _fmt(title: str, examples: list, target: str) -> str:
    lines = [title, '', 'Infer the hidden rule from examples and solve the target.', '']
    for i, (inp, out) in enumerate(examples, 1):
        lines += [f'Example {i}:', f'Input: {inp}', f'Output: {out}', '']
    lines += ['Target:', f'Input: {target}', 'Output:']
    return '\n'.join(lines)


def gen_bit(rng: random.Random) -> dict:
    shift = rng.choice([1, 2])
    rules = [
        ('inv', lambda s: ''.join('1' if c == '0' else '0' for c in s), 'Flips every bit.'),
        ('rev', lambda s: s[::-1], 'Reverses the bit string.'),
        ('rot', lambda s, k=shift: s[k:] + s[:k], f'Rotates left by {shift}.'),
    ]
    _, fn, expl = rng.choice(rules)
    bl = rng.choice([6, 7, 8])
    vals: list[str] = []
    while len(vals) < 4:
        s = ''.join(rng.choice('01') for _ in range(bl))
        if s not in vals:
            vals.append(s)
    ans = fn(vals[3])
    return {
        'prompt':  _fmt('Bit puzzle', [(s, fn(s)) for s in vals[:3]], vals[3]),
        'answer':  ans,
        'bucket':  'bit_like',
    }


def gen_cipher(rng: random.Random) -> dict:
    sh = rng.choice([1, 2, 3, 4])
    rules = [
        ('rev', lambda w: w[::-1], 'Reverses characters.'),
        ('cae', lambda w, k=sh: ''.join(ALPHABET[(ALPHABET.index(c) + k) % 26] for c in w), f'Shifts by {sh}.'),
        ('atb', lambda w: ''.join(ALPHABET[25 - ALPHABET.index(c)] for c in w), 'Atbash cipher.'),
    ]
    _, fn, _ = rng.choice(rules)
    ws = rng.sample(WORD_BANK, 4)
    ans = fn(ws[3])
    return {
        'prompt':  _fmt('Cipher puzzle', [(w, fn(w)) for w in ws[:3]], ws[3]),
        'answer':  ans,
        'bucket':  'cipher_like',
    }


def gen_unit(rng: random.Random) -> dict:
    rules = [
        ('km',  1000, 'km',   'm'),
        ('kg',  1000, 'kg',   'g'),
        ('hr',    60, 'hour', 'min'),
        ('day',   24, 'day',  'hr'),
        ('m',    100, 'm',    'cm'),
    ]
    _, f, s, d = rng.choice(rules)
    vs = rng.sample(range(2, 25), 4)
    ans = str(vs[3] * f)
    return {
        'prompt':  _fmt(f'Unit puzzle ({s}->{d})', [(f'{v} {s}', str(v * f)) for v in vs[:3]], f'{vs[3]} {s}'),
        'answer':  ans,
        'bucket':  'unit_like',
    }


def gen_numeral(rng: random.Random) -> dict:
    rules = [
        ('bin', lambda n: format(n, 'b'), 'Converts to binary.'),
        ('hex', lambda n: format(n, 'x'), 'Converts to hex.'),
    ]
    _, fn, _ = rng.choice(rules)
    ns = rng.sample(range(6, 40), 4)
    ans = fn(ns[3])
    return {
        'prompt':  _fmt('Numeral puzzle', [(f'{n} (base 10)', fn(n)) for n in ns[:3]], f'{ns[3]} (base 10)'),
        'answer':  ans,
        'bucket':  'numeral_like',
    }


def gen_eq(rng: random.Random) -> dict:
    a = rng.choice([1, 2, 3, 4, 5, 6])

    def mk():
        x = rng.choice(range(-8, 9))
        b = rng.choice(range(-9, 10))
        c = a * x + b
        lhs = f'{a}x+{b}={c}' if b >= 0 else f'{a}x-{abs(b)}={c}'
        return lhs, str(x)

    exs = [mk() for _ in range(3)]
    tgt = mk()
    return {
        'prompt':  _fmt('Equation puzzle', exs, tgt[0]),
        'answer':  tgt[1],
        'bucket':  'equation_like',
    }


GENERATORS = {
    'bit_like':      gen_bit,
    'cipher_like':   gen_cipher,
    'unit_like':     gen_unit,
    'numeral_like':  gen_numeral,
    'equation_like': gen_eq,
}
SYNTH_PER_BUCKET = 2400


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-csv',  default='data/train.csv',       help='Competition train.csv')
    ap.add_argument('--out',        default='data/v0.5_train.jsonl', help='Output JSONL')
    ap.add_argument('--seed',       type=int, default=SEED)
    ap.add_argument('--synth-per-bucket', type=int, default=SYNTH_PER_BUCKET)
    args = ap.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    records: list[dict] = []

    # ── competition data ───────────────────────────────────────────────────────
    train_path = pathlib.Path(args.train_csv)
    if not train_path.exists():
        print(f'ERROR: {train_path} not found', file=sys.stderr)
        sys.exit(1)

    with open(train_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            bucket = prompt_bucket(row['prompt'])
            records.append(make_record(row['prompt'], row['answer'], bucket))

    n_real = len(records)
    print(f'Competition rows: {n_real}', file=sys.stderr)

    bucket_counts: dict[str, int] = {}
    for r in records:
        bucket_counts[r['bucket']] = bucket_counts.get(r['bucket'], 0) + 1
    for b, n in sorted(bucket_counts.items()):
        print(f'  {b}: {n}', file=sys.stderr)

    # ── synthetic data ─────────────────────────────────────────────────────────
    n_synth = 0
    for bucket, gen_fn in GENERATORS.items():
        for _ in range(args.synth_per_bucket):
            ex = gen_fn(rng)
            records.append(make_record(ex['prompt'], ex['answer'], ex['bucket']))
            n_synth += 1

    print(f'Synthetic rows:   {n_synth}', file=sys.stderr)
    print(f'Total:            {len(records)}', file=sys.stderr)

    # ── write output ───────────────────────────────────────────────────────────
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'Written: {out_path}', file=sys.stderr)

    # ── spot-check ─────────────────────────────────────────────────────────────
    with open(out_path, encoding='utf-8') as f:
        first = json.loads(f.readline())
    print('\nSpot-check (first record):', file=sys.stderr)
    print(f'  bucket: {first["bucket"]}', file=sys.stderr)
    print(f'  user[-60:]: {repr(first["messages"][0]["content"][-60:])}', file=sys.stderr)
    print(f'  assistant:  {repr(first["messages"][1]["content"])}', file=sys.stderr)


if __name__ == '__main__':
    main()
