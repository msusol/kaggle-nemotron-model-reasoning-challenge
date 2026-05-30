#!/usr/bin/env python3
"""Decode the pre-tokenized huikang corpus to standard {id,prompt,response,system} JSONL.

Reads corpus/corpus/{problem_id}/synthetic.jsonl entries from the zip, decodes token
IDs using the Nemotron tokenizer, strips chat-template markers, and writes train/valid
JSONL files compatible with train_lora.py / format_example().

Must run inside the nemotron-gb10 Docker container (needs transformers + cached tokenizer).

Usage (via runner script):
    bash scripts/run_extract_corpus.sh

Or directly inside container:
    python scripts/extract_huikang_corpus.py \\
        --zip /workspace/.cache/huikang-artifacts/huikang-nemotron-artifacts.zip \\
        --out-train /workspace/data/v0.4_train.jsonl \\
        --out-valid /workspace/data/v0.4_valid.jsonl
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys
import zipfile
from collections import Counter

# ── chat-template boundary patterns ───────────────────────────────────────────
# Record 0 (masked):  <|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{PROMPT}\n<|im_end|>\n<|im_start|>assistant\n<think>\n
# Record 1 (unmasked): {COT}\n</think>\n\boxed{ANSWER}<|im_end|>

_USER_RE = re.compile(
    r"<\|im_start\|>user\n(.*?)\n?<\|im_end\|>",
    re.DOTALL,
)
_SYS_RE = re.compile(
    r"<\|im_start\|>system\n(.*?)<\|im_end\|>",
    re.DOTALL,
)
_IM_END = "<|im_end|>"


def parse_masked(decoded: str) -> tuple[str, str]:
    """Return (system_text, user_text) from decoded masked tokens."""
    sys_m = _SYS_RE.search(decoded)
    system = sys_m.group(1).strip() if sys_m else ""
    user_m = _USER_RE.search(decoded)
    prompt = user_m.group(1).strip() if user_m else ""
    return system, prompt


def parse_unmasked(decoded: str) -> str:
    """Return response text from decoded unmasked tokens (strip trailing <|im_end|>)."""
    text = decoded
    if text.endswith(_IM_END):
        text = text[: -len(_IM_END)]
    return text.rstrip()


def is_valid(problem_id: str, valid_frac: float) -> bool:
    """Deterministic 95/5 split via MD5 hash — reproducible across runs."""
    h = int(hashlib.md5(problem_id.encode()).hexdigest(), 16)
    return (h % 10000) < int(valid_frac * 10000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="Path to huikang-nemotron-artifacts.zip")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-valid", required=True)
    ap.add_argument("--valid-split", type=float, default=0.05)
    ap.add_argument("--model-id",
                    default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                    help="Tokenizer model ID or local path")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print("Loading tokenizer...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    print(f"  vocab_size={tok.vocab_size}", file=sys.stderr)

    zip_path = pathlib.Path(args.zip)
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Reading corpus index...", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as zf:
        corpus_meta: dict[str, dict] = {}
        with zf.open("corpus.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r.get("included", True):
                    corpus_meta[r["problem_id"]] = r
        print(f"  {len(corpus_meta)} included problems", file=sys.stderr)

        out_train = pathlib.Path(args.out_train)
        out_valid = pathlib.Path(args.out_valid)
        out_train.parent.mkdir(parents=True, exist_ok=True)
        out_valid.parent.mkdir(parents=True, exist_ok=True)

        n_train = n_valid = n_skip = 0
        category_counts: Counter = Counter()

        with open(out_train, "w", encoding="utf-8") as f_tr, \
             open(out_valid, "w", encoding="utf-8") as f_va:

            for i, (pid, meta) in enumerate(corpus_meta.items()):
                if i % 1000 == 0:
                    print(f"  {i}/{len(corpus_meta)} ...", file=sys.stderr, flush=True)

                seg_path = f"corpus/corpus/{pid}/synthetic.jsonl"
                try:
                    with zf.open(seg_path) as f:
                        raw = f.read().decode("utf-8", errors="replace")
                    segments = [json.loads(l) for l in raw.strip().splitlines() if l.strip()]
                except KeyError:
                    n_skip += 1
                    continue

                masked_tokens: list[int] = []
                unmasked_tokens: list[int] = []
                for seg in segments:
                    toks = seg.get("tokens", [])
                    if seg.get("type") == "masked":
                        masked_tokens.extend(toks)
                    else:
                        unmasked_tokens.extend(toks)

                if not masked_tokens or not unmasked_tokens:
                    n_skip += 1
                    continue

                masked_text = tok.decode(masked_tokens, skip_special_tokens=False)
                unmasked_text = tok.decode(unmasked_tokens, skip_special_tokens=False)

                system, prompt = parse_masked(masked_text)
                response = parse_unmasked(unmasked_text)

                if not prompt or not response:
                    n_skip += 1
                    continue

                record = {
                    "id":       pid,
                    "prompt":   prompt,
                    "response": response,
                    "system":   system,
                    "category": meta.get("category", "unknown"),
                }
                line = json.dumps(record, ensure_ascii=False) + "\n"

                if is_valid(pid, args.valid_split):
                    f_va.write(line)
                    n_valid += 1
                else:
                    f_tr.write(line)
                    n_train += 1

                category_counts[meta.get("category", "unknown")] += 1

    print(f"\nExtraction complete:", file=sys.stderr)
    print(f"  train : {n_train}", file=sys.stderr)
    print(f"  valid : {n_valid}", file=sys.stderr)
    print(f"  skipped: {n_skip}", file=sys.stderr)
    print(f"  → {out_train}", file=sys.stderr)
    print(f"  → {out_valid}", file=sys.stderr)

    print("\nCategory breakdown (train+valid):", file=sys.stderr)
    total = n_train + n_valid
    for cat, n in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<30} {n:>6}  ({100*n/total:.1f}%)", file=sys.stderr)


if __name__ == "__main__":
    main()
