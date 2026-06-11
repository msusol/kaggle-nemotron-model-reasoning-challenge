#!/usr/bin/env python3
"""Convert pre-generated huikang reasoning and augmentation files to Format 4 JSONL.

Sources (from huikang/huikang-nemotron-repository-snapshot):
  reasoning/{pid}.txt       — CoT trace ending with \\boxed{answer}
  augmentations/{pid}.txt   — [category]/[prompt]/[completion] sections
  problems/{pid}.jsonl      — problem prompt for reasoning categories

Output record format (matches v0.5_train.jsonl used by Kaggle notebook and DGX Spark):
  {
    "messages": [
      {"role": "user", "content": "{prompt}\\nPlease put your final answer inside \\\\boxed{}."},
      {"role": "assistant", "content": "{trace}\\n</think>\\n\\\\boxed{{answer}}"}
    ],
    "category": "...",
    "source": "huikang_v12",
    "id": "...",
    "bucket": "..."
  }

Deduplicates against an existing JSONL file (--existing) by problem ID to avoid
re-adding examples already present in the training dataset.

After generating, mix with existing data to produce the final training file:
  cat data/v0.5_train.jsonl data/v0.12_augmented.jsonl \\
    | python scripts/shuffle_jsonl.py > data/v0.12_train.jsonl

Usage:
    python scripts/generate_reasoner_data.py \\
        --repo-root /tmp/huikang-repo/nemotron-master \\
        --existing  data/v0.5_train.jsonl \\
        --out       data/v0.12_augmented.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROMPT_SUFFIX = "\nPlease put your final answer inside \\boxed{}."

TYPE_TO_BUCKET = {
    "bit_manipulation":        "bit_like",
    "cipher":                  "cipher_like",
    "concatenation":           "other",
    "cryptarithm_deduce":      "equation_like",
    "cryptarithm_guess":       "equation_like",
    "equation_numeric_deduce": "equation_like",
    "equation_numeric_guess":  "equation_like",
    "equation_numeric":        "equation_like",
    "equation_symbolic":       "equation_like",
    "gravity":                 "other",
    "lstrip":                  "other",
    "matching":                "other",
    "numeral":                 "numeral_like",
    "spelling":                "cipher_like",
    "splitting":               "other",
    "unit_conversion":         "unit_like",
}

THINK_ONLY_CATS = {"concatenation", "lstrip", "matching", "spelling", "splitting"}


def extract_reasoning_answer(trace: str) -> str:
    """Extract the final \\boxed{} answer from a reasoning trace."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", trace)
    non_empty = [m.strip() for m in matches if m.strip() and m.strip() != "–"]
    return non_empty[-1] if non_empty else ""


def extract_think_only_answer(completion: str, category: str) -> str:
    """Extract the final answer from an augmentation completion trace."""
    text = completion.rstrip()
    if text.endswith("</think>"):
        text = text[: -len("</think>")].rstrip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""
    if category == "matching":
        bests = re.findall(r"Best:\s*(.+?):\s*\d+", text)
        return bests[-1].strip() if bests else lines[-1]
    last = lines[-1]
    if " -> " in last:
        return last.split(" -> ", 1)[1].strip()
    return last


def parse_augmentation_file(path: Path) -> tuple[str, str, str]:
    """Return (category, prompt, completion) from an augmentation file."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"\[category\]\n|\[prompt\]\n|\[completion\]\n", text)
    # parts[0] = '' (before [category]), parts[1]=cat, parts[2]=prompt, parts[3]=completion
    if len(parts) < 4:
        raise ValueError(f"Unexpected format in {path}: {len(parts)} parts")
    category   = parts[1].strip()
    prompt     = parts[2].rstrip("\n")
    completion = parts[3].rstrip("\n")
    return category, prompt, completion


def load_existing_fingerprints(path: Path) -> set[str]:
    """Return prompt-content fingerprints (MD5 hex) from an existing JSONL file.

    Handles both messages-format rows (v0.5_train.jsonl) and id-bearing rows.
    Uses a prompt-hash approach because v0.5_train.jsonl has no 'id' field.
    """
    import hashlib
    fps: set[str] = set()
    if not path.exists():
        return fps
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                msgs = r.get("messages", [])
                if msgs and msgs[0]["role"] == "user":
                    fp = hashlib.md5(msgs[0]["content"].encode()).hexdigest()
                    fps.add(fp)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    return fps


def _fp(user_content: str) -> str:
    import hashlib
    return hashlib.md5(user_content.encode()).hexdigest()


def process_reasoning(repo_root: Path, existing_fps: set[str]) -> list[dict]:
    """Convert rule_found reasoning/*.txt files to Format 4 records."""
    index_path = repo_root / "problems.jsonl"
    problems_dir = repo_root / "problems"
    reasoning_dir = repo_root / "reasoning"

    # Build status index from root problems.jsonl
    with open(index_path, encoding="utf-8") as f:
        status_index = {json.loads(l)["id"]: json.loads(l) for l in f}

    records = []
    skipped_dup = skipped_no_answer = skipped_no_file = 0

    for pid, meta in status_index.items():
        if meta.get("status") != "rule_found":
            continue

        rfile = reasoning_dir / f"{pid}.txt"
        if not rfile.exists():
            skipped_no_file += 1
            continue

        # Get prompt from the per-problem jsonl
        pfile = problems_dir / f"{pid}.jsonl"
        if not pfile.exists():
            skipped_no_file += 1
            continue
        prob = json.loads(pfile.read_text(encoding="utf-8"))
        prompt = prob.get("prompt", "")
        if not prompt:
            skipped_no_file += 1
            continue

        user_content = prompt + PROMPT_SUFFIX
        if _fp(user_content) in existing_fps:
            skipped_dup += 1
            continue

        trace = rfile.read_text(encoding="utf-8").rstrip()
        answer = extract_reasoning_answer(trace)
        if not answer:
            skipped_no_answer += 1
            continue

        cat = meta["category"]
        assistant = f"{trace}\n</think>\n\\boxed{{{answer}}}"
        existing_fps.add(_fp(user_content))  # prevent intra-batch dups
        records.append({
            "messages": [
                {"role": "user",      "content": prompt + PROMPT_SUFFIX},
                {"role": "assistant", "content": assistant},
            ],
            "category": cat,
            "bucket":   TYPE_TO_BUCKET.get(cat, "other"),
            "source":   "huikang_v12",
            "id":       pid,
        })

    print(f"  reasoning: {len(records)} new  "
          f"(dup={skipped_dup}, no_answer={skipped_no_answer}, no_file={skipped_no_file})",
          flush=True)
    return records


def process_augmentations(repo_root: Path, existing_fps: set[str]) -> list[dict]:
    """Convert augmentations/*.txt files to Format 4 records."""
    aug_dir = repo_root / "augmentations"
    records = []
    skipped_dup = skipped_no_answer = skipped_parse_err = 0

    for afile in sorted(aug_dir.iterdir()):
        pid = afile.stem

        try:
            cat, prompt, completion = parse_augmentation_file(afile)
        except (ValueError, IndexError):
            skipped_parse_err += 1
            continue

        user_content = prompt + PROMPT_SUFFIX
        if _fp(user_content) in existing_fps:
            skipped_dup += 1
            continue

        answer = extract_think_only_answer(completion, cat)
        if not answer:
            skipped_no_answer += 1
            continue

        assistant = f"{completion}\n</think>\n\\boxed{{{answer}}}"
        existing_fps.add(_fp(user_content))  # prevent intra-batch dups
        records.append({
            "messages": [
                {"role": "user",      "content": prompt + PROMPT_SUFFIX},
                {"role": "assistant", "content": assistant},
            ],
            "category": cat,
            "bucket":   TYPE_TO_BUCKET.get(cat, "other"),
            "source":   "huikang_v12_aug",
            "id":       pid,
        })

    print(f"  augmentations: {len(records)} new  "
          f"(dup={skipped_dup}, no_answer={skipped_no_answer}, parse_err={skipped_parse_err})",
          flush=True)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repo-root",
        default="/tmp/huikang-repo/nemotron-master",
        help="Root of the huikang-nemotron-repository-snapshot (default: %(default)s)",
    )
    ap.add_argument(
        "--existing",
        default="data/v0.5_train.jsonl",
        help="Existing JSONL to deduplicate against (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        default="data/v0.12_augmented.jsonl",
        help="Output JSONL file (default: %(default)s)",
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        choices=["reasoning", "augmentations", "both"],
        default=["both"],
        help="Which sources to include (default: both)",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    if not repo_root.exists():
        sys.exit(f"ERROR: --repo-root not found: {repo_root}")

    existing_path = Path(args.existing)
    print(f"Loading existing fingerprints from: {existing_path}", flush=True)
    existing_fps = load_existing_fingerprints(existing_path)
    print(f"  {len(existing_fps):,} existing prompt fingerprints loaded", flush=True)

    use_reasoning = "both" in args.sources or "reasoning" in args.sources
    use_augmentations = "both" in args.sources or "augmentations" in args.sources

    all_records: list[dict] = []

    if use_reasoning:
        print("Processing reasoning/*.txt ...", flush=True)
        rec = process_reasoning(repo_root, existing_fps)
        all_records.extend(rec)

    if use_augmentations:
        print("Processing augmentations/*.txt ...", flush=True)
        # Pass the same fps set (already updated by process_reasoning) for cross-source dedup
        aug = process_augmentations(repo_root, existing_fps)
        all_records.extend(aug)

    # Per-category summary
    cat_counts: Counter = Counter(r["category"] for r in all_records)
    print(f"\nNew examples by category ({len(all_records):,} total):")
    for cat in sorted(cat_counts):
        print(f"  {cat:<35} {cat_counts[cat]:>5}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_records):,} records → {out_path}", flush=True)


if __name__ == "__main__":
    main()
