import argparse
import json
import math
import re
from pathlib import Path

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


def maybe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def is_correct(pred: str, truth: str, rel_tol: float = 1e-4) -> bool:
    p = normalize(pred)
    t = normalize(truth)
    if p == t:
        return True
    pf = maybe_float(p)
    tf = maybe_float(t)
    if pf is not None and tf is not None:
        return math.isclose(pf, tf, rel_tol=rel_tol, abs_tol=0.0)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--rel-tol", type=float, default=1e-4)
    args = ap.parse_args()

    preds = {}
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            preds[row["id"]] = extract_answer(row["output"])

    total = 0
    correct = 0
    rows = []
    with open(args.labels, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1
            pred = preds.get(row["id"], "")
            ok = is_correct(pred, row["answer"], rel_tol=args.rel_tol)
            correct += int(ok)
            rows.append({"id": row["id"], "pred": pred, "truth": row["answer"], "correct": ok})

    acc = correct / total if total else 0.0
    print(json.dumps({"accuracy": acc, "correct": correct, "total": total}, indent=2))

    out_path = Path(args.predictions).with_suffix(".scored.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote detailed scores to {out_path}")


if __name__ == "__main__":
    main()
