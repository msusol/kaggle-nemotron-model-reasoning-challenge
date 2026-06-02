#!/usr/bin/env python3
"""Convert v0.4 text JSONL to NeMo pre-tokenized SFT format.

Reads data/v0.4_train.jsonl and data/v0.4_valid.jsonl (prompt/response/system fields),
applies the same system-prompt logic as train_lora.py (SYSTEM_PROMPT constant), tokenizes
with the Nemotron chat template, and writes NeMo-compatible JSONL:

    {"input_ids": [int, ...], "labels": [int, ...]}

labels: -100 for the system+user+<think> prefix (masked/no loss),
        real token IDs for the CoT+answer region (loss computed here).

This produces a NeMo dataset consistent with v0.4b PEFT training (same system prompt,
same format_example logic). Use this instead of prepare_nemo_dataset.py when you want
NeMo and PEFT training to share the same prompt format.

Usage:
    python scripts/convert_jsonl_to_nemo.py
    python scripts/convert_jsonl_to_nemo.py \\
        --train data/v0.4_train.jsonl \\
        --valid data/v0.4_valid.jsonl \\
        --out-train data/nemo_dataset/nemo_train.jsonl \\
        --out-valid data/nemo_dataset/nemo_valid.jsonl \\
        --model-id nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
        --max-seq-length 8192
"""
import argparse
import json
import sys

_IGNORE_INDEX = -100


def build_messages(example: dict) -> list[dict]:
    # Use empty system throughout — matches 0.87 training format. The \boxed{}
    # instruction is in the user prompt for boxed categories; augmenter categories
    # intentionally have no \boxed{} in prompt or response.
    system = example.get("system", "")
    answer = example["response"]
    if not answer.startswith("<think>"):
        answer = "<think>\n" + answer
    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": example["prompt"]},
        {"role": "assistant", "content": answer},
    ]


def tokenize_example(example: dict, tokenizer) -> dict | None:
    messages = build_messages(example)

    # Get text for both full sequence and prefix, then tokenize separately.
    # Using tokenize=False + manual encode avoids BOS-token double-counting
    # that occurs when apply_chat_template is called twice with tokenize=True.
    full_text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prefix_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )

    if not full_text.startswith(prefix_text):
        return None  # template mismatch — skip

    full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    prefix_len = len(prefix_ids)
    if prefix_len >= len(full_ids):
        return None  # degenerate: no response tokens

    labels = [_IGNORE_INDEX] * prefix_len + full_ids[prefix_len:]
    return {"input_ids": full_ids, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train",          default="data/v0.4_train.jsonl")
    ap.add_argument("--valid",          default="data/v0.4_valid.jsonl")
    ap.add_argument("--out-train",      default="data/nemo_dataset/nemo_train.jsonl")
    ap.add_argument("--out-valid",      default="data/nemo_dataset/nemo_valid.jsonl")
    ap.add_argument("--model-id",       default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--max-seq-length", type=int, default=8192)
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model_id}", file=sys.stderr)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for src, dst, split in [
        (args.train, args.out_train, "train"),
        (args.valid, args.out_valid, "valid"),
    ]:
        n_ok = n_skip = n_filtered = 0
        with open(src, encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                ex = json.loads(line)
                rec = tokenize_example(ex, tok)
                if rec is None:
                    n_skip += 1
                    continue
                if len(rec["input_ids"]) > args.max_seq_length:
                    n_filtered += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
        print(f"{split}: {n_ok} written, {n_filtered} filtered (>{args.max_seq_length} tok), "
              f"{n_skip} skipped", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
