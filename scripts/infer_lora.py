"""
Run inference with a fine-tuned LoRA adapter against a JSONL prompt file.

Input JSONL: {"id": ..., "prompt": ..., "system": ...}
Output JSONL: {"id": ..., "output": <full model generation>}

Usage:
  python scripts/infer_lora.py \
    --adapter-dir /workspace/output/adapter \
    --data-file   /workspace/data/v0.3_valid.jsonl \
    --output-file /workspace/output/predictions.jsonl
"""
import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("BASE_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
MAX_MEMORY = {0: "115GiB"}


def _get_mamba_cache_cls(model):
    mod = sys.modules.get(model.__class__.__module__)
    if mod is None:
        return None
    return getattr(mod, "HybridMambaAttentionDynamicCache", None)


def build_prompt(row: dict, tokenizer) -> str:
    system = row.get(
        "system",
        "You are a careful reasoning model. Solve the problem step by step and end with Final answer: \\boxed{...}.",
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["prompt"]},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"system: {system}\nuser: {row['prompt']}\nassistant:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--output-file", default="/workspace/output/predictions.jsonl")
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=MAX_MEMORY,
    )
    print(f"Loading adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.eval()

    cache_cls = _get_mamba_cache_cls(model)

    rows = []
    with open(args.data_file, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as out_fh:
        for i in tqdm(range(0, len(rows), args.batch_size), desc="Inference"):
            batch = rows[i : i + args.batch_size]
            prompts = [build_prompt(r, tokenizer) for r in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(model.device)
            pkv = None
            if cache_cls is not None:
                try:
                    pkv = cache_cls(
                        model.config,
                        batch_size=len(batch),
                        dtype=torch.bfloat16,
                        device=model.device,
                    )
                except Exception:
                    pkv = None
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    past_key_values=pkv,
                )
            # Decode only the newly generated tokens (strip the prompt).
            input_len = inputs["input_ids"].shape[1]
            for row, output_ids in zip(batch, outputs):
                text = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
                out_fh.write(json.dumps({"id": row["id"], "output": text}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
