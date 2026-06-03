#!/usr/bin/env python3
"""GRPO self-improvement training for Nemotron-3-Nano-30B.

Loads the v0.4-r3 SFT adapter as the starting point, then runs Group Relative
Policy Optimization on all 9,500 competition problems. Rewards are binary:
1.0 if the extracted answer matches ground truth, 0.0 otherwise.

Usage (via run_grpo.sh):
    python scripts/train_grpo.py --config configs/nemotron_grpo.yaml [--run-name NAME]
"""

import argparse
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

warnings.filterwarnings("ignore", message=r".*save_embedding_layers.*")

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BEST_RE = re.compile(r"Best:\s+([\d\s]+):")  # matching category: "Best: 3 4 5 6 7 0 1 2: 8"

# Categories with clean \boxed{} answers — reward computed normally.
BOXED_CATEGORIES = {
    "bit_manipulation", "cipher", "gravity", "numeral", "unit_conversion",
    "cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_deduce",
    "equation_numeric_guess",
}
# Categories where we can extract a compact answer from the reasoning chain.
MATCHING_CATEGORIES = {"matching"}
# Categories with multi-line structured output — no extractable scalar answer.
# These are included in training for KL regularisation but yield reward=0.
NO_REWARD_CATEGORIES = {"splitting", "concatenation", "lstrip", "spelling"}


# ── answer extraction ─────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    """Extract final answer from model output — mirrors validate_metric.py."""
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1].strip()
    words = text.strip().split()
    return words[-1].strip() if words else ""


def extract_training_answer(response: str, category: str) -> str:
    """Extract ground-truth answer from a huikang training response."""
    if category in BOXED_CATEGORIES:
        boxed = BOXED_RE.findall(response)
        return boxed[-1].strip() if boxed else ""
    if category in MATCHING_CATEGORIES:
        # "Best: 3 4 5 6 7 0 1 2: 8" → "3 4 5 6 7 0 1 2"
        matches = BEST_RE.findall(response)
        return matches[-1].strip() if matches else ""
    return ""  # NO_REWARD_CATEGORIES — no extractable scalar answer


def normalize(s: str) -> str:
    return s.strip().replace("$", "")


def answers_match(pred: str, truth: str, rel_tol: float = 1e-4) -> bool:
    if not truth:
        return False  # no ground truth → never reward
    p, t = normalize(pred), normalize(truth)
    if p == t:
        return True
    try:
        return math.isclose(float(p), float(t), rel_tol=rel_tol, abs_tol=0.0)
    except (ValueError, TypeError):
        return False


# ── reward function ────────────────────────────────────────────────────────────

def make_reward_fn(ground_truths: list[str]):
    """Returns a reward function closed over the ground-truth answers."""
    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion, truth in zip(completions, ground_truths):
            pred = extract_answer(completion)
            rewards.append(1.0 if answers_match(pred, truth) else 0.0)
        return rewards
    return reward_fn


# ── data loading ──────────────────────────────────────────────────────────────

def load_jsonl_data(jsonl_path: str, tokenizer, system: str = "") -> Dataset:
    """Load v0.4_train.jsonl → HF Dataset with 'prompt' and 'answer' columns.

    Extracts ground-truth answers from huikang responses:
      - Boxed categories: last \\boxed{} content
      - Matching: Best sequence extracted from reasoning chain
      - Splitting/concatenation/lstrip/spelling: answer="" → reward always 0
        (included for KL regularisation; model retains SFT knowledge via KL penalty)
    """
    rows = []
    no_answer_count = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            category = ex.get("category", "unknown")
            answer = extract_training_answer(ex["response"], category)
            if not answer:
                no_answer_count += 1
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": ex["prompt"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            rows.append({
                "prompt": prompt,
                "answer": answer,
                "id": ex["id"],
                "category": category,
            })
    print(f"  {len(rows)} examples loaded, {no_answer_count} with no extractable answer "
          f"(reward=0 for splitting/concatenation/lstrip/spelling)")
    return Dataset.from_list(rows)


# ── config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/nemotron_grpo.yaml")
    ap.add_argument("--run-name", default="grpo_v5")
    ap.add_argument("--test-steps", type=int, default=0,
                    help="If >0, stop after this many steps (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"]).parent / f"adapter_{args.run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {cfg['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])

    print(f"Loading base model: {cfg['base_model']}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    # Patch mamba fast path (same as train_lora.py)
    for name, mod in base_model.named_modules():
        if hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = True

    print(f"Loading init adapter: {cfg['init_adapter']}")
    model = PeftModel.from_pretrained(
        base_model, cfg["init_adapter"], is_trainable=True
    )

    print(f"Loading dataset: {cfg['train_file']}")
    dataset = load_jsonl_data(cfg["train_file"], tokenizer)
    print(f"  {len(dataset)} problems loaded")

    ground_truths = dataset["answer"]
    reward_fn = make_reward_fn(ground_truths)

    max_steps = args.test_steps if args.test_steps > 0 else -1

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        num_generations=cfg["num_generations"],
        temperature=cfg["temperature"],
        max_new_tokens=cfg["max_new_tokens"],
        max_prompt_length=cfg["max_prompt_length"],
        kl_coeff=cfg["kl_coeff"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_steps=max_steps,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"Saving adapter to {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
