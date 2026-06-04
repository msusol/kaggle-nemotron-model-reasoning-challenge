#!/usr/bin/env python3
"""v0.6 GRPO training — self-improvement via reward-based RL.

Loads best v0.5 SFT adapter as init, generates N responses per problem,
rewards correct \boxed{} answers, and updates via Group Relative Policy
Optimization (GRPO). No CoT labels needed — only competition problems +
ground-truth answers.

Usage (via run_grpo.sh):
    python scripts/train_grpo.py \
        --adapter-dir  /workspace/output/adapter_v5_sft_unsloth \
        --train-file   /workspace/data/train.csv \
        --output-dir   /workspace/output/adapter_grpo_v6_YYYYMMDD \
        --num-steps    500 \
        --num-generations 4
"""

import argparse
import csv
import functools
import math
import re
import sys
import types
import threading
import warnings
from datetime import datetime
from pathlib import Path

# Inject a stub llm_blender into sys.modules BEFORE importing TRL.
# llm_blender internally imports TRANSFORMERS_CACHE which was removed in
# transformers 5.5.3, causing GRPOTrainer to fail to import. File-patching
# judges.py fails because the container runs as non-root. Injecting a stub
# module is safe — we never call any llm_blender functions.
import importlib.machinery

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    return m

for _stub_name in (
    "llm_blender", "llm_blender.blender",
    "llm_blender.blender.blender", "llm_blender.blender.blender_utils",
    "vllm", "vllm.lora", "vllm.lora.request",
):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _make_stub(_stub_name)

# grpo_trainer.py does `from vllm import LLM, SamplingParams` when is_vllm_available().
# Our stub makes it available, so add sentinel attrs — GRPOTrainer never uses them
# when use_vllm=False.
sys.modules["vllm"].LLM = None
sys.modules["vllm"].SamplingParams = None

warnings.filterwarnings("ignore", message=r".*save_embedding_layers.*")
warnings.filterwarnings("ignore", message=r".*Could not find a config file.*")
warnings.filterwarnings("ignore", message=r".*Unable to fetch remote file.*")
warnings.filterwarnings("ignore", message=r".*use_return_dict.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*torchvision.*")

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_answer(text: str) -> str:
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def answers_match(pred: str, truth: str, rel_tol: float = 1e-4) -> bool:
    p, t = pred.strip(), truth.strip()
    if p == t:
        return True
    try:
        return math.isclose(float(p), float(t), rel_tol=rel_tol, abs_tol=0.0)
    except Exception:
        return False


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir",     required=True, help="Best v0.5 SFT adapter to init from")
    ap.add_argument("--train-file",      required=True, help="data/train.csv")
    ap.add_argument("--output-dir",      default=None)
    ap.add_argument("--model-id",        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--num-steps",       type=int,   default=500)
    ap.add_argument("--num-generations", type=int,   default=4)
    ap.add_argument("--learning-rate",   type=float, default=1e-6)
    ap.add_argument("--max-new-tokens",  type=int,   default=1024)
    ap.add_argument("--kl-coeff",        type=float, default=0.04)
    ap.add_argument("--batch-size",      type=int,   default=1)
    ap.add_argument("--seed",            type=int,   default=3407)
    return ap.parse_args()


def _make_cache_dropper(interval: float = 30.0):
    import torch as _torch
    stop = threading.Event()
    def _loop():
        while not stop.wait(interval):
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3\n")
            except OSError:
                pass
            try:
                _torch.cuda.empty_cache()
            except Exception:
                pass
    t = threading.Thread(target=_loop, daemon=True, name="cache-dropper")
    t.start()
    return stop


import torch
from datasets import Dataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(args.adapter_dir).parent / f"adapter_grpo_v6_{ts}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)

    # ── load model ──────────────────────────────────────────────────────────
    _stop = _make_cache_dropper()
    try:
        from unsloth import FastLanguageModel
        print(f"Loading via FastLanguageModel (Unsloth): {args.model_id}", flush=True)
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            full_finetuning=False,
            trust_remote_code=True,
            unsloth_force_compile=False,
            attn_implementation="eager",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("FastLanguageModel loaded — MoE layers patched", flush=True)
    except Exception as e:
        print(f"Unsloth unavailable ({e}), falling back to AutoModelForCausalLM", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True,
        )
    _stop.set()
    torch.cuda.empty_cache()
    print(f"Base model loaded. GPU free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB", flush=True)

    # ── Mamba fast path ──────────────────────────────────────────────────────
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = True
            print("Mamba fast path enabled", flush=True)
            break

    # ── load SFT adapter as init ─────────────────────────────────────────────
    print(f"Loading SFT init adapter: {args.adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=True)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)

    _gc_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    try:
        model.base_model.model._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=_gc_func
        )
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled", flush=True)
    except Exception as _e:
        print(f"Warning: gradient checkpointing unavailable: {_e}", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,}", flush=True)

    # ── dataset — prompts + ground truths only ───────────────────────────────
    print(f"Loading dataset: {args.train_file}", flush=True)
    records = []
    with open(args.train_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", row.get("question", ""))
            answer = str(row.get("answer", row.get("label", ""))).strip()
            if prompt and answer:
                records.append({"prompt": prompt, "answer": answer})
    dataset = Dataset.from_list(records)
    print(f"Dataset: {len(dataset)} problems", flush=True)

    # ── reward function ──────────────────────────────────────────────────────
    def reward_fn(completions, prompts=None, answer=None, **kwargs):
        ground_truths = answer if answer is not None else []
        rewards = []
        for completion, truth in zip(completions, ground_truths):
            text = completion if isinstance(completion, str) else (
                completion[-1].get("content", "") if completion else "")
            pred = extract_answer(text)
            rewards.append(1.0 if (pred and answers_match(pred, str(truth))) else 0.0)
        return rewards

    # ── GRPO config ──────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        kl_coeff=args.kl_coeff,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        use_vllm=False,
        seed=args.seed,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    print("Starting GRPO training...", flush=True)
    trainer.train()
    print("GRPO training complete", flush=True)

    # ── save bf16 ────────────────────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    from safetensors import safe_open
    from safetensors.torch import save_file
    st_path = Path(args.output_dir) / "adapter_model.safetensors"
    if st_path.exists():
        tensors = {}
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k).to(torch.bfloat16)
        save_file(tensors, str(st_path))
        size_gb = st_path.stat().st_size / 1e9
        print(f"Saved {len(tensors)} keys ({size_gb:.2f} GB bf16) → {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
