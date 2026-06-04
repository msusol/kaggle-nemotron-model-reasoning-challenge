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
import importlib.machinery
import math
import re
import sys
import types
import threading
import warnings
from datetime import datetime
from pathlib import Path

# ── Import Unsloth FIRST so its optimizations apply before TRL/transformers ──
# Then stub ONLY llm_blender (leaves vllm absent so Unsloth's metadata check
# doesn't trigger and FastLanguageModel loads correctly).
try:
    from unsloth import FastLanguageModel as _FLM  # noqa: F401 — primes patches
except Exception:
    pass

def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    return m

# llm_blender: TRANSFORMERS_CACHE removed in transformers 5.5.3 — stub it out.
# Do NOT stub vllm — leaving it absent lets is_vllm_available() return False,
# which skips the `from vllm import LLM, SamplingParams` line cleanly.
for _stub_name in (
    "llm_blender", "llm_blender.blender",
    "llm_blender.blender.blender", "llm_blender.blender.blender_utils",
):
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = _make_stub(_stub_name)

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

    # ── load model + SFT adapter via Unsloth ────────────────────────────────
    # MUST use FastLanguageModel.from_pretrained(adapter_dir) — NOT base model
    # + PeftModel.from_pretrained.  The v5_sft_unsloth adapter was saved by
    # Unsloth with keys like `backbone.layers.X.mixer.in_proj.lora_A.weight`
    # (no `.default.` adapter name, `backbone` not `model` in path).  Standard
    # PeftModel.from_pretrained expects `model.layers.X.*.lora_A.default.weight`
    # and silently reports ALL 12,008 keys missing → model trains from base
    # weights, generates 1024-token CoT responses → OOM.
    _stop = _make_cache_dropper()
    model = None
    try:
        from unsloth import FastLanguageModel
        print(f"Loading SFT adapter via FastLanguageModel: {args.adapter_dir}", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter_dir,  # Unsloth reads adapter_config.json → loads base + adapter
            dtype=torch.bfloat16,
            load_in_4bit=False,
            full_finetuning=False,
            trust_remote_code=False,  # MUST be False — hub cache has buggy prepare_inputs_for_generation
            unsloth_force_compile=False,
            attn_implementation="eager",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("FastLanguageModel loaded with SFT adapter — MoE layers patched", flush=True)
    except Exception as e:
        print(f"Unsloth load failed ({e}), falling back to base + PeftModel", flush=True)

    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=True)

    _stop.set()
    torch.cuda.empty_cache()
    print(f"Model loaded. GPU free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB", flush=True)

    # ── Mamba fast path ──────────────────────────────────────────────────────
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = True
            print("Mamba fast path enabled", flush=True)
            break

    # Make LoRA params trainable in fp32 (Unsloth loads adapter in inference_mode=True)
    lora_count = 0
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.requires_grad_(True)
            param.data = param.data.to(torch.float32)
            lora_count += 1
    print(f"LoRA params made trainable: {lora_count}", flush=True)

    _gc_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    try:
        # Try PeftModel path first, fall back to base model direct call
        _gc_target = getattr(model, "base_model", model)
        _gc_target = getattr(_gc_target, "model", _gc_target)
        _gc_target._set_gradient_checkpointing(
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
    # TRL 0.15.2 GRPOTrainer requires global_batch_size % num_generations == 0.
    # global_batch_size = per_device_train_batch_size × num_processes (1 GPU = 1).
    # Simplest valid config: per_device_train_batch_size = num_generations.
    # Each step: one problem, N completions generated and compared for advantage.
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.num_generations,  # must equal num_generations
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        max_prompt_length=1024,
        temperature=0.7,
        beta=args.kl_coeff,
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

    # GRPOTrainer sets model.warnings_issued["estimate_tokens"] = True at init.
    # NemotronHForCausalLM doesn't have this attribute — add it.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    # TRL 0.15.2 _get_train_sampler() takes no arguments, but transformers 5.5.3
    # Trainer.get_train_dataloader() now calls sampler_fn(dataset). Patch the
    # method to accept and ignore the extra positional argument.
    _orig_sampler = trainer._get_train_sampler
    trainer._get_train_sampler = lambda dataset=None: _orig_sampler()

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
