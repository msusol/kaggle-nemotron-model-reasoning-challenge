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
    ap.add_argument("--adapter-dir",     required=True,
                    help="SFT adapter in Unsloth format (backbone.layers, per-expert LoRA)")
    ap.add_argument("--grpo-warmstart",  default=None,
                    help="Pre-remapped adapter in GRPO format (model.layers, .default. keys). "
                         "Used by the PeftModel fallback path if Unsloth fails. "
                         "Build with: python scripts/remap_sft_adapter_for_grpo.py")
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
from trl import GRPOConfig, GRPOTrainer


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(args.adapter_dir).parent / f"adapter_grpo_v6_{ts}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)

    # ── load base model + SFT LoRA adapter ──────────────────────────────────
    # Unsloth's FastLanguageModel.from_pretrained always creates fused-MoE LoRA
    # (232 modules, model.layers naming) in GRPO mode — even when loading a
    # per-expert SFT adapter (12,008 keys, backbone.layers naming). The SFT keys
    # are simply "missing" from the GRPO model structure and never load.
    #
    # Fix: accept the 232 Unsloth modules, then manually inject the 232 warm-start
    # weights from the pre-remapped adapter (scripts/remap_sft_adapter_for_grpo.py).
    # The remapped adapter has model.layers keys with .default. infix that exactly
    # match the GRPO model's parameter names.
    #
    # Do NOT try a fallback to AutoModelForCausalLM — loading a second copy of
    # the 30B model while the first is in GPU memory causes OOM (exit 137).
    _stop = _make_cache_dropper()
    from unsloth import FastLanguageModel

    print(f"Loading base + SFT adapter via FastLanguageModel: {args.adapter_dir}", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        trust_remote_code=False,      # MUST be False — hub cache has buggy prepare_inputs_for_generation
        unsloth_force_compile=False,
        attn_implementation="eager",
        # NOTE: do NOT pass is_trainable=True — it propagates to NemotronHForCausalLM.__init__
        # which rejects it as an unexpected kwarg.
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Mark LoRA params trainable for GRPO
    for _n, _p in model.named_parameters():
        if ".lora_" in _n:
            _p.requires_grad_(True)

    n_lora = sum(1 for n, _ in model.named_parameters() if ".lora_" in n)
    print(f"Unsloth created {n_lora} LoRA modules (GRPO fused-MoE mode)", flush=True)

    # Inject SFT warm-start weights into the Unsloth GRPO modules.
    # The remapped adapter was built by scripts/remap_sft_adapter_for_grpo.py:
    #   backbone.layers → model.layers, lora_A.weight → lora_A.default.weight
    # These 232 keys match exactly the GRPO model's parameter names.
    if args.grpo_warmstart:
        from safetensors.torch import load_file as _st_load
        _ws_path = Path(args.grpo_warmstart) / "adapter_model.safetensors"
        if _ws_path.exists():
            print(f"Injecting warm-start weights from {_ws_path}", flush=True)
            _ws = _st_load(str(_ws_path), device="cpu")
            _params = dict(model.named_parameters())
            _loaded, _missing = 0, []
            for _k, _v in _ws.items():
                if _k in _params:
                    _params[_k].data.copy_(_v.to(_params[_k].dtype).to(_params[_k].device))
                    _loaded += 1
                else:
                    _missing.append(_k)
            print(f"Warm-start: injected {_loaded}/{len(_ws)} keys", flush=True)
            if _missing:
                print(f"Warm-start: {len(_missing)} keys not found: {_missing[:3]}", flush=True)
            del _ws, _params
            torch.cuda.empty_cache()
        else:
            print(f"WARNING: remapped adapter not found at {_ws_path} — LoRA starts cold", flush=True)
    else:
        print("WARNING: --grpo-warmstart not provided — LoRA starts from random init. "
              "Run scripts/remap_sft_adapter_for_grpo.py first.", flush=True)

    print("Model ready for GRPO training", flush=True)

    _stop.set()
    torch.cuda.empty_cache()
    print(f"Model loaded. GPU free={torch.cuda.mem_get_info()[0]/1e9:.1f}GB", flush=True)

    # ── Silence generation_config warnings ───────────────────────────────────
    # (1) max_length=262144 from model's generation_config.json conflicts with
    #     max_new_tokens passed by GRPOTrainer — clear it so max_new_tokens wins cleanly.
    # (2) pad_token_id set on generation_config prevents TRL from passing it as
    #     a separate kwarg (deprecated mixing of config + kwargs).
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = args.max_new_tokens
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    # ── Mamba fast path ──────────────────────────────────────────────────────
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = True
            print("Mamba fast path enabled", flush=True)
            break

    # Cast LoRA params to fp32 for stable training
    lora_count = 0
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.requires_grad_(True)
            param.data = param.data.to(torch.float32)
            lora_count += 1
    print(f"LoRA params trainable (fp32): {lora_count}", flush=True)

    _gc_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    try:
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
