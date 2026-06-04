#!/usr/bin/env python3
"""v0.5 SFT training — replicates kuangyicheng/nemotron-087-training approach.

Warmstarts from huikang/nemotron-adapter v27 and fine-tunes 240 steps on
competition train.csv + synthetic data (short responses, no <think> tags).

Key differences from train_lora.py (v0.4):
  - Warmstart: loads v27 adapter instead of training LoRA from scratch
  - Response format: short one-sentence trace + Final answer: \\boxed{answer}.
  - max_steps=240 (not full epoch)
  - max_seq_length=6144 (short responses, not 8192)
  - Stratified batching across category buckets
  - LoRA params cast to fp32 after warmstart load
  - lm_head key rename fix applied before saving

Usage (via run_train_v5.sh):
    python scripts/train_v5_sft.py \\
        --warmstart-dir /workspace/output/adapter_huikang_v27 \\
        --train-file    /workspace/data/v0.5_train.jsonl \\
        --output-dir    /workspace/output/adapter_v5_YYYYMMDD_HHMMSS
"""

import argparse
import sys
import threading
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*save_embedding_layers.*")
warnings.filterwarnings("ignore", message=r".*Could not find a config file.*")
warnings.filterwarnings("ignore", message=r".*Unable to fetch remote file.*")
warnings.filterwarnings("ignore", message=r".*use_return_dict.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*torchvision.*")
warnings.filterwarnings("ignore", message=r".*unsloth_force_compile.*")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmstart-dir", required=True, help="Path to huikang v27 adapter")
    ap.add_argument("--train-file",    required=True, help="data/v0.5_train.jsonl")
    ap.add_argument("--output-dir",    default=None,  help="Adapter output dir (auto-named if omitted)")
    ap.add_argument("--model-id",      default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--max-steps",     type=int, default=240)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-seq-length", type=int, default=6144)
    ap.add_argument("--batch-size",    type=int, default=1)
    ap.add_argument("--grad-accum",    type=int, default=16)
    ap.add_argument("--seed",          type=int, default=3407)
    return ap.parse_args()


def _make_cache_dropper(interval: float = 20.0) -> threading.Event:
    """Background thread that drops Linux page cache and CUDA freed blocks every interval s.

    During from_pretrained, temporary dtype-conversion tensors accumulate as
    reserved-but-freed CUDA allocator cache (~41 GB at 80% load) and crowd out
    the remaining weight shards. empty_cache() releases them back to the driver.
    Page cache drop clears safetensors mmap residue from Linux RAM.
    Returns a stop Event; set it to halt the thread after loading completes.
    """
    import torch as _torch

    stop = threading.Event()

    def _loop():
        while not stop.wait(interval):
            try:
                with open("/proc/sys/vm/drop_caches", "w") as fh:
                    fh.write("3\n")
            except OSError:
                pass
            try:
                free_before = _torch.cuda.mem_get_info()[0]
                _torch.cuda.empty_cache()
                free_after = _torch.cuda.mem_get_info()[0]
                reclaimed = (free_after - free_before) / 1e9
                alloc = _torch.cuda.memory_allocated() / 1e9
                reserv = _torch.cuda.memory_reserved() / 1e9
                print(
                    f"[dropper] reclaimed={reclaimed:+.1f}GB"
                    f" alloc={alloc:.1f}GB reserv={reserv:.1f}GB"
                    f" free={free_after/1e9:.1f}GB",
                    flush=True,
                )
            except Exception:
                pass

    t = threading.Thread(target=_loop, daemon=True, name="cache-dropper")
    t.start()
    return stop


def main():
    args = parse_args()

    import json
    import math
    import random
    from collections import defaultdict

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from torch.utils.data import DataLoader, Sampler

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── output dir ─────────────────────────────────────────────────────────────
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(args.train_file).parent.parent / "output" / f"adapter_v5_{ts}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)

    # ── mamba fast-path patch ───────────────────────────────────────────────────
    def _patch_mamba_fastpath(model):
        for name, mod in sys.modules.items():
            if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
                mod.is_fast_path_available = True
                print("Mamba fast path enabled", flush=True)
                return
        print("Mamba fast path: module not found yet (will retry after model load)", flush=True)

    # ── load base model ─────────────────────────────────────────────────────────
    # Prefer FastLanguageModel (Unsloth) which patches MoE expert tensors and Mamba
    # projections as trainable nn.Linear-like LoRA targets. Without this, PeftModel
    # silently drops ~232 of v27's 418 adapter keys and only attention layers train.
    # Falls back to AutoModelForCausalLM if Unsloth is not installed.
    # See docs/investigate/v0.5-unsloth-peft-key-mismatch.md for full explanation.
    _unsloth_loaded = False
    _stop_dropper = _make_cache_dropper(interval=20.0)
    try:
        from unsloth import FastLanguageModel
        print(f"Loading base model + tokenizer via FastLanguageModel (Unsloth): {args.model_id}", flush=True)
        base_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            dtype=torch.bfloat16,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
            trust_remote_code=True,
            unsloth_force_compile=False,
            attn_implementation="eager",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _unsloth_loaded = True
        print("FastLanguageModel loaded — MoE/Mamba expert layers patched for LoRA", flush=True)
    except Exception as _e:
        print(f"Unsloth unavailable ({_e}), falling back to AutoModelForCausalLM", flush=True)
        print(f"WARNING: v27 MoE/Mamba keys will be silently dropped — expect ~0.56 score", flush=True)
        print(f"Loading tokenizer: {args.model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loading base model: {args.model_id}", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

    _stop_dropper.set()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Base model loaded (unsloth={_unsloth_loaded}). GPU free={free_gb:.1f}GB", flush=True)

    _patch_mamba_fastpath(base_model)

    # ── load v27 warmstart ──────────────────────────────────────────────────────
    # With Unsloth: all 418 v27 keys load (MoE experts + Mamba layers visible).
    # Without Unsloth: only 186 standard-path keys load; 232 dropped silently.
    print(f"Loading warmstart adapter: {args.warmstart_dir}", flush=True)
    model = PeftModel.from_pretrained(
        base_model,
        args.warmstart_dir,
        is_trainable=True,
    )

    # Cast LoRA params to fp32 for precision (base model stays bf16)
    n_cast = 0
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)
            n_cast += 1
    print(f"Cast {n_cast} LoRA param tensors to fp32", flush=True)

    # Enable gradient checkpointing via NemotronH's native GradientCheckpointingLayer.
    # NemotronHForCausalLM.supports_gradient_checkpointing=False blocks the standard
    # gradient_checkpointing_enable() path, but _set_gradient_checkpointing() is fully
    # implemented and walks all modules that inherit GradientCheckpointingLayer.
    # SFTConfig must keep gradient_checkpointing=False so SFTTrainer does not call
    # gradient_checkpointing_enable() again and raise the ValueError.
    import functools
    _gc_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    try:
        model.base_model.model._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=_gc_func
        )
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled (NemotronH native, use_reentrant=False)", flush=True)
    except Exception as _e:
        print(f"Warning: gradient checkpointing unavailable: {_e}", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}", flush=True)

    # ── load dataset ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.train_file}", flush=True)
    records = []
    buckets = []
    with open(args.train_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({"messages": r["messages"]})
            buckets.append(r.get("bucket", "other"))

    dataset = Dataset.from_list(records)
    print(f"Dataset: {len(dataset)} examples", flush=True)

    # ── formatting function ─────────────────────────────────────────────────────
    def formatting_func(example):
        msgs = example["messages"]
        if msgs and isinstance(msgs[0], dict):
            conversations = [msgs]
        else:
            conversations = msgs
        texts = []
        for conv in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False, enable_thinking=True
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
            texts.append(text)
        return texts

    # ── stratified batching ─────────────────────────────────────────────────────
    def build_stratified_order(labels, batch_size, seed):
        by_label = defaultdict(list)
        for idx, label in enumerate(labels):
            by_label[label].append(idx)
        rng = random.Random(seed)
        for v in by_label.values():
            rng.shuffle(v)
        n_batches = max(1, math.ceil(len(labels) / batch_size))
        batches = [[] for _ in range(n_batches)]
        order = list(range(n_batches))
        rng.shuffle(order)
        i = 0
        for label in sorted(by_label.keys()):
            for idx in by_label[label]:
                batches[order[i % n_batches]].append(idx)
                i += 1
        return [idx for b in batches for idx in b]

    class OrderedSampler(Sampler):
        def __init__(self, order):
            self.order = list(order)
        def __iter__(self):
            return iter(self.order)
        def __len__(self):
            return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *a, stratified_order=None, **kw):
            super().__init__(*a, **kw)
            self._stratified_order = stratified_order

        def get_train_dataloader(self):
            if self._stratified_order is None:
                return super().get_train_dataloader()
            # Explicitly keep only tokenized columns — _remove_unused_columns is a
            # no-op when remove_unused_columns=False, so the 'text' string column
            # (added by formatting_func before tokenization) survives into the dataset.
            # DataCollatorForLanguageModeling then tries to tensorize it and raises
            # "too many dimensions 'str'". Select only what the collator needs.
            keep = [c for c in self.train_dataset.column_names
                    if c in ("input_ids", "attention_mask", "labels")]
            dataset = self.train_dataset.select_columns(keep)
            return DataLoader(
                dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=OrderedSampler(self._stratified_order),
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    eff_bs = args.batch_size * args.grad_accum
    strat_order = build_stratified_order(buckets, eff_bs, args.seed)
    print(f"Effective batch: {eff_bs}, stratified order built", flush=True)

    # ── training config ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_seq_length=args.max_seq_length,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1e9,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=False,       # GC enabled manually above via _set_gradient_checkpointing()
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
        report_to="none",
        packing=False,
    )

    trainer = StratifiedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        stratified_order=strat_order,
    )

    # ── train ───────────────────────────────────────────────────────────────────
    print("Starting training...", flush=True)
    trainer.train()
    print("Training complete", flush=True)

    # ── save adapter ────────────────────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Key count report — confirms whether Unsloth loaded all v27 keys or not.
    # With Unsloth: expect ~418+ keys (all MoE/Mamba keys trained and saved).
    # Without Unsloth: expect ~232 keys (MoE/Mamba silently dropped) → ~0.56 score.
    st_path = Path(args.output_dir) / "adapter_model.safetensors"
    if st_path.exists():
        saved = load_file(str(st_path))
        print(f"Saved adapter key count: {len(saved)} ({'full coverage' if len(saved) >= 400 else 'PARTIAL — MoE keys missing, expect ~0.56'})", flush=True)

    print(f"Saved adapter to {args.output_dir}", flush=True)
    for f in sorted(Path(args.output_dir).iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
