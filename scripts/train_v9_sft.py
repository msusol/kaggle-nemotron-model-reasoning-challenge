#!/usr/bin/env python3
"""v0.9 SFT training — Format 4 from base model, no warmstart.

Trains from the base Nemotron model on data/v0.9_train.jsonl (18,603 examples,
14 categories). Assistant content = {trace}\n</think>\n\boxed{answer}.

Usage (via run_train_v9.sh):
    python scripts/train_v9_sft.py \\
        --train-file /workspace/data/v0.9_train.jsonl \\
        --output-dir /workspace/output/adapter_v9_YYYYMMDD_HHMMSS
"""

try:
    import unsloth  # must precede trl/transformers/peft to apply all optimizations
except ImportError:
    pass

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
    ap.add_argument("--train-file",      required=True, help="data/v0.9_train.jsonl")
    ap.add_argument("--output-dir",      default=None,  help="Adapter output dir (auto-named if omitted)")
    ap.add_argument("--model-id",        default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    ap.add_argument("--max-steps",       type=int,   default=1000)
    ap.add_argument("--learning-rate",   type=float, default=2e-4)
    ap.add_argument("--max-seq-length",  type=int,   default=8192)
    ap.add_argument("--batch-size",      type=int,   default=1)
    ap.add_argument("--grad-accum",      type=int,   default=16)
    ap.add_argument("--lora-r",          type=int,   default=32)
    ap.add_argument("--lora-alpha",      type=int,   default=32)
    ap.add_argument("--lora-dropout",    type=float, default=0.0)
    ap.add_argument("--seed",            type=int,   default=3407)
    return ap.parse_args()


def _make_cache_dropper(interval: float = 20.0) -> threading.Event:
    """Background thread that drops Linux page cache and CUDA freed blocks every interval s."""
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
    from trl import SFTConfig, SFTTrainer
    from torch.utils.data import DataLoader, Sampler

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── output dir ─────────────────────────────────────────────────────────────
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path(args.train_file).parent.parent / "output" / f"adapter_v9_{ts}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output: {args.output_dir}", flush=True)

    # ── mamba fast-path patch ───────────────────────────────────────────────────
    def _patch_mamba_fastpath(model):
        for name, mod in sys.modules.items():
            if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
                mod.is_fast_path_available = True
                print("Mamba fast path enabled", flush=True)
                return
        print("Mamba fast path: module not found yet", flush=True)

    # ── Step 1: load base model weights (one attempt — never reload) ───────────────
    # FastLanguageModel patches MoE expert tensors and Mamba projections so they are
    # visible to PEFT as trainable LoRA targets. If it fails, fall back to
    # AutoModelForCausalLM. Either way, weights are loaded exactly once.
    _unsloth_loaded = False
    _lora_via_unsloth = False
    _stop_dropper = _make_cache_dropper(interval=20.0)

    try:
        from unsloth import FastLanguageModel
        print(f"Loading base model via FastLanguageModel (Unsloth): {args.model_id}", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
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
        print(f"FastLanguageModel load failed ({_e}), using AutoModelForCausalLM", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

    _stop_dropper.set()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Base model loaded (unsloth={_unsloth_loaded}). GPU free={free_gb:.1f}GB", flush=True)

    # ── Step 2: apply LoRA (on the already-loaded model — no reload on failure) ───
    # Try FastLanguageModel.get_peft_model first (covers MoE/Mamba expert layers).
    # Unsloth 2026.6.x raises "No layers to finetune" for Nemotron-H with all-linear;
    # fall back to standard peft.get_peft_model on the Unsloth-patched model instead.
    if _unsloth_loaded:
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules="all-linear",
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
            _lora_via_unsloth = True
            print("LoRA initialized via FastLanguageModel.get_peft_model", flush=True)
        except Exception as _e:
            print(f"FastLanguageModel.get_peft_model failed ({_e})", flush=True)
            print("Falling back to standard PEFT on Unsloth-patched model", flush=True)

    if not _lora_via_unsloth:
        from peft import LoraConfig, get_peft_model as _std_get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = _std_get_peft_model(model, lora_config)
        print("LoRA initialized via PEFT get_peft_model", flush=True)

    _patch_mamba_fastpath(model)

    # ── gradient checkpointing (NemotronH native path) ──────────────────────────
    # NemotronHForCausalLM.supports_gradient_checkpointing=False blocks the standard
    # gradient_checkpointing_enable() path, but _set_gradient_checkpointing() is fully
    # implemented and walks all GradientCheckpointingLayer modules.
    # SFTConfig must keep gradient_checkpointing=False to avoid a second call.
    # Unsloth's use_gradient_checkpointing="unsloth" handles this when _lora_via_unsloth.
    if not _lora_via_unsloth:
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

    # ── override sequence-length caps before Unsloth tokenizes ──────────────────
    # Unsloth caps max_seq_length to the smallest of: tokenizer.model_max_length,
    # model.config.max_position_embeddings, and several other aliases. For Nemotron-H
    # (Mamba architecture) these are all set to 2048 by default. Override ALL of them
    # before the dataset is tokenized so full 8192-token traces are preserved.
    try:
        _base_cfg = model.base_model.model.config
    except AttributeError:
        _base_cfg = model.config

    # Model config — check every alias Unsloth might read
    _cfg_attrs = ("max_position_embeddings", "max_seq_length", "max_seqlen",
                  "seq_length", "max_sequence_length")
    for _attr in _cfg_attrs:
        _val = getattr(_base_cfg, _attr, None)
        if _val is not None:
            print(f"  model.config.{_attr} = {_val}", flush=True)
            if _val < args.max_seq_length:
                setattr(_base_cfg, _attr, args.max_seq_length)
                print(f"  → overrode to {args.max_seq_length}", flush=True)

    # Tokenizer model_max_length — the primary value Unsloth reads
    _tok_max = getattr(tokenizer, "model_max_length", None)
    print(f"  tokenizer.model_max_length = {_tok_max}", flush=True)
    if _tok_max is not None and _tok_max < args.max_seq_length:
        tokenizer.model_max_length = args.max_seq_length
        print(f"  → overrode tokenizer.model_max_length to {args.max_seq_length}", flush=True)

    # ── load dataset ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.train_file}", flush=True)
    records = []
    strat_labels = []
    with open(args.train_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({"messages": r["messages"]})
            # stratify by category (14 groups) rather than bucket (6 groups) so each
            # gradient-accumulation window sees all categories proportionally
            strat_labels.append(r.get("category") or r.get("bucket", "other"))

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
    strat_order = build_stratified_order(strat_labels, eff_bs, args.seed)
    n_labels = len(set(strat_labels))
    print(f"Effective batch: {eff_bs}, stratified over {n_labels} categories", flush=True)

    # ── training config ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=50,
        max_seq_length=args.max_seq_length,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1e9,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=False,
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

    from safetensors.torch import load_file
    st_path = Path(args.output_dir) / "adapter_model.safetensors"
    if st_path.exists():
        saved = load_file(str(st_path))
        print(f"Saved adapter key count: {len(saved)}", flush=True)

    print(f"Saved adapter to {args.output_dir}", flush=True)
    for f in sorted(Path(args.output_dir).iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
