#!/usr/bin/env python3
"""v0.9 SFT training — Format 4, supports warmstart from a prior adapter.

Trains from the base Nemotron model (or an existing LoRA adapter) on
data/v0.9_train.jsonl. Assistant content = {trace}\n</think>\n\boxed{answer}.

Two-run coalescing workflow:
  Run 1 (short):  --max-seq-length 2048  (covers ~3,892 examples ≤ 2048 tok)
  Run 2 (long):   --min-seq-length 2048 --max-seq-length 7680 \\
                  --warmstart-adapter /workspace/output/adapter_v9_sft_2k
  This makes run 2 continue training run 1's adapter on the complementary
  9,838 examples (2049–7680 tokens), yielding a single coalesced adapter.

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
    ap.add_argument("--max-seq-length",  type=int,   default=7680)
    ap.add_argument("--min-seq-length",  type=int,   default=0,
                    help="Skip examples with token count <= this value (use to train on the "
                         "'balance' dataset after a short-context run). Default 0 = no lower bound.")
    ap.add_argument("--warmstart-adapter", default=None,
                    help="Path to a completed LoRA adapter dir. Loads and continues training "
                         "that adapter instead of creating a new one. Use with --min-seq-length "
                         "to coalesce two runs into one adapter.")
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
        import traceback as _tb
        print(f"FastLanguageModel load failed — falling back to AutoModelForCausalLM", flush=True)
        _tb.print_exc()
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
    # "all-linear" triggers "No layers to finetune" in Unsloth 2026.6.x for NemotronH
    # because Unsloth's discovery logic can't enumerate the patched MoE/Mamba linears.
    # Explicit names bypass that check and let Unsloth apply its full memory opts
    # (gradient offloading, etc.) which are only active when _lora_via_unsloth=True.
    _LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]

    if args.warmstart_adapter:
        # Load a completed adapter and continue training it on the new data slice.
        # The Unsloth-patched model already has MoE experts accessible; PeftModel
        # simply re-attaches the existing LoRA A/B weights as trainable parameters.
        from peft import PeftModel
        print(f"Warmstart: loading adapter from {args.warmstart_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.warmstart_adapter, is_trainable=True)
        _lora_via_unsloth = _unsloth_loaded  # base is still Unsloth-patched
        print("Adapter loaded and set to trainable (warmstart mode)", flush=True)
    elif _unsloth_loaded:
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=_LORA_TARGETS,
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
            _lora_via_unsloth = True
            print("LoRA initialized via FastLanguageModel.get_peft_model", flush=True)
        except Exception as _e:
            print(f"FastLanguageModel.get_peft_model failed ({_e})", flush=True)
            print("Falling back to standard PEFT on Unsloth-patched model", flush=True)

    if not _lora_via_unsloth and not args.warmstart_adapter:
        from peft import LoraConfig, get_peft_model as _std_get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=_LORA_TARGETS,
            task_type="CAUSAL_LM",
        )
        model = _std_get_peft_model(model, lora_config)
        print("LoRA initialized via PEFT get_peft_model", flush=True)

    _patch_mamba_fastpath(model)

    # ── gradient checkpointing (NemotronH native path — always apply) ─────────────
    # NemotronH-native GC must run regardless of LoRA path.
    # Unsloth's use_gradient_checkpointing="unsloth" in get_peft_model calls the
    # standard gradient_checkpointing_enable(), which NemotronHForCausalLM blocks via
    # supports_gradient_checkpointing=False → silent no-op, no GC active.
    # The native _set_gradient_checkpointing() bypasses that guard and actually works.
    # Without this: activation memory 20–40 GB → step-0 OOM at seq_len ≥ 4096.
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

    # ── silence PretrainedConfig.use_return_dict FutureWarning ─────────────────
    # Newer Transformers turns use_return_dict into a property descriptor that fires
    # a FutureWarning on every access. The NemotronH forward() reads it on every call.
    # Shadow the class property with a plain instance attribute so the descriptor
    # is never invoked, and re-apply the warnings filter after Unsloth imports
    # (which can reset the filter list).
    _cfg = getattr(model, "config", None)
    if _cfg is not None and "use_return_dict" not in _cfg.__dict__:
        _cfg.__dict__["use_return_dict"] = getattr(_cfg, "return_dict", True)
    warnings.filterwarnings("ignore", category=FutureWarning, message=r".*use_return_dict.*")

    # ── override model.max_seq_length before Unsloth tokenizes ─────────────────
    # Unsloth injects a check into SFTTrainer.__init__ that reads getattr(model,
    # 'max_seq_length', None) and caps args.max_seq_length to that value.
    # For NemotronH, Unsloth's own patching sets model.max_seq_length = 2048
    # (a conservative hardcode — the actual model supports 262144 context).
    # Override the attribute directly before trainer creation.
    _unsloth_cap = getattr(model, "max_seq_length", None)
    print(f"  model.max_seq_length (Unsloth) = {_unsloth_cap}", flush=True)
    if _unsloth_cap is not None and _unsloth_cap < args.max_seq_length:
        model.max_seq_length = args.max_seq_length
        print(f"  → overrode model.max_seq_length to {args.max_seq_length}", flush=True)

    # ── load dataset ────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.train_file}", flush=True)
    records = []
    strat_labels = []
    with open(args.train_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({"messages": r["messages"]})
            # stratify by category (16 groups) rather than bucket (6 groups) so each
            # gradient-accumulation window sees all categories proportionally
            strat_labels.append(r.get("category") or r.get("bucket", "other"))

    # ── pre-filter: drop examples outside [min_seq_length+1, max_seq_length] ────
    # Upper bound: truncated examples lose </think>\n\boxed{} from labels — bad signal.
    # Lower bound (optional): skip examples already covered by a prior short-context run
    # so this run trains only the complementary "balance" slice of the dataset.
    _filter_desc = f"<= {args.min_seq_length} or > {args.max_seq_length}" if args.min_seq_length else f"> {args.max_seq_length}"
    print(f"Pre-filtering examples {_filter_desc} tokens...", flush=True)
    kept_records, kept_labels, n_short, n_long = [], [], 0, 0
    for r, label in zip(records, strat_labels):
        msgs = r["messages"]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=True
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        n_tok = len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])
        if n_tok <= args.min_seq_length:
            n_short += 1
        elif n_tok > args.max_seq_length:
            n_long += 1
        else:
            kept_records.append(r)
            kept_labels.append(label)
    if args.min_seq_length:
        print(f"  Kept {len(kept_records):,} / skipped-short {n_short:,} (<={args.min_seq_length} tok) / dropped-long {n_long:,} (>{args.max_seq_length} tok)", flush=True)
    else:
        print(f"  Kept {len(kept_records):,} / dropped {n_long:,} (>{args.max_seq_length} tok)", flush=True)
    records, strat_labels = kept_records, kept_labels

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

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            try:
                return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            except RuntimeError as _err:
                if "view and is being modified inplace" not in str(_err):
                    raise
                # UnslothFusedLossBackward returns a view tensor; transformers Trainer's
                # `loss *= num_processes` (a no-op scalar for single GPU) triggers the
                # autograd constraint. Fall back to base Trainer.compute_loss which
                # avoids the inplace multiply by going through the standard loss path.
                from transformers import Trainer as _BaseTrainer
                print("Warning: UnslothFusedLossBackward inplace error — falling back to "
                      "standard loss computation", flush=True)
                return _BaseTrainer.compute_loss(
                    self, model, inputs, return_outputs, num_items_in_batch
                )

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
