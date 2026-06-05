#!/usr/bin/env python3
"""Modal.com training — kuangyicheng 0.87/0.88 SFT approach on RTX Pro 6000 (96GB).

Replicates the short-response SFT that produces 0.87/0.88:
  1. Unsloth FastLanguageModel patches Nemotron-H MoE experts as individual nn.Linear
  2. v27 adapter loaded as warmstart (11,962 keys in backbone.layers format)
  3. 240-step SFT on competition data with short responses (~100 tokens)

GPU: RTX Pro 6000 Blackwell (96GB, sm_120, $3.03/hr). ~2h run ≈ $6.
CUDA 12.8 image required for sm_120 (Blackwell) kernel compilation.

Prerequisites (local machine):
  pip install modal
  modal token new
  modal secret create huggingface-token HF_TOKEN=hf_xxx

Run:
  modal run scripts/train_modal.py

Download result:
  mkdir -p output/adapter_v5_modal
  modal volume get nemotron-output /adapter_v5_modal output/adapter_v5_modal

Then package and submit:
  RUN_NAME=v5_modal bash scripts/package_submission.sh output/adapter_v5_modal
"""

import modal
from pathlib import Path

# ── Image ──────────────────────────────────────────────────────────────────────
# RTX Pro 6000 Blackwell is sm_120; CUDA 12.8+ required to compile mamba-ssm for it.
# PyTorch 2.6 has cu128 wheels. mamba-ssm will build from source for sm_120.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "ninja-build")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==5.5.3",
        "datasets==3.2.0",
        "accelerate==1.3.0",
        "peft==0.14.0",
        "trl==0.15.2",
        "huggingface_hub>=1.5.0,<2.0",
        "sentencepiece",
        "safetensors",
        "scipy",
        "pandas",
        "tqdm",
        "causal-conv1d",
        "mamba-ssm",
        "bitsandbytes",
    )
    .pip_install("unsloth", "unsloth_zoo", extra_options="--no-deps")
    # Re-pin after unsloth potentially drifted versions
    .pip_install(
        "transformers==5.5.3",
        "peft==0.14.0",
        "trl==0.15.2",
        "accelerate==1.3.0",
        extra_options="--force-reinstall --no-deps",
    )
    # Disable vllm in TRL (pydantic conflict in this image)
    .run_commands(
        "python3 -c \""
        "import pathlib; "
        "p = pathlib.Path('/usr/local/lib/python3.11/dist-packages/trl/import_utils.py'); "
        "c = p.read_text(); "
        "old = '_vllm_available = _is_package_available(\\\"vllm\\\")'; "
        "new = '_vllm_available = False  # disabled'; "
        "p.write_text(c.replace(old, new, 1)) if old in c else print('already patched')\""
    )
)

# ── App + volumes ──────────────────────────────────────────────────────────────
app = modal.App("nemotron-sft", image=image)

# HF model cache — persists across runs; first run downloads ~60 GB base model
hf_cache_vol = modal.Volume.from_name("nemotron-hf-cache", create_if_missing=True)
# Output adapter
output_vol = modal.Volume.from_name("nemotron-output", create_if_missing=True)

# Local files to upload at run time
_PROJECT = Path(__file__).parent.parent
_TRAIN_DATA = _PROJECT / "data" / "v0.5_train.jsonl"         # 13 MB, 21500 examples
_WARMSTART  = _PROJECT / "output" / "adapter_huikang_v27_unsloth"  # 1.7 GB, 11962 keys


@app.function(
    gpu="RTX_PRO_6000",
    timeout=4 * 3600,   # 4-hour cap; 240-step run takes ~2 h on RTX Pro 6000
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    mounts=[
        modal.Mount.from_local_file(
            local_path=str(_TRAIN_DATA),
            remote_path="/data/v0.5_train.jsonl",
        ),
        modal.Mount.from_local_dir(
            local_path=str(_WARMSTART),
            remote_path="/warmstart",
        ),
    ],
)
def train_sft(
    num_steps: int = 240,
    output_name: str = "adapter_v5_modal",
    model_id: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    learning_rate: float = 2e-4,
    max_seq_length: int = 6144,
    seed: int = 3407,
):
    import functools
    import json
    import math
    import random
    import sys
    import threading
    import warnings
    from pathlib import Path as _Path

    import torch
    from datasets import Dataset
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from torch.utils.data import DataLoader, Sampler
    from trl import SFTConfig, SFTTrainer

    warnings.filterwarnings("ignore", message=r".*save_embedding_layers.*")
    warnings.filterwarnings("ignore", message=r".*Could not find a config file.*")
    warnings.filterwarnings("ignore", message=r".*use_return_dict.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*torchvision.*")
    warnings.filterwarnings("ignore", message=r".*unsloth_force_compile.*")

    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = _Path("/output") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}", flush=True)

    # ── cache dropper ────────────────────────────────────────────────────────
    def _make_cache_dropper(interval=20.0):
        stop = threading.Event()
        def _loop():
            while not stop.wait(interval):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        threading.Thread(target=_loop, daemon=True, name="cache-dropper").start()
        return stop

    # ── load base model ──────────────────────────────────────────────────────
    _stop = _make_cache_dropper()
    print(f"Loading base model: {model_id}", flush=True)
    from unsloth import FastLanguageModel

    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
        trust_remote_code=False,   # MUST be False — hub cache has buggy prepare_inputs_for_generation
        unsloth_force_compile=False,
        attn_implementation="eager",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _stop.set()
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"Base model loaded. GPU free={free_gb:.1f}GB", flush=True)

    # Mamba fast path
    for name, mod in sys.modules.items():
        if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = True
            print("Mamba fast path enabled", flush=True)
            break

    # ── load v27 warmstart via get_peft_model + set_peft_model_state_dict ────
    # v27 adapter is in Unsloth backbone.layers format (11,962 keys).
    # Must use get_peft_model to create the 12,008-module structure first,
    # then load saved weights; PEFT auto-handles lora_A.weight → lora_A.default.weight.
    adapter_cfg = json.loads((_Path("/warmstart") / "adapter_config.json").read_text())
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=adapter_cfg.get("r", 32),
        lora_alpha=adapter_cfg.get("lora_alpha", 32),
        target_modules=adapter_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=adapter_cfg.get("lora_dropout", 0),
        bias=adapter_cfg.get("bias", "none"),
        use_gradient_checkpointing=False,
        random_state=seed,
    )
    n_modules = sum(1 for n, _ in model.named_parameters() if ".lora_" in n)
    print(f"LoRA structure: {n_modules} params", flush=True)

    saved_state = load_file("/warmstart/adapter_model.safetensors")
    load_result = set_peft_model_state_dict(model, saved_state)
    n_missing = len(load_result.missing_keys) if load_result.missing_keys else 0
    print(f"v27 loaded: {len(saved_state)} keys, missing={n_missing}", flush=True)

    # Cast LoRA to fp32 for stable training
    for _, param in model.named_parameters():
        if ".lora_" in _:
            param.requires_grad_(True)
            param.data = param.data.to(torch.float32)

    # Gradient checkpointing (NemotronH native path)
    _gc_func = functools.partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    try:
        _gc_target = getattr(model, "base_model", model)
        _gc_target = getattr(_gc_target, "model", _gc_target)
        _gc_target._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=_gc_func)
        model.enable_input_require_grads()
        print("Gradient checkpointing enabled", flush=True)
    except Exception as _e:
        print(f"Warning: gradient checkpointing unavailable: {_e}", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,}", flush=True)

    # ── dataset ──────────────────────────────────────────────────────────────
    print("Loading dataset: /data/v0.5_train.jsonl", flush=True)
    records, buckets = [], []
    with open("/data/v0.5_train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append({"messages": r["messages"]})
            buckets.append(r.get("bucket", "other"))
    dataset = Dataset.from_list(records)
    print(f"Dataset: {len(dataset)} examples across {len(set(buckets))} buckets", flush=True)

    def formatting_func(example):
        msgs = example["messages"]
        conv = [msgs] if msgs and isinstance(msgs[0], dict) else msgs
        texts = []
        for c in conv:
            try:
                t = tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False, enable_thinking=True)
            except TypeError:
                t = tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            texts.append(t)
        return texts

    # Stratified batching — equal category mix per batch
    def build_stratified_order(labels, batch_size, rng_seed):
        from collections import defaultdict
        by_label = defaultdict(list)
        for i, lab in enumerate(labels):
            by_label[lab].append(i)
        rng = random.Random(rng_seed)
        for v in by_label.values():
            rng.shuffle(v)
        n_batches = max(1, math.ceil(len(labels) / batch_size))
        batches = [[] for _ in range(n_batches)]
        order = list(range(n_batches))
        rng.shuffle(order)
        pos = 0
        for lab in sorted(by_label.keys()):
            for idx in by_label[lab]:
                batches[order[pos % n_batches]].append(idx)
                pos += 1
        return [i for b in batches for i in b]

    class OrderedSampler(Sampler):
        def __init__(self, order): self.order = list(order)
        def __iter__(self): return iter(self.order)
        def __len__(self): return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *a, stratified_order=None, **kw):
            super().__init__(*a, **kw)
            self._stratified_order = stratified_order

        def get_train_dataloader(self):
            if self._stratified_order is None:
                return super().get_train_dataloader()
            keep = [c for c in self.train_dataset.column_names if c in ("input_ids", "attention_mask", "labels")]
            return DataLoader(
                self.train_dataset.select_columns(keep),
                batch_size=self.args.per_device_train_batch_size,
                sampler=OrderedSampler(self._stratified_order),
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    strat_order = build_stratified_order(buckets, batch_size=16, rng_seed=seed)

    # ── training config ───────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_steps=num_steps,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_seq_length=max_seq_length,
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
        seed=seed,
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

    print("Starting training...", flush=True)
    trainer.train()
    print("Training complete", flush=True)

    # ── save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    output_vol.commit()  # flush writes to Modal volume

    st_path = output_dir / "adapter_model.safetensors"
    if st_path.exists():
        saved = load_file(str(st_path))
        coverage = "full" if len(saved) >= 11000 else "PARTIAL — MoE keys missing"
        print(f"Saved {len(saved)} keys ({coverage}) → {output_dir}", flush=True)
    else:
        print(f"WARNING: adapter_model.safetensors not found in {output_dir}", flush=True)


@app.local_entrypoint()
def main(
    num_steps: int = 240,
    output_name: str = "adapter_v5_modal",
):
    print(f"Launching Modal SFT: {num_steps} steps → /output/{output_name}")
    print(f"  Warmstart:  output/adapter_huikang_v27_unsloth  ({_WARMSTART})")
    print(f"  Train data: data/v0.5_train.jsonl  ({_TRAIN_DATA})")
    print(f"  GPU:        RTX Pro 6000 Blackwell 96GB  (~$3.03/hr, ~2h = ~$6)")
    train_sft.remote(num_steps=num_steps, output_name=output_name)
    print(f"\nDone. Download the adapter:")
    print(f"  mkdir -p output/{output_name}")
    print(f"  modal volume get nemotron-output /{output_name} output/{output_name}")
