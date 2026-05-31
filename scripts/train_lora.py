import argparse
import math
import os
import sys
import warnings

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = os.environ.get("BASE_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")


def format_example(example, tokenizer):
    system = example.get(
        "system",
        "You are a careful reasoning model. Solve the problem step by step and end with Final answer: \\boxed{...}.",
    )
    user = example["prompt"]
    answer = example["response"]
    # Corpus extraction strips the <think> prefix (it was in the masked/no-loss region).
    # Re-add it so the assistant turn matches Nemotron's pre-training format.
    if not answer.startswith("<think>"):
        answer = "<think>\n" + answer

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        text = f"system: {system}\nuser: {user}\nassistant: {answer}"
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--valid-file", default=None)
    ap.add_argument("--output-dir", default="/workspace/output/adapter")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--num-epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--early-stopping-patience", type=int, default=0,
                    help="EarlyStoppingCallback patience (epochs). 0 = disabled.")
    ap.add_argument("--use-4bit", action="store_true")
    args = ap.parse_args()

    if args.lora_r > 32:
        raise ValueError("Competition constraint violated: --lora-r must be <= 32")

    if args.use_4bit:
        raise RuntimeError(
            "--use-4bit is not supported for Nemotron-H: quantized layers cause "
            "incompatible tensor shapes with both the fast and slow Mamba paths "
            "(fast path → shape error; slow path → Half/Float dtype error). "
            "Run without --use-4bit; the GB10 has sufficient memory for BF16."
        )

    # Suppress FutureWarning from transformers NemotronH causal-mask internals
    # (input_embeds → inputs_embeds rename; fires every forward pass, not our code).
    warnings.filterwarnings(
        "ignore",
        message=r".*input_embeds.*deprecated.*",
        category=FutureWarning,
        module=r"transformers\.models\.nemotron_h\..*",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = {"": 0}  # force all layers onto GPU 0 — no CPU offload on GB10
    free, total = torch.cuda.mem_get_info()
    print(f"GPU before load: free={free/1e9:.1f}GB total={total/1e9:.1f}GB used={(total-free)/1e9:.1f}GB", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=device,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Sync model config token IDs from tokenizer to avoid SFTTrainer alignment warning.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.use_cache = False

    # Force Mamba fast-path CUDA kernels (mamba_ssm / causal_conv1d installed in container).
    # Matches what the 0.85 reference notebook does explicitly. [cite:148]
    for _name, _mod in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_mod, "is_fast_path_available"):
            _mod.is_fast_path_available = True

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|up_proj|down_proj|lm_head)$",
    )
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    data_files = {"train": args.train_file}
    if args.valid_file:
        data_files["validation"] = args.valid_file

    ds = load_dataset("json", data_files=data_files)
    ds["train"] = ds["train"].map(lambda x: format_example(x, tokenizer), remove_columns=ds["train"].column_names)
    if "validation" in ds:
        ds["validation"] = ds["validation"].map(lambda x: format_example(x, tokenizer), remove_columns=ds["validation"].column_names)

    use_early_stopping = args.early_stopping_patience > 0 and args.valid_file

    steps_per_epoch = math.ceil(len(ds["train"]) / (args.batch_size * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.num_epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    train_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if args.valid_file else "no",
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        bf16=True,
        report_to="none",
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_seq_length=args.max_seq_length,
        # use_reentrant=False uses PyTorch generic checkpointing — does not call
        # gradient_checkpointing_enable() and works without native NemotronH support.
        # Required at seq_len=8192: without this, all layer activations are held in
        # memory simultaneously, exceeding the 121 GB unified pool at step 0.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    callbacks = []
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience
        ))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=train_args,
        callbacks=callbacks if callbacks else None,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
