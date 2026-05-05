import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
    ap.add_argument("--use-4bit", action="store_true")
    args = ap.parse_args()

    if args.lora_r > 32:
        raise ValueError("Competition constraint violated: --lora-r must be <= 32")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = {"": 0}  # force all layers onto GPU 0 — no CPU offload on GB10
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map=device,
            dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)

    data_files = {"train": args.train_file}
    if args.valid_file:
        data_files["validation"] = args.valid_file

    ds = load_dataset("json", data_files=data_files)
    ds["train"] = ds["train"].map(lambda x: format_example(x, tokenizer), remove_columns=ds["train"].column_names)
    if "validation" in ds:
        ds["validation"] = ds["validation"].map(lambda x: format_example(x, tokenizer), remove_columns=ds["validation"].column_names)

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
        bf16=True,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        args=train_args,
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
