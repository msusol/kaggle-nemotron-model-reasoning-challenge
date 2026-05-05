import glob
import os
import threading
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

warnings.filterwarnings("ignore", category=UserWarning, module="torch.library")


def _triton_watcher(stop: threading.Event) -> None:
    """Poll the Triton kernel cache and print a line each time new kernels finish compiling."""
    cache = os.environ.get("TRITON_CACHE_DIR", os.path.expanduser("~/.triton/cache"))
    prev = 0
    while not stop.is_set():
        n = len(glob.glob(os.path.join(cache, "**", "*.cubin"), recursive=True))
        if n != prev:
            print(f"[Triton JIT] {n} kernel(s) compiled", flush=True)
            prev = n
        stop.wait(3)

MODEL_ID = os.environ.get("BASE_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")

# GB10 has 121.7 GiB unified GPU memory (Grace-Blackwell). Keep everything on GPU.
# Leave ~7 GiB for activations and KV cache; never touch CPU or disk.
MAX_MEMORY = {0: "115GiB"}


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}  |  devices: {torch.cuda.device_count()}")
    if cuda_ok:
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")

    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=MAX_MEMORY,
    )

    messages = [
        {"role": "system", "content": "You are a careful math solver. Show brief reasoning and end with Final answer: \\boxed{...}."},
        {"role": "user", "content": "What is 17 + 25?"},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Generating (first run compiles Triton kernels — this may take 5–15 min) …", flush=True)
    stop_watcher = threading.Event()
    watcher = threading.Thread(target=_triton_watcher, args=(stop_watcher,), daemon=True)
    watcher.start()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    stop_watcher.set()
    watcher.join(timeout=5)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== SAMPLE GENERATION ===")
    print(text)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    peft_model = get_peft_model(model, config)
    save_dir = "/workspace/output/smoke_adapter"
    os.makedirs(save_dir, exist_ok=True)
    peft_model.save_pretrained(save_dir)
    print(f"Saved PEFT smoke adapter to: {save_dir}")


if __name__ == "__main__":
    main()
