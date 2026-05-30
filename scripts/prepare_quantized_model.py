"""One-time prep: quantize the base model to 4-bit NF4 and save to disk.

Run once via run_prepare.sh. Subsequent training runs load from
QUANTIZED_CACHE_DIR (~15 GB, ~1 min) instead of re-quantizing the
full BF16 checkpoint (~60 GB, ~6 min) every time.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = os.environ.get("BASE_MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
OUTPUT_DIR = "/workspace/.cache/nemotron_4bit"


def main():
    if os.path.exists(os.path.join(OUTPUT_DIR, ".ready")):
        print(f"Quantized model already exists at {OUTPUT_DIR} — nothing to do.")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {MODEL_ID} in 4-bit NF4 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Saving quantized model to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    # Sentinel written last — run_train.sh checks for this to confirm a complete save.
    open(os.path.join(OUTPUT_DIR, ".ready"), "w").close()
    print("Done. Future training runs will load from cache.")


if __name__ == "__main__":
    main()
