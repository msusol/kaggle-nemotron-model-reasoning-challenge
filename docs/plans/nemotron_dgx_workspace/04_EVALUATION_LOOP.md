# Step 4: Checkpoint Conversion and Evaluation

## Convert NeMo checkpoint to HuggingFace PEFT format

After training completes, convert the `.nemo` checkpoint to a HuggingFace-compatible
LoRA adapter so it can be packaged and submitted to Kaggle.

```bash
python /opt/NeMo/scripts/checkpoint_converters/convert_nemo_to_hf_lora.py \
    --input_nemo_file /data/nemotron_runs/nemotron3-nano-reasoning-lora-v0.4b/checkpoints/last.nemo \
    --output_hf_directory /data/nemotron_hf_lora/
```

The output directory should contain `adapter_config.json` and `adapter_model.safetensors`.

## Load and evaluate (BF16, no quantization)

> **Important**: Do NOT use 4-bit or 8-bit quantization — BitsAndBytes quantization is
> broken for NemotronH (MoE expert tensors cause shape mismatches). Load in BF16.
> Do NOT use `trust_remote_code=True` — `transformers>=5.5.3` includes native NemotronH
> support and the remote code overrides the fixed KV-cache implementation.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
adapter_dir = "/data/nemotron_hf_lora/"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

# Use the same system prompt as training
SYSTEM_PROMPT = "Solve the problem step by step. Put your final answer in \\boxed{}."
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": "Solve the cryptarithm: SEND + MORE = MONEY."},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Score against the validation set

```bash
# Run inference on validation set
python scripts/infer_lora.py \
    --adapter-dir /data/nemotron_hf_lora/ \
    --data-file   data/v0.4_valid.jsonl \
    --output-file output/predictions_nemo_v0.4b.jsonl

# Score with validate_metric.py (uses last \boxed{} — correct for CoT responses)
python scripts/validate_metric.py \
    --predictions output/predictions_nemo_v0.4b.jsonl \
    --labels      data/v0.4_valid_labels.jsonl
```

## Package for Kaggle submission

```bash
bash scripts/package_submission.sh /data/nemotron_hf_lora/ output/submission/
# → output/submission/submission.zip
kaggle competitions submit \
    -c nvidia-nemotron-model-reasoning-challenge \
    -f output/submission/submission.zip \
    -m "v0.4b-nemo: NeMo SFT on huikang corpus, seq=8192, r=32"
```
