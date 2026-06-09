# 06 — Checkpoint export

## Overview

NeMo RL with the Megatron backend saves checkpoints in Megatron format
(distributed tensor shards, `iter_XXXXXXX/` directories). To use the trained
adapter with standard HuggingFace tooling or submit to Kaggle, you need to:

1. Merge the LoRA adapter weights into the base model
2. Export as a standard HuggingFace checkpoint

---

## Step 1 — Identify the checkpoint to export

```bash
# List available checkpoints
ls -lt /workspace/results/$EXP_NAME/policy/weights/
# Example output:
#   iter_0000200/   ← step 200
#   iter_0000150/
#   iter_0000100/

# Pick the step you want to export
CKPT_STEP=200
CKPT_DIR=/workspace/results/$EXP_NAME/policy/weights/iter_$(printf '%07d' $CKPT_STEP)
echo "Exporting: $CKPT_DIR"
ls $CKPT_DIR
```

---

## Step 2 — Convert base model to Megatron format (if not already done)

The LoRA merger needs both the base model checkpoint (Megatron format) and the
LoRA adapter checkpoint. The base Megatron checkpoint is created during training
initialisation and stored alongside the adapter:

```bash
# Check if base Megatron checkpoint exists
ls /workspace/results/$EXP_NAME/policy/weights/iter_0000000/
# If iter_0000000 exists, it's the base checkpoint — proceed to Step 3
```

If `iter_0000000` is missing, convert the HF model to Megatron format first:

```bash
cd /opt/nemo-rl
uv run --extra mcore python examples/converters/convert_hf_to_megatron.py \
  --hf-model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --hf-ckpt-path /workspace/model \
  --megatron-ckpt-path /workspace/results/base_megatron/iter_0000000 \
  --config /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml
```

---

## Step 3 — Merge LoRA adapter into base model and export to HF

```bash
cd /opt/nemo-rl

BASE_CKPT=/workspace/results/$EXP_NAME/policy/weights/iter_0000000
ADAPTER_CKPT=$CKPT_DIR
HF_OUT=/workspace/results/lora-merged-hf-step${CKPT_STEP}

mkdir -p $HF_OUT

uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
  --base-ckpt    $BASE_CKPT \
  --adapter-ckpt $ADAPTER_CKPT \
  --hf-model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --hf-ckpt-path $HF_OUT

echo "Export complete: $HF_OUT"
ls -lh $HF_OUT
```

---

## Step 4 — Verify the exported model

```bash
python3 - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/workspace/results/lora-merged-hf-step${CKPT_STEP}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,     # critical — avoids double-allocation OOM
)

# Quick inference test
prompt = "What is 15 × 17?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
```

---

## Step 5 — (Optional) Re-enable thinking mode

If you want the exported model to generate reasoning traces:

```bash
# Re-enable thinking in the exported tokenizer config
TCFG=$HF_OUT/tokenizer_config.json
sed -i 's/enable_thinking=false/enable_thinking=true/g' $TCFG
echo "Thinking mode re-enabled"
```

---

## Kaggle submission

The merged HF checkpoint at `$HF_OUT` can be:

1. **Pushed to HuggingFace Hub** for Kaggle to pull:
   ```bash
   huggingface-cli upload \
     <your-hf-username>/nemotron-nano-30b-grpo-kaggle \
     $HF_OUT
   ```

2. **Compressed and uploaded** as a Kaggle dataset:
   ```bash
   tar -czf nemotron-nano-grpo-step${CKPT_STEP}.tar.gz -C $HF_OUT .
   kaggle datasets create --path nemotron-nano-grpo-step${CKPT_STEP}.tar.gz
   ```

3. **Used directly** from the `/workspace` volume if Kaggle inference runs on
   the same Spark.
