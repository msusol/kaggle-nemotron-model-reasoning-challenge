# Step 4: Extraction and Evaluation
Extract the `.nemo` checkpoint to Hugging Face format:
```bash
python /opt/NeMo/scripts/checkpoint_converters/convert_nemo_to_hf_lora.py \
    --input_nemo_file /data/nemotron_runs/checkpoints/nemotron3-nano-reasoning.nemo \
    --output_hf_directory /data/nemotron_hf_extracted_lora/
```

Run evaluation script to verify your 12 tasks and guard against catastrophic forgetting:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "/data/nemotron_hf_extracted_lora/")

# Test prompt assertion block
inputs = tokenizer("Solve the cryptarithm: SEND + MORE = MONEY.", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```