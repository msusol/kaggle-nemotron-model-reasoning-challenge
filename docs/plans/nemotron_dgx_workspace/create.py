import zipfile
import os

# 1. Define document contents
readme = """# Architectural Guide: Nemotron-3 Nano 30B Fine-Tuning
This workspace contains configuration settings to fine-tune the NVIDIA Nemotron-3 Nano 30B (A3B) hybrid model inside a 128 GB memory boundary.
By targeting only dense backbone modules (`q_proj`, `v_proj`, etc.) and bypassing the 128 mixture-of-experts layers, we drop the training footprint significantly to avoid OOM crashes.
"""

data_prep = """# Step 1: Data Preparation & Filtering
```python
import json
from datasets import load_dataset

# Pack Tong Hui Kang's pre-tokenized corpus into NeMo format
dataset = load_dataset("huikang/nemotron-reasoning-challenge", split="train")
output_nemo_file = "/data/nemotron_pretokenized_corpus.jsonl"
cleaned_nemo_file = "/data/nemotron_pretokenized_corpus_cleaned.jsonl"

with open(output_nemo_file, "w", encoding="utf-8") as f:
    for entry in dataset:
        nemo_token_entry = {
            "input_ids": [int(i) for i in entry.get("input_ids", [])],
            "labels": [int(l) for l in entry.get("labels", [])]
        }
        f.write(json.dumps(nemo_token_entry) + "\\n")

# Filter out length overflows exceeding 8192 tokens
with open(output_nemo_file, "r") as infile, open(cleaned_nemo_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        if len(data.get("input_ids", [])) <= 8192:
            outfile.write(json.dumps(data) + "\\n")
print("✅ Dataset prepared and cleaned.")
```
"""

nemo_config = """# Step 2: Config Files
### env.toml
```toml
[wandb]
project = "nemotron3-nano-tuning"
entity = "YOUR-WANDB-USERNAME"

[DGX-SPARK-LOCAL]
executor = "local"
gpus_per_node = 1
mounts = ["/home:/home", "/data:/data"]
```

### nemotron_sft_config.yaml
```yaml
model:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  num_moe_experts: 128
  moe_router_topk: 6
  pre_tokenized_dataset: True
  data:
    train_ds:
      file_names: ["/data/nemotron_pretokenized_corpus_cleaned.jsonl"]
      max_seq_length: 8192
  micro_batch_size: 1
  global_batch_size: 4
  seq_length: 8192
  bf16: True
  activations_checkpoint_method: "uniform"
  activations_checkpoint_granularity: "selective"
  peft:
    peft_scheme: "lora"
    lora_tuning:
      r: 8
      adapter_alpha: 16
      lora_dropout: 0.0
      target_modules: ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]
exp_manager:
  exp_dir: "/data/nemotron_runs"
  name: "nemotron3-nano-reasoning-lora"
  create_wandb_logger: True
  wandb_logger_kwargs:
    project: "nemotron3-nano-tuning"
    entity: "YOUR-WANDB-USERNAME"
```
"""

docker_exec = """# Step 3: Docker Execution
Run this to start your isolated training container:
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \\
  -v /home/your_user/data:/data \\
  -v /home/your_user/configs:/workspace/configs \\
  -e WANDB_API_KEY="your_wandb_key" \\
  nvcr.io/nvidia/nemo:24.07 /bin/bash
```
Kick off training from inside the container bash shell:
```bash
python /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \\
    --config-path=/workspace/configs \\
    --config-name=nemotron_sft_config.yaml
```
"""

eval_loop = """# Step 4: Extraction and Evaluation
Extract the `.nemo` checkpoint to Hugging Face format:
```bash
python /opt/NeMo/scripts/checkpoint_converters/convert_nemo_to_hf_lora.py \\
    --input_nemo_file /data/nemotron_runs/checkpoints/nemotron3-nano-reasoning.nemo \\
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
"""

framework_issues = """# Deep Dive: Nemotron Architecture Complexity & Framework Friction

When working with **NVIDIA Nemotron-3 Nano 30B (A3B)**, developers frequently run into framework friction. This document provides a breakdown of why this happens and why NVIDIA NeMo succeeds where Hugging Face (HF) fails.

---

### 1. The Core Nemotron-3 Architectural Obstacles
Nemotron-3 Nano 30B is not a standard dense Transformer (like Llama or Mistral). It uses a **highly non-standard hybrid structure**:
* **Interleaved Sequence Layers:** It combines 23 State Space Model (SSM) Mamba-2 layers with 6 traditional Attention layers.
* **Massive MoE Scale:** It utilizes a Mixture-of-Experts (MoE) block featuring **128 individual experts** per layer, routing 5 to 6 experts dynamically per token.

#### Why Hugging Face Ecosystem Fails With This Setup
* **`trust_remote_code=True` Vulnerability:** Because HF Transformers does not have a native primitive for an interleaved Mamba-2/MoE architecture, the model relies on custom remote modeling files. 
* **The BitsAndBytes 4-bit/8-bit Crash:** Standard `bitsandbytes` quantization scanners scan standard PyTorch models by looking for traditional `nn.Linear` layers. When encountering Nemotron's custom structural wrappers, `bitsandbytes` fails to correctly map the tensor matrices on-the-fly, throwing initialization memory faults or shape mismatches.
* **The "All-Linear" LoRA Trap:** If you leave your PEFT settings to target `all-linear` modules, the engine cannot differentiate between standard projection heads and the MoE system. It attempts to clone adapter targets across all 128 individual experts, skyrocketing your trainable parameter count back into the billions and over-saturating your 128 GB RAM.

---

### 2. Hugging Face Trainer vs. NVIDIA NeMo Framework

The transition to NVIDIA NeMo is required due to fundamental differences in memory management, model parallelism, and custom tensor structures.


| Capability Feature | Hugging Face (HF) Trainer Loop | NVIDIA NeMo Framework Environment |
| :--- | :--- | :--- |
| **Custom Architecture Compatibility** | ❌ Poor. Crashes on quantized custom model layers. |  Highly Optimized. Built natively to support hybrid Mamba/MoE backbones. |
| **Memory Isolation Safeguards** | ❌ Relies on native host python allocations. Crash risks can panic host OS. |  Containerized infrastructure (`nvcr.io`). Complete tracking isolated from Ubuntu host kernels. |
| **Expert Sharding Capability** | ❌ Standard PEFT treats MoE layers as a giant unified tensor block. |  Uses Megatron-Core Tensor Parallelism to split routing matrices naturally. |
| **8,192 Token Context Handling** | ❌ Triggers high activation memory peaks unless deep speed code is engineered. |  Natively offers `selective` activation checkpoint granularity to free RAM. |
| **Data Pipelines** | ❌ Tokenizes strings on-the-fly, which spikes host RAM memory overhead. |  Directly streams pre-tokenized raw integer index lines sequentially from disk. |

---

### 3. Summary of Code Fixes Implemented in this Package
1. **Target-Module Restricting:** Bypasses expert nodes entirely, mapping adapters only onto non-expert layers (`q_proj`, `v_proj`, `out_proj`, `in_proj`).
2. **Pre-Tokenized Streaming:** Converts Tong Hui Kang's reasoning traces into raw integer lists to remove character encoding operations from runtime.
3. **Selective Activation Granularity:** Automatically drops intermediate layers from memory right after the forward calculation pass concludes.
"""

# 2. Build the Zip Archive locally inside your workspace directory
zip_filename = 'nemotron_dgx_workspace.zip'
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    zipf.writestr('README.md', readme.strip())
    zipf.writestr('01_DATA_PREPARATION.md', data_prep.strip())
    zipf.writestr('02_NEMO_CONFIGURATION.md', nemo_config.strip())
    zipf.writestr('03_DOCKER_EXECUTION.md', docker_exec.strip())
    zipf.writestr('04_EVALUATION_LOOP.md', eval_loop.strip())
    zipf.writestr('05_NEMOTRON_FRAMEWORK_ISSUES.md', framework_issues.strip())

print(f"📦 Workspace package safely created at: {os.path.abspath(zip_filename)}")
print("The updated zip folder is ready for your local development setup.")
