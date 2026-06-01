# Step 1: Data Preparation & Filtering
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
        f.write(json.dumps(nemo_token_entry) + "\n")

# Filter out length overflows exceeding 8192 tokens
with open(output_nemo_file, "r") as infile, open(cleaned_nemo_file, "w") as outfile:
    for line in infile:
        data = json.loads(line)
        if len(data.get("input_ids", [])) <= 8192:
            outfile.write(json.dumps(data) + "\n")
print("✅ Dataset prepared and cleaned.")
```