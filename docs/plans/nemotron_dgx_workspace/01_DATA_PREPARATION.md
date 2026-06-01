# Step 1: Data Preparation

The pre-tokenized NeMo SFT dataset is published on Kaggle and ready to use directly —
no conversion needed unless you want to regenerate it locally.

## Option A — Download from Kaggle (recommended)

```bash
kaggle datasets download gdataranger/huikang-nemotron-nemo-sft-r32 -p /data --unzip
# Produces:
#   /data/nemo_train.jsonl   15,159 examples
#   /data/nemo_valid.jsonl      820 examples
#   /data/valid_labels.jsonl    820 ground-truth answers for scoring
```

Dataset card: https://www.kaggle.com/datasets/gdataranger/huikang-nemotron-nemo-sft-r32

## Option B — Regenerate locally from v0.4 text JSONL

Use this if you want to regenerate with different settings (e.g. different system prompt
or sequence length cutoff). Requires the Nemotron tokenizer and the v0.4 text JSONL files.

```bash
# First generate the text JSONL from the huikang corpus zip:
bash scripts/run_extract_huikang_corpus.sh   # → data/v0.4_train.jsonl + data/v0.4_valid.jsonl

# Then tokenize to NeMo format (runs inside nemotron-gb10 container):
bash scripts/run_convert_jsonl_to_nemo.sh
# → data/nemo_dataset/nemo_train.jsonl
# → data/nemo_dataset/nemo_valid.jsonl
```

See `scripts/convert_jsonl_to_nemo.py` for implementation details. The script uses the
same `SYSTEM_PROMPT` and `format_example` logic as `train_lora.py` so NeMo and PEFT
training are consistent.

## Dataset format

Each line is a JSON object:

```json
{
  "input_ids": [1, 2, 3, ...],
  "labels":    [-100, -100, ..., 42, 99, ...]
}
```

- `input_ids`: full token sequence (system + user + assistant CoT + final answer)
- `labels`: `-100` for the masked prompt region (no loss), real token IDs for the
  assistant response region (loss computed here)

System prompt baked into the masked tokens:
`"Solve the problem step by step. Put your final answer in \boxed{}."`

All sequences fit within 8,192 tokens (longest: 7,999 tokens).
