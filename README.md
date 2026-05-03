# Nemotron LoRA Training on GB10 for the Kaggle Reasoning Challenge

https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

This README packages a working setup for training a Nemotron LoRA adapter on a GB10-style NVIDIA system using Hugging Face, PEFT, TRL, and DSPy, then exporting a Kaggle-compatible `submission.zip`.[cite:74][cite:1] The Kaggle competition requires a Nemotron LoRA adapter with `adapter_config.json`, and Kaggle examples commonly package LoRA weights plus that config into the final archive.[cite:1][cite:119][cite:121]

## Goal

The target competition is the **NVIDIA Nemotron Model Reasoning Challenge** on Kaggle.[cite:1] The required submission is a LoRA adapter for **Nemotron-3-Nano-30B** with rank at most 32, evaluated under vLLM with deterministic generation settings and a metric that prefers answers inside `\\boxed{}`.[cite:1]

## Repository layout

```text
.
├── README.md
├── Dockerfile.gb10
├── data/
│   ├── train.jsonl
│   └── valid.jsonl
├── scripts/
│   ├── smoke_test_nemotron.py
│   ├── train_lora.py
│   ├── validate_metric.py
│   └── package_submission.sh
└── output/
    └── submission.zip
```

## Commands

### 1. Build the image

```bash
docker build -f Dockerfile.gb10 -t nemotron-gb10:latest .
```

### 2. Run the container

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace \
  -e HF_TOKEN="$HF_TOKEN" \
  nemotron-gb10:latest
```

### 3. Smoke-test Nemotron + PEFT

```bash
python /workspace/scripts/smoke_test_nemotron.py
```

### 4. First LoRA training run

```bash
python /workspace/scripts/train_lora.py \
  --train-file /workspace/data/train.jsonl \
  --valid-file /workspace/data/valid.jsonl \
  --output-dir /workspace/output/adapter \
  --max-seq-length 2048 \
  --batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-4 \
  --num-epochs 1 \
  --lora-r 32 \
  --lora-alpha 64
```

### 5. Local metric sanity check

```bash
cat > /workspace/output/preds.jsonl <<'JSONL'
{"id": 1, "output": "After solving, Final answer: \\boxed{56}"}
{"id": 2, "output": "The area is 30. Final answer: \\boxed{30}"}
JSONL

cat > /workspace/output/labels.jsonl <<'JSONL'
{"id": 1, "answer": "56"}
{"id": 2, "answer": "30"}
JSONL

python /workspace/scripts/validate_metric.py \
  --predictions /workspace/output/preds.jsonl \
  --labels /workspace/output/labels.jsonl
```

### 6. Package adapter into `submission.zip`

```bash
bash /workspace/scripts/package_submission.sh \
  /workspace/output/adapter \
  /workspace/output/submission
```