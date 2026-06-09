# 03 — Data preparation

## Overview

NeMo RL GRPO expects data in JSONL format with prompt/answer pairs.
NeMo Gym provides pre-built environments (math, code, tool-use, structured output)
that wrap these datasets and compute verifiable rewards during rollout.

For the Kaggle Nemotron Model Reasoning Challenge the primary domain is **math
reasoning**, which maps directly to the DAPO/SkyWorks math datasets used in
NVIDIA's own Nano v3 training.

---

## Option A — NeMo Gym managed datasets (recommended)

NeMo Gym can download and prepare the same datasets used in Nemotron 3 Nano's
RLVR training:

```bash
cd /opt/nemo-rl

# Download-only pass — prepares data without starting training
HF_HOME=/workspace/.cache/huggingface \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml \
  data.download_only=true

# Verify
ls -lh /workspace/data/
# Expected: train-split.jsonl, val-split.jsonl
wc -l /workspace/data/train-split.jsonl
```

### Available NeMo Gym environments

| Environment    | Domain              | Dataset source              | Reward signal    |
|----------------|---------------------|-----------------------------|------------------|
| `math`         | Competition math    | DAPO + SkyWorks (17K+104K)  | Exact match / SymPy |
| `code`         | Competitive coding  | Filtered competition problems (22K) | Unit tests |
| `science_qa`   | STEM multiple choice | Curated QA                 | Exact match      |
| `json_schema`  | Structured output   | NeMo Data Designer synthetic | Schema validation |

For the Kaggle challenge, start with `math`. Add `science_qa` if the challenge
includes STEM reasoning.

---

## Option B — Custom JSONL dataset

If you have your own dataset or want to use Kaggle-provided problems directly:

### Format

```jsonl
{"prompt": "Solve: If 3x + 7 = 22, what is x?", "answer": "5"}
{"prompt": "What is the derivative of x^3 + 2x?", "answer": "3x^2 + 2"}
```

Each line must be valid JSON with at minimum `prompt` and `answer` fields.

### Train/val split

```bash
# Shuffle and split 90/10
shuf your-dataset.jsonl > shuffled.jsonl
TOTAL=$(wc -l < shuffled.jsonl)
TRAIN=$(( TOTAL * 9 / 10 ))

head -n $TRAIN shuffled.jsonl > /workspace/data/train-split.jsonl
tail -n +$(( TRAIN + 1 )) shuffled.jsonl > /workspace/data/val-split.jsonl

echo "Train: $(wc -l < /workspace/data/train-split.jsonl) samples"
echo "Val:   $(wc -l < /workspace/data/val-split.jsonl) samples"
```

### Validation

NeMo RL provides a data validation script:

```bash
cd /opt/nemo-rl
uv run python -c "
from nemo_rl.data.utils import validate_jsonl
validate_jsonl('/workspace/data/train-split.jsonl')
validate_jsonl('/workspace/data/val-split.jsonl')
print('Data validation passed')
"
```

---

## Reward function

For math GRPO, NeMo Gym uses symbolic equivalence checking via SymPy by default.
For the Kaggle challenge you may want to customize the reward function to match
the competition's scoring metric.

Custom reward functions live in `examples/nemo_gym/` and are registered in the
Gym config. See `docs/guides/reward_functions.md` in the NeMo RL repo for details.

---

## Data flow during GRPO

```
JSONL prompts
     │
     ▼
NeMo Gym Agent Server
     │  (generates rollout prompts per step)
     ▼
Megatron Inference (rollout generation)
     │  (model generates N completions per prompt)
     ▼
NeMo Gym Resources Server
     │  (verifies completions, computes rewards)
     ▼
NeMo RL GRPO training loop
     │  (computes advantages, updates LoRA weights)
     ▼
Checkpoint saved
```
