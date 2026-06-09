# 05 — Training launch

## Pre-launch checklist

Run these checks **inside the container** before every training session:

```bash
# 1. Verify HBM headroom (need at least 80 GB free after model load)
python3 -c "
import torch
torch.cuda.init()
f, t = torch.cuda.mem_get_info()
print(f'HBM free: {f/1e9:.1f} GB / {t/1e9:.1f} GB total')
if f < 65e9:
    print('WARNING: low HBM — run cleanup sequence in 01-docker-build.md')
"

# 2. Verify model files exist
ls -lh /workspace/model/*.safetensors | wc -l   # expect 13
ls /workspace/model/config.json                  # expect: file exists

# 3. Verify training data
wc -l /workspace/data/train-split.jsonl
wc -l /workspace/data/val-split.jsonl

# 4. Verify NeMo RL branch
cd /opt/nemo-rl && git branch   # expect: * nano-v3

# 5. Check config exists
ls /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml
```

---

## Step 1 — Validation run (3 steps)

Always run 3 steps first to confirm the config loads, the model fits in HBM,
and rollout generation works before committing to a multi-hour full run.

```bash
cd /opt/nemo-rl

EXP_NAME="$(date +%Y%m%d)/lora-grpo-spark/validation-run-001"
mkdir -p /workspace/results/$EXP_NAME

TORCH_CUDA_ARCH_LIST="12.1" \
HF_HOME=/workspace/.cache/huggingface \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml \
  grpo.max_num_steps=3 \
  checkpointing.checkpoint_dir=/workspace/results/$EXP_NAME \
  logger.log_dir=/workspace/results/$EXP_NAME/logs \
  2>&1 | tee /workspace/results/$EXP_NAME/logs/output.log

echo "Exit code: $?"
```

### What to look for in the first 3 steps

```
[Step 1] rollout generation ... OK
[Step 1] reward computation ... mean_reward: 0.xx
[Step 1] policy update ...     loss: x.xx
[Step 2] ...
[Step 3] ...
Validation run complete.
```

If you see OOM errors, go to `07-troubleshooting.md` — OOM section.

---

## Step 2 — Full training run

Once the 3-step validation passes, launch the full run. The official Nemotron 3
Nano training ran ~hundreds of steps across multiple RL environments. For the
Kaggle challenge, ~200–500 steps on math data is a reasonable starting point.

```bash
cd /opt/nemo-rl

EXP_NAME="$(date +%Y%m%d)/lora-grpo-spark/full-run-001"
mkdir -p /workspace/results/$EXP_NAME/logs

TORCH_CUDA_ARCH_LIST="12.1" \
HF_HOME=/workspace/.cache/huggingface \
nohup uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml \
  checkpointing.checkpoint_dir=/workspace/results/$EXP_NAME \
  logger.log_dir=/workspace/results/$EXP_NAME/logs \
  > /workspace/results/$EXP_NAME/logs/output.log 2>&1 &

echo "Training PID: $!"
echo "Monitor: tail -f /workspace/results/$EXP_NAME/logs/output.log"
```

Using `nohup` is recommended — if your SSH session drops, training continues.

---

## Monitoring

```bash
# Live log tail
tail -f /workspace/results/$EXP_NAME/logs/output.log

# Watch HBM usage during training (in a second terminal)
watch -n 5 'python3 -c "
import torch
f,t = torch.cuda.mem_get_info()
print(f\"HBM: {(t-f)/1e9:.1f} GB used / {t/1e9:.1f} GB total ({100*(t-f)/t:.0f}%)\")
"'

# Check checkpoint saves
watch -n 30 'ls -lht /workspace/results/$EXP_NAME/policy/weights/ | head -5'
```

### Key metrics to watch

| Metric                  | Healthy range       | Warning sign                    |
|-------------------------|---------------------|---------------------------------|
| `mean_reward`           | Increasing over steps | Flat or decreasing from step 1 |
| `reward_std`            | > 0 (variance exists) | 0 → all completions identical  |
| `policy_loss`           | Decreasing          | Exploding (NaN / Inf)           |
| `kl_divergence`         | < 0.5               | > 1.0 → reduce lr or kl_coeff  |
| HBM used                | < 125 GB            | > 128 GB → OOM imminent         |
| Tokens/sec (generation) | > 100 tok/s         | < 50 → chunked prefill issue   |

---

## Resuming from checkpoint

```bash
CKPT_STEP=100   # adjust to your last good checkpoint step

TORCH_CUDA_ARCH_LIST="12.1" \
HF_HOME=/workspace/.cache/huggingface \
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
  --config /workspace/docs/plans/lora-grpo-spark/04-lora-grpo-config.yaml \
  checkpointing.checkpoint_dir=/workspace/results/$EXP_NAME \
  checkpointing.resume_from_checkpoint=true \
  logger.log_dir=/workspace/results/$EXP_NAME/logs \
  2>&1 | tee -a /workspace/results/$EXP_NAME/logs/output.log
```

NeMo RL auto-detects the latest checkpoint in `checkpoint_dir` when
`resume_from_checkpoint=true`.
