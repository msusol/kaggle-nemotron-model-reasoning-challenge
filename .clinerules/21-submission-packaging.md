# Rule 21 — Submission Packaging

## Always run `scripts/package_submission.sh` via `docker exec` into the live training container

The script calls `python3` with `torch` and `safetensors` directly — the host Python has neither. Run it inside the live training container:

```zsh
docker exec <container_name> bash scripts/package_submission.sh \
  output/adapter_<RUN_NAME>_ckpt \
  output/sub_<RUN_NAME>_step<N>
```

The active training container for run17 is `nemotron-trainer-v14`. Check with `docker ps --filter name=nemotron`.

**Why:** The project directory is bind-mounted as `/workspace` inside the training container. Checkpoints written by the trainer are immediately visible at the same relative path inside the container. `docker exec` runs the script in that environment where `torch`/`safetensors` are available, without interrupting training.

**Never use `docker run`** (a new container) — use `docker exec` into the already-running trainer.

## Submission workflow

```zsh
# 1. Package (host shell, from project root)
bash scripts/package_submission.sh \
  output/adapter_<RUN_NAME>_ckpt \
  output/sub_<RUN_NAME>_step<N>

# 2. Verify (optional)
ls -lh output/sub_<RUN_NAME>_step<N>/submission.zip

# 3. Check for duplicate before submitting
kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge | head -3

# 4. Submit
kaggle competitions submit nvidia-nemotron-model-reasoning-challenge \
  -f output/sub_<RUN_NAME>_step<N>/submission.zip \
  -m "<version> <run_name> step<N> <key metrics>"
```

Package from **named checkpoints** (`output/adapter_<RUN_NAME>/checkpoint-N/`) — they are permanent and never overwritten. Always wait for the `[moe-lora] Saved ... expert_lora_weights.pt` log line before packaging; the expert weights (1.6 GB) finish ~45s after the tqdm loss line.

**If the training container has stopped** (training complete), use `docker run --rm` with the same image:

```zsh
WORKSPACE=/home/msusol/LosusAI/Projects/Kaggle/kaggle-nemotron-model-reasoning-challenge
docker run --rm \
  --privileged -e NVIDIA_VISIBLE_DEVICES=all \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  nemotron-gb10:latest \
  bash scripts/package_submission.sh \
    output/adapter_<RUN_NAME>/checkpoint-N \
    output/sub_<RUN_NAME>_stepN
```
