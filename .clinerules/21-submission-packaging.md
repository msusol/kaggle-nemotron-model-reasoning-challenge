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

Package from the **rolling checkpoint** (`output/adapter_<RUN_NAME>_ckpt/`) immediately after each 100-step notification — it is overwritten at the next checkpoint. Named snapshots (e.g., `output/adapter_v12_run15_step200/`) are permanent.
