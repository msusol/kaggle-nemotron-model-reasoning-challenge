# Rule 21 — Submission Packaging

## Always use `scripts/package_submission.sh` directly from the host

To package a checkpoint for submission, run from the project root:

```zsh
bash scripts/package_submission.sh \
  output/adapter_<RUN_NAME>_ckpt \
  output/sub_<RUN_NAME>_step<N>
```

The script internally handles its own Docker exec call — it does **not** need to be wrapped in a separate `docker run` or `docker exec` invocation. Running it from a host shell is the correct and only way.

**Why:** `scripts/package_submission.sh` spawns the torch/safetensors work inside the live training container (or a minimal Docker exec) using the `nemotron-gb10:latest` image, which has the required ML packages. The host Python has no `torch`. Do not attempt to call `docker run` or `docker exec` around this script.

**Docker live mount:** The project directory is bind-mounted as `/workspace` inside the training container. Any checkpoint files written by the trainer (e.g., `output/adapter_v14_run17_ckpt/`) are immediately visible on the host and inside any new container launched with the same `-v` mount — no copying required.

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
