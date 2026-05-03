# Nemotron LoRA Training on GB10 for the Kaggle Reasoning Challenge

https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

This README packages a working setup for training a Nemotron LoRA adapter on a GB10-style NVIDIA system using Hugging Face, PEFT, TRL, and DSPy, then exporting a Kaggle-compatible `submission.zip`.[cite:74][cite:1] The Kaggle competition requires a Nemotron LoRA adapter with `adapter_config.json`, and Kaggle examples commonly package LoRA weights plus that config into the final archive.[cite:1][cite:119][cite:121]

## Goal

The target competition is the **NVIDIA Nemotron Model Reasoning Challenge** on Kaggle.[cite:1] The required submission is a LoRA adapter for **Nemotron-3-Nano-30B** with rank at most 32, evaluated under vLLM with deterministic generation settings and a metric that prefers answers inside `\\boxed{}`.[cite:1]

## Repository layout

```text
.
├── .env                         # not tracked — HF_TOKEN goes here
├── .gitignore
├── CLAUDE.md
├── TODO.md
├── README.md
├── Dockerfile.gb10                      # primary build (26.01-py3)
├── Dockerfile.gb10-25                   # archived 25.12-py3 baseline
├── .clinerules/
│   ├── 01-global.md
│   ├── 02-plan-and-todo-sync.md
│   ├── 03-desync-cleanup.md
│   ├── 10-commit-description.md
│   ├── 11-markdown-codeblocks.md
│   ├── 12-docker-stop-failed.md
│   └── 13-docker-gpu-gb10.md
├── configs/
│   └── nemotron.yaml                # training hyperparameters
├── data/
│   ├── train.jsonl
│   └── valid.jsonl
├── plans/
│   ├── CITATIONS.md
│   ├── competition-overview.md
│   ├── dockerfile-gb10-adapation.md
│   ├── dockerfile-gb10-proposed.md
│   ├── dockerfile-gb10-review.md
│   ├── dspy-peft-migration.md
│   ├── implementation-plan.md
│   ├── submission-checklist.md
│   └── submission-layout.md
└── scripts/
    ├── build_image.sh               # builds Docker image with GPU-capable buildkitd
    ├── load_config.sh               # exports configs/nemotron.yaml as env vars
    ├── run_smoke_test.sh
    ├── run_train.sh
    ├── smoke_test_nemotron.py
    ├── train_lora.py
    ├── validate_metric.py
    └── package_submission.sh
```

## Commands

### 1. Build the image

```bash
bash scripts/build_image.sh
```

**GB10 / aarch64**: Always build directly on the GB10 — never import an image built on x86\_64.
The `causal_conv1d` and `mamba_ssm` CUDA extensions must be compiled for `aarch64 + sm_120`.

**`selective_scan_cuda` / mamba-ssm note**: Docker OCI workers have isolated device namespaces;
GPU devices are not accessible during `docker build` on this system (confirmed: privileged
buildkitd daemons, `[worker.oci] privileged=true`, and TCP remote builders were all attempted).
`selective_scan_cuda` therefore cannot compile at build time and is patched with `try/except` in
the Dockerfile so mamba-ssm imports cleanly. This is safe: Nemotron-H uses Mamba-2 Triton
kernels exclusively and never calls the legacy `selective_scan_cuda` path.

To force a full recompile of the mamba/causal-conv1d layers (e.g., after a base-image update):

```bash
bash scripts/build_image.sh "" --fresh
```

The archived 25.12-py3 baseline: `bash scripts/build_image.sh 25`
Force-recompile it: `bash scripts/build_image.sh 25 --fresh`

**OOM note**: CUDA extension compilation is memory-intensive. Both Dockerfiles cap parallel
jobs at `MAX_JOBS=8` for `pip install --no-binary` steps and `-j8` for the bitsandbytes cmake
build (previously `-j$(nproc)` = 72 jobs on Grace, which OOM'd the DGX Spark).

### 2. Run the smoke test

```bash
bash scripts/run_smoke_test.sh
```

**Expected timing**:
- First run (weights not yet cached): ~8–10 min model download + ~30 s shard load + ~5–15 min
  Triton JIT kernel compilation before the first token appears.
- Subsequent runs (weights cached in `~/.cache/huggingface`): ~30 s shard load + ~5–15 min
  Triton JIT, then generation completes quickly.

**Expected output** (abridged):

```text
Loading tokenizer: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
Loading model: ...
Loading checkpoint shards: 100%|██████████| 13/13
=== SAMPLE GENERATION ===
... 17 + 25 = 42 ... Final answer: \boxed{42}
Saved PEFT smoke adapter to: /workspace/output/smoke_adapter
```

After the run, `output/smoke_adapter/adapter_config.json` must exist — that confirms the full
tokenizer → model load → generation → PEFT export stack works end-to-end.

**Memory note**: `smoke_test_nemotron.py` passes `max_memory={0: "20GiB", "cpu": "110GiB"}` to
`from_pretrained`. This prevents `device_map="auto"` from spilling layers to NVMe (disk offload
makes 64-token generation take 60+ minutes). If your system has less than ~60 GB of
GPU + CPU RAM, the test will OOM rather than silently running for hours.

### 3. First LoRA training run

Edit `configs/nemotron.yaml` to set hyperparameters, then:

```bash
bash scripts/run_train.sh
```

### 4. Local metric sanity check

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

### 5. Package adapter into `submission.zip`

```bash
bash /workspace/scripts/package_submission.sh \
  /workspace/output/adapter \
  /workspace/output/submission
```