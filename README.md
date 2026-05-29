# Nemotron LoRA Training on GB10 for the Kaggle Reasoning Challenge

https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge

This README packages a working setup for training a Nemotron LoRA adapter on a GB10-style NVIDIA system using Hugging Face, PEFT, TRL, and DSPy, then exporting a Kaggle-compatible `submission.zip`.[cite:74][cite:1] The Kaggle competition requires a Nemotron LoRA adapter with `adapter_config.json`, and Kaggle examples commonly package LoRA weights plus that config into the final archive.[cite:1][cite:119][cite:121]

## Goal

The target competition is the **NVIDIA Nemotron Model Reasoning Challenge** on Kaggle.[cite:1] The required submission is a LoRA adapter for **Nemotron-3-Nano-30B** with rank at most 32, evaluated under vLLM with deterministic generation settings and a metric that prefers answers inside `\\boxed{}`.[cite:1]

## Hardware

Training runs on a **NVIDIA DGX Spark (GB10)** — 128 GB unified CPU/GPU memory, Blackwell GB10 GPU, aarch64.

![DGX Spark Dashboard — CPU/GPU utilisation during training](docs/images/dgx-spark-dashboard.png)

## Repository layout

```text
.
├── .env                         # not tracked — HF_TOKEN, KAGGLE_API_TOKEN go here
├── .gitignore
├── CLAUDE.md
├── README.md
├── Dockerfile.gb10                      # primary build (26.04-py3)
├── Dockerfile.gb10-26-01                # validated 26.01-py3 baseline (used for training)
├── Dockerfile.gb10-25-12                # archived 25.12-py3 baseline
├── .clinerules/                         # 17 rules (framework 01-12, project-specific 13-17)
├── configs/
│   └── nemotron.yaml                # training hyperparameters
├── data/                            # peer CoT dataset (v0.2-cot)
│   ├── train.jsonl          # 8,358 examples — Gemini-2.0-flash CoT traces + \boxed{} answers
│   ├── valid.jsonl          # 929 examples
│   └── valid_labels.jsonl   # {"id","answer"} pairs for validate_metric.py
├── docs/
│   ├── images/
│   │   └── dgx-spark-dashboard.png      # DGX Spark CPU/GPU usage during training
│   ├── investigate/
│   │   └── dataset-comparison.md        # raw competition data vs peer CoT dataset analysis
│   └── plans/
│       ├── TODO.md                      # central task checklist
│       ├── leaderboard.md               # run history and scores
│       ├── peer-cot-dataset-training.md # v0.2-cot pipeline plan
│       ├── implementation-plan.md
│       ├── competition-overview.md
│       ├── CITATIONS.md
│       └── ...                          # other plan files
├── notebook/
│   ├── kaggle_prize_eligibility_outline.ipynb   # public prize eligibility writeup
│   ├── kernel-metadata.json                     # push config for prize notebook
│   ├── nemotron_submission_demo.ipynb           # submission path 2: load adapter → /kaggle/working
│   ├── submission-demo-kernel-metadata.json     # push config for submission demo
│   └── scrapbook.ipynb
└── scripts/
    ├── build_image.sh               # builds Docker image (26.01 used for training)
    ├── load_config.sh               # exports configs/nemotron.yaml as env vars
    ├── download_data.py             # competition data download + JSONL conversion
    ├── download_peer_cot.py         # peer CoT dataset download + JSONL conversion
    ├── smoke_test_nemotron.py
    ├── train_lora.py
    ├── infer_lora.py                # run inference with a saved adapter
    ├── validate_metric.py           # score predictions against labels
    ├── plot_training.py
    ├── package_submission.sh        # zip adapter into submission.zip
    ├── run_download.sh              # runner: competition data
    ├── run_download_peer_cot.sh     # runner: peer CoT dataset
    ├── run_smoke_test.sh
    ├── run_train.sh                 # runner: training (reads configs/nemotron.yaml)
    ├── run_inference.sh
    └── run_validate.sh
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

Archived baselines:
- 26.01: `bash scripts/build_image.sh 26-01`
- 25.12: `bash scripts/build_image.sh 25-12`

**OOM note**: CUDA extension compilation is memory-intensive. Both Dockerfiles cap parallel
jobs at `MAX_JOBS=8` for `pip install --no-binary` steps and `-j8` for the bitsandbytes cmake
build (previously `-j$(nproc)` = 72 jobs on Grace, which OOM'd the DGX Spark).

### 2. Training data

`data/train.jsonl` (8,358 ex), `data/valid.jsonl` (929 ex), and `data/valid_labels.jsonl` are
already committed — a fresh clone is ready to train without re-downloading.

The committed data is the **peer CoT dataset** (`kienngx/nemotron-30b-competition-trainingdata-cot-labels`):
each `response` contains a Gemini-2.0-flash chain-of-thought trace followed by `Final answer: \boxed{...}`.
See [`docs/investigate/dataset-comparison.md`](docs/investigate/dataset-comparison.md) for a
side-by-side comparison with the raw competition data.

To regenerate, add `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` to `.env` and run:

```bash
# Peer CoT dataset (current — recommended)
bash scripts/run_download_peer_cot.sh

# Raw competition labels only (v0.1-baseline source)
bash scripts/run_download.sh
```

**kagglehub credential note:** `run_download_peer_cot.sh` maps `KAGGLE_API_TOKEN → KAGGLE_KEY`
automatically (kagglehub uses `KAGGLE_KEY`, not `KAGGLE_API_TOKEN`).

### 3. Run the smoke test

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

**Memory note**: `smoke_test_nemotron.py` passes `max_memory={0: "115GiB"}` to `from_pretrained`,
keeping all shards on the GB10's unified GPU memory and preventing `device_map="auto"` from
spilling to NVMe (disk offload makes generation take 60+ minutes). Adjust the cap if running
on a system with less GPU memory.

### 4. LoRA training run

Edit `configs/nemotron.yaml` to set hyperparameters, then:

```bash
# Basic run — outputs timestamped log and adapter dir
bash scripts/run_train.sh

# Named run — log and adapter dir include RUN_NAME prefix (recommended)
RUN_NAME=cot_v1 bash scripts/run_train.sh
# → output/train_cot_v1_YYYYMMDD_HHMMSS.log
# → output/adapter_cot_v1_YYYYMMDD_HHMMSS/
```

The script reads all hyperparameters from `configs/nemotron.yaml` via `load_config.sh`.

### 5. Validate

After training, run inference on the validation set and score it:

```bash
bash scripts/run_validate.sh
```

Or to score a specific predictions file:

```bash
bash scripts/run_validate.sh --predictions output/predictions_cot_v1.jsonl
```

### 6. Package adapter into `submission.zip`

```bash
bash scripts/package_submission.sh output/adapter_cot_v1_YYYYMMDD_HHMMSS
# → output/submission/submission.zip
```

### 7. Submit to Kaggle

```bash
source .env && kaggle competitions submit \
  -c nvidia-nemotron-model-reasoning-challenge \
  -f output/submission/submission.zip \
  -m "v0.2-cot"
```

### Kaggle Notebooks

| Notebook | URL | Purpose |
|---|---|---|
| Prize eligibility writeup | [nemotron-3-nano-30b-lora-reasoning-challenge](https://www.kaggle.com/code/gdataranger/nemotron-3-nano-30b-lora-reasoning-challenge) | Public writeup required for prizes |
| Submission demo | [nemotron-lora-submission-demo](https://www.kaggle.com/code/gdataranger/nemotron-lora-submission-demo) | Loads pre-trained adapter → saves to `/kaggle/working` for submission |

To push notebook updates:

```bash
# Prize eligibility notebook
source .env && KAGGLE_KEY="${KAGGLE_API_TOKEN}" kaggle kernels push -p notebook/

# Submission demo
cp notebook/nemotron_submission_demo.ipynb /tmp/demo-push/
cp notebook/submission-demo-kernel-metadata.json /tmp/demo-push/kernel-metadata.json
source .env && KAGGLE_KEY="${KAGGLE_API_TOKEN}" kaggle kernels push -p /tmp/demo-push/
```

### Leaderboard

See [`docs/plans/leaderboard.md`](docs/plans/leaderboard.md) for the full run history.

| Version | Data | Val Acc | Kaggle Score |
|---|---|---|---|
| v0.1-baseline | Competition labels only | 43.5% | 0.57 |
| v0.2-cot | Peer CoT dataset (Gemini-2.0-flash) | TBD | TBD |