# Adapting `Dockerfile.gb10`

## Objective

The purpose of adapting `Dockerfile.gb10` is to preserve the existing GPU-ready environment while swapping in the specific model, libraries, scripts, and packaging steps required by the Kaggle Nemotron competition.[cite:1][cite:34]

## What should stay the same

- CUDA and PyTorch support for the GB10 environment can stay if the current image already supports large-model Hugging Face fine-tuning.[cite:34]
- Existing PEFT, Transformers, Accelerate, TRL, and DSPy dependencies can stay if they are already stable in the current container.[cite:35]
- Existing patterns for mounted volumes, cache directories, and training entrypoints can stay if they are not tied to a different base model family.[cite:34]

## What should change

### 1. Base model assumption

Any previous model-specific configuration should be replaced with Nemotron-3-Nano-30B, preferably the Hugging Face checkpoint `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` if the current workflow is already based on Transformers.[cite:34]

### 2. Training scripts

Training scripts inside the image should be updated so they:

- load Nemotron instead of the prior model,[cite:34]
- enforce LoRA rank 32 or lower,[cite:1]
- produce reasoning-formatted outputs ending in `\\boxed{}` during training targets,[cite:1]
- save PEFT adapters with `adapter_config.json`.[cite:30]

### 3. Validation scripts

Add a lightweight validation script that mimics the Kaggle metric by extracting boxed answers first, then applying exact or numeric comparison logic.[cite:1] This is more reliable than using only loss or generic text-evaluation metrics.[cite:1]

### 4. Packaging step

The container should include a script that takes the saved adapter directory and packages it into `submission.zip` in the layout expected by the competition.[cite:1][cite:30]

## Suggested Docker-related additions

- An environment variable for the Nemotron base model ID.[cite:34]
- A dedicated output path for the final exported adapter.[cite:30]
- A packaging script such as `scripts/package_submission.sh` that zips only the adapter assets needed by Kaggle.[cite:1]
