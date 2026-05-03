# PyTorch Container Migration Plan for Nemotron LoRA on GB10 / DGX Spark

## Overview

This plan migrates the training image from `nvcr.io/nvidia/pytorch:25.12-py3` to a newer NVIDIA PyTorch container while preserving Hugging Face Nemotron 3 Nano compatibility, ARM64 support, and the current custom extension strategy for `causal-conv1d`, `mamba-ssm`, and `bitsandbytes`.[cite:126][cite:127]

The target state should keep Blackwell-ready support, since NVIDIA documents Blackwell optimization in newer framework containers and the 26.01 release family explicitly supports Blackwell compute capability 10.x and 12.x GPUs.[cite:126][cite:128][cite:129]

## Migration goal

The immediate goal is to test whether `nvcr.io/nvidia/pytorch:26.01-py3` reduces or removes the ABI mismatch that causes `causal-conv1d` 1.6.x to reference `TensorImpl::decref_pyobject()` against the older 25.12 NVIDIA torch ABI.[cite:126]

A secondary goal is to keep the model stack aligned with the Nemotron Hugging Face card, which states that the published examples were tested with Transformers 4.57.3 and use `trust_remote_code=True`.[cite:133]

## Why migrate

NVIDIA PyTorch 26.01 is a published container version, so there is a higher `nv26.#` series available beyond 25.12.[cite:126][cite:127]

The 26.01 release family moves to CUDA 13.1 and a newer PyTorch build on Ubuntu 24.04 / Python 3.12, which makes it a better candidate for Blackwell-era ARM64 systems and newer custom CUDA extension builds.[cite:126][cite:129][cite:130]

CUDA 13.1 also aligns with your custom arch targets because NVIDIA documents support focused on newer compute capabilities, and related 26.01 framework notes state support for compute capability 6.0 and later.[cite:128][cite:130]

## Main risks

A newer base image does not automatically guarantee that `causal-conv1d` 1.6.x will build cleanly, because the package may still expect a torch ABI newer than the one shipped in NVIDIA PyTorch 26.01.[cite:126]

Your image also depends on several source builds on ARM64, so any migration can fail even if the base container works, especially when wheels or compiled shared objects silently target `x86_64` instead of `aarch64`.[cite:127][cite:128]

NVIDIA also documents that the PyTorch containers ship with `/etc/pip/constraint.txt`, and from 25.03 onward this file constrains installs unless edited when overriding packaged versions.[cite:85]

## Phased plan

### Phase 1: Rebase only

1. Change the base image to `FROM --platform=linux/arm64 nvcr.io/nvidia/pytorch:26.01-py3`.[cite:127]
2. Keep all Python package pins unchanged for the first build so the rebase isolates container-level changes from Python dependency changes.[cite:133][cite:126]
3. Record the container-reported versions of Python, CUDA, torch, and architecture metadata during build validation.[cite:126][cite:129]

### Phase 2: Validate packaging behavior

1. Inspect `/etc/pip/constraint.txt` in the new image before installing overrides, because NVIDIA states it governs package resolution in these containers.[cite:85]
2. Remove or edit entries for packages you intentionally pin, especially `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `torchao`, and `huggingface_hub`.[cite:85]
3. Pin the Hugging Face stack consistently rather than letting `huggingface_hub` float, because version drift around Transformers 4.57.3 has caused compatibility issues.[cite:138]

### Phase 3: Test `causal-conv1d`

1. Attempt a clean source build of `causal-conv1d` 1.6.x first on 26.01 without your current patch.[cite:126]
2. If it still fails with the ABI symbol issue, fall back to your current `v1.5.0.post8` patched-source strategy.[cite:126]
3. Keep the patched arch list focused on modern targets such as `sm_80`, `sm_90`, and Blackwell `sm_120`, because older CUDA 13.x toolchains no longer fit the hardcoded legacy arch list in older setup logic.[cite:128][cite:130]

### Phase 4: Revalidate dependent extensions

1. Rebuild `mamba-ssm` from source and confirm the installed extension layout is `aarch64`-compatible.[cite:127][cite:128]
2. Rebuild `bitsandbytes` with the same explicit CUDA compute capability list used today, then verify that import and GPU detection work on the actual target host.[cite:128][cite:129]
3. Run import-time checks for `torch`, `transformers`, `peft`, `trl`, `mamba_ssm`, and `bitsandbytes` in the built image.[cite:133][cite:126]

### Phase 5: Model-level smoke tests

1. Load the Nemotron tokenizer and config from `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with `trust_remote_code=True`.[cite:133]
2. Confirm the model can initialize in the expected training mode without import errors or custom code failures.[cite:133]
3. Run a minimal LoRA wiring test with PEFT to verify adapter injection completes before attempting full training.[cite:133]

## Recommended Dockerfile changes

| Area | Current | Proposed | Reason |
|---|---|---|---|
| Base image | `25.12-py3` | `26.01-py3` | Newer NVIDIA PyTorch container with Blackwell-era support.[cite:126][cite:127][cite:128] |
| Pip constraints | Implicit | Explicitly inspect and edit `/etc/pip/constraint.txt` | NVIDIA says container package installs are constrained by that file.[cite:85] |
| HF stack | Mixed pins | Keep `transformers==4.57.3` and pin related HF packages tightly | Nemotron examples were tested with 4.57.3, and drift can break compatibility.[cite:133][cite:138] |
| `causal-conv1d` | Patched 1.5.x | Try 1.6.x first on 26.01, keep patched 1.5.x fallback | Newer torch ABI may help, but fallback is still needed if ABI mismatch remains.[cite:126] |
| Validation | Build success only | Add runtime import and model-load smoke tests | Source builds can succeed while runtime still fails.[cite:133][cite:126] |

## Acceptance criteria

The migration should be considered successful only if all of the following are true:

- The image builds successfully on `linux/arm64` using NVIDIA PyTorch 26.01.[cite:127]
- `torch.cuda.is_available()` works on the target system and reports the expected GPU environment.[cite:126][cite:129]
- `causal-conv1d` either builds cleanly at 1.6.x or the patched 1.5.x fallback works without ABI crashes at import time.[cite:126]
- `mamba-ssm` and `bitsandbytes` import successfully on ARM64 with no hidden `x86_64` artifact leakage.[cite:127][cite:128]
- The Nemotron tokenizer and model config load successfully with Transformers 4.57.3 and `trust_remote_code=True`.[cite:133]

## Rollback path

If 26.01 does not fix the `causal-conv1d` ABI issue or introduces new ARM64 regressions, revert to `25.12-py3`, retain the patched `v1.5.0.post8` workflow, and defer the migration until a later NVIDIA PyTorch release in the 26.xx line provides a better torch ABI match.[cite:126][cite:127]

This rollback keeps the current known-working architecture while preserving the migration work as a branch for retesting on future NVIDIA container releases.[cite:126]