# Nemotron v09 — Dependency Utility Script
#
# Installs all packages needed by the training notebook into /kaggle/working/python_packages/
# so they can be imported via sys.path — no internet or pip needed at training time.
#
# Run once with any GPU + Internet ON (P100 or T4 is fine).
# Packages are cross-compiled for SM 12.0 (RTX Pro 6000 Blackwell) via
# TORCH_CUDA_ARCH_LIST=12.0+PTX — the host GPU does not need to be an RTX Pro 6000.
#
# GPU compilation approach adapted from ryanholbrook/nvidia-utility-script.

import os, pathlib, subprocess, sys

# ── GPU / CUDA check ──────────────────────────────────────────────────────────
r = subprocess.run('/usr/local/cuda/bin/nvcc --version', shell=True, capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError('nvcc not found — enable GPU accelerator in Session Options')
print(r.stdout.strip())

r2 = subprocess.run(
    'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader',
    shell=True, capture_output=True, text=True,
)
if r2.returncode != 0:
    raise RuntimeError('nvidia-smi failed — GPU not available in this session')
gpu_line = r2.stdout.strip().split('\n')[0]
gpu_name, cc = gpu_line.rsplit(',', 1)
cc = cc.strip()
print(f'Host GPU: {gpu_name.strip()}  Compute capability: SM {cc}')
print('Target: SM 12.0 (RTX Pro 6000) — cross-compiling via TORCH_CUDA_ARCH_LIST=12.0+PTX')

# ── Build environment ─────────────────────────────────────────────────────────
TARGET = '/kaggle/working/python_packages'
pathlib.Path(TARGET).mkdir(exist_ok=True)

env = os.environ.copy()
env['TORCH_CUDA_ARCH_LIST'] = '12.0+PTX'
env['FLASH_ATTENTION_FORCE_BUILD'] = 'TRUE'
env['MAX_JOBS'] = '2'
env['PYTHONPATH'] = f"{TARGET}:{env.get('PYTHONPATH', '')}"

def _run(cmd, use_env=True):
    print(f'Running: {cmd[:120]}')
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       env=(env if use_env else None))
    if r.returncode != 0:
        print(f'FAILED (exit {r.returncode}):')
        print(r.stdout[-3000:] if r.stdout.strip() else '')
        print(r.stderr[-3000:] if r.stderr.strip() else '')
        raise RuntimeError(f'Command failed: {cmd}')
    print('  OK')

# ── Step 1: PyTorch nightly cu128 ─────────────────────────────────────────────
# RTX Pro 6000 (SM 12.0 Blackwell) requires cu128; the base image ships cu121/cu124.
# torchvision/torchaudio installed --no-deps to avoid nightly version-pinning conflicts.
print('\n=== Step 1: PyTorch nightly cu128 ===')
_run('uv pip uninstall torch torchvision torchaudio', use_env=False)
_run(f'uv pip install --target={TARGET} --system --pre torch '
     '--index-url https://download.pytorch.org/whl/nightly/cu128')
_run(f'uv pip install --target={TARGET} --system --pre --no-deps torchvision torchaudio '
     '--index-url https://download.pytorch.org/whl/nightly/cu128')

# ── Step 2: nvidia-cutlass ────────────────────────────────────────────────────
print('\n=== Step 2: nvidia-cutlass ===')
_run(f'uv pip install --target={TARGET} --system nvidia-cutlass')

# ── Step 3: causal-conv1d (source build, ~5 min) ─────────────────────────────
print('\n=== Step 3: causal-conv1d (source build, ~5 min) ===')
_run(f'uv pip install --target={TARGET} --system '
     '--no-build-isolation "causal-conv1d>=1.4.0"')

# ── Step 4: mamba-ssm from git (source build, ~10 min) ───────────────────────
# flash-attn intentionally skipped — not needed with attn_implementation='eager'
print('\n=== Step 4: mamba-ssm from git (source build, ~10 min) ===')
_run(f'uv pip install --target={TARGET} --system '
     '--no-build-isolation "git+https://github.com/state-spaces/mamba.git"')

# ── Step 5: trl, unsloth, unsloth_zoo (pure Python) ──────────────────────────
print('\n=== Step 5: trl, unsloth, unsloth_zoo (pure Python) ===')
r = subprocess.run(
    [sys.executable, '-m', 'pip', 'install',
     '--target', TARGET, '--no-deps',
     'trl', 'unsloth', 'unsloth_zoo'],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print('pip FAILED:')
    print(r.stdout[-2000:] if r.stdout.strip() else '')
    print(r.stderr[-2000:] if r.stderr.strip() else '')
    raise RuntimeError('pip install failed — ensure Internet is ON')
print('  OK')

# ── Step 6: bitsandbytes (unsloth dependency; pre-built cu12x wheel) ─────────
print('\n=== Step 6: bitsandbytes ===')
r = subprocess.run(
    [sys.executable, '-m', 'pip', 'install',
     '--target', TARGET, '--no-deps',
     'bitsandbytes'],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print('pip FAILED:')
    print(r.stdout[-2000:] if r.stdout.strip() else '')
    print(r.stderr[-2000:] if r.stderr.strip() else '')
    raise RuntimeError('pip install failed — ensure Internet is ON')
print('  OK')

# ── Verify ────────────────────────────────────────────────────────────────────
sys.path.insert(0, TARGET)

import importlib.metadata
print('\nInstalled package versions:')
for pkg in ['torch', 'mamba_ssm', 'causal_conv1d', 'trl', 'unsloth', 'unsloth_zoo', 'bitsandbytes']:
    try:
        print(f'  {pkg}: {importlib.metadata.version(pkg)} ✓')
    except Exception as e:
        print(f'  {pkg}: MISSING — {e}')

import torch
print(f'\nPyTorch: {torch.__version__}  CUDA: {torch.version.cuda}')
assert torch.cuda.is_available(), 'CUDA not available in nightly torch'
print(f'GPU: {torch.cuda.get_device_name(0)}  SM {torch.cuda.get_device_capability(0)}')

total = sum(f.stat().st_size for f in pathlib.Path(TARGET).rglob('*') if f.is_file())
print(f'\nTotal size: {total/1e6:.1f} MB')
print('\nDone — output is ready to use as kernel source gdataranger/nemotron-v09-deps')
