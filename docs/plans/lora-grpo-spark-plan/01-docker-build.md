# 01 — Docker build for DGX Spark (GB10 / arm64)

## Context

The DGX Spark runs a Blackwell GB10 GPU with compute capability `sm_121a` on an
aarch64 host (Ubuntu 24.04). Pre-built NGC containers (including `nemo-rl:v0.5.0`)
compile CUDA extensions for Hopper (`sm_90`) and earlier — they lack `sm_121` and
will fail at runtime with `no kernel image is available for execution on the device`.

This build:
- Targets `sm_120` (GB200) and `sm_121` (GB10) plus all prior arches
- Builds `causal-conv1d` and `mamba-ssm` from source (required for Mamba-H model)
- Builds `bitsandbytes` from source for Blackwell (required for QLoRA if needed)
- Uses `pytorch:26.04-py3` (arm64) as the base — first PyTorch image with full GB10 support
- Clones the `nano-v3` branch of NeMo RL

## Build command

```bash
# From the directory containing Dockerfile.spark
docker build \
  -f Dockerfile.spark \
  --build-arg MAX_JOBS=8 \
  -t nemo-rl-spark:latest \
  .
```

> **`MAX_JOBS=8`** is critical. The GB10 has 72 ARM cores — cmake without a job cap
> spawns 72 parallel compile jobs and exhausts RAM during `docker build`.

Expected build time: 30–90 minutes depending on network and storage speed.

## Run flags

The standard NVIDIA container runtime flags (`--gpus all`, `--runtime=nvidia`,
`--device nvidia.com/gpu=all`) all fail on GB10 with:
```
failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: no such file
```

Use `--privileged` instead:

```bash
docker run --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name nemo-rl-run \
  -v /path/to/your/workspace:/workspace \
  -v /path/to/hf/cache:/root/.cache/huggingface \
  -it nemo-rl-spark:latest bash
```

Flag explanations:
- `--privileged` — grants full host capabilities, bypasses the failing bind-mount
- `NVIDIA_VISIBLE_DEVICES=all` — exposes GPU to the container runtime
- `PYTORCH_CUDA_ALLOC_CONF` — `expandable_segments` reduces fragmentation;
  `max_split_size_mb:512` prevents large cached blocks being fragmented
- `--ipc=host` — shared IPC namespace, required for CUDA multi-process
- `--ulimit memlock=-1` — removes locked-memory ceiling for CUDA pinned allocations

## Pre-run HBM cleanup (run before every training session)

After any killed or OOM'd container, the CUDA driver retains orphaned HBM
allocations invisible to `nvidia-smi`. Each subsequent run starts with less HBM.

```bash
# Stop gnome-remote-desktop (holds ~6 GB CUDA context)
systemctl --user stop gnome-remote-desktop.service

# Remove any stale container (triggers driver GC of orphaned allocations)
docker rm -f nemo-rl-run 2>/dev/null || true

# Drop Linux page cache and reset swap
sync
docker run --rm --privileged -v /:/host alpine sh -c '
  echo 3 > /proc/sys/vm/drop_caches
  echo 1048576 > /proc/sys/vm/min_free_kbytes
  echo 500 > /proc/sys/vm/vfs_cache_pressure
  swapoff /host/swap.img 2>/dev/null
  swapon /host/swap.img 2>/dev/null
  true'

# Torch preflight: creating a CUDA context forces GC of orphaned HBM
python3 -c "
import torch
torch.cuda.init()
f, t = torch.cuda.mem_get_info()
print(f'HBM free: {f/1e9:.1f} GB / {t/1e9:.1f} GB total')
assert f > 60e9, f'Not enough HBM free ({f/1e9:.1f} GB) — check for orphaned allocations'
"
```

## Verifying GPU visibility inside the container

```bash
# Should show GB10 with ~130 GB HBM
nvidia-smi

# Cross-check with torch (use this, not nvidia-smi, for actual free memory)
python3 -c "
import torch
print(torch.cuda.get_device_name(0))
f, t = torch.cuda.mem_get_info()
print(f'Free: {f/1e9:.1f} GB  Total: {t/1e9:.1f} GB')
"
```

> **Note:** `free -h` shows CPU LPDDR5x (~121 GB), not GPU HBM. These are two
> separate physical memory pools linked by NVLink-C2C. Always use
> `torch.cuda.mem_get_info()` for GPU memory checks.
