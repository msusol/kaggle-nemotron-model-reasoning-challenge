# Docker Training on GB10 (Grace-Blackwell)

Operational runbook for building the training image, launching training runs, and tuning memory on the GB10 / DGX Spark system.

## Prerequisites

- Docker daemon running and accessible without `sudo` (user is in `docker` group)
- `HF_TOKEN` set in `.env` or the environment
- Image built: `nemotron-gb10:latest` (see Build section below)

---

## GPU flags — why `--privileged` instead of `--gpus all`

On this GB10 system, the standard NVIDIA container runtime patterns fail:

```bash
# These all fail with:
# "failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: no such file or directory"
--gpus all
--runtime=nvidia
--device nvidia.com/gpu=all   # CDI mode
```

The NVIDIA runtime's bind-mount logic requires host capabilities that are not present in the default Docker setup here. The working pattern is:

```bash
docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ...
```

- `--privileged` grants full host capabilities, bypassing the failing bind-mount
- `NVIDIA_VISIBLE_DEVICES=all` tells the runtime which GPUs to expose
- `--ipc=host` shares the host IPC namespace (required for multi-process GPU comms via shared memory)
- `--ulimit memlock=-1` removes the locked-memory ceiling (needed for CUDA pinned allocations)

These flags are baked into `scripts/run_train.sh` and should not be changed.

---

## Memory limits

The GB10 has **128 GB physical unified memory** shared between the Grace CPU and Blackwell GPU. The OS sees ~121 GB — the remaining ~7 GB is reserved by firmware and memory-mapped I/O, which is normal for this platform:

```
$ free -h
               total        used        free      shared  buff/cache   available
Mem:           121Gi        78Gi        27Gi        55Mi        17Gi        43Gi
Swap:           15Gi          0B        15Gi
```

There is no separate GPU VRAM pool — `nvidia-smi` reports `[N/A]` for `memory.total/free`. Docker's `--memory` cgroup limit covers the entire OS-visible pool (121 GB).

### Default limits in `run_train.sh`

| Flag | Default | Purpose |
|---|---|---|
| `--memory` | `120g` | Hard container ceiling; leaves ~1 GB for the host OS kernel/driver |
| `--memory-swap` | `128g` | Combined memory+swap ceiling; allows ~8 GB swap spill before hard kill |

Both are overridable via environment variables:

```bash
DOCKER_MEMORY_LIMIT=120g DOCKER_MEMORY_SWAP=128g RUN_NAME=experiment bash scripts/run_train.sh
```

### PyTorch allocator tuning

`PYTORCH_CUDA_ALLOC_CONF` is set in `run_train.sh`:

```
expandable_segments:True,max_split_size_mb:512
```

- `expandable_segments:True` — the allocator grows memory incrementally rather than reserving large contiguous blocks up front; significantly reduces peak RSS and fragmentation
- `max_split_size_mb:512` — prevents large cached blocks from being fragmented into sub-512 MB pieces; reduces the chance of failing a large allocation even when total free memory is sufficient

If training is still OOM after these settings, the next lever is `max_seq_length` in `configs/nemotron.yaml`. At `seq_len=8192` the activation memory for a full-length sample is the dominant cost; dropping to `4096` approximately halves it and still covers the data's P99 (the huikang corpus P99 is well under 8192 real tokens).

---

## run_prepare.sh — one-time 4-bit cache

`run_prepare.sh` quantizes the 30B BF16 model to 4-bit NF4 and saves it to `.cache/nemotron_4bit/`. Once the `.ready` sentinel exists, `run_train.sh` loads the 15 GB cache in ~1 min instead of re-doing the 6-min BF16 load every run.

### Why prepare has no memory limit

`run_prepare.sh` intentionally omits `--memory` / `--memory-swap`. Two reasons:

1. **`--ulimit memlock=-1` + a hard memory cap = silent OOM kill.** The ulimit allows CUDA to pin unlimited memory in RAM (no swap fallback). When the cap is hit the cgroup kills the container instantly — no traceback, no `OOMKilled` flag, the progress bar just stops. This is what caused repeated deaths at 72–94%.

2. **Prepare has no training overhead.** No gradients, no optimizer states, no activation buffers. The only spike is during BF16→4-bit conversion. The host has 120 GB free; removing the cap is safe.

### `low_cpu_mem_usage=True` — cuts the BF16 loading peak

Without this flag, `from_pretrained` loads each ~5 GB safetensors shard fully into memory before quantizing — 13 shards means up to ~5 GB of BF16 is held alongside the accumulating 4-bit model.

With `low_cpu_mem_usage=True` the meta-device path is used instead: the model is initialised as an empty shell and weights stream in **one tensor at a time**, so the peak BF16 footprint is a few hundred MB (the largest single tensor) rather than a full shard. This was the fix that allowed 401/401 tensors to load where 94% had been the previous ceiling.

> **Note:** `low_cpu_mem_usage=True` is set in both `prepare_quantized_model.py` and `train_lora.py`. Do not remove it.

### Swap extension — last-resort burst headroom

If the host still OOM-kills prepare (the final few tensors — `lm_head`, last SSM layers — are the largest), extend swap before running:

```bash
sudo fallocate -l 32G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
```

This adds ~32 GB of burst headroom for the loading peak. The swap is used only for the transient BF16 spike; once loading completes and the shard is released, the swapped pages are reclaimed. Remove after the cache is built:

```bash
sudo swapoff /swapfile2 && sudo rm /swapfile2
```

---

## Build the image

The primary image is built from `Dockerfile.gb10` (NVIDIA PyTorch 26.04):

```bash
bash scripts/build_image.sh                  # primary image → nemotron-gb10:latest
bash scripts/build_image.sh 26-01            # archived 26.01 image → nemotron-gb10-26-01:latest
bash scripts/build_image.sh <variant> --fresh  # force-recompile mamba/causal-conv1d
```

**Always build directly on the GB10.** Do not import images built on x86_64 — CUDA `.so` files carry the wrong platform tag (`x86_64-linux-gnu`) and Python silently ignores them on aarch64, causing `ModuleNotFoundError` at runtime.

### Known build quirks

| Issue | Root cause | Fix in Dockerfile |
|---|---|---|
| `mamba_ssm` `ModuleNotFoundError` at runtime | `selective_scan_cuda` `.so` in base image is x86_64; GPU is also unavailable in Docker `RUN` steps so source build skips it too | `try/except` patch on `selective_scan_interface.py`; safe because Nemotron-H uses Mamba-2 Triton kernels exclusively |
| `causal_conv1d` undefined symbol `decref_pyobject` (25.12 only) | Symbol added to standard PyTorch but absent from nv25.12 libtorch ABI | Use causal_conv1d 1.5.0.post8 (archived in `Dockerfile.gb10-25-12`) |
| `causal_conv1d` build fails on CUDA 13 (25.12 only) | 1.5.x hardcodes old arches (`compute_53/62`) which CUDA 13 dropped | Patch `setup.py` arch block (in `Dockerfile.gb10-25-12`) |
| OOM during `cmake` / pip source builds | `-j$(nproc)` spawns 72 jobs on the Grace CPU | `MAX_JOBS=8` and `-j8` set in all Dockerfiles |

---

## Pause and resume other services

Long training runs (~10 hours) benefit from stopping unrelated containers first to eliminate any
background CPU/IO competition and leave a clean memory baseline.

```bash
bash scripts/services.sh pause    # stop all non-training containers; save names to .paused_containers
RUN_NAME=huikang_v4 bash scripts/run_train.sh
bash scripts/services.sh resume   # restart every container that was running before the pause
```

`pause` snapshots whatever containers are running at that moment (excluding any `nemotron-gb10`
training container) and stops them. `resume` reads that snapshot and starts each one back up,
then deletes the state file. Both commands are no-ops if there is nothing to act on.

The state file `.paused_containers` is gitignored.

---

## Launch a training run

```bash
RUN_NAME=huikang_v4 bash scripts/run_train.sh
```

Logs go to `output/train_<RUN_NAME>_<timestamp>.log`. The adapter is saved to `output/adapter_<RUN_NAME>_<timestamp>/`.

Config is loaded from `configs/nemotron.yaml` via `scripts/load_config.sh`. Override individual settings inline:

```bash
RUN_NAME=test LEARNING_RATE=1e-5 MAX_SEQ_LENGTH=4096 bash scripts/run_train.sh
```

### Signs of OOM kill

A silent death (no Python traceback in the log, progress bar stops mid-step) is the OOM kernel kill pattern. Check:

```bash
dmesg | grep -i "oom\|killed process" | tail -20
```

---

## Force-stop a stuck container

`docker stop` can hang or be denied on containers launched with `--privileged`. Escalate:

```bash
# 1. Find the host PID
docker inspect <container-id> --format '{{.State.Pid}}'

# 2. Graceful SIGTERM
sudo kill <pid>

# 3. Force kill if still alive after ~10 s
sudo kill -9 <pid>
```

---

## Related documentation

| Document | Notes |
|---|---|
| [`.clinerules/14-docker-gpu-gb10.md`](../../.clinerules/14-docker-gpu-gb10.md) | Canonical rule: always use `--privileged -e NVIDIA_VISIBLE_DEVICES=all` |
| [`.clinerules/13-docker-stop-failed.md`](../../.clinerules/13-docker-stop-failed.md) | Force-stop procedure |
| [`docs/plans/archive/pytorch-container-migration-plan.md`](../plans/archive/pytorch-container-migration-plan.md) | 25.12 → 26.01 → 26.04 migration history |
| [`docs/plans/archive/dockerfile-gb10-review.md`](../plans/archive/dockerfile-gb10-review.md) | Initial Dockerfile audit |
| [`docs/plans/archive/dockerfile-gb10-proposed.md`](../plans/archive/dockerfile-gb10-proposed.md) | Proposed changes that became the current Dockerfile |
