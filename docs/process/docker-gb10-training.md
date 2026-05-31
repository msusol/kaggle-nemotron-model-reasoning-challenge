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

### Memory limits in `run_train.sh`

`run_train.sh` intentionally omits `--memory` / `--memory-swap` for the same reason as `run_prepare.sh`:

- **`--ulimit memlock=-1` + a cgroup cap = silent OOM kill.** CUDA pins all model weights in RAM (no swap fallback). Safetensors mmap also holds all 13 HF shards open simultaneously (~60 GB of page cache), which is charged to the container cgroup on top of the model allocation (~57 GB). The combined total peaks near or above any reasonable cap, and the cgroup kills the container with no traceback.
- **The kernel manages eviction correctly without a cap.** Training uses ~57 GB (model) + ~10 GB (activations/LoRA/optimizer) = ~67 GB, leaving ~54 GB for the host OS and desktop.

If you need to cap memory for a specific experiment, pass the flags directly:

```bash
docker run ... --memory 115g --memory-swap 120g ...
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

## run_prepare.sh — DISABLED (broken quantized cache)

`run_prepare.sh` is currently disabled and will exit with an error if invoked.

**Root cause:** `bitsandbytes load_in_4bit` only quantizes standard `nn.Linear` layers. Nemotron-H (30B-A3B MoE) stores its expert weight tensors as batched 3-D tensors, not `nn.Linear` modules, so they remain BF16. The resulting cache is ~57 GB instead of the expected ~15 GB — no useful memory saving, and worse OOM behavior than loading from the original 13-shard HF model directly.

**Do not re-enable until** `prepare_quantized_model.py` is fixed to quantize expert layers explicitly. See `docs/investigate/` for analysis.

### Why the model dies at ~76–98% of weight loading

During BF16 loading, two allocations grow simultaneously and compete for the same 121 GB pool:

| Allocation | Source | Size at 100% load |
|---|---|---|
| GPU (unified memory) | tensors placed on device as each is loaded | ~57 GB |
| Kernel page cache | safetensors `mmap` pages from the 13 HF shards | up to ~57 GB |

Both are backed by physical RAM on the GB10 unified-memory architecture. The GPU allocation is **pinned** (cannot be swapped or evicted). The mmap pages are evictable, but the kernel does not evict them fast enough when the demand curve is smooth — it prefers to serve pages from cache rather than evict clean pages proactively.

**At ~98% loaded (~56 GB on GPU) the combined pressure approaches 113 GB**, which leaves only ~8 GB for the OS, desktop, and driver. If anything else competes for that headroom — accumulated swap from prior crashed runs, CUDA driver workspace, a background process paging in — the OOM killer fires with no Python traceback.

**This is NOT about the last shard being an extra or optional.** All 13 shards / 401 tensors are required. The death near the end of loading is purely a memory arithmetic problem: GPU + mmap page cache ≈ 2× model size, and 2 × 57 GB = 114 GB sits right at the 121 GB ceiling.

The failure point varies (73%, 76%, 98%) depending on prior system state — accumulated swap from crashed runs, residual page cache, CUDA driver state — not on which tensors are loading.

### `low_cpu_mem_usage=True` — required for BF16 loading

Without this flag, `from_pretrained` first constructs the full model as empty (zero-filled) CPU tensors (~57 GB), then streams the safetensors data into those tensors. The empty model + the mmap page cache + the partial GPU allocation stack up to well over 121 GB even in the first half of loading.

With `low_cpu_mem_usage=True` the meta-device path is used: the model is initialised as an empty **shell with no backing memory**, and weights stream in one tensor at a time directly to their final device location. This eliminates the empty-model spike and moves the ceiling from ~76% to ~98% of loading — but the dual-pressure problem (GPU + mmap) is not fully resolved by this flag alone.

> **Note:** `low_cpu_mem_usage=True` is set in `train_lora.py`. Do not remove it.

### Page cache drop before training — required for BF16 loading

`run_train.sh` runs a throwaway privileged alpine container before the training container starts:

```bash
docker run --rm --privileged alpine \
  sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

This writes `3` to the kernel's `drop_caches` interface as root (privileged containers can write to host `/proc/sys`), which releases all clean page cache and reclaimable slabs. On a system where prior training crashes left residual page cache or accumulated swap pressure, this clears the headroom needed for mmap + GPU to coexist during the last few percent of loading.

**If the system has accumulated swap (from prior OOM-killed runs)**, interactive sudo is needed to fully reset before training:

```bash
sudo swapoff -a && sudo swapon -a          # evict all swap back to RAM, then re-enable
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'  # clear residual page cache
```

The swap accumulation pattern: each OOM-killed training container leaves some kernel-pinned pages that cannot be reclaimed without a privileged release. Over multiple crash/retry cycles this can push 3–5 GB into swap, shrinking the effective headroom for the next run.

### Swap extension — last-resort burst headroom

If loading still OOM-kills after the above steps, extend swap before running:

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
