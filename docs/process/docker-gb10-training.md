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

### Memory architecture — two separate pools

The GB10 has **two distinct memory pools** confirmed via `torch.cuda.mem_get_info()`:

| Pool | Size | What lives here |
|---|---|---|
| Linux RAM (LPDDR5x) | ~121 GB visible to OS | page cache, Python heap, Docker overlays |
| GPU HBM (Blackwell) | ~130.7 GB visible to CUDA | model tensors, CUDA contexts, activations |

These are **separate physical memories** linked by NVLink-C2C. Loading 57 GB to GPU draws from the HBM pool and does **not** reduce Linux RAM. The mmap page cache from safetensors shards goes into Linux RAM and does **not** reduce GPU memory. They do not directly compete.

This corrects an earlier assumption (documented before `torch.cuda.mem_get_info()` was run) that both pools shared the same 121 GB unified DRAM. The Linux `free -h` total (~121 GB) reflects only the CPU-side LPDDR5x; CUDA's 130.7 GB is the Blackwell HBM.

### Why the model dies at ~70–98% of weight loading

The failure point varies (70%, 81%, 86%, 98%) and the root cause is **stale CUDA allocations in the GPU HBM pool**, not page cache pressure.

When a training container is SIGKILL'd (e.g., OOM at training step 0), the NVIDIA driver does not immediately release its GPU HBM allocations. The dead container's model weights (~57 GB) and any training-step allocations stay resident in HBM as orphaned allocations. Each subsequent loading attempt uses a GPU with less available HBM:

| Run | GPU free at start | % loaded before OOM |
|---|---|---|
| Clean start | ~124 GB | 100% ✓ |
| After 1 SIGKILL'd run | less | ~86% |
| After 2 SIGKILL'd runs | less | ~81% |
| After 3 SIGKILL'd runs | less | ~70% |

The orphaned allocations accumulate across Docker container restarts because `docker run --rm` cleans up the container filesystem but does not reset CUDA driver state. A new CUDA context initialization (e.g., a fresh `torch.cuda.init()` call in a new container) forces the driver to GC orphaned allocations from dead processes.

**No Python traceback appears** because the OOM kill is issued by the CUDA driver or kernel before Python can print anything.

### GPU pre-flight — required after any SIGKILL'd training run

`run_train.sh` and `run_train_v5.sh` run a throwaway container before loading to force CUDA driver GC:

```python
import torch
torch.cuda.init()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
used = total - free
print(f'GPU free={free/1e9:.1f}GB total={total/1e9:.1f}GB used={used/1e9:.1f}GB')
if free < 70e9:
    print(f'PREFLIGHT_FAIL only {free/1e9:.1f}GB free — model load (60 GB) will OOM')
elif used > 20e9:
    print(f'PREFLIGHT_FAIL {used/1e9:.1f}GB stale GPU allocations — nvidia_uvm reload may help')
elif free < 90e9:
    print(f'WARNING: {free/1e9:.1f}GB free — some stale allocs present but training should fit')
```

This container initialises and immediately exits cleanly. Creating a new CUDA context forces the driver to GC orphaned HBM allocations from dead containers. It also prints the GPU baseline so the log shows how much HBM is available before loading.

**Expected baseline** (clean system, GRD stopped): `GPU free=~120GB total=130.7GB used=~10GB`

The `used` reading on a clean system is the preflight container's own CUDA context (~8–10 GB). This is the unavoidable minimum — every CUDA context costs ~8 GB of HBM overhead regardless of model size.

**Threshold rationale:**

| Check | Threshold | Meaning |
|---|---|---|
| `free < 70 GB` | FAIL | Model (60 GB) + startup overhead won't fit — definitive OOM |
| `used > 20 GB` | FAIL | >10 GB of orphaned allocs beyond context overhead — `nvidia_uvm` reload attempted |
| `free < 90 GB` | WARNING | Some stale allocs present but training (60 GB + ~15 GB) should still fit |

**Previous threshold (`used > 12 GB`) was wrong.** It was based on the (incorrect) assumption that GPU HBM and Linux page cache share the same 128 GB unified pool. In reality they are separate: HBM is 130.7 GB; Linux RAM is 121 GB. A small residue of orphaned HBM allocations (~6 GB from a previous SIGKILL'd run) that cannot be cleared by `nvidia_uvm` reload (fails in CUDA Forward Compat mode) is harmless as long as `free > 70 GB`.

**`nvidia_uvm` reload always fails in CUDA Forward Compatibility mode** (the GB10's default mode since it uses a newer driver than the kernel module). A reboot is the only way to clear truly stuck orphaned allocations. In practice, the allocations are small enough (~6 GB) that training proceeds normally.

### `low_cpu_mem_usage=True` — required for BF16 loading

Without this flag, `from_pretrained` first constructs the full model as empty (zero-filled) CPU tensors (~57 GB), then streams the safetensors data into those tensors. The empty model + the mmap page cache + the partial GPU allocation stack up to well over 121 GB even in the first half of loading.

With `low_cpu_mem_usage=True` the meta-device path is used: the model is initialised as an empty **shell with no backing memory**, and weights stream in one tensor at a time directly to their final device location. This eliminates the empty-model spike and moves the ceiling from ~76% to ~98% of loading — but the dual-pressure problem (GPU + mmap) is not fully resolved by this flag alone.

> **Note:** `low_cpu_mem_usage=True` is set in `train_lora.py`. Do not remove it.

### Pre-training memory clearing sequence — full ordered steps

The order of operations matters. Running these steps out of order can cause spurious
preflight passes (the preflight sees stale allocations that were already removed) or leave
the preflight container's own CUDA context in memory when training starts.

Correct sequence (implemented in both `run_train.sh` and `run_train_v5.sh`):

| Step | Command | Why |
|---|---|---|
| 1. Stop gnome-remote-desktop | `systemctl --user stop gnome-remote-desktop.service` | Frees ~6 GB GPU HBM held by its CUDA context |
| 2. Pause services | `bash scripts/services.sh pause` | Stops NIM/other containers; reduces CPU/IO competition |
| 3. Stage-1 GPU check | `nvidia-smi --query-compute-apps` | Fast check — catches any running compute process > 1 GB |
| 4. Remove stale container | `docker rm -f nemotron-trainer[-v5]` | Releases the dead container's filesystem overlay; must happen **before** preflight so preflight sees the post-cleanup GPU state, not pre-cleanup |
| 5. `drop_caches` pass 1 | `echo 3 > /proc/sys/vm/drop_caches` | Clears NIM/stopped-container file-backed pages from Linux RAM so the torch preflight threshold check isn't confused by RAM pressure |
| 6. Torch preflight | `torch.cuda.init(); torch.cuda.empty_cache()` | Initialising a new CUDA context forces the driver to GC orphaned HBM allocations from dead containers; 12 GB threshold |
| 7. `nvidia_uvm` reload (if needed) | `rmmod nvidia_uvm && modprobe nvidia_uvm` | Last resort if preflight still reports > 12 GB used after step 4–6 |
| 8. `drop_caches` pass 2 | `echo 3 > /proc/sys/vm/drop_caches` | Clears the ~8 GB CUDA context the preflight container itself added; this is the definitive baseline training will see |
| 9. Start training | `ionice -c 2 -n 7 docker run ...` | `ionice` lowers I/O scheduling priority; safetensors shard reads compete less aggressively with other host I/O |

**Why two `drop_caches` passes?** The first pass clears the pre-existing page cache so
the preflight threshold check is not confused. The preflight container then allocates a
fresh CUDA context (~8 GB HBM) and re-reads some model files. The second pass clears
those additions. Without it, training starts with an extra ~8 GB already consumed.

**Why `docker rm -f` before preflight?** A SIGKILL'd container holds 6–8 GB of driver-level
HBM allocations. `docker rm -f` removes the container record; the driver then releases
those allocations when the preflight opens a new CUDA context. If you run the preflight
first, it sees the stale allocations, may trigger the `nvidia_uvm` reload unnecessarily,
and then `docker rm -f` removes allocations the reload already handled — wasted steps.

### Page cache drop — Linux RAM hygiene

Each `drop_caches` pass (steps 5 and 8 above) is run via a privileged Alpine container:

```bash
docker run --rm --privileged -v /:/host alpine sh -c \
  'echo 3 > /proc/sys/vm/drop_caches \
   && echo 1048576 > /proc/sys/vm/min_free_kbytes \
   && echo 500 > /proc/sys/vm/vfs_cache_pressure \
   && swapoff /host/swap.img 2>/dev/null; swapon /host/swap.img 2>/dev/null; true'
```

- `drop_caches=3` — drops page cache, dentries, and inodes
- `min_free_kbytes=1048576` (1 GB) — nudges the kernel to proactively reclaim file-backed
  pages before they crowd out CUDA allocations during the loading window
- `vfs_cache_pressure=500` — makes file-backed page eviction 5× more aggressive
- `swapoff/swapon` — recycles swap to reclaim pages pushed there by prior OOM-killed runs

Defaults are restored after training: `min_free_kbytes=45166`, `vfs_cache_pressure=100`.

**Swap accumulation** from prior OOM-killed runs can push a few GB of pages to disk. If the
`swapoff/swapon` inside the Alpine container fails (e.g., swap file path differs), run manually:

```bash
sudo swapoff -a && sudo swapon -a
```

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

### v0.4 (huikang corpus SFT) — `run_train.sh`

```bash
RUN_NAME=huikang_v4 bash scripts/run_train.sh
```

Logs go to `output/train_<RUN_NAME>_<timestamp>.log`. The adapter is saved to `output/adapter_<RUN_NAME>_<timestamp>/`.

Config is loaded from `configs/nemotron.yaml` via `scripts/load_config.sh`. Override individual settings inline:

```bash
RUN_NAME=test LEARNING_RATE=1e-5 MAX_SEQ_LENGTH=4096 bash scripts/run_train.sh
```

### v0.5 (kuangyicheng warmstart SFT) — `run_train_v5.sh`

```bash
# Always inside a tmux session — survives SSH disconnect
tmux new -s train_v5    # or: tmux attach -t train_v5
RUN_NAME=v5_sft bash scripts/run_train_v5.sh
```

Logs go to `output/train_v5_sft.log`. Adapter saved to `output/adapter_v5_sft/`.

Key differences from `run_train.sh`:
- Warmstarts from `output/adapter_huikang_v27/` (huikang v27 PEFT adapter)
- Invokes `scripts/train_v5_sft.py` (not `train_lora.py`)
- 240 steps at `max_seq_length=6144` (short responses, not full huikang corpus)
- All memory clearing steps identical to `run_train.sh` (full preflight, two `drop_caches` passes)

**Prerequisite:** `output/adapter_huikang_v27/` and `data/v0.5_train.jsonl` must exist.
Generate the dataset: `python scripts/prepare_v5_sft_data.py`

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
