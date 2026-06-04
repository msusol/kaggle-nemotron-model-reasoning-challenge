# Loading nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 on DGX Spark (GB10)

**Forum thread:** https://forums.developer.nvidia.com/t/loading-nvidia-nvidia-nemotron-3-nano-30b-a3b-bf16-on-dgx-spark/372168

**Platform:** DGX Spark, Blackwell GB10, aarch64, Ubuntu 24.04  
**GPU HBM:** 130.7 GB (separate from 121 GB CPU LPDDR5x — NOT unified despite marketing)  
**CUDA:** 13.2 Forward Compatibility mode (driver 580.x, kernel 580.159.03)  
**Use case:** LoRA fine-tuning for the NVIDIA Nemotron Model Reasoning Challenge (Kaggle)

---

## 1. Docker flags — `--gpus all` does not work

The standard NVIDIA container runtime fails on GB10:

```bash
# These all fail with:
# "failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: no such file"
--gpus all
--runtime=nvidia
--device nvidia.com/gpu=all
```

Working pattern:

```bash
docker run --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ...
```

- `--privileged` grants full host capabilities, bypassing the failing bind-mount
- `NVIDIA_VISIBLE_DEVICES=all` tells the runtime which GPUs to expose
- `--ipc=host` shares the host IPC namespace (required for multi-process GPU comms)
- `--ulimit memlock=-1` removes the locked-memory ceiling (needed for CUDA pinned allocations)

---

## 2. Base image and Python packages

**Base image:** `nvcr.io/nvidia/pytorch:26.04-py3` (aarch64)

**Pinned packages — exact versions matter:**

```dockerfile
RUN pip install \
    "transformers==5.5.3" \
    "peft==0.14.0" \
    "trl==0.15.2" \
    "accelerate==1.3.0"
```

**Why `transformers==5.5.3` specifically:** This is the first version with native
`NemotronHForCausalLM` support including the KV-cache fix. Do NOT use `trust_remote_code=True`
— it pulls the model's bundled code which has the old broken KV-cache implementation.

### Source builds required — causal-conv1d and mamba-ssm

Docker `RUN` steps run in isolated OCI containers with no GPU access, so `mamba-ssm`
silently skips building `selective_scan_cuda.so` at install time. The extension is absent
from the installed package and fails at runtime.

```dockerfile
# causal-conv1d — must force-build, must include sm_120 and sm_121 (Blackwell)
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0;12.0;12.1+PTX" \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=8 \
    pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation

# mamba-ssm — same arch list; patch selective_scan_cuda import after build
RUN TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9;9.0;12.0;12.1+PTX" \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=8 \
    pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation \
 && python3 -c "
f='/usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/selective_scan_interface.py'
c=open(f).read()
needle='\nimport selective_scan_cuda\n'
open(f,'w').write(c.replace(needle,
  '\ntry:\n    import selective_scan_cuda\nexcept ImportError:\n    selective_scan_cuda = None\n',
  1)) if needle in c else None
print('selective_scan_cuda patched OK')"
```

**Why the patch is safe:** Nemotron-H uses Mamba-2 Triton kernels exclusively for its
forward pass and never calls the legacy Mamba-1 `selective_scan_cuda` CUDA path. Wrapping
the import in `try/except` is a no-op at runtime — the extension simply isn't called.

**`MAX_JOBS=8`:** Without this, `cmake` spawns one job per CPU core. The DGX Spark has
72 ARM cores — spawning 72 parallel compile jobs during `docker build` exhausts memory
and kills the build. `MAX_JOBS=8` keeps it sane.

### bitsandbytes — source build for Blackwell sm_120

Pre-built bitsandbytes wheels don't include Blackwell compute capabilities:

```dockerfile
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git /opt/bitsandbytes \
 && cd /opt/bitsandbytes \
 && cmake -DCOMPUTE_BACKEND=cuda \
          -DCOMPUTE_CAPABILITY="80;86;87;89;90;120;121" \
          -S . -B build \
 && cmake --build build --config Release -j8 \
 && pip install . --no-build-isolation
```

---

## 3. Model loading code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    dtype=torch.bfloat16,       # NOT torch_dtype= (deprecated in transformers 5.5.3)
    device_map={"": 0},         # single GPU, explicit device assignment
    low_cpu_mem_usage=True,     # REQUIRED — see note below
)
```

**`low_cpu_mem_usage=True` is critical.** Without it, `from_pretrained` first allocates
the full model as empty CPU tensors (~57 GB), then streams the safetensors data into them.
The empty model + mmap page cache + partial GPU allocation stack up to well over 121 GB
even halfway through loading → OOM.

With `low_cpu_mem_usage=True` the meta-device path is used: the model is initialised as
an empty shell with no backing memory, and weights stream in one tensor at a time directly
to their final device location. This eliminates the empty-model spike.

**Do not use `trust_remote_code=True`** — transformers 5.5.3 has native `NemotronHForCausalLM`
support. Using `trust_remote_code` pulls the model's bundled code which has the old broken
KV-cache implementation and will cause incorrect inference results.

---

## 4. CUDA allocator tuning

Pass as an environment variable to the container:

```bash
-e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
```

- `expandable_segments:True` — allocator grows memory incrementally rather than reserving
  large contiguous blocks up front; reduces peak RSS and fragmentation
- `max_split_size_mb:512` — prevents large cached blocks from being fragmented into
  sub-512 MB pieces; reduces the chance of failing a large allocation even when total
  free memory is sufficient

---

## 5. Memory architecture — two separate pools

Despite the "unified memory" marketing, the GB10 has **two distinct physical memory pools**:

| Pool | Size | What lives here |
|---|---|---|
| GPU HBM (Blackwell) | **130.7 GB** (via `torch.cuda.mem_get_info()`) | CUDA tensors, model weights, activations |
| CPU LPDDR5x | **~121 GB** (via `free -h`) | Linux page cache, Python heap, Docker overlays |

These pools are linked by NVLink-C2C but do NOT compete for the same physical memory.
`free -h` shows only the CPU-side pool — it tells you nothing about GPU HBM availability.
`torch.cuda.mem_get_info()` is the correct tool to check GPU memory.

---

## 6. Pre-loading memory cleanup — the hard part

**The failure pattern:** After any SIGKILL'd container (OOM, crash, `docker stop`), the
CUDA driver does not release HBM allocations. They persist as orphaned allocations completely
invisible to `nvidia-smi --query-compute-apps` (which only shows running processes).
Each subsequent run starts with less available HBM:

| Attempt | HBM available | % of model loaded before OOM |
|---|---|---|
| Clean start | ~122 GB | 100% ✓ |
| After 1 killed run | ~116 GB | ~97% |
| After 2 killed runs | ~110 GB | ~86% |
| After 3 killed runs | ~103 GB | ~73% |

**Required cleanup sequence before every training run:**

```bash
# Step 1: Stop gnome-remote-desktop — holds ~6 GB HBM CUDA context
systemctl --user stop gnome-remote-desktop.service

# Step 2: Remove stale container
# Driver releases orphaned HBM when the next CUDA context initialises
docker rm -f <previous-container-name> 2>/dev/null || true

# Step 3: Drop Linux page cache + reset swap + tune VM
# Run inside a privileged Alpine container (avoids needing sudo on host)
sync
docker run --rm --privileged -v /:/host alpine sh -c '
  echo 3 > /proc/sys/vm/drop_caches
  echo 1048576 > /proc/sys/vm/min_free_kbytes
  echo 500 > /proc/sys/vm/vfs_cache_pressure
  swapoff /host/swap.img 2>/dev/null
  swapon /host/swap.img 2>/dev/null
  true'

# Step 4: Torch GPU preflight
# Creating a new CUDA context forces the driver to GC orphaned HBM allocations
# from any prior dead containers. Check that free HBM > 70 GB before loading.
docker run --rm --privileged -e NVIDIA_VISIBLE_DEVICES=all <your-image> python3 -c "
import torch
torch.cuda.init()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
used = total - free
print(f'GPU free={free/1e9:.1f}GB  total={total/1e9:.1f}GB  used={used/1e9:.1f}GB')
if free < 70e9:
    print('ABORT: less than 70 GB free — stale allocations present, try rebooting')
else:
    print('OK: enough HBM to load 60 GB model')
"

# Step 5: Second drop_caches pass
# The preflight container itself adds ~8 GB (new CUDA context overhead).
# Without this second pass, training starts with that 8 GB already consumed.
sync
docker run --rm --privileged alpine sh -c '
  echo 3 > /proc/sys/vm/drop_caches
  echo 1048576 > /proc/sys/vm/min_free_kbytes
  echo 500 > /proc/sys/vm/vfs_cache_pressure'

# Step 6: Launch training with lower I/O priority
# safetensors shard reads compete with host I/O during the 6-minute loading window
ionice -c 2 -n 7 docker run --privileged \
  --oom-score-adj 300 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  ...
```

**Why `min_free_kbytes=1048576` (1 GB):** At the default (45 KB), the kernel barely
evicts file-backed pages until near-OOM. During model loading, safetensors mmap pages
accumulate in the CPU pool. Setting 1 GB tells the kernel to proactively reclaim pages,
keeping the CPU pool clean during the 6-minute load window.

**Why `vfs_cache_pressure=500`:** Makes file-backed page eviction 5× more aggressive.
Restore defaults after training: `min_free_kbytes=45166`, `vfs_cache_pressure=100`.

**Why `nvidia_uvm` module reload doesn't help:** In CUDA Forward Compatibility mode
(the GB10's default — driver 580.x with kernel module 580.x), `rmmod nvidia_uvm`
always fails. The torch preflight + `docker rm -f` is the only available cleanup path
short of a full reboot.

---

## 7. During-load CUDA cache dropper thread

Even with `low_cpu_mem_usage=True`, `from_pretrained` accumulates ~41 GB of
freed-but-cached CUDA blocks by the time loading reaches ~80%. These are temporary
tensors from dtype conversion that PyTorch has freed internally but not returned to
the driver. Without periodic `empty_cache()` calls they pile up and cause OOM at ~97%
loading despite having plenty of total HBM.

```python
import threading
import torch

def make_cache_dropper(interval: float = 20.0) -> threading.Event:
    """Background thread: drop Linux page cache + return freed CUDA blocks every interval s."""
    stop = threading.Event()

    def _loop():
        while not stop.wait(interval):
            # Drop Linux page cache (CPU pool hygiene)
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("3\n")
            except OSError:
                pass
            # Return freed CUDA allocator cache back to the driver
            try:
                free_before = torch.cuda.mem_get_info()[0]
                torch.cuda.empty_cache()
                free_after = torch.cuda.mem_get_info()[0]
                reclaimed = (free_after - free_before) / 1e9
                alloc = torch.cuda.memory_allocated() / 1e9
                print(f"[dropper] reclaimed={reclaimed:+.1f}GB  alloc={alloc:.1f}GB  "
                      f"free={free_after/1e9:.1f}GB", flush=True)
            except Exception:
                pass

    t = threading.Thread(target=_loop, daemon=True, name="cache-dropper")
    t.start()
    return stop


# Usage:
_stop = make_cache_dropper(interval=20.0)
model = AutoModelForCausalLM.from_pretrained(...)
_stop.set()
torch.cuda.empty_cache()
free_gb = torch.cuda.mem_get_info()[0] / 1e9
print(f"Model loaded. GPU free={free_gb:.1f}GB")
```

---

## 8. Gradient checkpointing bypass (fine-tuning)

`NemotronHForCausalLM.supports_gradient_checkpointing = False` is set at the class level,
which causes TRL/HF to raise `ValueError` when `gradient_checkpointing=True` is passed.

However, every `NemotronHBlock` inherits `GradientCheckpointingLayer` which fully
implements gradient checkpointing. The flag guard is the only blocker. Bypass it directly:

```python
import functools
import torch

_gc_func = functools.partial(
    torch.utils.checkpoint.checkpoint,
    use_reentrant=False
)
model.base_model.model._set_gradient_checkpointing(
    enable=True,
    gradient_checkpointing_func=_gc_func
)
model.enable_input_require_grads()

# IMPORTANT: keep gradient_checkpointing=False in SFTConfig / TrainingArguments
# so TRL does not call gradient_checkpointing_enable() again and hit the ValueError.
```

This reduces activation memory from ~20–40 GB to ~1–5 GB at `seq_len=6144`, which is
the difference between OOM at the first training step and stable training throughout.

---

## 9. Mamba fast path (optional but recommended for training speed)

After the model loads, enable the Mamba CUDA fast path:

```python
import sys

for name, mod in sys.modules.items():
    if "modeling_nemotron_h" in name and hasattr(mod, "is_fast_path_available"):
        mod.is_fast_path_available = True
        print("Mamba fast path enabled")
        break
```

---

## 10. Complete memory budget

| Component | HBM |
|---|---|
| Base model BF16 | ~60 GB |
| CUDA context overhead | ~8 GB |
| LoRA adapter r=32 (standard PEFT, attention only) | ~0.1 GB |
| LoRA adapter r=32 (Unsloth, all layers incl. 128 MoE experts) | ~1.5 GB |
| Training activations at seq=6144 with gradient checkpointing | ~5–10 GB |
| **Total at training step (Unsloth)** | **~75–80 GB** |
| **Available (130.7 GB − 8 GB context)** | **~122 GB** |
| **Headroom** | **~42–47 GB** |

The model loads and trains comfortably on DGX Spark once the memory cleanup sequence
is followed. Failures happen when stale CUDA allocations from prior killed containers
reduce available HBM below the ~75 GB needed.

---

## 11. LoRA fine-tuning — Unsloth required for full MoE coverage

Standard PEFT (`get_peft_model`) cannot add LoRA to Nemotron-H's MoE expert layers.
The 128 experts per MoE block are stored as batched 3-D `torch.Parameter` tensors, not
`nn.Linear` modules. PEFT's `named_modules()` walk misses them entirely — silently,
with no error.

**Unsloth** (`FastLanguageModel.from_pretrained`) patches the model before `get_peft_model`
runs, replacing the batched expert tensors with individual `nn.Linear`-like wrappers.
This exposes all 128 experts as trainable LoRA targets.

Without Unsloth: ~116 LoRA modules (27M trainable params, attention layers only)  
With Unsloth: ~6,004 LoRA modules (883M trainable params, all layers including MoE)

**Install note:** Unsloth requires GPU at import time — `from unsloth import FastLanguageModel`
fails in Docker `RUN` steps. Install with `--no-deps` and verify with `pip show` at build
time; the actual import works at runtime when the GPU is accessible.

```dockerfile
RUN pip install unsloth unsloth_zoo --no-deps && \
    pip install \
        "transformers==5.5.3" \
        "peft==0.14.0" \
        "trl==0.15.2" \
        "accelerate==1.3.0" \
        --force-reinstall --no-deps && \
    pip show unsloth unsloth_zoo | grep -E "^Name|^Version"
# Note: `from unsloth import FastLanguageModel` cannot be tested here —
# unsloth_zoo runs a GPU presence check at import time (device_type.py)
# which raises NotImplementedError in GPU-less build containers.
# Import is guarded by try/except in training script and succeeds at runtime.
```

---

## Summary checklist

- [ ] Use `--privileged -e NVIDIA_VISIBLE_DEVICES=all` (not `--gpus all`)
- [ ] Build `causal-conv1d` and `mamba-ssm` from source with `TORCH_CUDA_ARCH_LIST` including `12.0;12.1+PTX`
- [ ] Patch `selective_scan_interface.py` after `mamba-ssm` install
- [ ] Pin `transformers==5.5.3` — native NemotronH, no `trust_remote_code`
- [ ] Load with `dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True`
- [ ] Set `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"`
- [ ] Before each run: stop GRD, `docker rm -f` stale container, `drop_caches` ×2, torch preflight
- [ ] Wrap `from_pretrained` with cache dropper thread (20 s interval)
- [ ] Bypass gradient checkpointing flag via `_set_gradient_checkpointing()` directly
- [ ] Use Unsloth `FastLanguageModel` if fine-tuning with LoRA on MoE layers
