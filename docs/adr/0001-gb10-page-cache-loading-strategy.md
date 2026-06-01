# ADR-0001 — GB10 Page Cache Strategy for Model Loading

**Status:** Accepted

## Context

The GB10 (Grace-Blackwell Superchip) has a **unified memory architecture**: a single 130.7 GB
LPDDR5x pool shared between the Linux kernel (page cache, heap, stack) and the NVIDIA GPU
(CUDA allocations). Each allocator is blind to the other's usage.

Loading the 30B-parameter Nemotron model in BF16 requires approximately 60 GB of CUDA
allocations. The safetensors shards on NVMe also consume approximately 60 GB of Linux page
cache during and after a load (~7 minutes from NVMe).

This creates a resource conflict on every subsequent run:

| Item | Size |
|---|---|
| Shard pages (page cache from prior run) | ~60 GB |
| CUDA model allocation (current run) | ~60 GB |
| Kernel / other overhead | ~6 GB |
| **Total peak** | **~126 GB** |
| **Pool size** | **130.7 GB** |

With only ~4–5 GB headroom the OOM killer fired reproducibly at ~80% of weight loading
(`adapter_huikang_v4_20260531_111936`, `20260531_085517`, `20260531_080801`, overnight
`20260531_001134`). `torch.cuda.mem_get_info()` reports page cache as "free" because CUDA
is blind to it, giving a false sense of safety.

## Decision

Use `echo 3 > /proc/sys/vm/drop_caches` in `scripts/run_train.sh` before each training run.

`echo 3` flushes **all** reclaimable page cache (including the ~60 GB shard pages) plus
dentries and inodes. This reduces actual memory in use to ~6 GB before CUDA loading begins,
giving ~60+ GB of comfortable headroom during the 60 GB weight allocation.

The consequence is that every run reloads weights from NVMe (~7 minutes). This is accepted
because:
- Training itself takes 2–3 hours; 7 minutes is ~5% overhead.
- OOM failures during loading waste more time than the reload itself.
- The alternative (persistent container) has not yet been implemented.

## Alternatives Considered

### `echo 1` (drop page cache only) — **rejected**
Same as `echo 3` for shard eviction but leaves dentries/inodes. Tried in commit `ac1646c`
with a confusing commit message claiming it "keeps file pages" — it does not (`echo 1` drops
file-backed pages; `echo 2` would keep them). The mislabelling led to repeated confusion.

### `echo 2` (drop dentries/inodes, keep file pages) — **rejected for now**
Would keep the ~60 GB shard pages warm for a fast reload (~1 min from RAM). However, on
GB10 unified memory these warm pages compete with CUDA allocations for the same physical
pool. With `vfs_cache_pressure=500` the kernel should evict them under pressure, but
empirically it cannot keep up with the rate of CUDA allocation during loading. OOM fires
consistently at ~80% of weight loading. This approach may be revisited if a reliable
`POSIX_FADV_DONTNEED` advise step can be added after each forward-copy step in transformers.

### In-process page-cache dropper thread — **rejected**
An earlier approach (commits `12adcc9`→`b5ff533`) ran a daemon thread in `train_lora.py`
calling `echo 3` on `/proc/sys/vm/drop_caches` every 20 seconds during `from_pretrained`.
This works but defeats any warm-cache strategy, adds complexity to the training script, and
requires `/proc` access from inside the container. Removed in `cf672d8`. The single `echo 3`
at the shell level before `docker run` achieves the same result more cleanly.

### Persistent Docker container — **deferred**
Keep `nemotron-trainer` running between experiments and use `docker exec` to call
`train_lora.py`. The model stays in CUDA; reload time drops to zero. This is the right
long-term solution but requires workflow changes (start script, exec-based run_train.sh
mode) not yet implemented. See TODO Phase 5/6.

## Consequences

- Each training run costs ~7 minutes of NVMe reload before training starts.
- Loading is reliable: run `235341` (first successful 5-step v0.4 run) used this strategy.
- The gradient checkpointing fix (`gradient_checkpointing=False`, commit `ac1646c`) is
  orthogonal and retained — NemotronHForCausalLM does not implement
  `gradient_checkpointing_enable()`.
- If the persistent-container approach is implemented, this ADR should be superseded.
