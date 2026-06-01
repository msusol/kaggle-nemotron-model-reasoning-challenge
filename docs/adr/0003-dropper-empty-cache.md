# ADR-0003 — Add `torch.cuda.empty_cache()` to the Cache Dropper Thread

**Status:** Accepted

## Context

`scripts/train_lora.py` starts a background daemon thread (`_make_cache_dropper`)
that calls `echo 3 > /proc/sys/vm/drop_caches` every 20 seconds during
`from_pretrained`. This was introduced to prevent safetensors mmap pages from
crowding out CUDA allocations in the GB10 unified LPDDR5X pool (the OS and the CUDA
driver draw from the same physical memory). The dropper successfully addressed the
page-cache OOM pattern documented in ADR-0001/0002.

### The 2026-05-31 incident

On 2026-05-31, `RUN_NAME=huikang_v4 bash scripts/run_train.sh` was killed at 80%
weight loading (exit 137) even though the page-cache dropper was running and had
already reduced Linux `cached` to 0.1 GB. Full analysis in
`docs/investigate/v0.4-oom-loading.md`. The short summary:

PyTorch's CUDA allocator initialises a **63.2 GB pool** when `from_pretrained` starts.
As each of the 401 safetensors shards is loaded, temporary tensors (for dtype
conversion, contiguity checks, and buffer alignment) are allocated and freed, but the
allocator retains them as a freed-block cache rather than returning them via
`cudaFree`. By 80%, the `reserv − alloc` gap had grown to **41 GB**:

```
80%  cuda free=2.1GB  alloc=49.3GB  reserv=90.3GB  gap=41.0GB
     linux free=2.0GB  cached=0.1GB
     → kernel OOM kill (remaining shards need ~11.5 GB)
```

At this point, 41 GB of freed CUDA blocks were sitting in the allocator pool while
the kernel had only 2 GB available for the remaining 20% of the model. The existing
dropper had no mechanism to release those CUDA blocks.

### Why `drop_caches` alone is insufficient

`echo 3 > /proc/sys/vm/drop_caches` reclaims Linux **page-cache** (file-backed mmap
pages — the safetensors files). It has no effect on PyTorch's internal allocator
free-list. Only `torch.cuda.empty_cache()` (which calls the allocator's
`emptyCache()`) returns freed CUDA blocks to the CUDA driver, making them available
to the OS.

### Memory budget at 80% without fix

| Component | GB |
|---|---|
| CUDA reserved (pool, incl. 41 GB freed blocks) | 90.3 |
| OS + container + Python stack | ~38.4 |
| **Total used** | **128.7** |
| Physical memory free | **2.0** |
| Needed for remaining 20% of model | ~11.5 |
| **Shortfall** | **~9.5 GB** |

## Decision

Add `torch.cuda.empty_cache()` to the `_loop()` function in `_make_cache_dropper`,
called immediately after `drop_caches` and before the memory stats print:

```python
try:
    torch.cuda.empty_cache()
except Exception:
    pass
```

This is called every 20 seconds (the existing dropper interval) for the full duration
of `from_pretrained`. The `try/except` matches the style used for `drop_caches` and
ensures the dropper cannot crash the main training thread.

### Why this is safe

`torch.cuda.empty_cache()` only releases blocks in the allocator's **freed-block
cache** — tensors that have already been freed by Python and are awaiting reuse.
It never touches live (non-freed) tensors. Calling it from a separate thread while
the main thread is in `from_pretrained` is safe because PyTorch's CUDA allocator uses
internal locking.

The cost of each call is one pass over the allocator's free-list and one or more
`cudaFree` calls. At 20-second intervals during a ~6-minute load this adds no
measurable overhead.

### Expected effect

At each dropper tick, the freed-block gap (`reserv − alloc`) should collapse to near
zero. The pool expansion phase (55%–80% in the failed run) should not occur because
the accumulating freed blocks are returned to the driver before the allocator needs
to call `cudaMalloc` to expand. Projected state at 80% with fix:

| | Without fix | With fix (projected) |
|---|---|---|
| alloc | 49.3 GB | 49.3 GB |
| reserv | 90.3 GB | ~50–55 GB |
| gap | 41.0 GB | ~1–6 GB |
| Linux free | 2.0 GB | ~35–45 GB |

## Alternatives Considered

### Reduce dropper interval to 10 seconds — deferred

Halving the interval would call `empty_cache()` twice as often, potentially keeping
the gap even smaller. Deferred because 20 seconds is already sufficient to prevent
accumulation above ~3–4 GB at the observed per-shard allocation rate. If the next
run shows the gap growing materially between ticks, the interval can be reduced.

### Set `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` — rejected

The `garbage_collection_threshold` triggers automatic cache reclaim when
`allocated / reserved` exceeds the threshold. This would work but relies on a
fractional ratio that depends on the pool size, which varies across runs and model
sizes. The explicit `empty_cache()` call in the dropper is deterministic and
decoupled from allocator internals.

### Change `expandable_segments` to `False` — rejected

Disabling expandable segments prevents the allocator from expanding its pool, which
would force earlier `cudaFree` of freed blocks. However, disabling expandable segments
is known to cause increased fragmentation during training (many small alloc/free
cycles in the forward pass), outweighing the benefit during loading. The loading OOM
is better addressed directly.

### CPU offload during loading (`device_map="auto"`) — rejected

Specifying `device_map="auto"` would split the model across CPU and GPU, loading
layers to CPU if GPU fills up. This would avoid the loading OOM but degrade training
speed for every subsequent step (repeated CPU↔GPU tensor transfers). The GB10 has
sufficient HBM for the full model; the issue was allocator cache, not insufficient
total memory.

## Consequences

- **Loading OOM resolved**: the 41 GB freed-block accumulation is cleared every
  20 seconds. The remaining 20% of weights can load into the returned memory.
- **No training-phase impact**: `empty_cache()` is only called during `from_pretrained`
  (the dropper stop event is set immediately after). The training forward/backward
  pass is unaffected.
- **Dropper stats more informative**: with `empty_cache()` running before each stats
  print, the `reserv` value now reflects true live CUDA usage rather than an inflated
  cache-inclusive figure.
- **ADR-0001/ADR-0002 still apply**: the page-cache dropper (`drop_caches`) remains
  necessary for phase-1 loading (0%–43%) where safetensors mmap pages accumulate.
  The two mechanisms are complementary: `drop_caches` handles Linux page cache;
  `empty_cache()` handles the CUDA allocator free-list.
