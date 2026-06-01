# ADR-0002 — OOM Score Adjustment and Automatic Service Pause for Training

**Status:** Accepted

## Context

`scripts/run_train.sh` previously launched the `nemotron-trainer` container with
`--oom-score-adj -500`, which tells the Linux OOM killer to protect the container
and prefer killing other processes first.

The rationale at the time was defensive: a training run takes 2–3 hours; losing it
to an OOM kill is expensive, so keep it alive as long as possible.

### The 2026-05-31 incident

On 2026-05-31 at ~12:08, the `nemotron-trainer` container exhausted system memory
during model loading (the 12:00 run started with 13 GB of orphaned GPU HBM
allocations from the SIGKILL'd 11:19 run — see ADR-0001). Because `--oom-score-adj
-500` shielded the training container, the kernel OOM killer turned to system
processes instead:

```
Out of memory: Killed process 1653 (systemd-network)  anon-rss:160kB
Out of memory: Killed process 5106 (sshd)              anon-rss:1136kB
Out of memory: Killed process 1917 (smartd)            anon-rss:0kB
Out of memory: Killed process 1991 (wpa_supplicant)    anon-rss:320kB
Out of memory: Killed process 39201 (tailscaled)       anon-rss:1600kB
...
Out of memory: Killed process 37639 (python)           anon-rss:36908064kB
```

Killing `systemd-networkd` and `sshd` rendered the machine unreachable over the
network; `nvidia-persistenced` was also killed, disrupting the GPU driver. The
machine required a full hard reboot to recover.

The Python training process (PID 37639, 35 GB RSS, 358 GB virtual) was eventually
killed anyway — only after the system was already broken.

### Contributing factor: manual service pause never integrated

`scripts/services.sh pause` was documented as a manual step to run before training.
In practice it was called inconsistently. On the day of the incident, `rnaseq-server`
was running alongside the trainer and consuming additional system RAM, reducing the
headroom available during the already-strained loading phase.

## Decision

### 1. Change `--oom-score-adj` from `-500` to `+300`

A value of `+300` makes the training container a **preferred** OOM kill target
relative to system daemons (whose scores are typically 0 or below). The kernel will
now kill the trainer rather than `sshd` or `systemd-networkd`.

The original goal (keep training alive as long as possible) was correct in isolation
but wrong for a system where killing the trainer is a recoverable event and killing
networking daemons is not. Training state is lost either way once OOM fires; the
difference is whether the system stays reachable.

### 2. Integrate `services.sh pause/resume` into `run_train.sh`

`run_train.sh` now calls `services.sh pause` immediately before the GPU pre-flight
and `services.sh resume` after training (including on abort). This makes pausing
automatic and removes it from the manual runbook.

The pause stops `nginx-proxy`, `rnaseq-server`, and any other non-training
containers, freeing their RAM before model loading begins. Resume is called
unconditionally so containers are not left stopped on training failure.

### 3. GPU pre-flight is now a two-stage check

**Stage 1 — host-level (nvidia-smi), threshold 1 GB:**
Before any Docker container starts, `nvidia-smi --query-compute-apps` sums the GPU
memory held by all active compute processes. On a clean system this is ~176 MiB
(gnome-remote-desktop only). Above 1 GB means a zombie training container or other
GPU-heavy process is still running; the script aborts and resumes services.

This check uses the host view (process-attributed memory) rather than the CUDA
driver view, so it is unaffected by the ~6 GB CUDA context overhead that appears
inside any container.

**Stage 2 — in-container (torch), threshold 10 GB:**
The pre-flight container calls `torch.cuda.init(); torch.cuda.empty_cache()` to
trigger driver GC of orphaned allocations from prior SIGKILL'd containers, then
checks `torch.cuda.mem_get_info()`. If `used > 25 GB` after the flush the script
aborts. Rationale: ~6 GB is the expected CUDA context baseline; model load needs
~57 GB; training activations need ~20 GB → total peak ~83 GB, leaving 47 GB
headroom in the 130.7 GB HBM pool. Up to ~25 GB of orphaned allocations still
leaves enough room. Beyond 25 GB the load + training peak risks HBM exhaustion.

The incident run (12:00 on 2026-05-31) showed `used=13.0 GB` at the start of
train_lora.py — within the safe range for HBM. That OOM was caused by Linux RAM
pressure (safetensors page cache + Python heap), not HBM exhaustion, and is
addressed by the services.sh pause (frees rnaseq-server RAM).

**Important — abort must be in bash, not Python (`sys.exit(1)`):**
If the pre-flight Python script calls `sys.exit(1)` to signal failure, CUDA context
teardown is incomplete on container exit, and each pre-flight run *adds* ~2–3 GB of
orphaned allocations instead of clearing them. The abort marker must be a printed
line (`PREFLIGHT_FAIL`) that bash detects after the container exits cleanly (exit 0).

**Recovery when orphaned allocations persist across runs:**
If `used` remains above threshold after multiple pre-flight passes, the allocations
are stuck in the driver. Try in order:

```bash
# 1. Stop gnome-remote-desktop — releases its HBM context, often frees 6+ GB
systemctl --user stop gnome-remote-desktop.service
# restart after training: systemctl --user start gnome-remote-desktop.service

# 2. Reload nvidia_uvm via privileged container (requires no active CUDA users)
docker run --rm --privileged alpine sh -c \
  'rmmod nvidia_uvm && modprobe nvidia_uvm && echo reloaded'

# 3. Full reboot — guaranteed fallback
```

Stopping gnome-remote-desktop is sufficient in most cases: on 2026-05-31 it reduced
`used` from 14.6 GB to 8.4 GB, allowing training to proceed.

## Alternatives Considered

### Keep `-500`, rely on memory limits — rejected

Adding `--memory` cgroup limits to the training container causes silent OOM kills
with no traceback (CUDA pins weights; the kernel charges mmap page cache against the
cgroup; the combined peak exceeds any reasonable limit). Documented in
`docs/process/docker-gb10-training.md` under "Memory limits". The limit approach was
already evaluated and rejected for `run_prepare.sh` for the same reason.

### Keep `-500`, fix the root cause (orphaned GPU allocations) — insufficient

The orphaned-allocation problem is addressed by the GPU pre-flight (ADR-0001 +
threshold abort in this ADR). However, even a clean loading run can OOM at training
step 0 (first forward pass allocates activations on top of the already-loaded
model). Keeping `-500` leaves the system vulnerable to that case as well. The
`+300` change is a low-cost safety net that handles any OOM scenario, not just
the loading case.

### Positive value higher than 300 — deferred

Values approaching `+1000` maximise the probability of the trainer being killed
first but reduce the kernel's flexibility to kill other expendable processes (e.g.,
desktop apps) before training. `+300` is a pragmatic middle ground that ensures
training is clearly preferred over system daemons (score 0) while leaving room for
even-lower-priority processes to be killed first.

## Consequences

- **System stability**: A training OOM now kills the training container rather than
  networking daemons. The machine remains reachable and recoverable without a hard
  reboot.
- **Training reliability**: A SIGKILL'd training container still loses all progress
  for that run. This is unchanged — the difference is only in what the kernel kills
  first.
- **Service pause is automatic**: `services.sh pause/resume` no longer needs to
  appear in the manual launch workflow. The process doc "Pause and resume other
  services" section is updated to reflect this.
- **Abort on dirty GPU**: Two-stage check catches both active processes (> 1 GB via
  nvidia-smi host check) and orphaned driver allocations (> 10 GB via torch
  in-container check). If either fires, the script exits before launching a doomed
  run. Retry usually succeeds; if not, reboot to force a full driver reset.
- **Stale `.paused_containers`**: After a hard reboot, Docker restarts containers
  with `restart: unless-stopped` automatically. Any `.paused_containers` file left
  from before the reboot is stale — it should be deleted before running training
  again, or `services.sh resume` will attempt to `docker start` containers that are
  already running.
