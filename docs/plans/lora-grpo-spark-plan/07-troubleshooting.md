# 07 — Troubleshooting

## OOM during model load

**Symptom:** `torch.cuda.OutOfMemoryError` while loading weights.

**Cause:** `from_pretrained` without `low_cpu_mem_usage=True` allocates the full
model as empty CPU tensors first (~57 GB), then streams weights — the CPU+GPU
combined allocation spike exhausts memory.

**Fix:** NeMo RL handles this internally, but if loading manually always use:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,   # meta-device path — no empty tensor spike
)
```

**Also check** for orphaned HBM from previous runs:
```bash
# Run the HBM cleanup sequence from 01-docker-build.md
systemctl --user stop gnome-remote-desktop.service
docker rm -f nemo-rl-run 2>/dev/null || true
python3 -c "import torch; torch.cuda.init(); f,t=torch.cuda.mem_get_info(); print(f'{f/1e9:.1f}/{t/1e9:.1f} GB')"
```

---

## OOM during rollout generation

**Symptom:** OOM during the generation phase (after model loads fine).

**Cause:** KV cache allocation (`buffer_size_gb`) is too large, or sequence length
is too high for the remaining HBM after weights.

**Fix — reduce in this order:**

```yaml
# In 04-lora-grpo-config.yaml, try each change and re-test:

# 1. Reduce KV cache buffer
generation:
  mcore_generation_config:
    buffer_size_gb: 8       # was 12; try 8, then 6

# 2. Reduce sequence length
policy:
  max_total_sequence_length: 2048  # was 4096

# 3. Fewer rollouts per prompt
grpo:
  num_generations: 2        # was 4

# 4. Fewer CUDA graphs
generation:
  mcore_generation_config:
    num_cuda_graphs: 4      # was 8
```

---

## OOM during backward pass

**Symptom:** OOM after rollout succeeds but during gradient computation.

**Cause:** Activations are not fully checkpointed, or rollout batch is too large.

**Fix:**
```yaml
policy:
  megatron_cfg:
    activation_checkpointing: true   # must be true on single GPU

grpo:
  rollout_batch_size: 16    # reduce from 32
  global_batch_size: 8      # reduce from 16
```

---

## `sm_121a` / Triton PTXAS error

**Symptom:**
```
ptxas fatal: Value 'sm_121a' is not defined for option '--gpu-name'
```

**Cause:** Triton's bundled `ptxas` compiler doesn't support GB10's `sm_121a`
architecture. This affects the DTensor backend more than Megatron.

**Fix (Megatron backend):** The Megatron backend uses its own CUDA kernels, not
Triton, so this error should not appear. If it does, disable Triton explicitly:

```yaml
policy:
  megatron_cfg:
    use_triton: false
```

**Fix (DTensor backend):** If you fall back to the DTensor backend, LoRA requires:
```yaml
policy:
  dtensor_cfg:
    _v2: true
    tensor_parallel_size: 1
    use_triton: false        # Triton not supported for TP=1 on sm_121a yet
    lora_cfg:
      enabled: true
```

---

## `NemotronHConfig has no attribute 'rms_norm_eps'`

**Symptom:**
```
AttributeError: 'NemotronHConfig' object has no attribute 'rms_norm_eps'
```

**Cause:** Using an old container image or `transformers < 5.5.3`. The config
schema changed in 5.5.3 when native `NemotronHForCausalLM` was added.

**Fix:**
```bash
pip install "transformers==5.5.3" --break-system-packages
# Verify:
python3 -c "import transformers; print(transformers.__version__)"
```

Also: do **not** use `trust_remote_code=True` — it loads the model's bundled
code which has the old broken KV-cache implementation:
```python
# Wrong — loads old bundled code
model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

# Correct — uses native transformers 5.5.3 implementation
model = AutoModelForCausalLM.from_pretrained(name)
```

---

## Ray cluster fails to start / port conflicts

**Symptom:** Ray workers fail to connect, or errors about address already in use.

**Cause:** A previous training run left Ray processes running.

**Fix:**
```bash
# Kill all Ray processes
ray stop --force 2>/dev/null || true
pkill -f "ray::" 2>/dev/null || true
pkill -f "gcs_server" 2>/dev/null || true

# Wait and verify
sleep 3
pgrep -f ray || echo "Ray processes cleared"
```

---

## Training hangs after first step

**Symptom:** Step 1 completes but step 2 never starts; no error message.

**Cause:** Usually a deadlock in the Megatron generation / Ray actor communication.
Common with MoE models when `expert_model_parallel_size > 1` is set on a single GPU.

**Fix:** Confirm `expert_model_parallel_size: 1` in the config. If still hanging:
```bash
# Check for stuck Ray actors
ray status

# If Ray looks healthy, check for CUDA stream deadlock
# Add to your launch command:
NCCL_DEBUG=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
  uv run python examples/nemo_gym/run_grpo_nemo_gym.py ...
```

---

## Orphaned HBM — progressive memory loss across runs

**Symptom:** Each run starts with less free HBM than the last. After 3+ crashed
runs, the model no longer fits.

**Cause:** The CUDA driver does not release HBM on `SIGKILL`. Orphaned allocations
are invisible to `nvidia-smi --query-compute-apps`.

**Fix:** Run the full cleanup sequence from `01-docker-build.md` before each
training session, including:

```bash
# Step that actually GCs orphaned allocations:
# Create a fresh CUDA context to force the driver to clean up
python3 -c "
import torch
torch.cuda.init()
# This forces the driver to GC any allocations from dead processes
torch.cuda.empty_cache()
f, t = torch.cuda.mem_get_info()
print(f'HBM after GC: {f/1e9:.1f} GB free / {t/1e9:.1f} GB total')
"
```

---

## mamba-ssm / selective_scan_cuda ImportError

**Symptom:**
```
ImportError: ... selective_scan_cuda ... no module named selective_scan_cuda
```

**Cause:** `mamba-ssm` was installed from a pre-built wheel that skipped the CUDA
extension build (silent during `docker build` because no GPU is available).

**Fix:** The Dockerfile applies the `try/except` patch to make this import
optional — Nemotron-H uses Mamba-2 Triton kernels exclusively and never calls
the Mamba-1 `selective_scan_cuda` CUDA path. If the patch wasn't applied:

```bash
python3 -c "
f='/usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/selective_scan_interface.py'
c = open(f).read()
needle = '\nimport selective_scan_cuda\n'
repl = '\ntry:\n    import selective_scan_cuda\nexcept ImportError:\n    selective_scan_cuda = None\n'
open(f, 'w').write(c.replace(needle, repl, 1)) if needle in c else None
print('patch applied')
"
```

---

## Useful diagnostic commands

```bash
# Full system state for bug reports
echo "=== GPU ===" && nvidia-smi
echo "=== HBM ===" && python3 -c "import torch; f,t=torch.cuda.mem_get_info(); print(f'{f/1e9:.1f}/{t/1e9:.1f} GB free/total')"
echo "=== CPU RAM ===" && free -h
echo "=== Disk ===" && df -h /workspace
echo "=== Ray ===" && ray status 2>/dev/null || echo "Ray not running"
echo "=== NeMo RL branch ===" && cd /opt/nemo-rl && git branch && git log --oneline -3
echo "=== transformers ===" && python3 -c "import transformers; print(transformers.__version__)"
echo "=== torch ===" && python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```
