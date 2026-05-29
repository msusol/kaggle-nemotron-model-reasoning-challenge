# GPU access and build notes for Docker on GB10

## Runtime: GPU flags

On this GB10 system, `--gpus all --runtime=nvidia` and `--device nvidia.com/gpu=all` (CDI)
both fail with:

```
failed to fulfil mount request: open /usr/bin/nvidia-cuda-mps-control: no such file or directory
```

The NVIDIA container runtime's bind-mount logic requires capabilities that the default Docker
setup does not provide here. The working pattern is:

```bash
docker run --rm --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  ...
```

`--privileged` grants the container full host capabilities and bypasses the failing mount.
`NVIDIA_VISIBLE_DEVICES=all` tells the NVIDIA runtime which GPUs to expose.

## Build: aarch64 CUDA extension issues

The GB10 has an aarch64 CPU (NVIDIA Grace). Always build images directly on the GB10.
Do **not** import images built on x86\_64 — CUDA `.so` files have the wrong platform tag and
Python silently ignores them, causing `ModuleNotFoundError` at import time.

### mamba_ssm — `selective_scan_cuda` missing on aarch64

`selective_scan_cuda.cpython-312-x86_64-linux-gnu.so` ships in the base image's dist-packages.
Python's module finder ignores it on aarch64 (platform tag mismatch). Building mamba_ssm from
source with `--no-binary` also silently skips the extension.

**Fix**: patch `mamba_ssm/ops/selective_scan_interface.py` — wrap
`import selective_scan_cuda` in `try/except ImportError: selective_scan_cuda = None`.
The patch is **active in both Dockerfiles**. GPU access in Docker `RUN` steps is not achievable
on this system: Docker OCI workers have isolated device namespaces that don't inherit GPU
devices from the host even with `--privileged` on the outer container. Approaches tried and
confirmed non-working: `--gpus all` (fails), privileged `moby/buildkit` daemon +
`[worker.oci] privileged=true` (GPU still not in OCI workers), `--driver-opt privileged=true`
on `docker-container` buildx driver (invalid option). The patch is safe: Nemotron-H uses
Mamba-2 Triton kernels exclusively and never calls `selective_scan_cuda`.

### causal_conv1d 1.6.x — ABI gap (`decref_pyobject` missing)

causal_conv1d ≥ 1.6 references `c10::TensorImpl::decref_pyobject() const` which was added to
standard PyTorch but is **absent** from NVIDIA's custom `torch 2.10.0a0+b4e4ee81d3.nv25.12`.
Results in: `undefined symbol: _ZNK3c1010TensorImpl15decref_pyobjectEv`.

**Fix**: use causal_conv1d 1.5.0.post8 (does not reference this symbol).

### causal_conv1d 1.5.x — hardcoded old CUDA arches

`setup.py` ignores `TORCH_CUDA_ARCH_LIST` and hardcodes `compute_53/62/70/72`.
CUDA 13.0 dropped support for compute < 7.0 — build fails with `nvcc fatal: Unsupported gpu
architecture 'compute_53'`.

**Fix**: clone from GitHub, patch `setup.py` lines 172–187 to replace the arch block with
`sm_80/sm_90/sm_120`, build with `CAUSAL_CONV1D_FORCE_BUILD=TRUE`.

The causal_conv1d 1.5.x patch is in `Dockerfile.gb10-25-12` (25.12 archive). `Dockerfile.gb10`
(26.04 primary) and `Dockerfile.gb10-26-01` use causal_conv1d 1.6.x directly — `decref_pyobject`
is present in the nv26.01+ torch ABI. The mamba_ssm try/except patch is **active in all**
Dockerfiles. Pass
`--build-arg MAMBA_REBUILD=$(date +%s)` to force recompile after a cached foreign-arch layer.

### OOM during build — MAX_JOBS and cmake parallelism

CUDA compilation can OOM when too many jobs run in parallel. Both Dockerfiles now set
`MAX_JOBS=8` for causal-conv1d and mamba-ssm source builds, and use `-j8` for the
bitsandbytes cmake step (instead of `-j$(nproc)` which spawns 72 jobs on Grace CPU).
