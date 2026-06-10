#!/usr/bin/env python3
"""
Standalone HF -> Megatron checkpoint conversion for DGX Spark (GB10, 121 GB HBM).

WHY THIS EXISTS
---------------
MegatronPolicyWorker.__init__ performs a one-time HF->mcore conversion on first
run via:

    from nemo_rl.models.megatron.community_import import import_model_from_hf_name
    import_model_from_hf_name(hf_model_name, pretrained_path, megatron_cfg, **overrides)

which ends in `bridge.save_megatron_model(model, output_path)` -> writes
`iter_0000000/` (model weight shards + run_config.yaml). On GB10's 121 GB
unified HBM, that save OOMs: the ~54 GB mcore model is resident in HBM while
`save_megatron_model` makes a ~54 GB CPU copy, on top of freed HF safetensors
pages still in page cache. (See docs/investigate/v0.11-grpo-spark-smoke-tests.md.)

Skipping the save was tried (smoke11) and BREAKS the reload: the worker reloads
weights from `iter_0000000/run_config.yaml` to start training, so a skipped or
partial checkpoint -> FileNotFoundError -> exit 255.

The fix (per that doc's own follow-up): pre-seed a COMPLETE `iter_0000000/` in
an isolated, low-pressure process. This script runs ONLY the conversion -- no
Ray actors, no optimizer, no reference model, no activation buffers, no KV
cache -- so the entire machine (HBM + 512 GB system RAM + extra swap) is
available for the single 54 GB save. Once a complete checkpoint exists, the
real training run detects it (pt_checkpoint_exists == True) and SKIPS the
import entirely, taking the cheap `load_checkpoint` path. No worker patch
needed: this is the worker's own intended skip behavior.

Run this INSIDE the nemo-rl-spark container via scripts/run_convert_spark.sh.
It writes to the exact path the worker reads:
    $HF_HOME/nemo_rl/<model_name>/iter_0000000/
matching get_megatron_checkpoint_dir() + the worker's pretrained_path logic.

IMPORTANT: must be launched with the venv interpreter
(/opt/nemo-rl/.venv/bin/python3), NOT the system python3. megatron-bridge is an
EDITABLE install -- its import is wired up by the venv's site.py at interpreter
startup, so merely appending site-packages to sys.path (as run_grpo_wrapper.py
does for the *driver*) does not expose `megatron.bridge`. This mirrors how
NeMo-RL runs the real MegatronPolicyWorker (py_executable = venv python). The
container's LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2 loads system NCCL
first, so the ncclAlltoAll undefined-symbol crash does not occur under the venv
python. run_convert_spark.sh sets both.
"""

import os
import sys

# Under the venv interpreter these are already on sys.path; insert as a harmless
# fallback (and add the nemo-rl source root). This does NOT make editable
# megatron-bridge importable on its own -- the venv python is what does that.
import torch  # noqa: E402

_NEMO_RL = "/opt/nemo-rl"
_VENV_SITE = "/opt/nemo-rl/.venv/lib/python3.12/site-packages"
for _p in (_NEMO_RL, _VENV_SITE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# 1b. Stub nemo_rl.models.generation.fp8 (top-level imports vllm's _C.abi3.so,
#     compiled for x86_64; fails on aarch64 GB10). community_import does not use
#     it, but nemo_rl package __init__ chains may touch generation. fp8_cfg is
#     null in our config so the stubbed functions are never called.
# ---------------------------------------------------------------------------
import types as _types  # noqa: E402

_fp8_stub = _types.ModuleType("nemo_rl.models.generation.fp8")
_fp8_stub.convert_calibration_to_vllm_format = lambda *a, **kw: None
_fp8_stub.get_vllm_qkv_scale_names = lambda *a, **kw: {}
sys.modules.setdefault("nemo_rl.models.generation.fp8", _fp8_stub)

# ---------------------------------------------------------------------------
# 1c. Background page-cache dropper (every 10 s). Container runs --privileged,
#     so writing '3' to /proc/sys/vm/drop_caches reclaims clean file-backed
#     pages system-wide. Keeps the freed HF safetensors page cache from
#     stacking on top of the CUDA model + CPU save copy. (Same as wrapper 3c.)
# ---------------------------------------------------------------------------
import threading as _threading  # noqa: E402
import time as _time  # noqa: E402


def _drop_cache_loop():
    while True:
        _time.sleep(10)
        try:
            with open("/proc/sys/vm/drop_caches", "w") as _f:
                _f.write("3\n")
        except OSError:
            pass


threading_thread = _threading.Thread(
    target=_drop_cache_loop, daemon=True, name="cache-dropper"
)
threading_thread.start()

# ---------------------------------------------------------------------------
# 2. Imports that depend on the venv being on sys.path.
# ---------------------------------------------------------------------------
import argparse  # noqa: E402
import gc  # noqa: E402
import shutil  # noqa: E402

import yaml  # noqa: E402

from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy.utils import (  # noqa: E402
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
)


def _human(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone HF->Megatron pre-seed conversion for DGX Spark."
    )
    parser.add_argument(
        "--config",
        default="/workspace/docs/plans/lora-grpo-spark-plan/04-lora-grpo-config.yaml",
        help="Path to the GRPO YAML. policy.model_name and policy.megatron_cfg "
        "are read so the produced checkpoint matches the training run exactly.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Override the megatron checkpoint dir base. Defaults to the worker's "
        "path: get_megatron_checkpoint_dir()/<model_name>. Leave unset so it "
        "matches what MegatronPolicyWorker reads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete any existing iter_0000000/ first and re-convert.",
    )
    parser.add_argument(
        "--min-complete-gb",
        type=float,
        default=1.0,
        help="A checkpoint dir smaller than this is treated as incomplete/partial "
        "and removed before converting (the real model is ~54 GB).",
    )
    args = parser.parse_args()

    # --- Distributed: single process, single GPU --------------------------
    # community_import.import_model_from_hf_name calls
    # model_provider.initialize_model_parallel(seed=0), which requires
    # torch.distributed to already be initialized -- mirror the worker, which
    # calls init_process_group("nccl") before importing the model.
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")

    configure_dynamo_cache()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    # --- Load config (single source of truth = the training YAML) ---------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    policy = cfg["policy"]
    hf_model_name = policy["model_name"]
    megatron_cfg = policy["megatron_cfg"]
    hf_overrides = policy.get("hf_config_overrides") or {}

    # --- Resolve output path EXACTLY as the worker does -------------------
    # MegatronPolicyWorker.__init__:
    #   hf_model_subdir = hf_model_name
    #   if os.path.exists(hf_model_name):                 # local path case
    #       hf_model_subdir = f"model_{subdir.replace('/', '_')}"
    #   pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
    if args.output_path:
        pretrained_path = args.output_path
    else:
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = "model_" + hf_model_subdir.replace("/", "_")
        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"

    iter0 = os.path.join(pretrained_path, "iter_0000000")
    run_config = os.path.join(iter0, "run_config.yaml")

    print(f"[convert] model:        {hf_model_name}")
    print(f"[convert] output base:  {pretrained_path}")
    print(f"[convert] iter_0000000: {iter0}")

    # --- Idempotency / partial-checkpoint handling ------------------------
    if os.path.exists(iter0):
        size = _dir_size(iter0)
        complete = os.path.exists(run_config) and size >= args.min_complete_gb * 1e9
        print(
            f"[convert] existing iter_0000000: size={_human(size)} "
            f"run_config={'yes' if os.path.exists(run_config) else 'NO'} "
            f"complete={'yes' if complete else 'no'}"
        )
        if complete and not args.force:
            print(
                "[convert] Complete checkpoint already present. Nothing to do. "
                "The training worker will skip the HF->mcore import."
            )
            return 0
        print(f"[convert] Removing incomplete/old checkpoint: {iter0}")
        shutil.rmtree(iter0)

    os.makedirs(pretrained_path, exist_ok=True)

    # Free anything reclaimable before the big allocation/save.
    gc.collect()
    torch.cuda.empty_cache()

    print("[convert] Starting import_model_from_hf_name (this is the heavy step)...")
    # IDENTICAL call to MegatronPolicyWorker.__init__ so the produced checkpoint
    # is byte-format-compatible with what training expects to load.
    import_model_from_hf_name(
        hf_model_name,
        pretrained_path,
        megatron_cfg,
        **hf_overrides,
    )

    # --- Verify completeness ---------------------------------------------
    if not os.path.exists(run_config):
        print(
            f"[convert] ERROR: conversion finished but {run_config} is missing. "
            "Checkpoint is incomplete; training would FileNotFoundError on reload.",
            file=sys.stderr,
        )
        return 1

    size = _dir_size(iter0)
    print(f"[convert] DONE. iter_0000000 size={_human(size)}")
    print(f"[convert] run_config.yaml present: {run_config}")
    if size < args.min_complete_gb * 1e9:
        print(
            f"[convert] WARNING: checkpoint is only {_human(size)} -- smaller than "
            f"expected (~54 GB). Inspect before launching training.",
            file=sys.stderr,
        )
    print(
        "[convert] Pre-seed complete. Launch training normally; the worker will "
        "detect this checkpoint and SKIP the HF->mcore conversion."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
