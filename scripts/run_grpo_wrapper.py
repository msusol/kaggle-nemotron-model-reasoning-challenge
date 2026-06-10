#!/usr/bin/env python3
"""
Import system torch BEFORE adding the uv venv to sys.path.

When PYTHONPATH includes /opt/nemo-rl/.venv/lib/python3.12/site-packages,
Python's site.py finds the venv's sitecustomize.py (or a .pth file) and
preloads nvidia-nccl-cu12's libnccl.so.2 before libtorch_cuda.so is opened.
That older libnccl lacks ncclAlltoAll, causing:
  ImportError: libtorch_cuda.so: undefined symbol: ncclAlltoAll

Fix: run without PYTHONPATH, import torch first (from system Python, where
torch's bundled NCCL provides ncclAlltoAll), then add the venv to sys.path
manually (sys.path.insert bypasses site.py -- no .pth processing).
"""
import sys
import os
import runpy

# 1. Load system torch first -- sys.path contains no venv packages yet.
#    libtorch_cuda.so binds ncclAlltoAll from torch's own bundled libnccl.so.2.
import torch  # noqa: E402  pylint: disable=wrong-import-position

# 2. Add venv mcore packages and nemo-rl source to sys.path.
#    sys.path.insert() does NOT trigger site.py .pth processing,
#    so nvidia-nccl's __init__.py is never auto-executed.
_NEMO_RL = "/opt/nemo-rl"
_VENV_SITE = "/opt/nemo-rl/.venv/lib/python3.12/site-packages"
for _p in [_NEMO_RL, _VENV_SITE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 2b. Set PYTHONPATH for subprocess inheritance (Ray agent processes).
#
# Ray's raylet spawns runtime_env_agent and dashboard_agent as Python subprocesses.
# These subprocesses use the system Python which cannot find the 'ray' package
# (ray is only in the venv). Setting PYTHONPATH here — AFTER torch is already
# loaded in THIS process — lets the agents find ray without triggering the
# .pth-based NCCL preload (which only affects Python startup, not runtime).
os.environ["PYTHONPATH"] = (
    _VENV_SITE + ":" + _NEMO_RL + ":" + os.environ.get("PYTHONPATH", "")
)

# 3. Point sys.argv[0] at the real script so Hydra resolves config paths correctly.
_SCRIPT = os.path.join(_NEMO_RL, "examples", "run_grpo_math.py")
sys.argv[0] = _SCRIPT

# 3b. Auto-clean incomplete Megatron checkpoints (partial iter_0000000/ from a
#     crashed save).  Only common.pt (12 KB) with no model weight shards signals
#     an incomplete save.  Delete so HF→Megatron conversion runs fresh.
import shutil as _shutil
import pathlib as _pathlib_ck
_NEMO_CACHE_DIR = _pathlib_ck.Path("/workspace/.cache/huggingface/nemo_rl")
_ITER0 = _NEMO_CACHE_DIR / "nvidia" / "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" / "iter_0000000"
if _ITER0.exists():
    _iter0_bytes = sum(f.stat().st_size for f in _ITER0.rglob("*") if f.is_file())
    if _iter0_bytes < 100 * 1024 * 1024:  # < 100 MB → incomplete (no model weights)
        print(f"[wrapper] Incomplete Megatron checkpoint ({_iter0_bytes // 1024} KB) — removing {_ITER0}")
        _shutil.rmtree(_ITER0)

# 3c. Background thread: drop page cache every 10 s inside the training container.
#
#     On GB10 unified HBM (121 GB), reading the 60 GB HF safetensors fills page
#     cache as the model loads.  Combined with the 60 GB CUDA Megatron model this
#     exhausts all physical memory → OOM killer fires.
#
#     The container runs --privileged so writing '3' to /proc/sys/vm/drop_caches
#     from inside the container is equivalent to writing from the host — it reclaims
#     clean (read-side) file-backed pages system-wide.  Running every 10 s keeps
#     accumulated page cache < 20 GB between drops, leaving 60 GB headroom for the
#     Megatron model after the 60 GB HF weights are active.
#
#     NOTE: This replaces the former oom_score_adj reset thread.  That thread caused
#     host system services (nvidia-persistenced, systemd-journal, agetty) to be OOM-
#     killed before our worker — catastrophic collateral damage.  The kernel handles
#     page-cache reclamation automatically when under pressure; proactive dropping is
#     the correct lever for unified-memory systems where read cache and CUDA compete.
import threading as _threading_dc
import time as _time_dc

def _drop_cache_loop():
    _dc_path = "/proc/sys/vm/drop_caches"
    while True:
        _time_dc.sleep(10)
        try:
            with open(_dc_path, "w") as _f:
                _f.write("3\n")
        except OSError:
            pass  # non-privileged context — skip silently

_dc_thread = _threading_dc.Thread(target=_drop_cache_loop, daemon=True, name="cache-dropper")
_dc_thread.start()

# 4. Patch ray.init to disable the dashboard — virtual_cluster.py hardcodes
#    include_dashboard=True, which crashes when the dashboard binary fails to
#    write its .err file in a fresh container /tmp directory.
import ray as _ray
_orig_ray_init = _ray.init

def _patched_ray_init(*args, **kwargs):
    kwargs["include_dashboard"] = False
    return _orig_ray_init(*args, **kwargs)

_ray.init = _patched_ray_init

# 5. Stub nemo_rl.models.generation.fp8 in the driver process.
#
#    fp8.py top-level imports vllm (_C.abi3.so) which was compiled for x86_64
#    and fails on aarch64 GB10 Spark:
#      ImportError: vllm/_C.abi3.so: undefined symbol (torch CUDA symbol)
#    The two imported functions (convert_calibration_to_vllm_format,
#    get_vllm_qkv_scale_names) are only used for FP8 calibration.
#    Our config has fp8_cfg: null — they are never called.  No-op stubs are safe.
import types as _types
_fp8_stub = _types.ModuleType("nemo_rl.models.generation.fp8")
_fp8_stub.convert_calibration_to_vllm_format = lambda *a, **kw: None
_fp8_stub.get_vllm_qkv_scale_names = lambda *a, **kw: {}
sys.modules.setdefault("nemo_rl.models.generation.fp8", _fp8_stub)

# 5b. Write sitecustomize.py into the SYSTEM Python stdlib path so it runs for
#     ALL Python processes in the container (venv and system alike).
#
#     Target: /usr/lib/python3.12/sitecustomize.py
#     This path is processed by site.py BEFORE the venv's site-packages, so
#     writing to the venv site-packages would be shadowed.  The system file is
#     a simple apport hook — we preserve that and append our patches.
#
#     Patches applied to ALL worker Python processes:
#     1. fp8 stub: vllm ABI mismatch on aarch64 (fp8_cfg: null, never called)
#     2. ConfigModuleInstance cloudpickle reducer: torch.utils._config_module
#        creates a local class ConfigModuleInstance for each install_config_module()
#        call.  IsolatedWorkerInitializer cloudpickles MegatronPolicyWorker when
#        creating the worker actor; methods reference torch.distributed.config etc.
#        in their __globals__, causing TypeError: cannot pickle 'ConfigModuleInstance'.
import pathlib as _pathlib
_SC = _pathlib.Path("/usr/lib/python3.12/sitecustomize.py")
_SC.write_text(
    "# install the apport exception handler if available (original system content)\n"
    "try:\n"
    "    import apport_python_hook\n"
    "except ImportError:\n"
    "    pass\n"
    "else:\n"
    "    apport_python_hook.install()\n"
    "\n"
    "# Auto-generated by run_grpo_wrapper.py — do not edit below.\n"
    "import sys, types as _t\n"
    "\n"
    "# Stub fp8 (vllm ABI mismatch on aarch64; fp8_cfg: null so never called)\n"
    "_m = _t.ModuleType('nemo_rl.models.generation.fp8')\n"
    "_m.convert_calibration_to_vllm_format = lambda *a, **kw: None\n"
    "_m.get_vllm_qkv_scale_names = lambda *a, **kw: {}\n"
    "sys.modules.setdefault('nemo_rl.models.generation.fp8', _m)\n"
    "\n"
    "# Patch cloudpickle for ConfigModuleInstance (torch config-module singletons).\n"
    "# IsolatedWorkerInitializer cloudpickles MegatronPolicyWorker when creating\n"
    "# the worker actor; methods reference torch.distributed.config etc. in their\n"
    "# __globals__, causing TypeError: cannot pickle 'ConfigModuleInstance'.\n"
    "# torch.utils._config_module.install_config_module() creates a new local\n"
    "# class ConfigModuleInstance per call — multiple distinct types, all need fixing.\n"
    "try:\n"
    "    import cloudpickle as _cp\n"
    "    import torch as _torch\n"
    "    import torch.distributed, torch.compiler, torch._export\n"
    "    def _reconstruct_cmi(mod_name, attr_name):\n"
    "        import importlib\n"
    "        return getattr(importlib.import_module(mod_name), attr_name)\n"
    "    def _reduce_cmi(obj):\n"
    "        _paths = {}\n"
    "        try: _paths[id(_torch._export.config)] = ('torch._export', 'config')\n"
    "        except: pass\n"
    "        try: _paths[id(_torch.compiler.config)] = ('torch.compiler', 'config')\n"
    "        except: pass\n"
    "        try: _paths[id(_torch.distributed.config)] = ('torch.distributed', 'config')\n"
    "        except: pass\n"
    "        try:\n"
    "            import torch._meta_registrations as _mr\n"
    "            _paths[id(_mr.exp_config)] = ('torch._meta_registrations', 'exp_config')\n"
    "        except: pass\n"
    "        if id(obj) in _paths:\n"
    "            return (_reconstruct_cmi, _paths[id(obj)])\n"
    "        return (lambda: None, ())\n"
    "    _seen = set()\n"
    "    for _o in [_torch._export.config, _torch.compiler.config, _torch.distributed.config]:\n"
    "        _t2 = type(_o)\n"
    "        if _t2 not in _seen:\n"
    "            _seen.add(_t2)\n"
    "            _cp.CloudPickler.dispatch[_t2] = _reduce_cmi\n"
    "    try:\n"
    "        import torch._meta_registrations as _mr2\n"
    "        _t3 = type(_mr2.exp_config)\n"
    "        if _t3 not in _seen: _cp.CloudPickler.dispatch[_t3] = _reduce_cmi\n"
    "    except: pass\n"
    "except Exception:\n"
    "    pass\n"
    "\n"
    "# Patch megatron.bridge.training.checkpointing.save_checkpoint to drop page\n"
    "# cache and empty CUDA allocator cache before writing the checkpoint.\n"
    "#\n"
    "# Root cause (confirmed via journalctl OOM killer): on GB10 unified HBM\n"
    "# (121 GB), torch.save copies all GPU tensors to CPU first (~54 GB extra).\n"
    "# With 54 GB CUDA model + 57 GB anon (freed HF pages + Python + driver)\n"
    "# physical memory is full.  Extra swap (60 GB) in run_nemo_grpo_spark.sh\n"
    "# gives the kernel room to page out anon pages during the save.\n"
    "# Dropping page cache here removes any residual read-cache pressure before\n"
    "# the save, leaving maximum headroom for the CPU copy.\n"
    "#\n"
    "# IMPORTANT: the actual save call is in\n"
    "#   megatron.bridge.training.model_load_save.save_megatron_model\n"
    "# which imports save_checkpoint via 'from ... import'.  This hook fires\n"
    "# when 'megatron.bridge.training.checkpointing' is first imported (before\n"
    "# any 'from ... import'), so the module-level patch IS picked up by callers.\n"
    "#\n"
    "# Uses find_spec + wrapped loader (NOT find_module/load_module) to avoid\n"
    "# infinite recursion when importlib.util.find_spec re-enters our hook.\n"
    "import sys as _sys_msp\n"
    "class _MegatronSavePatcher:\n"
    "    def find_spec(self, name, path, target=None):\n"
    "        if name != 'megatron.bridge.training.checkpointing':\n"
    "            return None\n"
    "        for _f in _sys_msp.meta_path:\n"
    "            if _f is self:\n"
    "                continue\n"
    "            try:\n"
    "                _sp = _f.find_spec(name, path, target)\n"
    "                if _sp is None:\n"
    "                    continue\n"
    "                _orig_ldr = _sp.loader\n"
    "                class _PatchedLoader:\n"
    "                    def create_module(self2, spec):\n"
    "                        try: return _orig_ldr.create_module(spec)\n"
    "                        except AttributeError: return None\n"
    "                    def exec_module(self2, mod):\n"
    "                        _orig_ldr.exec_module(mod)\n"
    "                        _orig_sv = mod.save_checkpoint\n"
    "                        def _patched_sv(*a, **kw):\n"
    "                            # Drop page cache and CUDA allocator cache before\n"
    "                            # save_checkpoint writes the ~54 GB model checkpoint.\n"
    "                            # This frees residual read-side page cache from the\n"
    "                            # HF safetensors load, leaving maximum headroom for\n"
    "                            # the CPU copy that torch.save creates.  The OOM is\n"
    "                            # ultimately handled by 60 GB extra swap added in\n"
    "                            # run_nemo_grpo_spark.sh; this just reduces pressure.\n"
    "                            try:\n"
    "                                import torch as _tc\n"
    "                                _tc.cuda.empty_cache()\n"
    "                                import gc as _gc; _gc.collect()\n"
    "                            except Exception: pass\n"
    "                            try:\n"
    "                                with open('/proc/sys/vm/drop_caches', 'w') as _dc:\n"
    "                                    _dc.write('3\\n')\n"
    "                            except Exception: pass\n"
    "                            return _orig_sv(*a, **kw)\n"
    "                        mod.save_checkpoint = _patched_sv\n"
    "                _sp.loader = _PatchedLoader()\n"
    "                return _sp\n"
    "            except (AttributeError, TypeError):\n"
    "                continue\n"
    "        return None\n"
    "_sys_msp.meta_path.insert(0, _MegatronSavePatcher())\n"
)

# 5c. Patch cloudpickle in this driver process (catches ALL ConfigModuleInstance types
#     universally by type name — no need to enumerate the 7+ individual types).
#
#     CRITICAL: Ray uses ray.cloudpickle (ray/cloudpickle/__init__.py) — a DIFFERENT
#     module from the standalone cloudpickle package — for all actor serialization
#     (ray._common.serialization.pickle_dumps).  We must patch BOTH packages.
import cloudpickle as _cp
import ray.cloudpickle as _rcp
import sys as _sys_drv

def _reconstruct_cmi_drv(mod_name, attr_name):
    import importlib
    return getattr(importlib.import_module(mod_name), attr_name)

def _make_cmi_ro(orig_ro):
    def _ro(self, obj):
        result = orig_ro(self, obj)
        if result is not NotImplemented:
            return result
        if type(obj).__name__ == 'ConfigModuleInstance':
            for _k, _v in _sys_drv.modules.items():
                if _v is obj:
                    _pts = _k.rsplit('.', 1)
                    if len(_pts) == 2:
                        return (_reconstruct_cmi_drv, (_pts[0], _pts[1]))
        return NotImplemented
    return _ro

_cp.CloudPickler.reducer_override = _make_cmi_ro(_cp.CloudPickler.reducer_override)
_rcp.CloudPickler.reducer_override = _make_cmi_ro(_rcp.CloudPickler.reducer_override)

# 5d. Patch worker_groups.py to insert cloudpickle fix inline in create_worker().
#
#     sitecustomize.py (step 5b) is unreliable: torch import may fail silently at
#     Python startup time for some Ray worker process environments.  This step writes
#     the fix directly into create_worker() so it runs in the IsolatedWorkerInitializer
#     actor process right before worker_class.options().remote() — the exact moment
#     and process where MegatronPolicyWorker is cloudpickled.
#
#     The patched create_worker() is cloudpickled by the driver (step 5c handles any
#     ConfigModuleInstance in driver globals) and sent to the worker, where calling
#     create_worker() executes our patch inline as plain function bytecode.
_WG_PATH = _pathlib.Path("/opt/nemo-rl/nemo_rl/distributed/worker_groups.py")
_wg_src = _WG_PATH.read_text()
_CMI_MARKER = "# [cloudpickle-cmi-patch]"
if _CMI_MARKER not in _wg_src:
    _IND = "            "  # 12 spaces: body of create_worker inside two nested classes
    _WG_PATCH = (
        _IND + _CMI_MARKER + " Override reducer_override in BOTH cloudpickle packages.\n"
        + _IND + "# Ray uses ray.cloudpickle (NOT the standalone cloudpickle package) for all\n"
        + _IND + "# actor serialization (ray._common.serialization.pickle_dumps).  We must patch\n"
        + _IND + "# both.  Checking type(obj).__name__=='ConfigModuleInstance' catches all 7+\n"
        + _IND + "# distinct types created by install_config_module() universally.\n"
        + _IND + "try:\n"
        + _IND + "    import cloudpickle as _ccp, sys as _sys\n"
        + _IND + "    import ray.cloudpickle as _rccp\n"
        + _IND + "    def _recon_cmi(mn, an):\n"
        + _IND + "        import importlib as _il\n"
        + _IND + "        return getattr(_il.import_module(mn), an)\n"
        + _IND + "    def _make_ro(orig):\n"
        + _IND + "        def _ro(self, obj):\n"
        + _IND + "            result = orig(self, obj)\n"
        + _IND + "            if result is not NotImplemented: return result\n"
        + _IND + "            if type(obj).__name__ == 'ConfigModuleInstance':\n"
        + _IND + "                for _k, _v in _sys.modules.items():\n"
        + _IND + "                    if _v is obj:\n"
        + _IND + "                        _pts = _k.rsplit('.', 1)\n"
        + _IND + "                        if len(_pts) == 2: return (_recon_cmi, (_pts[0], _pts[1]))\n"
        + _IND + "            return NotImplemented\n"
        + _IND + "        return _ro\n"
        + _IND + "    _ccp.CloudPickler.reducer_override = _make_ro(_ccp.CloudPickler.reducer_override)\n"
        + _IND + "    _rccp.CloudPickler.reducer_override = _make_ro(_rccp.CloudPickler.reducer_override)\n"
        + _IND + "except Exception:\n"
        + _IND + "    pass\n"
        + _IND + "# Set up worker arguments and resources\n"
    )
    _wg_src = _wg_src.replace(
        _IND + "# Set up worker arguments and resources\n",
        _WG_PATCH,
    )
    _WG_PATH.write_text(_wg_src)

# 6. Patch MegatronPolicyWorker to use the pre-built venv Python directly.
#
#    Default: MCORE_EXECUTABLE = "uv run --locked --extra mcore ..."
#    This triggers create_local_venv_on_each_node() which tries to build a
#    worker-specific venv from scratch.  That build fails because
#    transformer-engine-torch==2.8.0 cannot compile in the new venv
#    (its setup.py tries `import torch`, gets the wrong PyPI torch, fails).
#
#    Fix: skip venv creation entirely and use the main venv Python directly.
#    The venv already has megatron-core, megatron-bridge, transformer-engine,
#    and all other mcore dependencies properly installed and editable-linked.
#    Worker py_executable not starting with "uv" → worker_groups.py bypasses
#    create_local_venv_on_each_node and uses it as-is.
#
#    NCCL: LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnccl.so.2 in the container
#    env preloads the system NCCL before any worker Python code runs, preventing
#    the venv's older libnccl.so.2 from being loaded first.
from nemo_rl.distributed import ray_actor_environment_registry as _reg  # noqa: E402
_VENV_PYTHON = "/opt/nemo-rl/.venv/bin/python3"
_reg.ACTOR_ENVIRONMENT_REGISTRY[
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
] = _VENV_PYTHON

# 7. Run the training script as __main__.
runpy.run_path(_SCRIPT, run_name="__main__")
