# Kaggle Notebook Setup — NVIDIA Nemotron Competition

Investigation log for non-obvious Kaggle environment requirements when running
Nemotron-3-Nano-30B notebooks in this competition.

---

## 1. nvidia-utility-script must be added as a kernel input

### Context

Competition notebooks that use Nemotron-H (the Mamba/attention hybrid architecture)
need CUTLASS DSL and other NVIDIA-provided utilities. These are distributed via a
companion notebook maintained by `ryanholbrook`:

- **Notebook:** https://www.kaggle.com/code/ryanholbrook/nvidia-utility-script
- **Competition discussion:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684244

### Investigation Checklist

- [x] Identified missing kernel input as root cause of `ModuleNotFoundError: No module named 'mamba_ssm'`
- [x] Confirmed fix: add `ryanholbrook/nvidia-utility-script` to `kernel_sources` in metadata
- [x] Verified `site.addsitedir(cutlass_pkg_path)` path matches what the utility script provides

### Findings

When a notebook references the CUTLASS path:

```python
cutlass_pkg_path = (
    "/kaggle/usr/lib/notebooks/ryanholbrook/"
    "nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
)
site.addsitedir(cutlass_pkg_path)
```

...this path only exists if `ryanholbrook/nvidia-utility-script` has been added as a
**kernel input** to the notebook. Without it, the path is silently missing and the
`mamba_ssm` import fails downstream.

The error chain is:
1. `site.addsitedir(cutlass_pkg_path)` — silently does nothing (path missing)
2. `trust_remote_code=True` triggers loading of `modeling_nemotron_h.py` from model repo
3. `modeling_nemotron_h.py` attempts `from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn`
4. `mamba_ssm` not available → `ImportError: mamba-ssm is required by the Mamba model`

### Actions Taken

1. Added `"ryanholbrook/nvidia-utility-script"` to `kernel_sources` in both metadata files:
   - `notebook/kernel-metadata.json` (prize eligibility notebook)
   - `notebook/submission-demo-kernel-metadata.json` (submission demo)

2. Separately fixed `trust_remote_code=True` → removed, with `transformers==5.5.3` pinned.
   Native NemotronH support in transformers ≥ 5.3.0 avoids the `mamba_ssm` dependency
   entirely, so the utility script is needed for CUTLASS kernels but no longer a hard
   requirement just to load the model.

### Resolution

**Status: Resolved**

Both notebooks now declare `ryanholbrook/nvidia-utility-script` as a kernel input via
`kernel_sources` in their metadata JSON. Pushed as Kaggle kernel versions 5 (prize
eligibility) and 4 (submission demo), commit `220d85b`.

### Follow-ups

- Verify that the CUTLASS path resolves correctly after re-running the notebooks.
- When pushing notebook updates via `kaggle kernels push`, the `kernel_sources` field
  in `kernel-metadata.json` must always include `ryanholbrook/nvidia-utility-script` —
  it will be stripped if the metadata is regenerated without it.

---

## 2. trust_remote_code=True breaks model loading on Kaggle (mamba_ssm missing)

### Context

The Kaggle T4/P100 environment does not have `mamba_ssm` pre-installed. Any notebook
that loads Nemotron-3-Nano-30B with `trust_remote_code=True` will fail because the
model repo's `modeling_nemotron_h.py` unconditionally imports `mamba_ssm`.

### Findings

- `transformers < 5.3.0` + `trust_remote_code=True` → loads `modeling_nemotron_h.py` → fails
- `transformers >= 5.3.0` + no `trust_remote_code` → uses native NemotronH implementation → works

The native implementation in transformers ≥ 5.3.0 uses Triton kernels directly and
does not require `mamba_ssm` to be installed separately.

### Actions Taken

- Pinned `transformers==5.5.3` in the pip install cell of both notebooks.
- Removed `trust_remote_code=True` from all `AutoModelForCausalLM.from_pretrained()` and
  `AutoTokenizer.from_pretrained()` calls in both notebooks.

### Resolution

**Status: Resolved** — commit `245c768` (prize eligibility), `508dc18` (submission demo).

### Follow-ups

- This mirrors the fix already applied to the GB10 training scripts
  (`train_lora.py`, `infer_lora.py`, `smoke_test_nemotron.py`) in commit `6374cd4`.
- Any new notebook or script that loads Nemotron-3-Nano-30B should use
  `transformers >= 5.3.0` without `trust_remote_code=True`.
