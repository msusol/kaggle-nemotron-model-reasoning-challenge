# Kaggle Notebook Setup — NVIDIA Nemotron Competition

Investigation log for non-obvious Kaggle environment requirements when running
Nemotron-3-Nano-30B notebooks in this competition.

---

## 1. nvidia-utility-script must be added as a kernel input (manually)

### Context

Competition notebooks that use Nemotron-H need CUTLASS DSL, `mamba_ssm`, and
`causal_conv1d` from a companion notebook maintained by `ryanholbrook`:

- **Notebook:** https://www.kaggle.com/code/ryanholbrook/nvidia-utility-script
- **Competition discussion:** https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684244

### Findings

The utility script makes its packages available under:

```
/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/
```

This path only exists when the script is added as a **kernel input**. Without it,
`site.addsitedir` silently does nothing and `mamba_ssm` fails to import downstream.

**`kernel_sources` in metadata JSON does NOT work via API push.** The field is present
in both `notebook/kernel-metadata.json` and `notebook/submission-demo-kernel-metadata.json`
and is accepted by `kaggle kernels push` without error — but Kaggle silently ignores it
and does not mount the utility script's output. This is a Kaggle API limitation, not a
metadata error.

**The manual add through the Kaggle UI is the only way to make it stick:**
Kaggle notebook editor → Inputs (right panel) → Add → search
`ryanholbrook/nvidia-utility-script` → Add. Once added this way, it persists across all
subsequent `kaggle kernels push` updates to the same kernel ID — you only need to do it
once per notebook.

### Actions Taken

- Added `"ryanholbrook/nvidia-utility-script"` to `kernel_sources` in both metadata files
  (has no effect on its own but documents the intent).
- Added the input **manually via Kaggle UI** to both notebooks — this is the actual fix.
- Updated `site.addsitedir` to scan **all** `python_packages` subdirs under the utility
  script root (not just the CUTLASS subdir), so `mamba_ssm` and `causal_conv1d` are also
  picked up regardless of where they live:

```python
utility_root = pathlib.Path(
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script"
)
if utility_root.exists():
    for pkg_path in sorted(utility_root.rglob("python_packages")):
        site.addsitedir(str(pkg_path))
    site.addsitedir(str(utility_root))
else:
    print("WARNING: nvidia-utility-script not found — add it as a kernel input")
```

### Resolution

**Status: Resolved** — utility script added manually via Kaggle UI; broad path scan
committed in `dade990`.

### Follow-ups

- Every time a new notebook is created for this competition, the utility script must be
  added manually via the Kaggle UI — API push alone is not sufficient.
- The `WARNING` print makes it obvious if the input is missing when the notebook is run.

---

## 2. trust_remote_code=True — required in Kaggle environment

### Context

The Kaggle model hub path (`kagglehub.model_download`) loads the model via the repo's
custom `modeling_nemotron_h.py`, which requires `mamba_ssm`. Dropping
`trust_remote_code=True` causes transformers to prompt interactively ("Do you wish to
run the custom code?") which hangs a notebook run.

### Findings

Two approaches were attempted:

**Approach A (failed):** Drop `trust_remote_code=True`, pin `transformers==5.5.3` for
native NemotronH support. This caused a `CalledProcessError` — Kaggle's base environment
rejects the transformers version pin with pip conflicts, and even when the install
succeeds the already-loaded `transformers` module in the running kernel is not replaced
without a restart. Result: the interactive "Do you wish to run the custom code?" prompt
still appeared.

**Approach B (working):** Keep `trust_remote_code=True` and supply `mamba_ssm` via the
utility script (Issue 1). This is the path the competition intended.

Note: the GB10 training pipeline correctly drops `trust_remote_code=True` and uses
`transformers==5.5.3` — this works on GB10 because the Docker image is rebuilt with the
correct transformers version and there is no conflicting base environment. The Kaggle
notebook environment is more constrained.

### Actions Taken

- Reverted `trust_remote_code=True` into both notebooks (`dade990`).
- Dropped `transformers==5.5.3` from the pip install cell; only `peft==0.14.0` is
  installed (other packages are pre-installed in Kaggle's base environment).
- pip install cell uses `capture_output=True` and prints stderr on failure instead of
  raising, so a pip issue does not abort the notebook.

### Resolution

**Status: Resolved** — `trust_remote_code=True` restored, pip cell simplified (`586d7fb`).

### Follow-ups

- If Kaggle's base environment ever ships `transformers >= 5.3.0`, Approach A becomes
  viable and would remove the `mamba_ssm` dependency entirely.
- The pip install cell should continue to use `capture_output=True` rather than
  `check=True` to avoid aborting on non-critical pip warnings.

---

## 3. pip install of pinned transformers fails in Kaggle environment

### Context

Pinning `transformers==5.5.3` via pip in a Kaggle notebook cell returns exit code 1.

### Findings

Kaggle's base Python environment has pinned packages with interdependencies. Attempting
to install a specific `transformers` version conflicts with those constraints and pip
exits non-zero. The `-q` flag suppresses the output, so the error only surfaces as a
`CalledProcessError` when `check=True` is used.

### Actions Taken

- Removed `transformers==5.5.3` from the pip install cell entirely (see Issue 2).
- Changed remaining `subprocess.run` call to use `capture_output=True` and print stderr
  on failure rather than `check=True`.

### Resolution

**Status: Resolved** — `586d7fb`.

### Follow-ups

- General rule for Kaggle notebook pip installs: never use `check=True`; always print
  stderr on failure so issues are visible without aborting the notebook.
- Only install packages that are genuinely absent from Kaggle's base environment.
  Use `importlib.util.find_spec("package_name")` to check before installing if unsure.
