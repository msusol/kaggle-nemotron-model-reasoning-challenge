# Kaggle Notebook Workflow

All Kaggle notebook changes (code, metadata, kernel sources, dataset sources) are managed
via `kaggle kernels push` from the local machine. **Never instruct the user to make changes
manually in the Kaggle UI** — all configuration lives in version-controlled metadata files.

---

## Push workflow — always use this pattern

```zsh
mkdir -p /tmp/nemotron-v09-kernel
cp notebook/v09_train_kaggle.ipynb /tmp/nemotron-v09-kernel/
cp notebook/v09-train-kaggle-kernel-metadata.json /tmp/nemotron-v09-kernel/kernel-metadata.json
kaggle kernels push -p /tmp/nemotron-v09-kernel
```

The staging dir is required because Kaggle's CLI expects the metadata file to be named
exactly `kernel-metadata.json`. Our source file has a descriptive name; copy it on push.

## GPU model selection — UI only, cannot be set in metadata

`enable_gpu: true` in the metadata means "use a GPU" but Kaggle assigns P100 (SM 6.0) by
default. **RTX Pro 6000 must be selected manually in the Kaggle editor UI** — there is no
metadata JSON field for specific GPU model.

`machine_shape` appears in metadata pulled from Kaggle (e.g. `"NvidiaRtxPro6000"`) but is
**read-only** — Kaggle ignores or overrides it on push, setting the accelerator to None.
Do NOT include `machine_shape` in kernel-metadata.json files.

**Correct workflow for both the training notebook and utility script:**

1. `kaggle kernels push` → updates the code; auto-starts a committed run on the wrong GPU
2. **Immediately stop** the auto-run in the Kaggle UI
3. Right panel → **Accelerator → GPU (RTX Pro 6000)**
4. **Save Version → Save & Run All**

Do not let the auto-run from `kernels push` complete — it wastes GPU quota on P100.

## Metadata file is the single source of truth

`notebook/v09-train-kaggle-kernel-metadata.json` controls everything:

| Field | What it controls |
|---|---|
| `kernel_sources` | Utility scripts mounted at runtime (mamba-ssm, trl, unsloth) |
| `dataset_sources` | Datasets mounted at `/kaggle/input/` |
| `competition_sources` | Competition data mounted at `/kaggle/input/` |
| `model_sources` | Model weights mounted at `/kaggle/input/` |
| `enable_gpu` | GPU accelerator (RTX Pro 6000) |
| `enable_internet` | Internet access (always false for RTX Pro 6000) |
| `is_private` | Notebook visibility |

To add or remove any input, edit the JSON and push — do not use the Kaggle UI.

## Committed runs vs interactive sessions

Kaggle has two execution modes with different behaviours:

| Setting | Committed run (Save & Run All) | Interactive session (editor) |
|---|---|---|
| `kernel_sources` | Applied automatically ✓ | **Must be added manually in UI** ✗ |
| `enable_internet` | Applied automatically ✓ | **Must be toggled manually in UI** ✗ |
| `dataset_sources` | Applied automatically ✓ | Applied automatically ✓ |

**Preferred mode for training: committed run.**
When the user wants to train, tell them to use **Save Version → Save & Run All** in the
Kaggle UI, or push via CLI which triggers a committed run. Do not guide them to run cells
interactively for training — kernel sources (trl, unsloth via nemotron-v09-build) will not
be mounted in an interactive session unless manually added.

## Current kernel inputs — v09 training notebook

| Type | Source | Purpose |
|---|---|---|
| kernel source | `gdataranger/nemotron-v09-build` | torch (nightly cu128), cutlass, causal-conv1d, mamba-ssm, trl, unsloth, unsloth_zoo |
| dataset source | `gdataranger/nemotron-v09-training-data` | v0.9_train.jsonl + v0.9_valid.jsonl |
| competition source | `nvidia-nemotron-model-reasoning-challenge` | competition data |
| model source | `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1` | base model weights |

`gdataranger/nemotron-v09-build` consolidates everything previously split between
`ryanholbrook/nvidia-utility-script` (torch nightly + compiled packages) and our own
deps script (trl, unsloth). GPU compilation adapted from ryanholbrook's approach.

## Updating the build script (when refreshing trl/unsloth/torch)

**IMPORTANT: kernel_type must be "script" — Kaggle blocks changing type after creation.**
`gdataranger/nemotron-v09-build` is a script-type kernel (`v09_utility_script.py`).
The old notebook-type `gdataranger/nemotron-v09-deps` was deleted (couldn't be converted).

```zsh
mkdir -p /tmp/nemotron-v09-build
cp notebook/v09_utility_script.py /tmp/nemotron-v09-build/
cp notebook/v09-build-script-kernel-metadata.json /tmp/nemotron-v09-build/kernel-metadata.json
kaggle kernels push -p /tmp/nemotron-v09-build
# After push: stop the auto-run (it runs on wrong GPU)
# → Go to kaggle.com/code/gdataranger/nemotron-v09-build
# → Right panel → Accelerator → any GPU with Internet ON (P100/T4 is fine — cross-compiles for SM 12.0)
# → Save Version → Save & Run All
# (script must complete a successful run before its output can be used as a kernel source)
```

## RTX Pro 6000 has no internet — dependency resolution order

The `cell-install` in `v09_train_kaggle.ipynb` resolves packages in this priority order:

1. `gdataranger/nemotron-v09-build` kernel source (committed run) → `sys.path.insert`, zero pip
2. PyPI → requires internet (T4/P100 tiers only, not RTX Pro 6000)

**`kernel_sources` are auto-applied for committed runs (Save & Run All).** The user does NOT need
to manually attach the utility script in the Kaggle UI when using Save Version → Save & Run All.
Manual attachment is only needed for interactive editor sessions (which are not used for training).

**`gdataranger/nemotron-v09-build` editor UI looks like a notebook** — Kaggle uses the same
cell-style editor for both script and notebook kernel types. Confirmed script type via API pull:
`kernel_type: "script"`, `code_file: "nemotron-v09-build.py"`. The `/notebook` URL suffix is
cosmetic; Kaggle applies it to all code regardless of type.
