#!/usr/bin/env bash
set -euo pipefail

ADAPTER_DIR="${1:-${ADAPTER_OUTPUT_DIR:-/workspace/output/adapter}}"
OUT_DIR="${2:-${SUBMISSION_DIR:-/workspace/output/submission}}"

mkdir -p "$OUT_DIR"
ZIP_PATH="$(cd "$OUT_DIR" && pwd)/submission.zip"

test -d "$ADAPTER_DIR"
test -f "$ADAPTER_DIR/adapter_config.json"

if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ] && [ ! -f "$ADAPTER_DIR/adapter_model.bin" ]; then
  echo "Missing adapter weights in $ADAPTER_DIR"
  exit 1
fi

# ── Rebuild adapter_config.json with standard PEFT fields only ───────────────
# Unsloth saves several non-standard fields that crash standard PEFT's LoraConfig:
#   target_parameters, use_qalora, qalora_group_size — cause TypeError in LoraConfig()
#   base_model_name_or_path — Kaggle-internal path not visible in eval environment
#   auto_mapping — custom class dict breaks eval env (modeling_nemotron_h not present)
# PEFT's _get_peft_type() catches any LoraConfig(**config_dict) exception and
# re-raises as "Can't find adapter_config.json", masking the real cause.
python3 - "$ADAPTER_DIR/adapter_config.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
raw = json.load(open(p))

STANDARD_FIELDS = {
    "peft_type", "task_type", "base_model_name_or_path", "revision",
    "r", "lora_alpha", "lora_dropout", "bias", "fan_in_fan_out",
    "target_modules", "modules_to_save", "init_lora_weights",
    "layers_to_transform", "layers_pattern", "rank_pattern", "alpha_pattern",
    "use_rslora", "use_dora", "inference_mode", "auto_mapping",
    "peft_version",
}
cfg = {k: v for k, v in raw.items() if k in STANDARD_FIELDS}
cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
cfg["auto_mapping"] = None

stripped = sorted(set(raw) - set(cfg))
json.dump(cfg, open(p, "w"), indent=2)
print(f"Rebuilt {p} (standard fields only)")
print(f"  base_model_name_or_path: {cfg['base_model_name_or_path']}")
print(f"  target_modules: {cfg.get('target_modules')}")
if stripped:
    print(f"  stripped Unsloth-specific fields: {stripped}")
PYEOF

# ── Filter safetensors: remove routed-expert keys ────────────────────────────
# Unsloth saves fused MoE expert LoRA as mixer.experts.lora_A/B.weight with shapes
# like [4096, 2688] — not r=32 matrices. No mixer.experts nn.Linear exists in the
# standard NemotronH model; PEFT raises a masked ValueError when it finds these keys.
# Filter them out. All other lora_A/B keys are standard-PEFT-compatible.
ST_IN="$ADAPTER_DIR/adapter_model.safetensors"
ST_OUT="$ADAPTER_DIR/adapter_model_filtered.safetensors"
python3 - "$ST_IN" "$ST_OUT" <<'PYEOF'
import sys
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
kept, dropped = {}, []
with safe_open(src, framework="pt") as f:
    for key in f.keys():
        if "mixer.experts.lora_" in key and "shared_" not in key:
            dropped.append(key)
        else:
            kept[key] = f.get_tensor(key)
save_file(kept, dst)
print(f"Safetensors: kept {len(kept)} keys, dropped {len(dropped)} fused MoE expert keys")
if dropped:
    print(f"  dropped pattern: {dropped[0].split('layers.')[1].split('.lora')[0]}...  ({len(dropped)} total)")
PYEOF

rm -f "$ZIP_PATH"

# ── Package submission.zip with adapter directory prefix ─────────────────────
# The Kaggle evaluator expects adapter files under a subdirectory, not at zip root.
# Use filtered safetensors; include all available tokenizer files.
python3 - "$ADAPTER_DIR" "$ST_OUT" "$ZIP_PATH" <<'PYEOF'
import sys, pathlib, zipfile

adapter_dir = pathlib.Path(sys.argv[1])
st_filtered  = pathlib.Path(sys.argv[2])
zip_path     = pathlib.Path(sys.argv[3])
prefix       = adapter_dir.name  # e.g. "adapter_v9_run7"

files = {
    "adapter_config.json":     adapter_dir / "adapter_config.json",
    "adapter_model.safetensors": st_filtered,
}
for fname in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]:
    p = adapter_dir / fname
    if p.exists():
        files[fname] = p

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname, src in files.items():
        zf.write(src, f"{prefix}/{fname}")
        print(f"  {prefix}/{fname}  ({src.stat().st_size/1e6:.1f} MB)")
print(f"Created {zip_path}  ({zip_path.stat().st_size/1e6:.1f} MB total)")
PYEOF

unzip -l "$ZIP_PATH"
