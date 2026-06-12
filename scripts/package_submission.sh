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
# Strip dead Mamba out_proj — zero-gradient path (UnslothCheckpointFunction guard)
if "out_proj" in cfg.get("target_modules", []):
    cfg["target_modules"] = [m for m in cfg["target_modules"] if m != "out_proj"]
    print("  removed dead 'out_proj' from target_modules")

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
        elif "mixer.out_proj.lora_" in key:
            # Dead path: Unsloth's UnslothCheckpointFunction guard blocks gradients when
            # Mamba scan output (args[0]) lacks requires_grad=True. lora_B stays exactly
            # zero throughout training; zero-contribution keys add filesize for no benefit.
            dropped.append(key)
        else:
            kept[key] = f.get_tensor(key)
save_file(kept, dst)
print(f"Safetensors: kept {len(kept)} keys, dropped {len(dropped)} keys (fused MoE expert + dead out_proj)")
if dropped:
    print(f"  sample dropped: {dropped[0]}  ({len(dropped)} total)")
PYEOF

# ── Inject per-routed-expert LoRA as standard PEFT keys ─────────────────────
# expert_lora_weights.pt stores fused tensors keyed as:
#   "{prefix}.lora_{A/B}_{up/down}" shape [128, r, dim]
# Kaggle's unfused NemotronH exposes individual nn.Linear per routed expert
# (mixer.experts.{j}.up_proj), so standard PEFT can load per-expert LoRA if
# the keys are saved in PEFT format. Without this step, the 856M expert params
# trained on DGX Spark are unused at inference.
ELO_PATH="$ADAPTER_DIR/expert_lora_weights.pt"
if [ -f "$ELO_PATH" ]; then
  ST_TMP="${ST_OUT}.tmp_elo"
  python3 - "$ST_OUT" "$ELO_PATH" "$ST_TMP" <<'PYEOF'
import sys, torch
from safetensors import safe_open
from safetensors.torch import save_file

st_in, elo_path, st_out = sys.argv[1], sys.argv[2], sys.argv[3]

# Load existing filtered safetensors
tensors = {}
with safe_open(st_in, framework="pt") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

# Load expert LoRA and convert to per-expert PEFT keys
elo = torch.load(elo_path, map_location="cpu", weights_only=True)
suffix_map = {
    "lora_A_up":   ("up_proj",   "lora_A"),
    "lora_B_up":   ("up_proj",   "lora_B"),
    "lora_A_down": ("down_proj", "lora_A"),
    "lora_B_down": ("down_proj", "lora_B"),
}
added_keys = 0
for src_key in sorted(elo.keys()):
    src_tensor = elo[src_key]
    prefix, param = src_key.rsplit(".", 1)
    if param not in suffix_map:
        print(f"  WARNING: unknown param '{param}' in {src_key} — skipped")
        continue
    proj_name, lora_name = suffix_map[param]
    E = src_tensor.shape[0]  # 128 experts
    for j in range(E):
        # e.g. base_model.model.model.layers.1.mixer.experts.0.up_proj.lora_A.weight
        peft_key = f"{prefix}.{j}.{proj_name}.{lora_name}.weight"
        tensors[peft_key] = src_tensor[j].contiguous()
        added_keys += 1

routed_layers = len(elo) // 4
n_experts = elo[next(iter(elo))].shape[0]
save_file(tensors, st_out)
print(f"Expert LoRA injected: {routed_layers} layers × {n_experts} experts × 4 tensors = {added_keys} keys added")
print(f"Total safetensors keys: {len(tensors)}")
PYEOF
  mv "$ST_TMP" "$ST_OUT"
  echo "  expert_lora_weights.pt injected into filtered safetensors"
else
  echo "  No expert_lora_weights.pt found — routed-expert LoRA not included (shared-expert LoRA still applies)"
fi

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
