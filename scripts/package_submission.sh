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

# ── Patch adapter_config.json ────────────────────────────────────────────────
# Unsloth writes a Kaggle-internal path for base_model_name_or_path and sets
# auto_mapping to a custom class dict. Both break the vLLM evaluator.
python3 - "$ADAPTER_DIR/adapter_config.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
cfg["base_model_name_or_path"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
cfg["auto_mapping"] = None
json.dump(cfg, open(p, "w"), indent=2)
print(f"Patched {p}")
print(f"  base_model_name_or_path: {cfg['base_model_name_or_path']}")
print(f"  auto_mapping: {cfg['auto_mapping']}")
PYEOF

rm -f "$ZIP_PATH"
# Build file list from what actually exists in the adapter dir.
# special_tokens_map.json is absent from Nemotron-H adapters; zip it only if present.
FILES=(adapter_config.json adapter_model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja)
OPTIONAL=(special_tokens_map.json)
for f in "${OPTIONAL[@]}"; do
  [[ -f "$ADAPTER_DIR/$f" ]] && FILES+=("$f")
done
# Zip with the adapter directory as a prefix (e.g. adapter_v9_run6/adapter_config.json).
# The Kaggle evaluator expects this subdirectory layout — a flat zip (files at root) errors.
ADAPTER_NAME="$(basename "$ADAPTER_DIR")"
(
  cd "$(dirname "$ADAPTER_DIR")"
  zip "$ZIP_PATH" "${FILES[@]/#/${ADAPTER_NAME}/}"
)

echo "Created $ZIP_PATH"
unzip -l "$ZIP_PATH"
