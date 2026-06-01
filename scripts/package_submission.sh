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

rm -f "$ZIP_PATH"
# Build file list from what actually exists in the adapter dir.
# special_tokens_map.json is absent from Nemotron-H adapters; zip it only if present.
FILES=(adapter_config.json adapter_model.safetensors tokenizer.json tokenizer_config.json chat_template.jinja)
OPTIONAL=(special_tokens_map.json)
for f in "${OPTIONAL[@]}"; do
  [[ -f "$ADAPTER_DIR/$f" ]] && FILES+=("$f")
done
(
  cd "$ADAPTER_DIR"
  zip "$ZIP_PATH" "${FILES[@]}"
)

echo "Created $ZIP_PATH"
unzip -l "$ZIP_PATH"
