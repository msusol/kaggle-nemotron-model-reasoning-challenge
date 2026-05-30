#!/usr/bin/env bash
# Source this file to export all keys from configs/nemotron.yaml as uppercase env vars.
# Usage: source scripts/load_config.sh [path/to/config.yaml]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

FILES=("${@:-${WORKSPACE}/configs/nemotron.yaml}")

eval "$(python3 - "${FILES[@]}" <<'PYEOF'
import sys, shlex, yaml

merged = {}
for path in sys.argv[1:]:
    with open(path) as f:
        merged.update(yaml.safe_load(f) or {})

for k, v in merged.items():
    # Export booleans as lowercase so bash [[ "$VAR" == "true" ]] comparisons work.
    val = str(v).lower() if isinstance(v, bool) else str(v)
    print(f"export {k.upper()}={shlex.quote(val)}")
PYEOF
)"
