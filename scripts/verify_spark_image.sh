#!/usr/bin/env zsh
# Verify nemo-rl-spark:latest — GPU, HBM, key packages, and nano-v3 branch.
# Run on DGX Spark after build completes.
#
# Usage:
#   bash scripts/verify_spark_image.sh
set -euo pipefail

IMAGE="nemo-rl-spark:latest"

docker run --rm --privileged -e NVIDIA_VISIBLE_DEVICES=all "${IMAGE}" \
  python3 -c "
import torch, transformers, causal_conv1d
from mamba_ssm.ops import selective_scan_interface

print('=== GPU ===')
print(f'  Device:         {torch.cuda.get_device_name(0)}')
f, t = torch.cuda.mem_get_info()
print(f'  HBM free:       {f/1e9:.1f} GB / {t/1e9:.1f} GB total')

print()
print('=== Packages ===')
print(f'  transformers:   {transformers.__version__}')
print(f'  causal_conv1d:  OK')
print(f'  mamba_ssm:      OK')

import nemo_rl
print(f'  nemo_rl:        OK')

import subprocess
result = subprocess.run(
    ['git', '-C', '/opt/nemo-rl', 'branch', '--show-current'],
    capture_output=True, text=True)
branch = result.stdout.strip()
print()
print('=== NeMo RL ===')
print(f'  branch:         {branch}')
assert branch == 'nano-v3', f'Expected nano-v3, got {branch}'
print()
print('All checks passed.')
"
