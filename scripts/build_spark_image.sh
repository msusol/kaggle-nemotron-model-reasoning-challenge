#!/usr/bin/env zsh
# Build the NeMo RL Spark image for v0.11 LoRA GRPO on DGX Spark (GB10, aarch64).
#
# Expected build time: 30–90 min (network + causal-conv1d / mamba-ssm source builds).
#
# MAX_JOBS=4 caps parallel cmake/nvcc processes. The GB10 has 72 ARM cores and
# cmake without a job cap will OOM the host during CUDA extension source builds.
#
# --memory 100g caps the build container's cgroup so it is killed before the
# OS runs out of its ~121 GB pool (kills build, not sshd/systemd).
#
# IMPORTANT: Always run inside tmux to survive SSH disconnects.
#
# Usage:
#   tmux new -s build_spark
#   bash scripts/build_spark_image.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

DOCKERFILE="${WORKSPACE}/docs/plans/lora-grpo-spark-plan/Dockerfile.spark"
IMAGE="nemo-rl-spark:latest"
MAX_JOBS="${MAX_JOBS:-4}"

echo "Building ${IMAGE} from ${DOCKERFILE}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "Memory cap: 100 GB (build container killed before OS if OOM)"
echo ""

docker build \
  -f "${DOCKERFILE}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  --memory 100g \
  -t "${IMAGE}" \
  "${WORKSPACE}"

echo ""
echo "Build complete: ${IMAGE}"
echo "Verify with:"
echo "  docker run --rm --privileged -e NVIDIA_VISIBLE_DEVICES=all ${IMAGE} \\"
echo "    python3 -c \"import torch; print(torch.cuda.get_device_name(0))\""
