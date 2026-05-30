#!/usr/bin/env bash
# Build the nemotron training image.
#
# GPU access in Docker RUN steps is not possible on this system: Docker OCI workers
# have isolated device namespaces that don't inherit GPU devices from the host, even
# with a privileged buildkitd daemon. The mamba-ssm selective_scan_cuda extension
# therefore cannot be compiled at build time; a try/except patch in the Dockerfile
# wraps the import so the package loads cleanly at runtime. This is safe for Nemotron,
# which uses Mamba-2 Triton kernels exclusively and never calls the legacy CUDA path.
#
# Usage:
#   bash scripts/build_image.sh          # build primary image (Dockerfile.gb10, 26.04)
#   bash scripts/build_image.sh --fresh  # force-recompile mamba/causal-conv1d
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

FRESH="${1:-}"

DOCKERFILE="Dockerfile.gb10"
IMAGE="nemotron-gb10:latest"

MAMBA_ARG=""
if [[ "$FRESH" == "--fresh" ]]; then
    MAMBA_ARG="--build-arg MAMBA_REBUILD=$(date +%s)"
fi

echo "Building $IMAGE from $DOCKERFILE …"
docker build \
  -f "${WORKSPACE}/${DOCKERFILE}" \
  ${MAMBA_ARG} \
  -t "$IMAGE" \
  "${WORKSPACE}"
