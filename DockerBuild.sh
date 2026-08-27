#!/bin/bash
set -euo pipefail

print_help() {
  cat <<'EOF'
DockerBuild.sh: build the HockeyMON CUDA Docker image

Usage:
  ./DockerBuild.sh [options...]
  ./DockerBuild.sh --help

Options are passed to: scripts/hm_cuda_container.py build

Common options:
  --tag TAG                     Docker image tag (default: env/tag if present)
  --network default|host|none    Build network mode

The CUDA, PyTorch, and TensorRT versions are pinned together in the Dockerfile
because their binary interfaces must match.

Examples:
  ./DockerBuild.sh
  ./DockerBuild.sh --tag hm
  ./DockerBuild.sh --network host
EOF
}

if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

PYTHONPATH="${REPO_ROOT}" exec python "${REPO_ROOT}/scripts/hm_cuda_container.py" build "$@"
