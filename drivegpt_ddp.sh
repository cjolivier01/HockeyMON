#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous host reachable by all nodes}"
: "${NODE_RANK:?Set NODE_RANK to the zero-based rank of this node}"

exec "${PYTHON_BIN_PATH:-python}" -m torch.distributed.run \
  --nnodes="${NNODES:-2}" \
  --nproc-per-node="${NPROC_PER_NODE:-1}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT:-29517}" \
  -m hmlib.cli.camgpt_train \
  --config "${ROOT_DIR}/hmlib/config/training/drivegpt.yaml" "$@"
