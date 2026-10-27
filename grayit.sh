#!/usr/bin/env bash
# Export the gray-background dataset using the tested RTX 4090 batch size.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-${HOME}/miniforge3/envs/ubuntu/bin/python}"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "--team-id" ]]; then
    shift
    inputs=(--team-id-root /data/datasets/koshkina/team-id/York
            --output /data/datasets/koshkina/team-id-gray)
    batch_size=1
else
    inputs=(--annotations /data/datasets/hockeyTrackingDataset-coco/annotations/train.json
                          /data/datasets/hockeyTrackingDataset-coco/annotations/val.json
            --image-root /data/datasets/hockeyTrackingDataset
            --output /data/datasets/hockeyTrackingDataset-gray)
    batch_size=8
fi

exec "${python_bin}" -u "${repo_dir}/scripts/gray_hockey_background.py" \
    "${inputs[@]}" \
    --device cuda:0 \
    --batch-size "${batch_size}" \
    --workers 4 \
    "$@"
