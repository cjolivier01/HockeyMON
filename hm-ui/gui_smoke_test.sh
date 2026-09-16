#!/usr/bin/env bash

set -euo pipefail

hm_ui="$1"
spec="$2"

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "Skipping hm-ui GUI smoke test: xvfb-run is not installed" >&2
  exit 0
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "Skipping hm-ui GUI smoke test: timeout is not installed" >&2
  exit 0
fi

state_dir="$(mktemp -d "${TEST_TMPDIR:-/tmp}/hm-ui-gui-smoke.XXXXXX")"
trap 'rm -rf "${state_dir}"' EXIT

set +e
LIBGL_ALWAYS_SOFTWARE=1 WINIT_UNIX_BACKEND=x11 timeout 5s xvfb-run -a \
  "${hm_ui}" \
  --spec "${spec}" \
  --state "${state_dir}/state.json" \
  --title "HM UI GUI Smoke Test"
status=$?
set -e

if [[ "${status}" -ne 124 ]]; then
  echo "hm-ui exited during the GUI smoke interval with status ${status}" >&2
  exit 1
fi
