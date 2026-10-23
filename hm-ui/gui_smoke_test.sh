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
env -u WAYLAND_DISPLAY LIBGL_ALWAYS_SOFTWARE=1 WINIT_UNIX_BACKEND=x11 timeout 5s xvfb-run -a \
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

# Surviving the interval is not enough: a spec that fails to parse leaves the UI
# running with no windows and exits 124 too. State is only written after a
# successful parse, so require the spec's controls to appear in it.
state="${state_dir}/state.json"
for control in Seam_Blend_Mode Seam_Feather_Percent Shadow_Lift_Black_Point; do
  if ! grep -q "${control}" "${state}" 2>/dev/null; then
    echo "hm-ui did not render ${control} from the smoke spec" >&2
    cat "${state}" >&2 2>/dev/null || echo "(no state file)" >&2
    exit 1
  fi
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "Skipping the hm-ui state assertions: python3 is not installed" >&2
  exit 0
fi

# Two separate checks. The spec gives Seam_Blend_Mode a value past its last
# label, so reading back a clamped index shows clamp_spec_choices ran - a plain
# slider clamps to max_value too, so this catches a regression in this build,
# not an older binary. The reported spec_version is what catches that, because
# only a build that knows the field emits it.
if ! python3 - "${state}" <<'EOF'
import json
import sys

state = json.load(open(sys.argv[1]))
selected = state["windows"]["Stitch Blend"]["Seam_Blend_Mode"]
if selected != 2:
    raise SystemExit(f"expected the out-of-range choice index to clamp to 2, got {selected}")
reported = state.get("spec_version")
if reported != 3:
    raise SystemExit(f"expected spec_version 3 from the sidecar, got {reported!r}")
EOF
then
  cat "${state}" >&2 2>/dev/null || true
  exit 1
fi
