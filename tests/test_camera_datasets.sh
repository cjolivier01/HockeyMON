#!/usr/bin/env bash

set -euo pipefail

repo_root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

source "${repo_root}/tests/python_runtime.sh"
python_bin="$(resolve_repo_python pytest torch numpy pandas)"

"${python_bin}" -m pytest \
  "${repo_root}/tests/${1:?Expected a camera test filename}" \
  -q \
  -c "${repo_root}/pyproject.toml"
