#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/..")"

find_bazel_workspace_root() {
  local directory

  directory="$(pwd -P)"
  while true; do
    if [[ -f "${directory}/MODULE.bazel" ||
          -f "${directory}/REPO.bazel" ||
          -f "${directory}/WORKSPACE.bazel" ||
          -f "${directory}/WORKSPACE" ]]; then
      printf '%s\n' "${directory}"
      return
    fi

    if [[ "${directory}" == "/" ]]; then
      return
    fi
    directory="$(dirname "${directory}")"
  done
}

source "${SCRIPT_DIR}/../.bazel_setup.sh"

if [[ -z "${PYTHON_BIN_PATH:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN_PATH="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN_PATH="$(command -v python3)"
  else
    echo "error: could not find python or python3 on PATH" >&2
    exit 1
  fi
fi

# pybind11_bazel's python_configure looks for this env var.
export PYTHON_BIN_PATH

BAZEL_WORKSPACE_ROOT="$(find_bazel_workspace_root)"

for arg in "$@"; do
  if [[ "${arg}" != "//..." || -z "${BAZEL_WORKSPACE_ROOT}" ]]; then
    continue
  fi

  workspace_link="${BAZEL_WORKSPACE_ROOT}/bazel-$(basename "${BAZEL_WORKSPACE_ROOT}")"
  if [[ -L "${workspace_link}" ]]; then
    if ! expected_execution_root="$(bazelisk info execution_root --color=no 2>/dev/null)"; then
      printf '%s\n' \
        "Warning: could not determine Bazel's execution root; leaving ${workspace_link} unchanged." >&2
    elif [[ "$(readlink -f "${workspace_link}" || true)" != "${expected_execution_root}" ]]; then
      printf '%s\n' "Removing stale Bazel workspace symlink: ${workspace_link}" >&2
      unlink "${workspace_link}"
    fi
  fi
  break
done

case "${1:-}:${BAZEL_WORKSPACE_ROOT}" in
  build:"${REPO_ROOT}"|coverage:"${REPO_ROOT}"|run:"${REPO_ROOT}"|test:"${REPO_ROOT}")
    CUDA_BAZEL_ARCHS="$(detect_cuda_bazel_archs)"
    export CUDA_BAZEL_ARCHS
    BAZEL_FLAGS="${BAZEL_FLAGS} --@rules_cuda//cuda:archs=${CUDA_BAZEL_ARCHS}"
    ;;
esac

args=("$@")
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--" ]]; then
    before=("${args[@]:0:i}")
    after=("${args[@]:i+1}")
    exec bazelisk "${before[@]}" ${BAZEL_FLAGS} -- "${after[@]}"
  fi
done

exec bazelisk "${args[@]}" ${BAZEL_FLAGS}
