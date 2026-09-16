#
# Bazel-related utility code to be sourced by build scripts
#
if [ ! -e "$(which bazelisk)" ]; then
  echo "Need to install bazelisk."
  ./scripts/install_bazelisk.sh
fi

resolve_conda_prefix() {
  local prefix derived_prefix python_bin

  prefix="${CONDA_PREFIX:-}"
  if [ -n "${prefix}" ] && [ -x "${prefix}/bin/python" ]; then
    printf '%s' "${prefix}"
    return 0
  fi

  python_bin="$(command -v python 2>/dev/null || true)"
  if [ -n "${python_bin}" ]; then
    derived_prefix="$("${python_bin}" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
    if [ -n "${derived_prefix}" ] && [ -x "${derived_prefix}/bin/python" ]; then
      if [ -n "${prefix}" ]; then
        printf '%s\n' \
          "CONDA_PREFIX=${prefix} is invalid; using ${derived_prefix} derived from ${python_bin}." >&2
      else
        printf '%s\n' \
          "CONDA_PREFIX is not set; using ${derived_prefix} derived from ${python_bin}." >&2
      fi
      printf '%s' "${derived_prefix}"
      return 0
    fi
  fi

  if [ -n "${prefix}" ]; then
    printf '%s\n' \
      "Warning: CONDA_PREFIX=${prefix} does not contain bin/python, and no replacement prefix was derived from python on PATH." >&2
  fi

  return 1
}

resolve_torch_backend() {
  local python_bin

  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    python_bin="${CONDA_PREFIX}/bin/python"
  else
    python_bin="$(command -v python 2>/dev/null || true)"
  fi

  if [ -z "${python_bin}" ]; then
    return 1
  fi

  "${python_bin}" -c '
import os
import sys

cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", cwd)]

import torch

if getattr(torch.version, "hip", None):
    print("rocm")
elif getattr(torch.version, "cuda", None):
    print("cuda")
else:
    print("cpu")
' 2>/dev/null || true
}

die_bazel_setup() {
  printf '%s\n' "$*" >&2
  return 1 2>/dev/null || exit 1
}

validate_forced_torch_backend() {
  local forced="$1"
  local detected="$2"

  case "$forced" in
    rocm|cuda|cpu)
      ;;
    *)
      die_bazel_setup \
        "HM_FORCE_TORCH_BACKEND must be one of: rocm, cuda, cpu (got '${forced}')."
      ;;
  esac

  if [ -z "${detected}" ]; then
    die_bazel_setup \
      "HM_FORCE_TORCH_BACKEND=${forced} but the torch backend could not be detected. Activate a matching environment or unset HM_FORCE_TORCH_BACKEND."
  fi

  if [ "${detected}" != "${forced}" ]; then
    die_bazel_setup \
      "HM_FORCE_TORCH_BACKEND=${forced} but detected torch backend ${detected}. Activate a matching environment or unset HM_FORCE_TORCH_BACKEND."
  fi
}

valid_rocm_root() {
  local root="$1"
  local candidate=""

  [ -n "${root}" ] || return 1
  [ -x "${root}/bin/hipcc" ] || return 1
  [ -f "${root}/include/hip/hip_runtime.h" ] || return 1

  for candidate in \
    "${root}/lib/libamdhip64.so" \
    "${root}/lib64/libamdhip64.so" \
    "${root}"/lib/libamdhip64.so.* \
    "${root}"/lib64/libamdhip64.so.*
  do
    [ -f "${candidate}" ] && return 0
  done

  return 1
}

detect_rocm_path() {
  local candidate=""

  for candidate in "${ROCM_PATH:-}" "${HIP_PATH:-}"; do
    [ -n "${candidate}" ] || continue
    valid_rocm_root "${candidate}" && {
      printf '%s\n' "${candidate}"
      return 0
    }
  done

  if command -v hipcc >/dev/null 2>&1; then
    candidate="$(dirname "$(dirname "$(readlink -f "$(command -v hipcc)")")")"
    valid_rocm_root "${candidate}" && {
      printf '%s\n' "${candidate}"
      return 0
    }
  fi

  for candidate in /opt/rocm /opt/rocm-*; do
    valid_rocm_root "${candidate}" && {
      printf '%s\n' "${candidate}"
      return 0
    }
  done

  return 1
}

RESOLVED_CONDA_PREFIX="$(resolve_conda_prefix || true)"
if [ -n "${RESOLVED_CONDA_PREFIX}" ]; then
  export CONDA_PREFIX="${RESOLVED_CONDA_PREFIX}"
fi

CPU="$(uname -m)"
if [ "$CPU" == "x86_64" ]; then
  CPU="k8"
fi

# Get a clean "login" PATH from your login shell
LOGIN_SHELL="${SHELL:-/bin/bash}"
LOGIN_PATH=$(
  env -i HOME="$HOME" USER="$USER" LOGNAME="$LOGNAME" \
    "$LOGIN_SHELL" -lc 'printf %s "$PATH"'
)

# Maybe add the current conda env
if [ ! -z "${CONDA_PREFIX}" ]; then
  LOGIN_PATH="${CONDA_PREFIX}/bin:${LOGIN_PATH}"
fi

if [ -e "${HOME}/.profile" ]; then
  restore_nounset=0
  case "$-" in
    *u*)
      restore_nounset=1
      set +u
      ;;
  esac
  PATH="${LOGIN_PATH}"
  source "${HOME}/.profile"
  if [ "${restore_nounset}" -eq 1 ]; then
    set -u
  fi
  LOGIN_PATH="${PATH}"
fi

# echo "LOGIN_PATH=${LOGIN_PATH}"

BAZEL_FLAGS="--action_env=PATH=${LOGIN_PATH} --repo_env=PATH=${LOGIN_PATH}"

DETECTED_TORCH_BACKEND="$(resolve_torch_backend | tr -d '\n' || true)"
FORCED_TORCH_BACKEND="${HM_FORCE_TORCH_BACKEND:-}"

if [ -n "${FORCED_TORCH_BACKEND}" ]; then
  validate_forced_torch_backend "${FORCED_TORCH_BACKEND}" "${DETECTED_TORCH_BACKEND}" || return 1 2>/dev/null || exit 1
  TORCH_BACKEND="${FORCED_TORCH_BACKEND}"
else
  TORCH_BACKEND="${DETECTED_TORCH_BACKEND}"
fi

if [ -n "${TORCH_BACKEND}" ]; then
  BAZEL_FLAGS="${BAZEL_FLAGS} --define=torch_backend=${TORCH_BACKEND}"
fi
if [ "${TORCH_BACKEND}" = "rocm" ]; then
  ROCM_ROOT="$(detect_rocm_path || true)"
  if [ -n "${ROCM_ROOT}" ]; then
    export ROCM_PATH="${ROCM_ROOT}"
    export HIP_PATH="${ROCM_ROOT}"
    LOGIN_PATH="${ROCM_ROOT}/bin:${LOGIN_PATH}"
    BAZEL_FLAGS="${BAZEL_FLAGS} --repo_env=ROCM_PATH=${ROCM_ROOT} --repo_env=HIP_PATH=${ROCM_ROOT}"
    BAZEL_FLAGS="${BAZEL_FLAGS} --action_env=ROCM_PATH=${ROCM_ROOT} --action_env=HIP_PATH=${ROCM_ROOT}"
  fi
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  BAZEL_FLAGS="${BAZEL_FLAGS} --define=backend=hip --define=jetson_use_hip=true --repo_env=GPU_BACKEND=rocm"
  for sibling_repo in hm-cupano jetson-utils; do
    sibling_path="${REPO_ROOT}/../${sibling_repo}"
    if [ -d "${sibling_path}" ]; then
      sibling_path="$(cd "${sibling_path}" && pwd)"
      BAZEL_FLAGS="${BAZEL_FLAGS} --override_repository=${sibling_repo}=${sibling_path}"
    fi
  done
elif [ "${TORCH_BACKEND}" = "cuda" ]; then
  BAZEL_FLAGS="${BAZEL_FLAGS} --define=backend=cuda --define=jetson_use_hip=false --repo_env=GPU_BACKEND=cuda"
fi
normalize_cuda_arch_for_rules_cuda() {
  local arch="$1"

  arch="${arch#sm_}"
  arch="${arch#compute_}"

  case "$arch" in
    30|32|35|37|50|52|53|60|61|62|70|72|75|80|86|87|89|90|100|101|120)
      printf 'sm_%s\n' "$arch"
      return 0
      ;;
  esac

  if [ "$arch" -gt 120 ] 2>/dev/null; then
    printf '%s\n' 'sm_120'
    return 0
  fi

  return 1
}

find_nvcc() {
  local candidate cuda_root

  if [ -n "${NVCC:-}" ] && [ -x "${NVCC}" ]; then
    printf '%s\n' "${NVCC}"
    return 0
  fi

  candidate="$(command -v nvcc 2>/dev/null || true)"
  if [ -n "${candidate}" ]; then
    printf '%s\n' "${candidate}"
    return 0
  fi

  for cuda_root in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /opt/cuda; do
    if [ -n "${cuda_root}" ] && [ -x "${cuda_root}/bin/nvcc" ]; then
      printf '%s\n' "${cuda_root}/bin/nvcc"
      return 0
    fi
  done

  return 1
}

detect_cuda_archs_from_nvidia_smi() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
    | awk '
      NF {
        gsub(/[^0-9]/, "", $1)
        if (length($1) == 1) {
          $1 = $1 "0"
        }
        print $1
      }
    ' \
    | while IFS= read -r arch; do
        normalize_cuda_arch_for_rules_cuda "$arch" || true
      done \
    | awk 'NF && !seen[$0]++' \
    | paste -sd ';' -
}

detect_cuda_archs_from_nvcc() {
  local nvcc_bin

  nvcc_bin="$(find_nvcc)" || return 1

  "${nvcc_bin}" --list-gpu-code 2>/dev/null \
    | while IFS= read -r arch; do
        normalize_cuda_arch_for_rules_cuda "$arch" || true
      done \
    | awk 'NF && !seen[$0]++' \
    | paste -sd ';' -
}

filter_cuda_archs_by_supported() {
  local requested="$1"
  local supported="$2"

  printf '%s\n' "$requested" \
    | tr ';' '\n' \
    | awk -v supported="$supported" '
      BEGIN {
        supported_count = split(supported, supported_archs, ";")
        for (i = 1; i <= supported_count; i++) {
          is_supported[supported_archs[i]] = 1
        }
      }
      NF && is_supported[$0] && !seen[$0]++ { print }
    ' \
    | paste -sd ';' -
}

filter_cuda_archs_by_unsupported() {
  local requested="$1"
  local supported="$2"

  printf '%s\n' "$requested" \
    | tr ';' '\n' \
    | awk -v supported="$supported" '
      BEGIN {
        supported_count = split(supported, supported_archs, ";")
        for (i = 1; i <= supported_count; i++) {
          is_supported[supported_archs[i]] = 1
        }
      }
      NF && !is_supported[$0] && !seen[$0]++ { print }
    ' \
    | paste -sd ';' -
}

detect_cuda_bazel_archs() {
  local archs="" supported_archs="" filtered_archs="" unsupported_archs=""

  if [ -n "${CUDA_BAZEL_ARCHS:-}" ]; then
    printf '%s\n' "$CUDA_BAZEL_ARCHS"
    return 0
  fi

  archs="$(detect_cuda_archs_from_nvidia_smi)" || true
  if [ -n "$archs" ]; then
    supported_archs="$(detect_cuda_archs_from_nvcc)" || true
    if [ -z "$supported_archs" ]; then
      printf '%s\n' "$archs"
      return 0
    fi

    filtered_archs="$(filter_cuda_archs_by_supported "$archs" "$supported_archs")"
    if [ -n "$filtered_archs" ]; then
      unsupported_archs="$(filter_cuda_archs_by_unsupported "$archs" "$supported_archs")"
      if [ -n "$unsupported_archs" ]; then
        printf '%s\n' \
          "Warning: CUDA compiler does not support detected GPU arch(s): ${unsupported_archs}; using ${filtered_archs}." >&2
      fi
      printf '%s\n' "$filtered_archs"
      return 0
    fi

    printf '%s\n' \
      "Warning: CUDA compiler supports none of detected GPU arch(s): ${archs}; using compiler-supported arch(s): ${supported_archs}. Binaries may not run CUDA kernels on the detected GPU(s)." >&2
    printf '%s\n' "$supported_archs"
    return 0
  fi

  archs="$(detect_cuda_archs_from_nvcc)" || true
  if [ -n "$archs" ]; then
    printf '%s\n' "$archs"
    return 0
  fi

  printf '%s\n' 'sm_75;sm_80;sm_86;sm_87;sm_89;sm_90;sm_100;sm_101;sm_120'
}
