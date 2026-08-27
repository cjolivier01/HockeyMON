#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

print_help() {
  cat <<'EOF'
DockerRun.sh: run HockeyMON CUDA container commands

Usage:
  ./DockerRun.sh [global options] <command> [command args...]
  ./DockerRun.sh --help

Main commands:
  hmtrack   Run end-to-end tracking on a game/video
  hmstitch  Stitch left/right cameras into a panorama (auto-clamps NVENC width)

Other commands:
  list-cli  List installed hm* CLIs inside the container
  bash      Open an interactive shell inside the container
  console   Same as 'bash'
  python    Run python inside the container (args passed through)
  docs      Show container user docs (use --open to open in browser)
           Use --serve to host docs on localhost

Global options (passed to scripts/hm_cuda_container.py run):
  --tag TAG              Docker image tag (default: env/tag if present)
  --gpus VALUE           Docker --gpus value (default: auto)
  --no-gpus              Run without GPU flags
  --videos-mount DIR     Host videos dir (default: ~/Videos)
  --no-videos-mount      Disable videos mount
  --workdir DIR          Host work dir mounted as container cwd
  --dev-mount            Bind-mount this repo into /workspace/hm
  --name NAME            Container name
  --network NET          Container network (default: bridge)
  --shm-size SIZE        Shared memory (default: 2g)

hmstitch NVENC width clamp:
  When hmstitch writes a video with NVENC, panoramas wider than 8192 can fail.
  This wrapper auto-adds: --config-override aspen.stitching.max_output_width=8192
  unless you already set it, or you pass: --no-hwenc-clamp

Examples:
  ./DockerBuild.sh
  ./DockerRun.sh bash
  ./DockerRun.sh hmtrack --game-id stockton-r3 -t=60
  ./DockerRun.sh hmstitch --game-id stockton-r3 -o stitched.mp4
  ./DockerRun.sh hmstitch --no-hwenc-clamp --game-id stockton-r3 -o stitched.mp4
  ./DockerRun.sh --workdir /tmp/hm-work hmtrack --game-id stockton-r3
  ./DockerRun.sh list-cli
  ./DockerRun.sh docs --open
  ./DockerRun.sh docs --serve
EOF
}

if [[ $# -eq 0 ]] || [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

run_args=()
hwenc_clamp=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      print_help
      exit 0
      ;;
    --no-hwenc-clamp)
      hwenc_clamp=0
      shift
      ;;
    --tag|--gpus|--videos-mount|--workdir|--name|--network|--shm-size)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a value" >&2
        exit 2
      fi
      run_args+=("$1" "$2")
      shift 2
      ;;
    --no-gpus|--no-videos-mount|--dev-mount)
      run_args+=("$1")
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: Unknown option: $1" >&2
      echo "Run: ./DockerRun.sh --help" >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

cmd="${1:-}"
shift || true

runner=( "${ROOT}/p" "${ROOT}/scripts/hm_cuda_container.py" run )

ensure_tag() {
  # If the user didn't specify --tag and the default tag isn't built locally,
  # fall back to a commonly used local tag ("hm") to reduce confusion.
  local tag_in_args=0
  for ((i = 0; i < ${#run_args[@]}; i++)); do
    if [[ "${run_args[$i]}" == "--tag" ]]; then
      tag_in_args=1
      break
    fi
  done
  if [[ "${tag_in_args}" -eq 1 ]]; then
    return 0
  fi

  local default_tag="hm-cuda"
  if [[ -f "${ROOT}/env/tag" ]]; then
    local t
    t="$(<"${ROOT}/env/tag")"
    t="${t//$'\n'/}"
    if [[ -n "${t}" ]]; then
      default_tag="${t}"
    fi
  fi

  if ! docker image inspect "${default_tag}" >/dev/null 2>&1; then
    if docker image inspect "hm" >/dev/null 2>&1; then
      run_args+=( "--tag" "hm" )
      echo "NOTE: Default tag '${default_tag}' not found locally; using --tag hm" >&2
    fi
  fi
}

case "${cmd}" in
  hmtrack)
    ensure_tag
    exec "${runner[@]}" "${run_args[@]}" -- hmtrack "$@"
    ;;
  hmstitch)
    ensure_tag
    args=( "$@" )

    if [[ "${hwenc_clamp}" -eq 1 ]]; then
      has_output=0
      has_clamp_override=0
      configure_only=0
      for a in "${args[@]}"; do
        case "${a}" in
          -o|--output|--output-video)
            has_output=1
            ;;
          --configure-only)
            configure_only=1
            ;;
          --config-override)
            # handled when we see the next arg; keep scanning
            ;;
          aspen.stitching.max_output_width=*)
            has_clamp_override=1
            ;;
        esac
      done

      # If user supplied "--config-override key=value", the key=value is a separate arg.
      for a in "${args[@]}"; do
        case "${a}" in
          aspen.stitching.max_output_width=*)
            has_clamp_override=1
            ;;
        esac
      done

      if [[ "${has_output}" -eq 1 ]] && [[ "${configure_only}" -eq 0 ]] && [[ "${has_clamp_override}" -eq 0 ]]; then
        args+=( "--config-override" "aspen.stitching.max_output_width=8192" )
      fi
    fi

    exec "${runner[@]}" "${run_args[@]}" -- hmstitch "${args[@]}"
    ;;
  list-cli)
    ensure_tag
    exec "${runner[@]}" "${run_args[@]}" -- bash -lc 'ls -1 /opt/conda/envs/hm/bin | grep -E "^hm" | sort'
    ;;
  bash|shell|console)
    ensure_tag
    exec "${runner[@]}" "${run_args[@]}" -- bash "$@"
    ;;
  python)
    ensure_tag
    exec "${runner[@]}" "${run_args[@]}" -- python "$@"
    ;;
  docs)
    open=0
    serve=0
    serve_port=8000
    case "${1:-}" in
      "" )
        ;;
      --open )
        open=1
        shift
        ;;
      --serve )
        serve=1
        shift
        if [[ -n "${1:-}" ]]; then
          serve_port="${1}"
          shift
        fi
        ;;
      * )
        echo "ERROR: docs supports: --open | --serve [port]" >&2
        exit 2
        ;;
    esac
    if [[ -n "${1:-}" ]]; then
      echo "ERROR: docs supports: --open | --serve [port]" >&2
      exit 2
    fi

    md="${ROOT}/docs/container.md"
    cli_md="${ROOT}/hmlib/cli/README.md"
    echo "Container docs:"
    echo "  ${md}"
    echo "CLI docs:"
    echo "  ${cli_md}"

    if [[ "${serve}" -eq 1 ]]; then
      echo
      echo "Serving docs from ${ROOT} on http://127.0.0.1:${serve_port}/"
      echo "  http://127.0.0.1:${serve_port}/docs/container.md"
      echo "  http://127.0.0.1:${serve_port}/hmlib/cli/README.md"
      echo "(Press Ctrl-C to stop)"
      exec python3 -m http.server "${serve_port}" --bind 127.0.0.1 --directory "${ROOT}"
    fi

    if [[ "${open}" -eq 1 ]]; then
      if [[ -z "${DISPLAY:-}" ]] && [[ -z "${WAYLAND_DISPLAY:-}" ]]; then
        echo "NOTE: No GUI display detected (DISPLAY/WAYLAND_DISPLAY empty); cannot auto-open a browser." >&2
        echo "      Use: ./DockerRun.sh docs --serve   then open the printed URL in your browser." >&2
        exit 0
      fi
      if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${md}" >/dev/null 2>&1 || true
        xdg-open "${cli_md}" >/dev/null 2>&1 || true
      elif command -v gio >/dev/null 2>&1; then
        gio open "${md}" >/dev/null 2>&1 || true
        gio open "${cli_md}" >/dev/null 2>&1 || true
      elif command -v open >/dev/null 2>&1; then
        open "${md}" >/dev/null 2>&1 || true
        open "${cli_md}" >/dev/null 2>&1 || true
      else
        echo "NOTE: No opener found (xdg-open/gio/open); open the paths above manually." >&2
      fi
    fi
    exit 0
    ;;
  *)
    echo "ERROR: Unknown command: ${cmd}" >&2
    echo "Run: ./DockerRun.sh --help" >&2
    exit 2
    ;;
esac
