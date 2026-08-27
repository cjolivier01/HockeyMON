#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_default_tag(repo_root: Path) -> str:
    tag_path = repo_root / "env" / "tag"
    if tag_path.is_file():
        tag = tag_path.read_text(encoding="utf-8").strip()
        if tag:
            return tag
    return "hm-cuda"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=check,
        text=True,
    )


def _is_transient_dns_failure(text: str) -> bool:
    lower = str(text).lower()
    return any(
        marker in lower
        for marker in (
            "temporary failure in name resolution",
            "could not resolve host",
            "failed to resolve host",
            "failure in name resolution",
        )
    )


def _run_streaming(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, bool]:
    print("+", shlex.join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    saw_transient_dns_failure = False
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if _is_transient_dns_failure(line):
            saw_transient_dns_failure = True
    return int(proc.wait()), saw_transient_dns_failure


def _docker_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("DOCKER_BUILDKIT", "1")
    return env


def _docker_image_user(tag: str) -> str | None:
    """Return the configured USER for the image, if available."""
    try:
        out = subprocess.check_output(
            ["docker", "image", "inspect", "--format", "{{.Config.User}}", tag],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _container_home_for_user(user: str | None) -> str:
    if not user or user in ("root", "0", "0:0"):
        return "/root"
    # Common case: USER is a name ("hm") or "uid:gid" ("1000:1000").
    if ":" in user:
        return "/root"
    return f"/home/{user}"


def _has_nvidia_gpu() -> bool:
    # Best-effort: if /dev/nvidiactl exists, NVIDIA driver is loaded.
    if Path("/dev/nvidiactl").exists():
        return True
    # Fallback: nvidia-smi should succeed if a usable NVIDIA GPU/driver is present.
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _docker_bridge_interface_exists() -> bool:
    # Docker's default bridge network typically uses an interface named "docker0".
    # If that interface is missing, `docker run`/`docker build` with bridge networking
    # can fail with "adding interface veth... to bridge docker0 failed: Device does not exist".
    return Path("/sys/class/net/docker0").exists()


def _default_build_network() -> str:
    # For `docker build` (BuildKit), valid network modes include: default, host, none.
    # "bridge" is a `docker run` network mode and fails with:
    #   network mode "bridge" not supported by buildkit
    return "default" if _docker_bridge_interface_exists() else "host"


def _default_run_network() -> str:
    # For `docker run`, "bridge" is the typical default network.
    return "bridge" if _docker_bridge_interface_exists() else "host"


def cmd_build(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    tag = args.tag or _read_default_tag(repo_root)

    network = args.network or _default_build_network()
    network_was_explicit = args.network is not None
    if network == "bridge":
        network = "default"

    build_cmd = [
        "docker",
        "build",
        "--network",
        network,
        "-f",
        str(repo_root / "env" / "Dockerfile"),
        "-t",
        tag,
        "--build-arg",
        f"USERNAME={args.username}",
        "--build-arg",
        f"UID={args.uid}",
        "--build-arg",
        f"GID={args.gid}",
        str(repo_root),
    ]
    returncode, saw_transient_dns_failure = _run_streaming(build_cmd, env=_docker_env())
    if (
        returncode
        and not network_was_explicit
        and network == "default"
        and saw_transient_dns_failure
    ):
        retry_cmd = list(build_cmd)
        retry_cmd[retry_cmd.index("--network") + 1] = "host"
        print(
            "NOTE: docker build hit a transient DNS failure; retrying with --network host",
            file=sys.stderr,
        )
        returncode, _ = _run_streaming(retry_cmd, env=_docker_env())
    if returncode:
        raise SystemExit(returncode)


def cmd_run(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    tag = args.tag or _read_default_tag(repo_root)

    network = args.network
    if network == "default":
        network = "bridge"

    docker_cmd = ["docker", "run", "--rm", "-i"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        docker_cmd.append("-t")
    docker_cmd += ["--network", network, "--shm-size", args.shm_size]

    gpus: str | None
    if args.gpus == "auto":
        gpus = "all" if _has_nvidia_gpu() else None
        if gpus is None:
            print(
                "NOTE: No NVIDIA GPU/driver detected on the host; running without Docker GPU flags. "
                "Use --gpus=all to force.",
                file=sys.stderr,
            )
    elif args.gpus in ("none", ""):
        gpus = None
    else:
        gpus = args.gpus

    if gpus is not None:
        docker_cmd += ["--gpus", gpus]
        # Ensure NVIDIA Container Toolkit mounts NVENC/NVDEC libs (e.g. libnvidia-encode.so.1).
        docker_cmd += ["-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video"]

    if args.name:
        docker_cmd += ["--name", args.name]

    if args.videos_mount:
        host_videos = Path(args.videos_mount).expanduser().resolve()
        image_user = _docker_image_user(tag)
        # Prefer the image USER for mounts so `hmtrack` resolves `$HOME/Videos`
        # correctly even when the image was built with USERNAME != host $USER.
        container_home = _container_home_for_user(image_user or args.username)
        container_videos = f"{container_home}/Videos"
        print(
            f"NOTE: Mounting videos: {host_videos} -> {container_videos}"
            + (f" (image USER={image_user})" if image_user else ""),
            file=sys.stderr,
        )
        docker_cmd += ["-v", f"{host_videos}:{container_videos}:rw"]
        # Also mount at a stable path for convenience.
        docker_cmd += ["-v", f"{host_videos}:/Videos:rw"]
        # If the image user differs from the host username, keep the old mount
        # location too so both paths work.
        if image_user and image_user != args.username and ":" not in image_user:
            docker_cmd += ["-v", f"{host_videos}:/home/{args.username}/Videos:rw"]

    container_workdir: str | None = None

    if args.dev_mount:
        docker_cmd += ["-v", f"{repo_root}:/workspace/hm:rw"]
        docker_cmd += ["-e", "PYTHONPATH=/workspace/hm:/workspace/hm/src"]
        container_workdir = "/workspace/hm"

    if args.workdir:
        host_workdir = Path(args.workdir).expanduser().resolve()
        host_workdir.mkdir(parents=True, exist_ok=True)
        container_workdir = "/workspace/workdir"
        print(
            f"NOTE: Mounting workdir: {host_workdir} -> {container_workdir} "
            "(relative outputs like ./output_workdirs are written here)",
            file=sys.stderr,
        )
        docker_cmd += ["-v", f"{host_workdir}:{container_workdir}:rw"]

    if container_workdir:
        docker_cmd += ["-w", container_workdir]

    cmd = args.command if args.command else ["bash"]
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    docker_cmd.append(tag)
    docker_cmd += cmd

    proc = _run(docker_cmd, check=False)
    if proc.returncode:
        raise SystemExit(proc.returncode)


def main(argv: list[str]) -> int:
    repo_root = _repo_root()
    default_tag = _read_default_tag(repo_root)
    default_build_network = _default_build_network()
    default_run_network = _default_run_network()

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tag", default=default_tag, help=f"Docker tag (default: {default_tag})")
    common.add_argument(
        "--username", default=os.environ.get("USER", "hm"), help="Username in image"
    )
    common.add_argument("--uid", type=int, default=os.getuid(), help="UID for user in image")
    common.add_argument("--gid", type=int, default=os.getgid(), help="GID for user in image")

    parser = argparse.ArgumentParser(
        prog="hm_cuda_container.py",
        description=(
            "Build and run a CUDA Docker image with HockeyMON fully installed (Bazel-built wheels). "
            "By default, only $HOME/Videos is mounted into the container."
        ),
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    build = subparsers.add_parser("build", parents=[common], help="Build the CUDA image")
    build.add_argument(
        "--network",
        default=None,
        help=(
            "Docker networking mode for build containers "
            f"(default: auto -> {default_build_network})"
        ),
    )
    build.set_defaults(func=cmd_build)

    run = subparsers.add_parser(
        "run", parents=[common], help="Run the CUDA container with GPU access"
    )
    run.add_argument(
        "--network",
        default=default_run_network,
        help=f"Docker networking mode for the container (default: {default_run_network})",
    )
    run.add_argument("--name", default=None, help="Optional container name")
    run.add_argument(
        "--gpus",
        default="auto",
        help='Docker --gpus value (default: auto; use "none" to disable)',
    )
    run.add_argument(
        "--no-gpus",
        dest="gpus",
        action="store_const",
        const="none",
        help="Disable GPU access (omit --gpus)",
    )
    run.add_argument(
        "--videos-mount",
        default=str(Path.home() / "Videos"),
        help="Host videos directory to mount into $HOME/Videos (and /Videos) (default: ~/Videos)",
    )
    run.add_argument(
        "--no-videos-mount",
        dest="videos_mount",
        action="store_const",
        const=None,
        help="Do not mount any host videos directory",
    )
    run.add_argument(
        "--dev-mount",
        action="store_true",
        help="Bind-mount this repo into /workspace/hm for development (overrides PYTHONPATH)",
    )
    run.add_argument(
        "--workdir",
        default=None,
        help=(
            "Host directory to mount as /workspace/workdir and use as container cwd. "
            "Relative output paths (for example ./output_workdirs) are created there."
        ),
    )
    run.add_argument("--shm-size", default="2g", help="Shared memory size (default: 2g)")
    run.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        default=["bash"],
        help="Command to run in the container (default: bash)",
    )
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
