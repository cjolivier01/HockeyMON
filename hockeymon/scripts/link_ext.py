#!/usr/bin/env python3
import glob
import os
import sys
import sysconfig


def resolve_workspace_root() -> str:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY") or os.getcwd()
    if os.path.isdir(os.path.join(workspace, "bazel-bin")):
        return workspace

    # If we're running from the Bazel execroot, resolve the real workspace root
    # via the symlinked source tree.
    hm_dir = os.path.join(workspace, "hockeymon")
    if os.path.exists(hm_dir):
        real_root = os.path.dirname(os.path.realpath(hm_dir))
        if os.path.isdir(os.path.join(real_root, "bazel-bin")):
            return real_root

    return workspace


def resolve_extension(workspace: str, so_name: str):
    bazel_bin = os.path.join(workspace, "bazel-bin", "hockeymon", so_name)
    if os.path.exists(bazel_bin):
        return bazel_bin

    matches = glob.glob(os.path.join(workspace, "bazel-out", "*", "bin", "hockeymon", so_name))
    if matches:
        matches.sort()
        return matches[0]

    return None


def main() -> int:
    workspace = resolve_workspace_root()

    # Prefer SOABI passed from Bazel (matches the built extension),
    # fall back to the runtime Python's SOABI if not provided.
    soabi = sys.argv[1] if len(sys.argv) > 1 else sysconfig.get_config_var("SOABI")
    if not soabi:
        print("Error: SOABI is not defined for this Python", file=sys.stderr)
        return 1

    so_name = f"_hockeymon.{soabi}.so"

    # Location of the built extension as seen from the workspace.
    src = resolve_extension(workspace, so_name)
    if not src:
        print(f"Error: built extension not found for {so_name}", file=sys.stderr)
        return 1

    pkg_dir = os.path.join(workspace, "hockeymon")
    os.makedirs(pkg_dir, exist_ok=True)
    dst = os.path.join(pkg_dir, so_name)

    # Skip relinking when the symlink already points at the right target.
    rel_src = os.path.relpath(src, pkg_dir)
    if os.path.islink(dst) and os.readlink(dst) == rel_src:
        print(f"Symlink already present: {dst} -> {rel_src}")
        return 0

    # Remove any existing file so we can replace it with a symlink.
    try:
        if os.path.lexists(dst):
            os.remove(dst)
    except OSError as exc:  # pragma: no cover - best-effort cleanup
        print(f"Warning: could not remove existing {dst}: {exc}", file=sys.stderr)

    # Create a relative symlink so it remains valid if the workspace is moved.
    os.symlink(rel_src, dst)
    print(f"Linked {dst} -> {rel_src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
