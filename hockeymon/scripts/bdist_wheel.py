#!/usr/bin/env python3
import glob
import os
import shutil


def find_wheel():
    # Look for wheel in various possible locations
    patterns = [
        "hockeymon-*.whl",
        "*/hockeymon-*.whl",
        "*/*/hockeymon-*.whl",
        "bazel-out/*/bin/hockeymon/hockeymon-*.whl",
    ]

    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return files[0]

    # Check runfiles
    runfiles_dir = os.environ.get("RUNFILES_DIR", "")
    if runfiles_dir:
        for pattern in patterns:
            files = glob.glob(os.path.join(runfiles_dir, pattern))
            if files:
                return files[0]

    return None


def main():
    wheel_path = find_wheel()
    if not wheel_path:
        print("Error: Could not find wheel file")
        return 1

    # Get workspace root
    workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    dist_dir = os.path.join(workspace_root, "dist")

    # Create dist directory
    os.makedirs(dist_dir, exist_ok=True)

    # Copy wheel
    wheel_name = os.path.basename(wheel_path)
    dest_path = os.path.join(dist_dir, wheel_name)
    shutil.copy2(wheel_path, dest_path)
    os.chmod(dest_path, 0o644)
    # Remove execute flag
    print(f"Installed {wheel_name} to {dist_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())
