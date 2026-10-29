#!/usr/bin/env python3
"""
Install hm Python packages in editable/develop mode.

This writes a ``.pth`` file pointing at the workspace root and creates lightweight
console entry-point shims in the current Python environment's scripts directory.
"""

import argparse
import logging
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from textwrap import dedent

ENTRY_POINTS = {
    "hmtrack": "hmlib.cli.hmtrack:main",
    "hmstitch": "hmlib.cli.stitch:main",
    "hmcreate_control_points": "hmlib.cli.create_control_points:main",
    "hmplayers": "hmlib.cli.players:main",
    "hmfind_ice_rink": "hmlib.cli.find_ice_rink:main",
    "hmfind_rink_landmarks": "hmlib.cli.find_rink_landmarks:main",
    "hmpostprocess_shifts": "hmlib.cli.postprocess_shifts:main",
    "hmorientation": "hmlib.cli.hmorientation:main",
    "hmscoreboard": "hmlib.scoreboard.selector:main",
    "hmconcatenate_videos": "hmlib.cli.concatenate_videos:main",
    "hmcamera_annotate": "hmlib.cli.camera_annotate:main",
}

LOCAL_EDITABLE_PACKAGES = (
    "openmm/mmengine",
    "openmm/mmeval",
    "openmm/mmdetection",
    "openmm/mmpose",
    "openmm/mmocr",
    "openmm/mmsegmentation",
    "openmm/mmyolo",
    "openmm/mmaction2",
    "xmodels/str/parseq",
)


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _find_workspace_root() -> Path:
    env_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if env_root:
        return Path(env_root).resolve()
    marker_names = ("WORKSPACE", "WORKSPACE.bazel")
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if any((parent / name).exists() for name in marker_names):
            return parent
    return Path.cwd().resolve()


def _write_pth(site_packages: Path, workspace_root: Path) -> Path:
    pth_path = site_packages / "hmlib-development.pth"
    pth_path.write_text(f"{workspace_root}\n", encoding="utf-8")
    return pth_path


def _write_script(scripts_dir: Path, name: str, target: str) -> Path:
    module_name, func_name = target.split(":")
    script_path = scripts_dir / name
    script_body = dedent(f"""\
        #!{sys.executable}
        import sys
        from importlib import import_module


        def _main():
            mod = import_module("{module_name}")
            func = getattr(mod, "{func_name}")
            return func()


        if __name__ == "__main__":
            sys.exit(_main())
        """)
    script_path.write_text(script_body, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _install_editable_package(
    python: str, package_root: Path, *, no_deps: bool, no_build_isolation: bool = False
) -> int:
    cmd = [python, "-m", "pip", "install", "-e", str(package_root)]
    if no_deps:
        cmd.insert(4, "--no-deps")
    if no_build_isolation:
        cmd.insert(4, "--no-build-isolation")
    logger.info("Running %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=package_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install hm Python packages in develop/editable mode."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Override the workspace root (defaults to auto-detection).",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", "python"),
        help="Python executable for the active development environment.",
    )
    parser.add_argument(
        "--legacy-pth",
        action="store_true",
        help="Write the historical .pth and console-script shims instead of using pip editable install.",
    )
    args = parser.parse_args()

    workspace_root = (args.workspace or _find_workspace_root()).resolve()
    if not args.legacy_pth:
        status = _install_editable_package(args.python, workspace_root, no_deps=False)
        if status != 0:
            return status
        for rel_path in LOCAL_EDITABLE_PACKAGES:
            package_root = workspace_root / rel_path
            status = _install_editable_package(
                args.python,
                package_root,
                no_deps=True,
                no_build_isolation=True,
            )
            if status != 0:
                return status
        return 0

    site_packages = Path(sysconfig.get_paths()["purelib"])
    scripts_dir = Path(sysconfig.get_paths()["scripts"])
    site_packages.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    pth_file = _write_pth(site_packages, workspace_root)
    scripts_written = [
        _write_script(scripts_dir, name, target) for name, target in ENTRY_POINTS.items()
    ]

    logger.info("Added %s pointing to %s", pth_file, workspace_root)
    logger.info("Installed %d console entry points into %s", len(scripts_written), scripts_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
