"""Leave Bazel's build-tool Python before loading the processing environment.

Wheel console scripts bypass this bootstrap and retain their installed interpreter.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def launch(module: str, workspace: Path) -> None:
    configured = os.environ.get("PYTHON_BIN_PATH")
    conda = os.environ.get("CONDA_PREFIX")
    selected = configured or (
        str(Path(conda) / "bin" / "python") if conda else shutil.which("python3")
    )
    if not selected:
        raise RuntimeError("Set PYTHON_BIN_PATH or activate the HockeyMON Python environment")
    # Preserve a virtual environment's executable symlink: resolving it would
    # select the base interpreter and lose that environment's dependencies.
    interpreter = Path(selected).expanduser().absolute()
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RuntimeError(f"The selected HockeyMON Python is not executable: {interpreter}")

    runfiles = workspace.parent
    environment = os.environ.copy()
    user_paths = [
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and not Path(entry).absolute().is_relative_to(runfiles)
    ]
    # Carry the application's runfiles, but do not import packages for Bazel's
    # hermetic interpreter into a different processing Python ABI.
    environment["PYTHONPATH"] = os.pathsep.join([str(workspace), *user_paths])
    if module == "hmlib.ui.desktop":
        environment.setdefault("HM_CLI_BIN", str(workspace / "hm-ui" / "hm-cli"))
    os.execve(interpreter, [str(interpreter), "-m", module, *sys.argv[1:]], environment)
