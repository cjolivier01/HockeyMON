"""Resolve the native executable behind HStream's packaged shell launcher."""

from __future__ import annotations

import os
from pathlib import Path


def native_runner(explicit: str | None = None) -> Path:
    from hmlib.ui.desktop import executable

    runner = executable(explicit, "hstream-cli", (Path("/opt/hstream/bin/hstream-cli"),))
    # Debian exposes /usr/bin/hstream-cli as a symlink to <prefix>/run.sh.
    # Companion tools/configs belong to the paired binary, not that script.
    binary = runner.parent / "bin" / "hstream-cli"
    if runner.name == "run.sh" and binary.is_file() and os.access(binary, os.X_OK):
        return binary.resolve()
    return runner
