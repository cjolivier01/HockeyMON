"""Run a Python pipeline with cooperative desktop process-group shutdown."""

from __future__ import annotations

import os
import runpy
import signal
from pathlib import Path


def main() -> None:
    if "HM_WEB_PROGRAM_REQUEST" in os.environ:
        from hmlib.ui.hstream_web import verify_worker_environment

        verify_worker_environment()
    from hmlib.ui.hstream_runtime import write_document

    session = Path(os.environ["HM_UI_SESSION_DIR"])

    def stop(signum, frame):
        write_document(session / "operator-actions" / "signal-stop.json", {"kind": "stop"})

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    runpy.run_module("hmlib.cli.hmtrack", run_name="__main__")


if __name__ == "__main__":
    main()
