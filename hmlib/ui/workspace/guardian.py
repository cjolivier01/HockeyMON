"""Keep a job's descendants owned if the desktop process disappears."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        # The whole group can exit between poll() and the signal; its ownership
        # obligation is already satisfied in that case.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", type=int, required=True)
    parser.add_argument("--child-state", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A child command is required")
    if os.getppid() != args.owner:
        raise RuntimeError("The desktop exited before its job could start")
    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    child = subprocess.Popen(command, start_new_session=True)
    temporary = args.child_state.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": child.pid}))
    temporary.replace(args.child_state)
    deadline = None
    while True:
        code = child.poll()
        if code is not None:
            # Retire background descendants even after their original parent
            # has exited. This process remains outside the pipeline group.
            _signal_group(child.pid, signal.SIGKILL)
            return code if code >= 0 else 128 - code
        if stopping or os.getppid() != args.owner:
            if deadline is None:
                # The guardian stays running when the pipeline is paused.
                # Losing its owner can therefore still retire paused workers.
                _signal_group(child.pid, signal.SIGCONT)
                _signal_group(child.pid, signal.SIGTERM)
                deadline = time.monotonic() + 8
            elif time.monotonic() >= deadline:
                _signal_group(child.pid, signal.SIGKILL)
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
