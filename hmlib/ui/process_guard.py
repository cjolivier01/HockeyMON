"""Own a subprocess tree without removing it from the desktop's process group.

The small guard remains Torch-free. Linux parent-death notification and child
subreaping let it clean up decoders even when the supervising UI/backend dies or
when its immediate child crashes while grandchildren still hold output pipes.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _pidfd_open(pid: int) -> int:
    # These Linux syscall numbers are shared by x86_64 and aarch64. Calling
    # libc.syscall also supports Python/glibc builds predating the wrappers.
    descriptor = ctypes.CDLL(None, use_errno=True).syscall(434, pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def guarded_command(command: list[str]) -> list[str]:
    # Execute the file directly: importing the hmlib.ui package initializes
    # visualization dependencies before this guard can own any descendants.
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--parent-pid",
        str(os.getpid()),
        "--",
        *command,
    ]


class ProcessCancelledError(RuntimeError):
    """The owner cancelled an operation after its subprocess tree was reaped."""


class ProcessCleanupError(RuntimeError):
    """An owned helper has not confirmed complete subprocess-tree cleanup."""


def run_cancellable(
    command: list[str], *, cancelled: threading.Event, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Capture an owned helper, retaining its guard until cancellation cleanup ends."""
    if cancelled.is_set():
        raise ProcessCancelledError("Subprocess operation was cancelled before startup")
    deadline = time.monotonic() + timeout if timeout is not None else None
    process = subprocess.Popen(
        guarded_command(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        while True:
            if cancelled.is_set():
                raise ProcessCancelledError("Subprocess operation was cancelled")
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=0.05)
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    finally:
        if process.poll() is None:
            # Signal the guard, never kill it ahead of its descendants. Its
            # bounded TERM/KILL cleanup must reap the whole tree before the
            # calling selector thread can exit or report itself closed.
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired as error:
                raise ProcessCleanupError("Owned subprocess cleanup did not complete") from error


def _configure_owner(parent: int) -> bool:
    if sys.platform != "linux":
        raise RuntimeError("The shared desktop process guard requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    # prctl does not inherit the parent-death setting across fork. This guard
    # stays alive as the direct child, and owns descendants via subreaping.
    for option, value in ((36, 1), (1, signal.SIGTERM)):
        if libc.prctl(option, value, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    # Refuse an unsupported kernel before starting anything we cannot own.
    os.close(_pidfd_open(os.getpid()))
    return os.getppid() == parent


def _descendants() -> set[int]:
    pending, discovered = [os.getpid()], set()
    while pending:
        parent = pending.pop()
        try:
            tasks = list((Path("/proc") / str(parent) / "task").iterdir())
        except FileNotFoundError:
            continue
        for task in tasks:
            try:
                children = [int(pid) for pid in (task / "children").read_text().split()]
            except FileNotFoundError:
                continue
            for pid in children:
                if pid not in discovered:
                    discovered.add(pid)
                    pending.append(pid)
    return discovered


def _signal_tree(kind: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    for pid in _descendants():
        # Bind the identity before rechecking ownership, so PID reuse between
        # /proc discovery and signaling cannot target an unrelated process.
        try:
            descriptor = _pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            if pid not in _descendants():
                continue
            if libc.syscall(424, descriptor, kind, None, 0) != 0:
                error = ctypes.get_errno()
                if error != errno.ESRCH:
                    raise OSError(error, os.strerror(error))
        finally:
            os.close(descriptor)


def _reap_orphans() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if not pid:
            return


def run(command: list[str], parent: int, *, grace_seconds: float = 2) -> int:
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for kind in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(kind, stop)
    if not _configure_owner(parent):
        return 128 + signal.SIGTERM
    environment = os.environ.copy()
    if "HSTREAM_UI_PARENT_PID" in environment:
        environment["HSTREAM_UI_PARENT_PID"] = str(os.getpid())
    process = subprocess.Popen(command, env=environment)
    try:
        while process.poll() is None and not stopped:
            time.sleep(0.02)
        # Normal completion also cleans up an accidentally leaked decoder.
        # Orphaned descendants are adopted here before the guard can exit.
        deadline = time.monotonic() + grace_seconds
        while _descendants():
            _signal_tree(signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL)
            process.poll()
            if process.returncode is not None:
                _reap_orphans()
            if time.monotonic() >= deadline + 2:
                raise RuntimeError("Subprocess descendants did not exit after SIGKILL")
            time.sleep(0.02)
        process.wait()
        if stopped:
            return 128 + signal.SIGTERM
        return process.returncode if process.returncode >= 0 else 128 - process.returncode
    finally:
        # Exception paths must retain the same ownership guarantee.
        _signal_tree(signal.SIGKILL)
        process.wait()
        while _descendants():
            _signal_tree(signal.SIGKILL)
            _reap_orphans()
            time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A subprocess command is required")
    try:
        return run(command, args.parent_pid)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"HockeyMON subprocess ownership failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
