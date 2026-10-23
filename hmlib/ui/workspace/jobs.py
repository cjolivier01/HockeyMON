"""Process-group supervision for local desktop jobs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .files import read_json, write_json

ACTIVE_STATES = {"running", "paused", "stopping"}


class Jobs:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen] = {}
        self.deadlines: dict[str, float] = {}
        for metadata in sorted(directory.glob("*/job.json")):
            record = read_json(metadata)
            if record["status"] in ACTIVE_STATES:
                record.update(status="interrupted", error="The supervising desktop session ended")
                write_json(metadata, record)
            self.records[record["id"]] = record

    def _save(self, record: dict) -> None:
        write_json(self.directory / record["id"] / "job.json", record)

    def start(
        self,
        command: list[str],
        *,
        game: Path,
        kind: str,
        cwd: Path,
        environment: dict[str, str],
        extra: dict | None = None,
    ) -> dict:
        if not command or not all(isinstance(arg, str) and "\0" not in arg for arg in command):
            raise ValueError("A job command must be a nonempty list of strings")
        job_id = uuid.uuid4().hex
        directory = self.directory / job_id
        directory.mkdir()
        record = {
            "id": job_id,
            "kind": kind,
            "game": str(game),
            "command": command,
            "cwd": str(cwd),
            "status": "starting",
            "created_at": time.time(),
            "log": str(directory / "output.log"),
            "returncode": None,
            **(extra or {}),
        }
        self.records[job_id] = record
        self._save(record)
        child_state = directory / "child.json"
        try:
            with open(record["log"], "wb") as output:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("guardian.py")),
                        "--owner",
                        str(os.getpid()),
                        "--child-state",
                        str(child_state),
                        "--",
                        *command,
                    ],
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except OSError as error:
            record.update(status="failed", error=str(error), finished_at=time.time())
            self._save(record)
            raise
        self.processes[job_id] = process
        record.update(status="running", pid=process.pid)
        deadline = time.monotonic() + 5
        while not child_state.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if child_state.exists():
            record["child_pid"] = read_json(child_state)["pid"]
        elif process.poll() is None:
            self.stop(job_id)
            raise RuntimeError("The pipeline supervisor did not acknowledge startup; see its log")
        self._save(record)
        return dict(record)

    def get(self, job_id: str) -> dict:
        if job_id not in self.records:
            raise ValueError(f"Unknown job: {job_id}")
        return self.records[job_id]

    def _signal(self, job_id: str, sig: int) -> bool:
        process = self.processes.get(job_id)
        if process is None:
            return False
        child_pid = self.records[job_id].get("child_pid")
        delivered = False
        if child_pid is not None:
            try:
                os.killpg(child_pid, sig)
                delivered = True
            except ProcessLookupError:
                # A completed child may already have been reaped by its guardian.
                delivered = False
        if sig not in {signal.SIGTERM, signal.SIGKILL}:
            return delivered
        try:
            os.kill(process.pid, sig)
        except ProcessLookupError:
            return delivered
        return True

    def request_stop(self, job_id: str, *, grace_seconds: float = 45) -> dict:
        """Allow the pipeline to drain/mux before escalating to process signals."""
        self.poll()
        record = self.get(job_id)
        if record["status"] in ACTIVE_STATES and record["status"] != "stopping":
            self._signal(job_id, signal.SIGCONT)
            record.update(status="stopping", graceful_stop=True, stop_requested_at=time.time())
            self.deadlines[job_id] = time.monotonic() + grace_seconds
            self._save(record)
        return dict(record)

    def stop(self, job_id: str) -> dict:
        self.poll()
        record = self.get(job_id)
        if record["status"] in ACTIVE_STATES:
            self._signal(job_id, signal.SIGCONT)
            self._signal(job_id, signal.SIGTERM)
            record["status"] = "stopping"
            self.deadlines[job_id] = time.monotonic() + 8
            self._save(record)
        return dict(record)

    def pause(self, job_id: str, paused: bool) -> dict:
        self.poll()
        record = self.get(job_id)
        expected = "running" if paused else "paused"
        if record["status"] != expected:
            raise ValueError(f"Cannot {'pause' if paused else 'resume'} a {record['status']} job")
        if not self._signal(job_id, signal.SIGSTOP if paused else signal.SIGCONT):
            self.poll()
            raise RuntimeError("The job exited before the requested state change")
        record["status"] = "paused" if paused else "running"
        self._save(record)
        return dict(record)

    def poll(self) -> list[dict]:
        completed = []
        for job_id, process in list(self.processes.items()):
            record = self.records[job_id]
            if job_id in self.deadlines and time.monotonic() >= self.deadlines[job_id]:
                if record.get("graceful_stop") and not record.get("forced_stop"):
                    record.update(
                        forced_stop=True,
                        error="The pipeline did not finish within its stop deadline; forced termination may leave incomplete outputs",
                    )
                    self._signal(job_id, signal.SIGTERM)
                    self.deadlines[job_id] = time.monotonic() + 8
                    self._save(record)
                else:
                    self._signal(job_id, signal.SIGKILL)
            code = process.poll()
            if code is None:
                continue
            # A command that left background children does not transfer their
            # ownership to the desktop or make them safe to leave running.
            self._signal(job_id, signal.SIGKILL)
            if record.get("graceful_stop") and not record.get("forced_stop"):
                status = "completed" if code == 0 else "failed"
                record["stopped"] = True
            else:
                status = (
                    "cancelled"
                    if record["status"] == "stopping"
                    else "completed" if code == 0 else "failed"
                )
            record.update(status=status, returncode=code, finished_at=time.time())
            if status == "failed":
                record["error"] = f"Process exited with status {code}; see {record['log']}"
            del self.processes[job_id]
            self.deadlines.pop(job_id, None)
            self._save(record)
            completed.append(record)
        return completed

    def log(self, job_id: str, offset: int = 0, limit: int = 65536) -> dict:
        record = self.get(job_id)
        if offset < 0 or not 1 <= limit <= 1024 * 1024:
            raise ValueError("Invalid log offset or size")
        with open(record["log"], "rb") as stream:
            stream.seek(offset)
            data = stream.read(limit)
            return {"text": data.decode("utf-8", errors="replace"), "offset": stream.tell()}

    def close(self) -> None:
        for job_id in list(self.processes):
            record = self.records[job_id]
            if record["status"] != "stopping":
                self.stop(job_id)
        end = max([time.monotonic(), *self.deadlines.values()]) + 9
        while self.processes and time.monotonic() < end:
            self.poll()
            time.sleep(0.025)
        for job_id, process in list(self.processes.items()):
            self._signal(job_id, signal.SIGKILL)
            process.wait(timeout=5)
        self.poll()
