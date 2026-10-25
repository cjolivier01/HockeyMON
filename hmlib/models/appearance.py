"""Execute the exact TensorRT appearance engine prepared by the shared native cache."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from hmlib.models.player_analytics import validate_model_gpu
from hmlib.utils.checksum import sha256_file


def read_tracker_model(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError(f"Missing or oversized prepared appearance configuration: {path}")
    text = path.read_text().replace("%YAML:1.0", "")
    settings = yaml.safe_load(text).get("ReID", {})
    if (
        settings.get("inferDims") != [3, 256, 128]
        or settings.get("reidFeatureSize") != 256
        or settings.get("inputOrder", 0) != 0
    ):
        raise ValueError("Appearance model requires NCHW [N,3,256,128] -> [N,256]")
    if settings.get("colorFormat", 0) not in (0, 1) or settings.get("keepAspc", 0) not in (0, 1):
        raise ValueError("Unsupported appearance color or aspect-ratio convention")
    offsets = settings.get("offsets", [0.0, 0.0, 0.0])
    if (
        len(offsets) != 3
        or not np.isfinite(offsets).all()
        or not np.isfinite(settings.get("netScaleFactor", 1.0))
    ):
        raise ValueError("Appearance normalization must contain finite channel offsets and scale")
    settings.setdefault("offsets", [0.0, 0.0, 0.0])
    settings.setdefault("netScaleFactor", 1.0)
    engine = Path(settings["modelEngineFile"]).expanduser()
    engine = engine.resolve() if engine.is_absolute() else (path.parent / engine).resolve()
    if not engine.is_file() or not 0 < engine.stat().st_size <= 2 * 1024**3:
        raise ValueError(f"Missing or oversized prepared appearance engine: {engine}")
    manifest = path.parent / "manifest.json"
    if manifest.is_file():
        document = yaml.safe_load(manifest.read_text())
        if document.get("feature") != "reid" or sha256_file(engine) != document.get(
            "engine", {}
        ).get("sha256"):
            raise ValueError("Prepared appearance engine does not match its native manifest")
    return engine, settings


class TensorRtAppearanceSession:
    """Run prepared engines in the matching native TensorRT runtime, independent of Python's ABI."""

    maximum_batch = 32

    def __init__(self, path: Path, gpu: int = 0):
        import os
        import subprocess
        import tempfile
        from contextlib import ExitStack

        from hmlib.ui.desktop import executable

        gpu = validate_model_gpu(gpu)
        configured = os.environ.get("HM_HSTREAM_MODEL_INFER")
        runner = os.environ.get("HSTREAM_NATIVE_RUNNER")
        candidates = []
        if runner:
            parent = Path(runner).resolve().parent
            candidates = [
                parent / "hstream-model-infer",
                parent.parent / "hstream-model-infer/hstream-model-infer",
            ]
        helper = executable(configured, "hstream-model-infer", tuple(candidates))
        self.inputs = [
            SimpleNamespace(name="images", shape=[-1, 3, 256, 128], type="tensor(float)")
        ]
        self.outputs = [SimpleNamespace(name="features", shape=[-1, 256], type="tensor(float)")]
        self._resources = ExitStack()
        self.stderr = self._resources.enter_context(
            tempfile.TemporaryFile()  # noqa: SIM115 - owned until close()
        )
        try:
            self.process = subprocess.Popen(
                [
                    str(helper),
                    "--engine",
                    str(path),
                    "--gpu-id",
                    str(gpu),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                env={**os.environ, "HSTREAM_UI_PARENT_PID": str(os.getpid())},
                bufsize=0,
            )
        except OSError:
            self._resources.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def _failure(self, message):
        import os

        self.stderr.seek(0, os.SEEK_END)
        self.stderr.seek(max(0, self.stderr.tell() - 8192))
        detail = self.stderr.read().decode("utf-8", errors="replace").strip()
        return RuntimeError(
            f"{message}; native appearance helper status={self.process.poll()}: {detail}"
        )

    def _transfer(self, payload, count, deadline):
        import os
        import select
        import time

        result = bytearray()
        offset = 0
        while offset < len(payload) or len(result) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._failure("Native appearance inference timed out")
            read_fds = [self.process.stdout.fileno()] if len(result) < count else []
            write_fds = [self.process.stdin.fileno()] if offset < len(payload) else []
            readable, writable, _ = select.select(read_fds, write_fds, [], min(remaining, 1.0))
            if writable:
                try:
                    offset += os.write(writable[0], payload[offset:])
                except BrokenPipeError as error:
                    raise self._failure("Native appearance helper closed its input") from error
            if readable:
                block = os.read(readable[0], count - len(result))
                if not block:
                    raise self._failure(
                        "Native appearance helper ended before completing its response"
                    )
                result.extend(block)
            if (
                self.process.poll() is not None
                and not readable
                and (offset < len(payload) or len(result) < count)
            ):
                raise self._failure("Native appearance helper exited during inference")
        return bytes(result)

    def run(self, names, feeds):
        import struct
        import time

        values = np.ascontiguousarray(feeds[self.inputs[0].name], dtype="<f4")
        count = len(values)
        if (
            values.shape != (count, 3, 256, 128)
            or not 0 < count <= self.maximum_batch
            or not np.isfinite(values).all()
        ):
            raise ValueError("Appearance inference requires at most 32 finite float32 RGB crops")
        deadline = time.monotonic() + 120
        response = self._transfer(
            struct.pack("<I", count) + values.tobytes(), 4 + count * 256 * 4, deadline
        )
        if struct.unpack("<I", response[:4])[0] != count:
            raise self._failure("Native appearance response batch does not match its input")
        output = np.frombuffer(response[4:], dtype="<f4").reshape(count, 256).copy()
        if not np.isfinite(output).all():
            raise self._failure("Native appearance inference returned nonfinite embeddings")
        return [output]

    def close(self):
        import struct
        import subprocess
        import time

        if self.process.stdin is None:
            return
        shutdown_error = None
        if self.process.poll() is None:
            try:
                # A later fork can inherit a pipe writer, so EOF alone does not
                # guarantee shutdown. The protocol reserves batch zero for it.
                self._transfer(struct.pack("<I", 0), 0, time.monotonic() + 5)
            except RuntimeError as error:
                shutdown_error = error
        self.process.stdin.close()
        self.process.stdin = None
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()
        try:
            if self.process.returncode != 0:
                raise self._failure("Native appearance helper failed during finalization")
            if shutdown_error is not None:
                raise shutdown_error
        finally:
            self._resources.close()
