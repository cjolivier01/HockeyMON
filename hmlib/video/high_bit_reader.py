"""Precision-preserving software decode for the shared desktop's grading path."""

from __future__ import annotations

import subprocess
import tempfile

import numpy as np
import torch


class HighBitVideoReader:
    """Decode BGR48 and retain fractional 8-bit units through floating grading.

    FFmpeg performs the source color-space conversion before quantization. The
    output uses the pipeline's existing 0..255 scale, without reducing it to
    256 integer levels. This deliberately selects software decoding because the
    current hardware Python bindings expose only 8-bit RGB surfaces.
    """

    def __init__(self, filename, width, height, *, seconds=0, batch_size=1):
        self.width, self.height = int(width), int(height)
        self.batch_size = batch_size
        self.frames_delivered_count = 0
        self._errors = tempfile.TemporaryFile()  # noqa: SIM115 - owned by the reader
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "2",
                "-ss",
                str(seconds),
                "-i",
                str(filename),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-threads",
                "2",
                "-pix_fmt",
                "bgr48le",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=self._errors,
        )

    def __iter__(self):
        return self

    def __next__(self):
        frames = []
        size = self.width * self.height * 6
        for _ in range(self.batch_size):
            data = self._process.stdout.read(size)
            if len(data) != size:
                code = self._process.wait()
                if data or code:
                    self._errors.seek(0)
                    detail = self._errors.read(16384).decode(errors="replace")
                    raise RuntimeError(f"High-bit-depth decode failed ({code}): {detail}")
                if not frames:
                    raise StopIteration
                break
            array = np.frombuffer(data, dtype="<u2").reshape(self.height, self.width, 3)
            tensor = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1) / 257.0
            frames.append(tensor)
            self.frames_delivered_count += 1
        return torch.stack(frames)

    def close(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._process.stdout.close()
        self._errors.close()
