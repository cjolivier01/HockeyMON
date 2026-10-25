"""Software 4:4:4 output for canvases that cannot be represented by YUV420."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from hmlib.utils.gpu import unwrap_tensor
from hmlib.video.video_stream import VideoStreamWriterInterface


class ExactRasterWriter(VideoStreamWriterInterface):
    def __init__(
        self,
        filename: str,
        fps: float,
        width: int,
        height: int,
        bit_rate: int,
        audio: dict | None = None,
    ) -> None:
        self.filename = Path(filename)
        self.width, self.height = width, height
        self.rate = Fraction(str(fps)).limit_denominator(1_000_000)
        self.audio = audio
        self._count = 0
        self._container = av.open(str(self.filename), mode="w")
        try:
            self._stream = self._container.add_stream("libx264", rate=self.rate)
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = "yuv444p"
            self._stream.bit_rate = bit_rate
            self._stream.options = {"preset": "veryfast", "threads": "2"}
        except BaseException:
            self._container.close()
            self._container = None
            raise

    def isOpened(self) -> bool:
        return self._container is not None

    def write(self, images, *, frame_ids=None) -> None:
        if self._container is None:
            raise RuntimeError("The exact-raster encoder has already closed")
        images = unwrap_tensor(images)
        if torch.is_tensor(images):
            images = images.detach().cpu().numpy()
        if images.ndim == 3:
            images = images[None]
        if images.shape[1:] != (self.height, self.width, 3) or images.dtype != np.uint8:
            raise ValueError("Exact-raster output requires uint8 BGR frames at the encoded size")
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = self._count
            frame.time_base = 1 / self.rate
            self._count += 1
            for packet in self._stream.encode(frame):
                self._container.mux(packet)

    def close(self) -> None:
        if self._container is None:
            return
        container = self._container
        try:
            for packet in self._stream.encode():
                container.mux(packet)
        finally:
            container.close()
            self._container = None
        if self.audio and self._count:
            from hmlib.ui.output_audio import ensure_archive_audio

            ensure_archive_audio(self.filename, **self.audio)
