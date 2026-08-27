from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def _ensure_hockeymon_ext(repo_root: Path) -> None:
    """
    When running tests via plain pytest (outside Bazel), prefer the Bazel-built
    hockeymon extension if available so that new symbols are present.
    """
    ext_key = "hockeymon._hockeymon"
    if ext_key in sys.modules and hasattr(sys.modules[ext_key], "bgr_to_i420_cuda"):
        return

    candidate_paths = [repo_root / "bazel-bin" / "hockeymon" / "src" / "_hockeymon.so"]
    bazel_out = repo_root / "bazel-out"
    if bazel_out.is_dir():
        candidate_paths.extend(sorted(bazel_out.glob("*/bin/hockeymon/src/_hockeymon.so")))

    ext_path = None
    for path in candidate_paths:
        if path.is_file():
            ext_path = path
            break
    if ext_path is None:
        return

    import importlib.util

    spec = importlib.util.spec_from_file_location(ext_key, ext_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[arg-type]
    except ImportError:
        # Best-effort: the Bazel-built extension may depend on a libpython/ABI
        # that is not compatible with the interpreter running this test.
        return
    sys.modules[ext_key] = module


def _require_torch_cuda():
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _ensure_hockeymon_ext(_ensure_repo_on_path())
    return torch


def should_write_h265_with_pynvencoder(tmp_path: Path):
    """
    Basic smoke test that exercises the GPU-only PyNvVideoEncoderWriter path.
    """
    torch = _require_torch_cuda()
    from hmlib.video.video_stream import PyNvVideoEncoderWriter

    filename = tmp_path / "pynvencoder_raw.mp4"
    writer = PyNvVideoEncoderWriter(
        filename=str(filename),
        fps=30.0,
        width=640,
        height=360,
        codec="hevc_nvenc",
        device=torch.device("cuda", 0),
        bit_rate=int(5e6),
        batch_size=1,
    )
    writer.open()

    frame = torch.randint(
        0,
        256,
        (1, 360, 640, 3),
        dtype=torch.uint8,
        device=writer.device,
    )
    writer.write(frame)
    writer.close()

    assert filename.is_file()
    assert filename.stat().st_size > 0


def should_support_mkv_container_with_pynvencoder(tmp_path: Path):
    """
    Verify that writing to a .mkv path produces a valid container.
    """
    torch = _require_torch_cuda()
    from hmlib.video.video_stream import PyNvVideoEncoderWriter

    filename = tmp_path / "pynvencoder_out.mkv"

    writer = PyNvVideoEncoderWriter(
        filename=str(filename),
        fps=30.0,
        width=640,
        height=360,
        codec="hevc_nvenc",
        device=torch.device("cuda", 0),
        bit_rate=int(5e6),
        batch_size=1,
    )
    writer.open()

    frame = torch.randint(
        0,
        256,
        (1, 360, 640, 3),
        dtype=torch.uint8,
        device=writer.device,
    )
    writer.write(frame)
    writer.close()

    assert filename.is_file()
    assert filename.stat().st_size > 0


def should_accept_frame_ids_with_extra_dim(tmp_path: Path):
    """
    Regression test: some callers produce frame_ids with shape [B, 1] instead of [B].
    The NVENC writer should accept these without crashing and enforce consecutiveness.
    """
    torch = _require_torch_cuda()
    from hmlib.video.video_stream import PyNvVideoEncoderWriter

    filename = tmp_path / "pynvencoder_frame_ids_bx1.mkv"

    writer = PyNvVideoEncoderWriter(
        filename=str(filename),
        fps=30.0,
        width=640,
        height=360,
        codec="hevc_nvenc",
        device=torch.device("cuda", 0),
        bit_rate=int(5e6),
        batch_size=1,
    )
    writer.open()

    frame = torch.randint(
        0,
        256,
        (1, 360, 640, 3),
        dtype=torch.uint8,
        device=writer.device,
    )
    writer.write(frame, frame_ids=torch.tensor([[1]], dtype=torch.int64))
    writer.write(frame, frame_ids=torch.tensor([[2]], dtype=torch.int64))
    writer.close()

    assert filename.is_file()
    assert filename.stat().st_size > 0


def should_use_pyav_backend_with_pynvencoder(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"):
    """
    Ensure that the PyAV backend is exercised when HM_VIDEO_ENCODER_BACKEND=pyav.

    This test imports av at module level, so it will fail if PyAV is not installed.
    """
    import pytest

    pytest.importorskip("av")
    torch = _require_torch_cuda()
    from hmlib.video.video_stream import PyNvVideoEncoderWriter

    monkeypatch.setenv("HM_VIDEO_ENCODER_BACKEND", "pyav")

    filename = tmp_path / "pynvencoder_pyav.mkv"
    writer = PyNvVideoEncoderWriter(
        filename=str(filename),
        fps=30.0,
        width=640,
        height=360,
        codec="hevc_nvenc",
        device=torch.device("cuda", 0),
        bit_rate=int(5e6),
        batch_size=1,
    )
    writer.open()

    frame = torch.randint(
        0,
        256,
        (1, 360, 640, 3),
        dtype=torch.uint8,
        device=writer.device,
    )
    writer.write(frame)
    writer.close()

    assert filename.is_file()
    assert filename.stat().st_size > 0


def _ffprobe_stream_metadata(video_path: Path) -> dict:
    """Return ffprobe JSON metadata for the first video stream."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,width,height,duration",
        "-of",
        "json",
        str(video_path),
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(raw.decode("utf-8"))
    assert data.get("streams"), f"ffprobe did not return streams for {video_path}"
    return data["streams"][0]


def _write_test_video_cv2(path: Path, *, fps: float, width: int, height: int, frames: int) -> None:
    """Create a tiny mp4 using OpenCV (CPU) for transcoding tests."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened()
    for i in range(frames):
        # Horizontal gradient + frame index to make frames distinct.
        x = np.linspace(0, 255, width, dtype=np.uint8)
        gradient = np.tile(x, (height, 1))
        frame = np.stack(
            [
                gradient,  # blue
                np.full_like(gradient, i * 10 % 255),  # green
                255 - gradient,  # red
            ],
            axis=2,
        )
        writer.write(frame)
    writer.release()


def should_transcode_with_reader_and_pynvencoder(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"):
    """
    Transcode a CPU-written test clip via VideoStreamReader -> PyNvVideoEncoderWriter
    and ensure fps/frame counts/durations match the source.
    """
    import pytest

    pytest.importorskip("av")
    _ = pytest.importorskip("cv2")
    torch = _require_torch_cuda()
    from hmlib.video.video_stream import PyNvVideoEncoderWriter, VideoStreamReader

    monkeypatch.setenv("HM_VIDEO_ENCODER_BACKEND", "pyav")

    fps = 15.0
    frame_count = 12
    width, height = 320, 240
    src = tmp_path / "source.mp4"
    dst = tmp_path / "transcoded.mp4"

    _write_test_video_cv2(src, fps=fps, width=width, height=height, frames=frame_count)

    reader = VideoStreamReader(str(src), type="cv2", device=torch.device("cpu"))
    collected = []
    for batch in reader:
        batch_t = torch.as_tensor(batch)
        if batch_t.ndim == 3:
            batch_t = batch_t.unsqueeze(0)
        collected.append(batch_t.permute(0, 2, 3, 1).contiguous())
    reader.close()

    assert collected, "VideoStreamReader returned no frames"
    frames = torch.cat(collected, dim=0)
    assert frames.shape[0] == frame_count

    writer = PyNvVideoEncoderWriter(
        filename=str(dst),
        fps=fps,
        width=width,
        height=height,
        codec="hevc_nvenc",
        device=torch.device("cuda", 0),
        bit_rate=int(5e6),
        batch_size=frame_count,
    )
    writer.open()
    writer.write(frames.to(writer.device))
    writer.close()

    src_meta = _ffprobe_stream_metadata(src)
    dst_meta = _ffprobe_stream_metadata(dst)

    def _rate_to_float(rate: str) -> float:
        return float(Fraction(rate)) if rate else 0.0

    def _frame_count(meta: dict) -> int:
        return int(meta.get("nb_frames") or meta.get("nb_read_frames") or 0)

    assert _frame_count(src_meta) == frame_count
    assert _frame_count(dst_meta) == frame_count
    assert int(dst_meta["width"]) == width
    assert int(dst_meta["height"]) == height
    assert _rate_to_float(dst_meta.get("r_frame_rate")) == pytest.approx(fps, rel=0, abs=1e-3)
    assert _rate_to_float(dst_meta.get("avg_frame_rate")) == pytest.approx(fps, rel=0, abs=1e-3)
    # Duration matches to within one frame.
    src_dur = float(src_meta.get("duration", frame_count / fps))
    dst_dur = float(dst_meta.get("duration", frame_count / fps))
    assert dst_dur == pytest.approx(src_dur, rel=0, abs=1.0 / fps)


def should_synchronize_dlpack_stream_between_producer_and_consumer():
    """Ensure CUDA stream handoff is synchronized when exporting via __dlpack__().

    PyNvVideoCodec requests a DLPack capsule with a consumer stream pointer.
    If the producer ignores that pointer, NVENC (or any other consumer running
    on a different stream) can read incomplete frames, producing temporal
    jitter and RGB flashing artifacts.
    """

    torch = _require_torch_cuda()
    from torch.utils.dlpack import from_dlpack

    from hmlib.video.py_nv_encoder import _DLPackFrame

    device = torch.device("cuda", 0)
    expected = 123

    # Use explicit stream priorities to increase determinism: ensure the consumer
    # stream gets scheduled while the producer stream is still sleeping.
    producer_stream = torch.cuda.Stream(device=device, priority=1)
    consumer_stream = torch.cuda.Stream(device=device, priority=-1)

    tensor = torch.zeros((2048, 2048), dtype=torch.uint8, device=device)

    # Produce on a non-default stream, delaying the write so the consumer can race.
    with torch.cuda.stream(producer_stream):
        torch.cuda._sleep(200_000_000)  # ~68ms on current hardware
        tensor.fill_(expected)
        # Export the capsule to a different consumer stream (positional arg matches
        # PyNvVideoCodec's call signature).
        capsule = _DLPackFrame(tensor).__dlpack__(consumer_stream.cuda_stream)

    # Consume on the consumer stream; this should block until the producer stream
    # finishes writing `tensor`.
    with torch.cuda.stream(consumer_stream):
        view = from_dlpack(capsule)
        result = view.sum(dtype=torch.int64)

    consumer_stream.synchronize()
    producer_stream.synchronize()

    assert int(result.item()) == expected * tensor.numel()
