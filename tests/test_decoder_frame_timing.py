"""CPU-only PTS alignment: fake NVIDIA surfaces and actual FFmpeg decode."""

import contextlib
import json
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hmlib.datasets.dataset.mot_video import MOTLoadVideoWithOrig
from hmlib.ui.preview_overlays import source_metadata
from hmlib.video import video_stream
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.frame_timing import DecodedTimestamp, FFmpegFrameTiming, timing_ns
from hmlib.video.high_bit_reader import HighBitVideoReader
from hmlib.video.py_amd_codec import PyAmdVideoDecoder
from hmlib.video.video_stream import CVVideoCaptureIterator, PyNvVideoCodecIterator


@pytest.fixture(params=[0, 5])
def vfr_video(tmp_path, request):
    path = tmp_path / "vfr.mp4"
    frames = np.stack([np.full((24, 32, 3), index * 10, np.uint8) for index in range(20)])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "32x24",
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-vf",
            f"setpts='PTS+gte(N,5)*2+gte(N,11)*3+{request.param * 10}'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-qp",
            "1",
            "-bf",
            "2",
            "-x264-params",
            "b-adapt=0:scenecut=0:keyint=50",
            "-video_track_timescale",
            "10000",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
        capture_output=True,
    )
    metadata = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pts,pict_type:stream=time_base,start_time",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    assert any(frame["pict_type"] == "B" for frame in metadata["frames"])
    return path, metadata


@pytest.mark.parametrize("kind", ["amd", "high-bit"])
@pytest.mark.parametrize("seconds", [0, 0.5, 1.2])
def should_preserve_actual_vfr_pts_and_order_after_accurate_input_seek(vfr_video, kind, seconds):
    path, metadata = vfr_video
    time_base = Fraction(metadata["streams"][0]["time_base"])
    start = Fraction(metadata["streams"][0]["start_time"])
    expected = [
        frame["pts"]
        for frame in metadata["frames"]
        if frame["pts"] * time_base >= start + Fraction(str(seconds))
    ]
    if kind == "amd":
        reader = PyAmdVideoDecoder(
            path, device=torch.device("cpu"), backend="software", batch_size=3
        )
        if seconds:
            reader.seek_to_index(round(seconds * float(reader.fps)))
        read = lambda: reader.read_batch()
    else:
        reader = HighBitVideoReader(path, 32, 24, seconds=seconds, batch_size=3)
        read = lambda: next(reader, None)
    actual, means = [], []
    try:
        while (batch := read()) is not None:
            assert len(batch) == len(reader.last_frame_timing)
            actual.extend(reader.last_frame_timing)
            means.extend(float(frame.mean()) for frame in batch.float())
    finally:
        reader.close()
    assert actual == [DecodedTimestamp(pts, time_base) for pts in expected]
    offset = len(metadata["frames"]) - len(expected)
    assert means == pytest.approx([index * 10 for index in range(offset, 20)], abs=2)


def should_capture_nvidia_pts_before_tensor_conversion_without_gpu_calls(monkeypatch):
    monkeypatch.setattr(torch.cuda, "stream", lambda _: contextlib.nullcontext())
    source = [
        SimpleNamespace(getPTS=lambda pts=pts: pts, cuda=lambda: [np.zeros((2, 3), np.uint8)] * 3)
        for pts in (9000, 12600, -(2**63))
    ]
    decoder = SimpleNamespace(get_batch_frames=lambda _: source)
    iterator = PyNvVideoCodecIterator(
        decoder,
        device=torch.device("cpu"),
        cuda_stream=None,
        time_base=Fraction(1, 90000),
        batch_size=3,
    )
    batch = next(iterator)
    assert batch.shape == (3, 3, 2, 3)
    assert [stamp.ns for stamp in iterator.last_frame_timing] == [100_000_000, 140_000_000, None]
    source[0].getPTS = lambda: 123
    assert iterator.last_frame_timing[0].pts == 9000


@pytest.mark.parametrize("async_mode", [False, True])
def should_preserve_chapter_local_timing_through_sampling_and_loader_seek(vfr_video, async_mode):
    path, metadata = vfr_video
    # Chapter indices are explicit even when two immutable aliases have equal bytes.
    dataset = MOTLoadVideoWithOrig(
        path=[str(path), str(path)],
        original_image_only=True,
        device=torch.device("cpu"),
        decoder_device=torch.device("cpu"),
        decoder_type="pyamdcodec",
        async_mode=async_mode,
        frame_step=2,
        batch_size=2,
    )
    try:
        batches = list(dataset)
    finally:
        dataset.close()
    stamps = [item for batch in batches for item in batch["decoded_timing"]]
    first_chapter = [item for item in stamps if item["chapter_index"] == 0]
    second_chapter = [item for item in stamps if item["chapter_index"] == 1]
    assert len(first_chapter) == len(second_chapter) == 10
    assert dataset._chapter_frames[0] != len(metadata["frames"])
    assert [value for batch in batches for value in batch["source_frame_ids"]] == list(
        range(1, 40, 2)
    )
    for chapter in (first_chapter, second_chapter):
        assert [int(item["pts"]) for item in chapter] == [
            frame["pts"] for frame in metadata["frames"][1::2]
        ]
    for batch in batches:
        captured = source_metadata(batch)
        assert captured["decoded_pts_ns"] == timing_ns(batch["decoded_timing"][-1])
        assert captured["decoded_timing"] == batch["decoded_timing"][-1]


@pytest.mark.parametrize("kind", ["amd", "high-bit"])
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("frame_step", [1, 2])
def should_keep_source_ordinal_unknown_after_vfr_seek_through_sampling_and_chapters(
    vfr_video, monkeypatch, kind, async_mode, frame_step
):
    path, metadata = vfr_video
    monkeypatch.setenv("HM_SHARED_HIGH_BIT_DEPTH", "1" if kind == "high-bit" else "0")
    dataset = MOTLoadVideoWithOrig(
        path=[str(path), str(path)],
        original_image_only=True,
        device=torch.device("cpu"),
        decoder_device=torch.device("cpu"),
        decoder_type="pyamdcodec",
        async_mode=async_mode,
        frame_step=frame_step,
        start_frame_number=12 // frame_step,
    )
    try:
        batches = list(dataset)
    finally:
        dataset.close()
    assert len(batches) == 30 // frame_step
    expected_first_ordinal = 10 + frame_step - 1
    first = batches[0]
    assert first["decoded_timing"][0]["pts"] == str(
        metadata["frames"][expected_first_ordinal]["pts"]
    )
    assert first["frame_ids"].tolist() == [12 // frame_step]
    assert first["pts_ns"] == [round((12 + frame_step - 1) * 1e9 / dataset._video_info.fps)]
    assert all(batch["source_frame_ids"] == [None] for batch in batches)
    assert {batch["decoded_timing"][0]["chapter_index"] for batch in batches} == {0, 1}
    captured = source_metadata(first)
    assert captured["source_frame_id"] is None
    assert captured["decoded_pts_ns"] == timing_ns(first["decoded_timing"][0])


@pytest.mark.parametrize("async_mode", [False, True])
def should_not_claim_an_exact_global_ordinal_when_initial_chapters_were_skipped(
    vfr_video, async_mode
):
    path, metadata = vfr_video
    dataset = MOTLoadVideoWithOrig(
        path=[str(path), str(path)],
        original_image_only=True,
        device=torch.device("cpu"),
        decoder_device=torch.device("cpu"),
        decoder_type="pyamdcodec",
        async_mode=async_mode,
        start_frame_number=BasicVideoInfo(str(path)).frame_count,
    )
    try:
        first = next(iter(dataset))
    finally:
        dataset.close()
    assert first["source_frame_ids"] == [None]
    assert first["decoded_timing"][0]["chapter_index"] == 1
    assert first["decoded_timing"][0]["pts"] == str(metadata["frames"][0]["pts"])


@pytest.mark.parametrize("async_mode", [False, True])
def should_invalidate_previously_proven_ordinals_on_runtime_seek(vfr_video, async_mode):
    path, metadata = vfr_video
    dataset = MOTLoadVideoWithOrig(
        path=[str(path)],
        original_image_only=True,
        device=torch.device("cpu"),
        decoder_device=torch.device("cpu"),
        decoder_type="pyamdcodec",
        async_mode=async_mode,
        prefetch_batches=1,
    )
    try:
        iterator = iter(dataset)
        assert next(iterator)["source_frame_ids"] == [0]
        dataset.seek(12)
        sought = next(iterator)
    finally:
        dataset.close()
    assert sought["source_frame_ids"] == [None]
    assert sought["frame_ids"].tolist() == [12]
    assert sought["decoded_timing"][0]["pts"] == str(metadata["frames"][10]["pts"])


def should_invalidate_decoder_continuity_when_nvidia_discards_a_surface(monkeypatch):
    monkeypatch.setattr(torch.cuda, "stream", lambda _: contextlib.nullcontext())
    source = [
        SimpleNamespace(cuda=lambda: None),
        SimpleNamespace(cuda=lambda: [np.zeros((2, 3), np.uint8)] * 3, getPTS=lambda: 9000),
    ]
    iterator = PyNvVideoCodecIterator(
        SimpleNamespace(get_batch_frames=lambda _: source),
        device=torch.device("cpu"),
        cuda_stream=None,
        time_base=Fraction(1, 90000),
        batch_size=2,
    )
    assert len(next(iterator)) == 1
    assert iterator.source_ordinals_complete is False
    assert iterator.last_frame_timing == [DecodedTimestamp(9000, Fraction(1, 90000))]


def should_invalidate_decoder_continuity_when_opencv_discards_a_partial_batch():
    reads = iter([(True, np.zeros((2, 3, 3), np.uint8)), (False, None)])
    iterator = CVVideoCaptureIterator(SimpleNamespace(read=lambda: next(reads)), batch_size=2)
    with pytest.raises(StopIteration):
        next(iterator)
    assert iterator.source_ordinals_complete is False


def should_not_certify_nvidia_simple_decoder_time_to_index_seek(vfr_video, monkeypatch):
    path, metadata = vfr_video
    seeks = []

    class SimpleDecoder:
        def get_index_from_time_in_seconds(self, seconds):
            assert seconds == pytest.approx(1.2)
            return 10

        def seek_to_index(self, index):
            seeks.append(index)

    class Frames:
        source_ordinals_complete = True
        last_frame_timing = (DecodedTimestamp(metadata["frames"][10]["pts"], Fraction(1, 10000)),)

        def __next__(self):
            return torch.zeros((1, 3, 24, 32), dtype=torch.uint8)

    class CpuReader(video_stream.VideoStreamReader):
        def __init__(self, **kwargs):
            self._type = "pynvcodec"
            self._video_in = SimpleDecoder()
            self._video_info = BasicVideoInfo(str(path))
            self._torchaudio_stream = self._gstreamer_stream = False

        def __iter__(self):
            return Frames()

    from hmlib.datasets.dataset import mot_video

    monkeypatch.setattr(
        video_stream, "_load_pynvcodec", lambda: SimpleNamespace(SimpleDecoder=SimpleDecoder)
    )
    monkeypatch.setattr(mot_video, "VideoStreamReader", CpuReader)
    dataset = MOTLoadVideoWithOrig(
        path=[str(path)],
        original_image_only=True,
        device=torch.device("cpu"),
        decoder_device=torch.device("cpu"),
        decoder_type="pynvcodec",
        async_mode=False,
        start_frame_number=12,
    )
    try:
        first = next(iter(dataset))
    finally:
        dataset.close()
    assert seeks == [10]
    assert first["source_frame_ids"] == [None]
    assert first["frame_ids"].tolist() == [12]
    assert first["decoded_timing"][0]["pts"] == str(metadata["frames"][10]["pts"])


@pytest.mark.parametrize(
    "bad_line",
    [
        "[Parsed_showinfo_0 @ x] n: 1 pts: 3 pts_time:0.1",
        "[Parsed_showinfo_0 @ x] n: 0 pts: broken pts_time:0.1",
    ],
)
def should_fail_missing_or_malformed_frame_timing_instead_of_guessing(bad_line):
    timing = FFmpegFrameTiming()
    timing.feed("[Parsed_showinfo_0 @ x] config in time_base: 1/90000, frame_rate: 30/1")
    timing.feed(bad_line)
    with pytest.raises(RuntimeError):
        timing.take()


def should_keep_lossless_signed_pts_when_v1_cannot_represent_them():
    for stamp in (DecodedTimestamp(-1, Fraction(1, 1000)), DecodedTimestamp(2**60, Fraction(1, 1))):
        record = stamp.document(3)
        assert record["pts"] == str(stamp.pts)
        assert timing_ns(record) is None
