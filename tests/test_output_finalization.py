from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hmlib.aspen.net import AspenNet
from hmlib.aspen.plugins.video_out_plugin import VideoOutPlugin
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.utils.finalization import FinalizationError, finalize_resources
from hmlib.video.video_out import VideoOutput


def _fail(message):
    def fail():
        raise OSError(message)

    return fail


def should_attempt_every_finalizer_and_report_all_errors():
    calls = []
    with pytest.raises(FinalizationError, match="encoder.*disk full.*CSV.*flush failed") as caught:
        finalize_resources(
            [
                ("encoder", _fail("disk full")),
                ("CSV", _fail("flush failed")),
                ("preview", lambda: calls.append("closed")),
            ]
        )
    assert calls == ["closed"]
    assert isinstance(caught.value.__cause__, OSError)
    assert len(caught.value.failures) == 2


def should_preserve_primary_failure_and_log_cleanup_failure(caplog):
    primary = ValueError("invalid input")
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        finally:
            finalize_resources([("encoder", _fail("disk full"))], primary_error=sys.exc_info()[1])
    assert caught.value is primary
    assert "encoder failed while handling ValueError" in caplog.text
    assert "disk full" in caplog.text


def should_finalize_other_video_outputs_after_encoder_failure():
    calls = []
    output = VideoOutput.__new__(VideoOutput)
    output._output_videos = {
        "main": SimpleNamespace(close=_fail("encoder flush")),
        "end_zones": SimpleNamespace(close=lambda: calls.append("end_zones")),
    }
    output._shower = SimpleNamespace(close=lambda: calls.append("preview"))
    with pytest.raises(FinalizationError, match="main.*encoder flush"):
        output.stop()
    assert calls == ["end_zones", "preview"]
    assert output._output_videos == {}
    assert output._shower is None


def should_propagate_video_plugin_finalization_errors():
    plugin = VideoOutPlugin()
    plugin._vo = SimpleNamespace(stop=_fail("mux failed"))
    with pytest.raises(OSError, match="mux failed"):
        plugin.finalize()


def should_finalize_all_aspen_plugins_after_failure():
    calls = []
    net = SimpleNamespace(
        threaded_trunks=True,
        threads=[object()],
        stop=lambda wait: calls.append("stop"),
        nodes=[
            SimpleNamespace(name="video", module=SimpleNamespace(finalize=_fail("disk full"))),
            SimpleNamespace(
                name="camera", module=SimpleNamespace(finalize=lambda: calls.append("camera"))
            ),
        ],
        stop_progress_graph=lambda: calls.append("progress"),
    )
    with pytest.raises(FinalizationError, match="Aspen plugin video.*disk full"):
        AspenNet.finalize(net)
    assert calls == ["stop", "camera", "progress"]


@pytest.mark.parametrize("input_error", [None, ValueError("decode failed")])
def should_propagate_tracking_finalization_errors_without_masking_input(
    monkeypatch, input_error, caplog
):
    from hmlib.tasks import tracking

    class Loader:
        batch_size = 1
        fps = 30

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 0

    class Iterator:
        def __next__(self):
            if input_error is not None:
                raise input_error
            raise StopIteration

    class Net:
        def __init__(self, name, config, shared):
            self.shared = shared

        def to(self, device):
            return self

        def finalize(self):
            raise OSError("final encoder flush failed")

    monkeypatch.setattr(tracking, "AspenNet", Net)
    monkeypatch.setattr(tracking, "CachedIterator", lambda **kwargs: Iterator())
    expected = FinalizationError if input_error is None else ValueError
    with pytest.raises(expected) as caught:
        tracking.run_mmtrack(
            model=None,
            config={"aspen": {"plugins": {}}},
            dataloader=Loader(),
            postprocessor=None,
            device="cpu",
            no_cuda_streams=True,
        )
    if input_error is not None:
        assert caught.value is input_error
        assert "final encoder flush failed" in caplog.text
    else:
        assert "final encoder flush failed" in str(caught.value)


def _camera(path):
    return CameraTrackingDataFrame(output_file=path, input_batch_size=1)


def _add_frame(dataframe, frame_id):
    dataframe.add_frame_records(frame_id=frame_id, tlwh=np.array([[1, 2, 3, 4]]))


def should_preserve_rows_across_manual_flushes_and_close(tmp_path):
    path = tmp_path / "camera.csv"
    dataframe = _camera(path)
    for frame in (1, 2, 3):
        _add_frame(dataframe, frame)
        dataframe.flush()
    dataframe.close()
    assert path.read_text().splitlines() == [f"{frame},1,2,3,4" for frame in (1, 2, 3)]


def should_materialize_empty_csv_and_close_idempotently(tmp_path):
    path = tmp_path / "camera.csv"
    dataframe = _camera(path)
    dataframe.close()
    assert path.read_bytes() == b""
    dataframe.close()
    assert path.read_bytes() == b""


def should_fail_permanently_after_uncertain_csv_append(tmp_path, monkeypatch):
    from hmlib.datasets import dataframe as dataframe_module

    path = tmp_path / "camera.csv"
    dataframe = _camera(path)
    _add_frame(dataframe, 1)
    dataframe.flush()
    _add_frame(dataframe, 2)
    monkeypatch.setattr(dataframe_module.os, "fsync", lambda fd: _fail("fsync failed")())
    with pytest.raises(OSError, match="fsync failed"):
        dataframe.flush()
    uncertain_bytes = path.read_bytes()
    with pytest.raises(RuntimeError, match="previously failed"):
        dataframe.close()
    assert path.read_bytes() == uncertain_bytes
    assert len(dataframe._dataframe_list) == 1


def should_drain_amd_encoder_even_when_input_flush_fails():
    from hmlib.video.py_amd_codec import PyAmdVideoEncoder

    calls = []
    process = SimpleNamespace(
        stdin=SimpleNamespace(close=_fail("input flush failed")),
        wait=lambda timeout: calls.append("wait"),
        returncode=0,
    )
    encoder = SimpleNamespace(
        _process=process,
        _opened=True,
        _stdout_thread=None,
        _stderr_thread=None,
        _stderr_tail=[],
        _check_background_error=lambda: calls.append("check worker"),
    )
    with pytest.raises(FinalizationError, match="input flush failed"):
        PyAmdVideoEncoder.close(encoder)
    assert calls == ["wait", "check worker"]
    assert encoder._process is None


def should_close_nvenc_bitstream_after_encoder_flush_failure(tmp_path):
    from hmlib.video.py_nv_encoder import PyNvVideoEncoder

    encoder = PyNvVideoEncoder.__new__(PyNvVideoEncoder)
    encoder._opened = True
    encoder._encoder = SimpleNamespace(EndEncode=_fail("EndEncode failed"))
    stream = (tmp_path / "video.h264").open("wb")
    stream.write(b"recoverable")
    encoder._bitstream_file = stream
    with pytest.raises(OSError, match="EndEncode failed"):
        encoder.close()
    assert stream.closed
    assert not encoder._opened
    assert (tmp_path / "video.h264").read_bytes() == b"recoverable"


def should_fail_requested_audio_mux_without_video_only_retry(tmp_path, monkeypatch):
    import shutil
    from hmlib.video import py_nv_encoder as nv

    audio = tmp_path / "audio.mp4"
    audio.write_bytes(b"audio")
    encoder = nv.PyNvVideoEncoder.__new__(nv.PyNvVideoEncoder)
    encoder.output_path = tmp_path / "ordinary-name.mp4"
    encoder._mux_audio_file = str(audio)
    encoder._mux_audio_stream = 0
    encoder._mux_audio_offset_seconds = 0
    encoder._mux_audio_aac_bitrate = "192k"
    encoder._ffmpeg_output_handler = None
    encoder._frames_in_current_bitstream = 3
    encoder.fps = 30
    encoder.codec = "h264"
    calls = []
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(nv.subprocess, "check_output", lambda *args, **kwargs: "aac")

    class Process:
        stdout = None
        stderr = None

        def __init__(self, cmd, **kwargs):
            calls.append(cmd)

        def wait(self):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(nv.subprocess, "Popen", Process)
    monkeypatch.setattr(
        nv,
        "build_ffmpeg_output_handler",
        lambda *args, **kwargs: SimpleNamespace(close=lambda code: None),
    )
    with pytest.raises(RuntimeError, match="ffmpeg muxer failed"):
        encoder._mux_bitstream_file_with_ffmpeg(tmp_path / "video.h264")
    assert len(calls) == 1
    assert str(audio) in calls[0]


def should_flush_fast_camera_when_follower_flush_fails(tmp_path):
    from hmlib.aspen.plugins.save_plugins import SaveCameraPlugin

    plugin = SaveCameraPlugin()
    plugin._camera_dataframe = SimpleNamespace(close=_fail("follower fsync failed"))
    plugin._camera_fast_dataframe = _camera(tmp_path / "camera_fast.csv")
    _add_frame(plugin._camera_fast_dataframe, 1)
    with pytest.raises(FinalizationError, match="follower fsync failed"):
        plugin.finalize()
    assert (tmp_path / "camera_fast.csv").read_text() == "1,1,2,3,4\n"


def should_propagate_camera_write_failure_immediately():
    from hmlib.aspen.plugins.save_plugins import SaveCameraPlugin

    plugin = SaveCameraPlugin(save_fast=False)
    plugin._camera_dataframe = SimpleNamespace(
        add_frame_records=lambda **kwargs: _fail("CSV full")()
    )
    with pytest.raises(OSError, match="CSV full"):
        plugin.forward({"frame_id": 1, "current_box": np.array([1, 2, 3, 4])})
