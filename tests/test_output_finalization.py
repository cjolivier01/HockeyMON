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
