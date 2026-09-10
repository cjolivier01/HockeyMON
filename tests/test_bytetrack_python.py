import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def should_keep_ids_across_frames_on_cpu():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCuda

    tracker = HmByteTrackerCuda(device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.tensor(
            [[10.0, 10.0, 30.0, 40.0], [100.0, 100.0, 140.0, 160.0]], dtype=torch.float32
        ),
        "labels": torch.tensor([1, 1], dtype=torch.long),
        "scores": torch.tensor([0.9, 0.85], dtype=torch.float32),
    }
    res0 = tracker.track(frame0)
    assert torch.equal(res0["ids"], torch.tensor([0, 1], dtype=torch.long))

    frame1 = {
        "frame_id": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor(
            [[12.0, 12.0, 32.0, 42.0], [103.0, 103.0, 143.0, 163.0]], dtype=torch.float32
        ),
        "labels": torch.tensor([1, 1], dtype=torch.long),
        "scores": torch.tensor([0.92, 0.8], dtype=torch.float32),
    }
    res1 = tracker.track(frame1)
    assert torch.equal(res1["ids"], torch.tensor([0, 1], dtype=torch.long))


def should_start_no_tracks_below_init_threshold():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCuda

    tracker = HmByteTrackerCuda(device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.tensor([[10.0, 10.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
        "scores": torch.tensor([0.6], dtype=torch.float32),  # default init_track_thr=0.7
    }
    res0 = tracker.track(frame0)
    assert res0["ids"].numel() == 0
    assert res0["bboxes"].shape == (0, 4)


def should_create_new_id_when_label_changes():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCuda

    tracker = HmByteTrackerCuda(device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.tensor([[10.0, 10.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.long),
        "scores": torch.tensor([0.9], dtype=torch.float32),
    }
    res0 = tracker.track(frame0)
    assert torch.equal(res0["ids"], torch.tensor([0], dtype=torch.long))

    frame1 = {
        "frame_id": torch.tensor([1], dtype=torch.long),
        "bboxes": torch.tensor([[10.0, 10.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([2], dtype=torch.long),  # label mismatch => no match
        "scores": torch.tensor([0.9], dtype=torch.float32),
    }
    res1 = tracker.track(frame1)
    assert torch.equal(res1["ids"], torch.tensor([1], dtype=torch.long))


def should_pad_static_outputs():
    from hmlib.tracking_utils.bytetrack import HmByteTrackerCudaStatic

    static = HmByteTrackerCudaStatic(max_detections=8, max_tracks=8, device="cpu")
    frame0 = {
        "frame_id": torch.tensor([0], dtype=torch.long),
        "bboxes": torch.zeros((8, 4), dtype=torch.float32),
        "labels": torch.zeros((8,), dtype=torch.long),
        "scores": torch.zeros((8,), dtype=torch.float32),
        "num_detections": torch.tensor([2], dtype=torch.long),
    }
    frame0["bboxes"][:2] = torch.tensor(
        [[10.0, 10.0, 30.0, 40.0], [100.0, 100.0, 140.0, 160.0]], dtype=torch.float32
    )
    frame0["labels"][:2] = torch.tensor([1, 1], dtype=torch.long)
    frame0["scores"][:2] = torch.tensor([0.9, 0.85], dtype=torch.float32)

    out = static.track(frame0)
    assert out["ids"].shape == (8,)
    assert out["bboxes"].shape == (8, 4)
    assert out["labels"].shape == (8,)
    assert out["scores"].shape == (8,)
    assert torch.equal(out["num_tracks"], torch.tensor([2], dtype=torch.long))
    assert torch.equal(out["num_detections"], torch.tensor([2], dtype=torch.long))

    assert torch.equal(out["ids"][:2], torch.tensor([0, 1], dtype=torch.long))
    assert torch.all(out["ids"][2:] == -1)

    frame0["frame_id"].fill_(1)
    frame0["bboxes"][:2] += 2.0
    updated = static.track(frame0)
    assert torch.equal(updated["ids"][:2], torch.tensor([0, 1], dtype=torch.long))
    assert torch.all(updated["ids"][2:] == -1)
    assert torch.isfinite(static._track_mean).all()
    assert torch.isfinite(static._track_covariance).all()


@pytest.mark.parametrize("tracker_name", ["HmByteTrackerCuda", "HmByteTrackerCudaStatic"])
def should_match_dense_kalman_updates_without_cpu_lapack(tracker_name, monkeypatch):
    from hmlib.tracking_utils import bytetrack

    def reject_lapack(*args, **kwargs):
        raise AssertionError("CPU tracking must not require PyTorch LAPACK support")

    monkeypatch.setattr(torch.linalg, "cholesky_ex", reject_lapack)
    monkeypatch.setattr(torch, "cholesky_solve", reject_lapack)
    tracker = getattr(bytetrack, tracker_name)(device="cpu")
    measurements = torch.tensor(
        [[20.0, 25.0, 2.0 / 3.0, 30.0], [120.0, 130.0, 0.8, 60.0]],
        dtype=torch.float32,
    )
    mean, covariance = tracker._kalman_initiate(measurements)
    for frame in range(20):
        mean, covariance = tracker._kalman_predict(mean, covariance)
        projected_mean, projected_cov = tracker._kalman_project(mean, covariance)
        torch.testing.assert_close(
            projected_cov,
            torch.diag_embed(projected_cov.diagonal(dim1=1, dim2=2)),
            rtol=0,
            atol=0,
        )
        measurements = measurements + torch.tensor(
            [[1.0, -0.5, 0.001, 0.2], [-0.7, 0.8, -0.001, -0.1]]
        )
        # Compare the specialized solve against a full matrix solve, including
        # position/velocity cross-covariance accumulated over multiple frames.
        prior = covariance.numpy().astype(np.float64)
        projected = projected_cov.numpy().astype(np.float64)
        gain = np.linalg.solve(projected, prior[:, :, :4].transpose(0, 2, 1))
        gain = gain.transpose(0, 2, 1)
        innovation = (measurements - projected_mean).numpy().astype(np.float64)
        expected_mean = mean.numpy() + (gain @ innovation[:, :, None])[:, :, 0]
        expected_cov = prior - gain @ projected @ gain.transpose(0, 2, 1)
        mean, covariance = tracker._kalman_update(mean, covariance, measurements)
        np.testing.assert_allclose(mean.numpy(), expected_mean, rtol=2e-6, atol=2e-5)
        np.testing.assert_allclose(covariance.numpy(), expected_cov, rtol=2e-5, atol=2e-5)
        assert torch.isfinite(mean).all(), frame
        assert torch.isfinite(covariance).all(), frame
