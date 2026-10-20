from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from hmlib.scoreboard import proposal


def should_use_cpu_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []
    monkeypatch.setattr(proposal, "_run_model", lambda path, tensor, device: calls.append(device))
    proposal.infer(Path("model.pt"), torch.zeros(1))
    assert calls == ["cpu"]


@pytest.mark.parametrize(
    "error",
    [
        torch.cuda.OutOfMemoryError("allocation failed"),
        torch.jit.Error("CUDA out of memory"),
        torch.jit.Error("CUDA error: out of memory"),
        torch.jit.Error("cudaErrorMemoryAllocation"),
        torch.jit.Error("HIP out of memory"),
        torch.jit.Error("HIP error: out of memory"),
        torch.jit.Error("hipErrorOutOfMemory"),
    ],
)
def should_release_gpu_and_retry_once_on_cpu_after_oom(monkeypatch, error, caplog):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def run(path, tensor, device):
        calls.append(device)
        if device.startswith("cuda"):
            raise error
        return ("classes", "masks")

    monkeypatch.setattr(proposal, "_run_model", run)
    monkeypatch.setattr(proposal.gc, "collect", lambda: calls.append("collect"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("release"))
    assert proposal.infer(Path("model.pt"), torch.zeros(1)) == ("classes", "masks")
    assert calls == ["cuda:0", "collect", "release", "cpu"]
    assert "retrying once on CPU" in caplog.text


def should_not_retry_unrelated_gpu_errors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def run(path, tensor, device):
        calls.append(device)
        raise RuntimeError("invalid model tensor shape")

    monkeypatch.setattr(proposal, "_run_model", run)
    with pytest.raises(RuntimeError, match="invalid model tensor shape"):
        proposal.infer(Path("model.pt"), torch.zeros(1))
    assert calls == ["cuda:0"]


def should_map_a_confident_mask_to_original_panorama_and_reject_background():
    classes = np.full((1, 100, 2), -20, dtype=np.float32)
    classes[:, :, 1] = 20
    masks = np.full((1, 100, 192, 512), -20, dtype=np.float32)
    assert proposal.postprocess(classes, masks, (1024, 384), (8192, 3072)) is None
    classes[0, 0] = [20, -20]
    masks[0, 0, 10:20, 30:50] = 20
    points = proposal.postprocess(classes, masks, (1024, 384), (8192, 3072))
    assert points is not None
    np.testing.assert_allclose(points, [[960, 320], [1592, 320], [1592, 632], [960, 632]], atol=16)
    masks[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        proposal.postprocess(classes, masks, (1024, 384), (8192, 3072))


def should_reject_corrupt_cached_models(monkeypatch, tmp_path):
    monkeypatch.delenv("HM_SCOREBOARD_TORCH_MODEL", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = tmp_path / ".cache/hstream/models" / proposal.MODEL_NAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a trusted model")
    with pytest.raises(ValueError, match="checksum mismatch"):
        proposal.model_path()


@pytest.mark.parametrize("saved", [None, [[4, 4], [40, 4], [40, 20], [4, 20]], [[0, 0]] * 4])
def should_propose_only_missing_selections_and_persist_only_after_review(monkeypatch, saved):
    from hmlib.scoreboard import selector

    config = {"rink": {"scoreboard": {"perspective_polygon": saved}}}
    monkeypatch.setattr(selector, "get_game_config", lambda **kwargs: config)
    proposals, writes = [], []
    predicted = [[5, 5], [45, 5], [45, 25], [5, 25]]

    def predict(image, size):
        proposals.append(size)
        return predicted

    monkeypatch.setattr(proposal, "propose", predict)
    monkeypatch.setattr(selector, "save_private_config", lambda **kwargs: writes.append(kwargs))

    def review(self):
        assert not writes
        if saved is None:
            assert self.points == [tuple(point) for point in predicted]
        self.points = [(6, 6), (44, 6), (44, 24), (6, 24)]

    monkeypatch.setattr(selector.ScoreboardSelector, "run", review)
    result = selector.configure_scoreboard("test", image=Image.new("RGB", (64, 48)), force=True)
    assert bool(proposals) == (saved is None)
    assert result == [[6, 6], [44, 6], [44, 24], [6, 24]]
    assert len(writes) == 1


def should_keep_manual_selection_available_and_explain_proposal_failure(monkeypatch):
    from hmlib.scoreboard import selector

    monkeypatch.setattr(selector, "get_game_config", lambda **kwargs: {})
    monkeypatch.setattr(selector, "save_private_config", lambda **kwargs: None)

    def fail(*args):
        raise RuntimeError("bad model")

    monkeypatch.setattr(proposal, "propose", fail)

    def review(self):
        assert self.points == []
        assert "bad model" in self.proposal_message
        self.points = [(0, 0)] * 4

    monkeypatch.setattr(selector.ScoreboardSelector, "run", review)
    assert selector.configure_scoreboard("test", image=Image.new("RGB", (64, 48))) == [[0, 0]] * 4


def should_reject_a_tilted_proposal_that_save_would_collapse(monkeypatch):
    from hmlib.scoreboard import selector

    classes = np.full((1, 100, 2), -20, dtype=np.float32)
    classes[:, :, 1] = 20
    classes[0, 0] = [20, -20]
    masks = np.full((1, 100, 192, 512), -20, dtype=np.float32)
    cv2.fillConvexPoly(masks[0, 0], np.array([[25, 10], [50, 5], [75, 10], [50, 30]]), 20)
    predicted = proposal.postprocess(classes, masks, (2048, 768), (2048, 768))
    assert predicted is not None and len({tuple(point) for point in predicted}) == 4
    monkeypatch.setattr(proposal, "propose", lambda *args: predicted)
    monkeypatch.setattr(selector, "get_game_config", lambda **kwargs: {})
    writes = []
    monkeypatch.setattr(selector, "save_private_config", lambda **kwargs: writes.append(kwargs))
    manual = [[10, 10], [90, 10], [90, 70], [10, 70]]

    def review(self):
        assert not writes
        assert self.points == []
        assert "cannot be ordered" in self.proposal_message
        self.process_ok(manual)

    monkeypatch.setattr(selector.ScoreboardSelector, "run", review)
    assert selector.configure_scoreboard("test", image=Image.new("RGB", (2048, 768))) == manual
    assert len(writes) == 1
