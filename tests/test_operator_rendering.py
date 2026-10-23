import numpy as np
import pytest
import torch

from hmlib.camera.rink_framing import RinkCameraGuide, draw_rink_map
from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.transforms.scoreboard_transforms import HmCaptureScoreboard, HmRenderScoreboard
from hmlib.vis.pt_text import _create_text_images, find_font_path


def mapping():
    return {
        "version": 1,
        "image_size": [1000, 500],
        "rink_size": [60.96, 25.908],
        "homography": [[0.06096, 0, 0], [0, 0.051816, 0], [0, 0, 1]],
        "goals": [],
    }


def should_anticipate_consistent_player_motion_and_reset_after_seek():
    guide = RinkCameraGuide(mapping(), {"anticipate": True}, (1000, 500))
    baseline = np.array([200, 100, 700, 400])
    arena = [0, 0, 1000, 500]
    boxes = np.array([[300, 180, 320, 220], [330, 190, 350, 230], [360, 180, 380, 220]])
    for index in range(30):
        shifted = boxes + [index * 3, 0, index * 3, 0]
        result = guide.step(index / 30, baseline, arena, shifted, [1, 2, 3], [250, 140, 700, 300])
        assert result[0] >= 0 and result[2] <= 1000
        assert result[1] >= 0 and result[3] <= 500
    assert guide.info["active"]
    assert result[0] > baseline[0] + 10
    np.testing.assert_array_equal(guide.step(0, baseline, arena, boxes, [1, 2, 3]), baseline)


def should_ignore_sparse_players_and_discard_disappeared_tracks():
    guide = RinkCameraGuide(mapping(), {"anticipate": True, "adaptive_zoom": True}, (1000, 500))
    baseline = [200, 100, 700, 400]
    for index in range(15):
        result = guide.step(
            index / 30,
            baseline,
            [0, 0, 1000, 500],
            [[300 + index, 100, 320 + index, 150]],
            [1],
            [250, 100, 600, 300],
        )
        np.testing.assert_array_equal(result, baseline)
    for index in range(15, 30):
        guide.step(index / 30, baseline, [0, 0, 1000, 500], [], [])
    assert not guide.history


@pytest.mark.parametrize(
    "position,where",
    [("upper-left", (20, 20)), ("upper-right", (20, 620)), ("lower-left", (330, 20))],
)
def should_place_preview_map_without_touching_recording(position, where):
    recording = torch.zeros((1, 360, 640, 3), dtype=torch.uint8)
    preview = draw_rink_map(
        recording.clone(),
        {"points": [[[30, 12]]], "size": [60.96, 25.908], "position": position, "map_size": "icon"},
    )
    assert torch.count_nonzero(recording) == 0
    assert torch.count_nonzero(preview[0, where[0], where[1]]) > 0


@pytest.mark.parametrize(
    "position,offset", [("upper-left", 0), ("upper-center", 8), ("upper-right", 16)]
)
def should_render_scoreboard_at_requested_program_position(position, offset):
    frame = torch.zeros((1, 10, 20, 3), dtype=torch.uint8)
    board = torch.full((1, 2, 4, 3), 137, dtype=torch.float32)
    rendered = HmRenderScoreboard(["img"])(
        {"img": frame, "scoreboard_img": board, "scoreboard_preferences": {"position": position}}
    )["img"]
    assert torch.all(rendered[:, :2, offset : offset + 4] == 137)
    assert int(rendered.sum()) == 137 * 2 * 4 * 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Scoreboard warp requires CUDA")
def should_resize_scoreboard_live_without_losing_source_dimensions():
    capture = HmCaptureScoreboard()
    config = {
        "scoreboard_points": [[2, 2], [22, 2], [22, 12], [2, 12]],
        "dest_width": 20,
        "dest_height": 10,
    }

    def frame(scale):
        return {
            "img": torch.full((1, 40, 60, 3), 127, dtype=torch.uint8, device="cuda"),
            "scoreboard_cfg": config,
            "scoreboard_preferences": {
                "scoreboard_scale": scale,
                "resize_filter": "bilinear",
                "sharpen_amount": 1,
            },
        }

    first = capture(frame(1))["scoreboard_img"]
    second = capture(frame(2))["scoreboard_img"]
    assert second.shape[1:3] == (first.shape[1] * 2, first.shape[2] * 2)
    assert config["dest_width"] == 20


def should_reject_invalid_scoreboard_filter_and_render_actual_font_glyphs():
    with pytest.raises(ValueError, match="resize filter"):
        Scoreboard(
            [[0, 0], [10, 0], [10, 10], [0, 10]], 10, 10, torch.float, resize_filter="missing"
        )
    glyphs = _create_text_images(find_font_path(), 16, (255, 255, 255), torch.device("cpu"))
    assert "A" in glyphs and "0" in glyphs and " " in glyphs
    assert "\n" not in glyphs


def should_translate_all_saved_scoreboard_corners_without_mutating_config(monkeypatch):
    from hmlib.transforms import scoreboard_transforms as module

    points = [[20, 30], [40, 30], [40, 50], [20, 50]]
    monkeypatch.setattr(module, "get_clip_box", lambda **kwargs: [10, 20, 100, 100])
    configure = module.HmConfigureScoreboard(game_id="fixture", allow_interactive_setup=False)
    result = configure(
        {
            "game_config": {
                "rink": {
                    "scoreboard": {
                        "perspective_polygon": points,
                        "projected_width": 20,
                        "projected_height": 20,
                    }
                }
            }
        }
    )
    assert result["scoreboard_cfg"]["scoreboard_points"] == [[10, 10], [30, 10], [30, 30], [10, 30]]
    assert points == [[20, 30], [40, 30], [40, 50], [20, 50]]


def should_record_bfloat16_detector_observations_as_numpy_compatible_float32():
    from mmdet.structures import DetDataSample, TrackDataSample
    from mmengine.structures import InstanceData

    from hmlib.aspen.plugins.detector_plugin import DetectorInferencePlugin
    from hmlib.telemetry.capture import array

    class Detector:
        def predict(self, inputs, samples):
            for sample in samples:
                sample.pred_instances = InstanceData(
                    bboxes=torch.tensor([[10, 20, 30, 40]], dtype=torch.bfloat16),
                    scores=torch.tensor([0.75], dtype=torch.bfloat16),
                    labels=torch.tensor([0]),
                )
            return samples

    frame = DetDataSample(metainfo={"img_id": 1})
    sample = TrackDataSample(video_data_samples=[frame])
    DetectorInferencePlugin().forward(
        {
            "inputs": torch.zeros(1, 3, 64, 64),
            "data_samples": sample,
            "detector_model": Detector(),
        }
    )
    assert array(frame.pred_instances.bboxes).dtype == np.float32
    np.testing.assert_array_equal(array(frame.pred_instances.bboxes), [[10, 20, 30, 40]])
    assert array(frame.pred_instances.scores).tolist() == [0.75]


def should_keep_desktop_first_run_out_of_the_legacy_scoreboard_selector(monkeypatch):
    from hmlib.transforms import scoreboard_transforms as module

    monkeypatch.setenv("HM_UI_SESSION_DIR", "/tmp/desktop-fixture")
    monkeypatch.setattr(
        module, "configure_scoreboard", lambda **kwargs: pytest.fail("Use the desktop editor")
    )
    result = module.HmConfigureScoreboard(game_id="fixture")(
        {"game_config": {"rink": {"scoreboard": {}}}}
    )
    assert "scoreboard_cfg" not in result
