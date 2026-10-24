import copy
import shutil
import subprocess

import pytest
import yaml

from hmlib.ui.runtime_setup import (
    OperatorWorkspace,
    apply_engine_preferences,
    inspect_pipeline,
    launch_arguments,
)
from hmlib.ui.workspace import Workspace


@pytest.fixture
def desktop(tmp_path):
    root = tmp_path / "games"
    with Workspace(root, tmp_path / "session") as workspace:
        workspace.dispatch("games.create", {"game_id": "operator-test"})
        yield workspace


def should_persist_operator_settings_and_translate_them_for_python_launch(desktop):
    editor = OperatorWorkspace(desktop.game())
    snapshot = editor.snapshot()
    changes = {
        "hm_ui.run.detector": "hockeyref",
        "hm_ui.run.precision": "trt-fp32",
        "hm_ui.run.analytics": "pose-jersey",
        "hm_ui.run.save_video": False,
        "video_out.output_width": "1920",
        "video_out.bit_rate": "8M",
        "team.backend": "lab",
        "team.enabled": True,
    }
    editor.update_settings(changes, expected_revision=snapshot["revision"])
    private = yaml.safe_load((desktop.game() / "config.yaml").read_text())
    assert private["video_out"]["output_width"] == 1920
    assert "stitching" not in private
    arguments = launch_arguments(desktop.game(), "track")
    assert "--detector-model" in arguments and "hockeyref" in arguments
    assert "--no-detector-trt-fp16" in arguments
    assert "--detect-jersey-numbers" in arguments
    assert "--no-save-video" in arguments
    config_index = arguments.index("--config")
    assert arguments[config_index + 1].endswith("tracking_pose_jersey.yaml")
    command = desktop.pipeline_command(desktop.game(), "track", ["--detector-model", "distilled"])
    assert command[-2:] == ["--detector-model", "distilled"]
    before = (desktop.game() / "config.yaml").read_bytes()
    desktop.poll()
    assert (desktop.game() / "config.yaml").read_bytes() == before


def should_fence_stale_operator_edits(desktop):
    editor = OperatorWorkspace(desktop.game())
    snapshot = editor.snapshot()
    editor.update_settings({"team.backend": "lab"})
    with pytest.raises(ValueError, match="changed"):
        editor.update_settings({"team.enabled": True}, expected_revision=snapshot["revision"])


@pytest.mark.parametrize(
    "change",
    [
        {"video_out.output_width": "1919"},
        {"video_out.output_height": "nan"},
        {"video_out.bit_rate": "8M;false"},
        {"video_out.youtube_stream_url": "http://example.com"},
        {"hm_ui.run.precision": "trt-int8"},
        {"hm_ui.run.batch_size": 0},
    ],
)
def should_reject_invalid_run_settings_without_mutation(desktop, change):
    path = desktop.game() / "config.yaml"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        OperatorWorkspace(desktop.game()).update_settings(change)
    assert path.read_bytes() == before


def should_validate_jersey_rate_at_the_runtime_boundary(desktop):
    from hmlib.aspen.plugins.native_analytics_plugin import ParseqJerseyPlugin

    editor = OperatorWorkspace(desktop.game())
    editor.update_settings({"pipeline.player-analytics.jersey.rate-hz": 120})
    interval, _ = ParseqJerseyPlugin()._settings(
        {"shared": {"game_config": editor._effective()}}, 2, 0.8
    )
    assert interval == round(1e9 / 120)
    with pytest.raises(ValueError):
        editor.update_settings({"pipeline.player-analytics.jersey.rate-hz": 121})


def should_clear_old_engines_without_constructing_missing_graph_nodes():
    config = {
        "hm_ui": {"run": {"precision": "pytorch", "pose_precision": "onnx"}},
        "aspen": {
            "plugins": {
                "detector_factory": {"params": {"trt": {"enable": True, "engine": "saved.engine"}}}
            }
        },
    }
    original = copy.deepcopy(config)
    apply_engine_preferences(config)
    assert config["aspen"]["plugins"]["detector_factory"]["params"]["trt"] == {
        "enable": False,
        "engine": "saved.engine",
    }
    assert "pose_factory" not in config["aspen"]["plugins"]
    assert config["hm_ui"] == original["hm_ui"]


def should_preserve_automatic_dimensions_after_reloading(desktop):
    editor = OperatorWorkspace(desktop.game())
    editor.update_settings({"video_out.output_width": "1920", "video_out.output_height": "1080"})
    editor.update_settings(
        {"video_out.output_width": "auto", "video_out.output_height": "", "video_out.bit_rate": ""}
    )
    assert editor.snapshot()["settings"]["video_out.output_width"] == "auto"
    assert editor.snapshot()["settings"]["video_out.output_height"] == ""


def should_inspect_unprepared_engines_without_loading_weights(desktop):
    OperatorWorkspace(desktop.game()).update_settings(
        {
            "hm_ui.run.precision": "onnx-int8",
            "hm_ui.run.pose_precision": "trt-fp32",
            "hm_ui.run.analytics": "pose",
        },
    )
    graph = inspect_pipeline(desktop.game())
    nodes = {node["name"]: node for node in graph["nodes"]}
    detector = nodes["detector_factory"]["params"]
    assert detector["onnx"]["enable"] is True
    assert detector["onnx"]["preparation"] == "missing"
    assert detector["trt"]["enable"] is False
    assert nodes["pose_factory"]["params"]["trt"]["fp16"] is False


@pytest.mark.parametrize(
    "analysis", ["tracking", "jersey", "pose", "pose-jersey", "pose-actions", "pose-jersey-actions"]
)
def should_keep_complete_output_graph_for_each_analysis_preset(
    desktop, analysis, monkeypatch, tmp_path
):
    import graphlib

    from hmlib.aspen import AspenNet

    OperatorWorkspace(desktop.game()).update_settings({"hm_ui.run.analytics": analysis})
    arguments = launch_arguments(desktop.game(), "track")
    paths = [arguments[index + 1] for index, value in enumerate(arguments) if value == "--config"]
    graph = inspect_pipeline(desktop.game())
    assert graph["graph"] == " + ".join(paths)
    nodes = {node["name"]: node for node in graph["nodes"]}
    order = list(
        graphlib.TopologicalSorter(
            {name: node["depends"] for name, node in nodes.items()}
        ).static_order()
    )
    assert {"detector_factory", "detector", "tracker", "apply_camera", "video_out"} <= nodes.keys()
    assert ("pose" in nodes) == ("pose" in analysis)
    assert ("jersey_numbers" in nodes) == ("jersey" in analysis)
    assert ("actions" in nodes) == ("actions" in analysis)
    if "actions" in nodes:
        assert order.index("actions") < order.index("overlays")
        assert order.index("actions") < order.index("play_tracker")
    if "jersey_numbers" in nodes:
        assert order.index("jersey_numbers") < order.index("overlays")
    monkeypatch.chdir(tmp_path)
    stubbed = copy.deepcopy(nodes)
    for node in stubbed.values():
        if "JoinPlugin" not in node["class"]:
            node["enabled"] = False
    AspenNet(analysis, {"plugins": stubbed})


def should_validate_custom_analysis_dependencies_and_apply_pose_model(desktop):
    editor = OperatorWorkspace(desktop.game())
    editor.update_settings(
        {"hm_ui.run.analytics": "custom", "pipeline.player-analytics.jersey.enable": True}
    )
    graph = inspect_pipeline(desktop.game())
    nodes = {node["name"]: node for node in graph["nodes"]}
    assert "pose" not in nodes
    assert nodes["jersey_numbers"]["params"]["roi_mode"] == "bbox"
    for changes in (
        {"pipeline.player-analytics.jersey.roi-mode": "pose"},
        {"pipeline.player-analytics.action.enable": True},
        {
            "pipeline.player-analytics.pose.enable": True,
            "pipeline.player-analytics.action.enable": True,
            "pipeline.player-analytics.pose.rate-hz": "5",
        },
        {"hm_ui.run.pose_model": "custom"},
    ):
        with pytest.raises(ValueError):
            editor.update_settings(changes)
    editor.update_settings(
        {
            "pipeline.player-analytics.pose.enable": True,
            "pipeline.player-analytics.action.enable": True,
            "pipeline.player-analytics.pose.rate-hz": "fps",
            "hm_ui.run.pose_model": "coco",
        }
    )
    nodes = {node["name"]: node for node in inspect_pipeline(desktop.game())["nodes"]}
    assert "coco" in nodes["pose_factory"]["params"]["pose_config"]
    assert "actions" in nodes


def should_toggle_individual_analysis_from_the_current_preset_and_resolve_custom_paths(desktop):
    editor = OperatorWorkspace(desktop.game())
    editor.update_settings({"hm_ui.run.analytics": "pose"})
    assert editor.snapshot()["settings"]["pipeline.player-analytics.pose.enable"] is True
    editor.update_settings({"pipeline.player-analytics.jersey.enable": True})
    saved = editor.snapshot()["settings"]
    assert saved["hm_ui.run.analytics"] == "custom"
    assert saved["pipeline.player-analytics.pose.enable"] is True
    assert saved["pipeline.player-analytics.jersey.enable"] is True
    editor.update_settings(
        {"hm_ui.run.jersey_model": "custom", "hm_ui.run.jersey_model_path": "models/jersey.onnx"}
    )
    nodes = {node["name"]: node for node in inspect_pipeline(desktop.game())["nodes"]}
    assert nodes["jersey_numbers"]["class"].endswith("ParseqJerseyPlugin")
    assert nodes["jersey_numbers"]["params"]["model_path"] == str(
        desktop.game() / "models/jersey.onnx"
    )
    editor.update_settings({"pipeline.player-analytics.jersey.roi-mode": "pose"})
    nodes = {node["name"]: node for node in inspect_pipeline(desktop.game())["nodes"]}
    assert "coco" in nodes["pose_factory"]["params"]["pose_config"]


def should_offer_baseline_camera_parameters_for_setup_and_experiments(desktop):
    from hmlib.config import ROOT_DIR, baseline_config

    editor = OperatorWorkspace(desktop.game())
    snapshot = editor.snapshot()
    values = snapshot["settings"]
    baseline = baseline_config(ROOT_DIR)["rink"]["camera"]
    assert values["rink.camera.max_speed_ratio_x"] == baseline["max_speed_ratio_x"]
    assert values["rink.camera.color.contrast"] == baseline["color"]["contrast"]
    keys = [spec["key"] for spec in snapshot["catalog"]]
    assert len(keys) == len(set(keys))
    editor.update_settings(
        {"rink.camera.follower_box_scale_width": 1.8, "rink.camera.color.gamma": 1.2}
    )
    values = editor.snapshot()["settings"]
    assert values["rink.camera.follower_box_scale_width"] == 1.8
    assert values["rink.camera.color.gamma"] == 1.2
    with pytest.raises(ValueError):
        editor.update_settings({"rink.camera.color.gamma": 0})


def should_inspect_unprepared_appearance_and_exact_native_pose_models(desktop):
    from hmlib.ui.analytics_graph import NATIVE_POSE_SHA256

    OperatorWorkspace(desktop.game()).update_settings(
        {"hm_ui.run.analytics": "pose-actions", "pipeline.tracker.reid-enable": True}
    )
    nodes = {node["name"]: node for node in inspect_pipeline(desktop.game())["nodes"]}
    assert nodes["tracker"]["depends"] == ["player_reid"]
    assert nodes["player_reid"]["params"]["onnx_path"].endswith("reid.onnx")
    assert nodes["pose_factory"]["params"]["pose_checkpoint_sha256"] == NATIVE_POSE_SHA256
    assert nodes["pose_factory"]["params"]["pose_config"].endswith("rtmpose-m_coco17-256x192.py")
    assert nodes["actions"]["class"].endswith("TemporalActionPlugin")


def should_keep_manual_video_synchronization_for_audio(monkeypatch):
    import hmlib.stitching.synchronize as sync
    from hmlib.cli.hmtrack import _resolve_mux_audio_source

    def unexpected_audio_matching(*args):
        pytest.fail("Manual/IMU synchronization must not perform audio matching")

    monkeypatch.setattr(sync, "synchronize_by_audio", unexpected_audio_matching)
    monkeypatch.setattr(
        "hmlib.ui.output_audio.prepare_chapter_audio",
        lambda paths, work_dir, **kwargs: (str(paths[0]), None),
    )
    assert _resolve_mux_audio_source(
        {"left": ["left.mp4"], "right": ["right.mp4"]}, frame_offsets=(23, 0)
    ) == ("right.mp4", None, 0.0)
    assert _resolve_mux_audio_source(["left.mp4", "right.mp4"], frame_offsets=(0, 17)) == (
        "left.mp4",
        None,
        0.0,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
@pytest.mark.parametrize("offsets", [(0, 20), (30, 60), (60, 30)])
def should_seek_concatenated_audio_using_selected_original_video_rate(tmp_path, offsets):
    import av

    from hmlib.cli.hmtrack import _resolve_mux_audio_source

    chapters = []
    for index in range(2):
        path = tmp_path / f"chapter-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=60:duration=1",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency={440 + index * 440}:duration=1",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                "-shortest",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        chapters.append(str(path))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    tmp_path.chmod(0o555)
    try:
        source, handle, offset = _resolve_mux_audio_source(
            {"left": chapters, "right": chapters}, frame_offsets=offsets, start_frame=15
        )
        try:
            with av.open(source) as container:
                assert len(container.streams.audio) == 1
                assert len(container.streams.video) == 0
            assert offset == pytest.approx(-(min(offsets) + 15) / 60)
            assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
        finally:
            handle.cleanup()
    finally:
        tmp_path.chmod(0o755)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
@pytest.mark.parametrize("partial_audio", [False, True])
def should_handle_silent_chapters_and_reject_inconsistent_audio(tmp_path, partial_audio):
    from hmlib.cli.hmtrack import _resolve_mux_audio_source

    chapters = []
    for index in range(2):
        path = tmp_path / f"chapter-{index}.mp4"
        command = [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=30:duration=1",
        ]
        if partial_audio and index == 1:
            command += ["-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac"]
        subprocess.run(
            [*command, "-c:v", "mpeg4", str(path)], check=True, capture_output=True, timeout=30
        )
        chapters.append(str(path))
    inputs = {"left": chapters, "right": chapters}
    if partial_audio:
        with pytest.raises(ValueError, match="disagree"):
            _resolve_mux_audio_source(inputs, frame_offsets=(0, 10), start_frame=15)
    else:
        assert _resolve_mux_audio_source(inputs, frame_offsets=(0, 10), start_frame=15) == (
            None,
            None,
            -0.5,
        )


def should_keep_single_source_chapters_distinct_from_stereo_inputs(monkeypatch):
    from hmlib.cli.hmtrack import _resolve_mux_audio_source

    selected = []

    def prepare(paths, work_dir, **kwargs):
        selected.extend(str(path) for path in paths)
        return "joined-audio.m4a", None

    def reject_stereo(*args):
        pytest.fail("Two single-source chapters must not trigger stereo synchronization")

    monkeypatch.setattr("hmlib.ui.output_audio.prepare_chapter_audio", prepare)
    monkeypatch.setattr("hmlib.stitching.synchronize.synchronize_by_audio", reject_stereo)
    result = _resolve_mux_audio_source(["part-1.mp4", "part-2.mp4"], single_playlist=True)
    assert selected == ["part-1.mp4", "part-2.mp4"]
    assert result == ("joined-audio.m4a", None, 0.0)
