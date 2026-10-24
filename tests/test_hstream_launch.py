"""Shared desktop launch contracts, initial invocation and exported jobs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from hmlib.ui.desktop import _anonymous_game, launch_desktop
from hmlib.ui.hstream_launch import export_job, make_plan, option_assignments, parser


@pytest.fixture
def game(tmp_path, monkeypatch):
    import hmlib.config as configuration

    root = tmp_path / "games"
    selected = root / "match"
    selected.mkdir(parents=True)
    (selected / "one.mp4").write_bytes(b"source")
    (selected / "two.mp4").write_bytes(b"source")
    (selected / "config.yaml").write_text(
        yaml.safe_dump({"game": {"videos": {"source": ["one.mp4", "two.mp4"]}}})
    )
    monkeypatch.setenv("HM_GAME_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setitem(
        configuration.get_game_config_private.__globals__, "GAME_DIR_BASE", str(root)
    )
    return selected


def should_preserve_yaml_commas_and_grouped_native_options():
    assert option_assignments('camera.x=[1,2],camera.y="one,two",camera.z={a: 1, b: 2}') == [
        "camera.x=[1,2]",
        'camera.y="one,two"',
        "camera.z={a: 1, b: 2}",
    ]
    with pytest.raises(ValueError, match="Malformed"):
        option_assignments("camera.x=[1,2")


@pytest.mark.parametrize(
    "native,python", [("auto", "automatic"), ("standard", "normal"), ("low", "low")]
)
def should_apply_shared_memory_profile_to_python_processing(native, python):
    from hmlib.ui.hstream_launch import _translate_native_settings

    config = {"runtime": {"gpu_memory_profile": native}}
    _translate_native_settings(config, set())
    assert config["hm_ui"]["run"]["memory_profile"] == python


def should_match_native_network_sink_address_and_rtmp_precedence():
    from hmlib.ui.hstream_launch import _translate_native_settings

    config = {
        "video_out": {"youtube_stream_url": "rtmps://default.invalid/live"},
        "pipeline": {"sink3": {"output-file": "rtmp://127.0.0.1/live/fixture", "rtsp-port": 9544}},
    }
    _translate_native_settings(config, {"RTMP", "RTSP"})
    assert config["video_out"]["youtube_stream_url"] == "rtmp://127.0.0.1/live/fixture"
    assert config["video_out"]["rtsp_port"] == 9544
    assert config["video_out"]["rtsp_program_mount"] == "/ds-test"
    assert config["video_out"]["rtsp_host"] == "0.0.0.0"


def should_launch_all_chapters_with_shared_color_and_audio_settings(game, tmp_path):
    args = parser().parse_args(
        [
            "-g",
            game.name,
            "--enable-sinks=ENCODE_STITCHED_FILE",
            "--options=hstream_ui.camera_controls.Exposure_x100=200",
            "--options=hstream_ui.camera_controls.Bring_Up_Shadows=40",
            "--options=pipeline.hmaudio.enable=false",
        ]
    )
    plan = make_plan(args)
    assert plan.sources == {"source": [game / "one.mp4", game / "two.mp4"]}
    assert plan.config["rink"]["camera"]["color"]["exposure_ev"] == 1
    assert plan.config["rink"]["camera"]["color"]["shadow_lift"] == 40
    assert "--no-audio" in plan.arguments
    assert plan.config["video_out"]["archive_stitched"]
    assert not plan.config["video_out"]["archive_program"]
    assert not plan.config["hm_shared_ui"]["telemetry"]
    assert plan.write(tmp_path)[-1] == "--camera-ui=1"


def should_preserve_initial_source_config_clock_and_cli_precedence(game, tmp_path):
    override = tmp_path / "overlay.yaml"
    override.write_text("aspen:\n  pipeline:\n    queue_size: 7\n")
    initial = tmp_path / "initial.json"
    initial.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": [
                    "--input-video",
                    str(game / "two.mp4"),
                    "--config",
                    override.name,
                    "--start-frame=30",
                    "--max-time=00:00:04",
                    "--deploy-dir=custom-output",
                    "--batch-size=2",
                    "--camera-ui=1",
                ],
            }
        )
    )
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=initial)
    assert plan.sources == {"source": [game / "two.mp4"]}
    assert plan.start_frame == 30
    assert plan.limit_seconds == 4
    assert plan.cwd == tmp_path
    assert plan.config["aspen"]["pipeline"]["queue_size"] == 7
    command = plan.write(tmp_path)
    assert not any(value.startswith("--start-time") for value in command)
    assert "--deploy-dir" not in command
    assert "--deploy-dir=custom-output" in command
    assert command.index(str(tmp_path / "launch.yaml")) < command.index(override.name)
    plan.start_seconds = 2
    sought = plan.write(tmp_path, seek_seconds=3)
    assert "--start-frame=30" not in sought
    assert "--start-time=5.000000000" in sought


def should_resolve_initial_config_root_and_private_opt_out(game, tmp_path):
    import hmlib.config as configuration

    # A private-only source and camera setting must not survive opt-out through
    # the adapter's generated --config, even when desktop user defaults exist.
    config_root = tmp_path / "custom-config"
    (config_root / "config" / "games").mkdir(parents=True)
    (config_root / "config" / "baseline.yaml").write_text(
        yaml.safe_dump(configuration.baseline_config(root_dir=None))
    )
    (config_root / "config" / "games" / f"{game.name}.yaml").write_text(
        yaml.safe_dump({"game": {"videos": {"source": ["two.mp4"]}}, "camera": {"tag": "public"}})
    )
    private = yaml.safe_load((game / "config.yaml").read_text())
    private["camera"] = {"tag": "private"}
    (game / "config.yaml").write_text(yaml.safe_dump(private))
    user = tmp_path / ".hstream" / "hstream.yaml"
    user.parent.mkdir()
    user.write_text("hstream_ui: {camera_controls: {Exposure_x100: 50}}\n")
    initial = tmp_path / "initial.json"
    initial.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": ["--root-dir", config_root.name, "--ignore-private-config=1"],
            }
        )
    )
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=initial)
    assert plan.sources == {"source": [game / "two.mp4"]}
    assert plan.config["camera"]["tag"] == "public"
    assert plan.config["rink"]["camera"]["color"]["exposure_ev"] == 0.25


def should_export_saved_recipe_and_execute_the_selected_backend(game, tmp_path):
    config = yaml.safe_load((game / "config.yaml").read_text())
    config["hstream_ui"] = {
        "job": {
            "arguments": [
                "--enable-sinks=RENDER,ENCODE_STITCHED_FILE,RTMP",
                "--render-realtime",
                "--start-time=00:00:12",
                "--options=stitching.blend_mode=alpha",
            ],
            "capture_telemetry": True,
        }
    }
    (game / "config.yaml").write_text(yaml.safe_dump(config))
    script = tmp_path / "job with spaces.sh"
    export_job(
        parser().parse_args(
            [
                "--export-job",
                "--game-dir",
                str(game),
                "--output",
                str(script),
                "--sbatch=--time=01:00:00",
            ]
        )
    )
    content = script.read_text()
    assert "#SBATCH --time=01:00:00" in content
    assert "--enable-sinks=ENCODE_STITCHED_FILE" in content
    assert "--start-time=00:00:12" in content
    assert "--options=stitching.blend_mode=alpha" in content
    assert "telemetry-db-dir=" + str(game) in content
    assert "--render-realtime" not in content
    result = subprocess.run(
        [str(script)], env={"HM_CLI_BIN": "/bin/echo"}, capture_output=True, text=True, check=True
    )
    assert "--game-id match" in result.stdout
    with pytest.raises(FileExistsError):
        export_job(
            parser().parse_args(["--export-job", "--game-dir", str(game), "--output", str(script)])
        )


def should_launch_one_shared_desktop_with_private_initial_envelope(game, monkeypatch):
    from hmlib.ui import desktop

    monkeypatch.setattr(
        desktop, "executable", lambda explicit, name, candidates=(): Path("/bin") / name
    )
    calls = []

    def call(command, env):
        envelope = Path(
            next(
                value.split("=", 1)[1] for value in command if value.startswith("--backend-launch=")
            )
        )
        calls.append((command, env, json.loads(envelope.read_text()), envelope))
        return 7

    monkeypatch.setattr(desktop.subprocess, "call", call)
    assert (
        launch_desktop(
            game_id=game.name,
            module="hmlib.cli.hmtrack",
            arguments=["--camera-ui=1"],
            root_dir=game.parent,
        )
        == 7
    )
    command, env, envelope, path = calls[0]
    assert command[0] == "/bin/hstream-ui"
    assert "--backend=hockeymon" in command and "--backend-runner=/bin/hm-cli" in command
    assert env["HM_GAME_DIR"] == str(game.parent)
    assert envelope["arguments"] == ["--camera-ui=1"]
    assert not path.exists()


def should_validate_anonymous_source_count_after_comma_expansion(tmp_path):
    sources = [tmp_path / f"{index}.mp4" for index in range(3)]
    for source in sources:
        source.touch()
    with pytest.raises(ValueError, match="left/right pair"):
        _anonymous_game(tmp_path, ["--input-video=" + ",".join(map(str, sources))])
    game_id = _anonymous_game(tmp_path, ["--input-video=" + ",".join(map(str, sources[:2]))])
    config = yaml.safe_load((tmp_path / "games" / game_id / "config.yaml").read_text())
    assert set(config["game"]["videos"]) == {"left", "right"}


def should_translate_initial_calibration_without_rewriting_saved_game(game, tmp_path):
    from hmlib.ui.hstream_initial import calibration_arguments

    before = (game / "config.yaml").read_bytes()
    envelope = tmp_path / "initial.json"
    envelope.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.stitch",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": [
                    "--game-id",
                    game.name,
                    "--camera-ui=1",
                    "--blend-mode=alpha",
                    "--stitch-frame-time=00:00:03",
                    "--lfo=1",
                    "--rfo=0",
                    "--max-time=5",
                ],
            }
        )
    )
    command, cwd = calibration_arguments(
        ["-g", game.name, "-c", "native.yaml", "--stitching-calibration-only"],
        envelope,
        Path("/native/hstream-cli"),
    )
    assert cwd == tmp_path
    assert command.count("-c") == 1
    assert "--time-limit=5" in command
    overlay = yaml.safe_load((tmp_path / "native-initial.yaml").read_text())
    assert overlay["stitching"]["blend_mode"] == "alpha"
    assert overlay["stitching"]["stitch_frame_time"] == "00:00:03"
    assert overlay["game"]["stitching"]["frame_offsets"] == {"left": 1, "right": 0}
    assert (game / "config.yaml").read_bytes() == before
