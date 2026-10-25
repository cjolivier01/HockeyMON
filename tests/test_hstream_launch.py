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


@pytest.mark.parametrize("disabled", [False, True])
def should_honor_initial_output_file_before_generating_recording_flags(game, tmp_path, disabled):
    from hmlib.cli.hmtrack import make_parser

    initial = tmp_path / "initial-output.json"
    initial.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": ["--output-video=explicit.mp4"]
                + (["--no-save-video"] if disabled else []),
            }
        )
    )
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=initial)
    worker = make_parser().parse_args(plan.write(tmp_path)[4:])
    assert worker.no_save_video is disabled
    assert worker.output_video == "explicit.mp4"
    assert plan.config["video_out"]["archive_program"] is not disabled
    assert ("ENCODE_FILE" in plan.sinks) is not disabled
    assert plan.launch_receipt()["sinks"] == sorted(plan.sinks)


def should_honor_initial_archive_config_and_freeze_resolved_routes(game, tmp_path):
    overlay = tmp_path / "outputs.yaml"
    overlay.write_text("video_out: {archive_program: true, archive_stitched: true}\n")
    initial = tmp_path / "initial-output.json"
    initial.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": ["--config", overlay.name],
            }
        )
    )
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=initial)
    assert plan.sinks == {"RENDER", "ENCODE_FILE", "ENCODE_STITCHED_FILE"}
    assert "--no-save-video" not in plan.arguments


@pytest.mark.parametrize("sinks", ["RENDER", "ENCODE_FILE", "RTMP"])
def should_not_start_saved_spare_stream_from_shared_output_selection(game, sinks):
    config = yaml.safe_load((game / "config.yaml").read_text())
    config["video_out"] = {"spare_rtmp_enabled": True, "spare_rtmp_url": "rtmp://saved/spare"}
    (game / "config.yaml").write_text(yaml.safe_dump(config))
    plan = make_plan(parser().parse_args(["-g", game.name, "--enable-sinks=" + sinks]))
    assert plan.config["video_out"]["spare_rtmp_enabled"] is False
    assert plan.config["video_out"]["show_youtube"] is (sinks == "RTMP")
    assert plan.sinks == {sinks}


def should_include_explicit_initial_spare_stream_in_network_admission(game, tmp_path):
    overlay = tmp_path / "spare.yaml"
    overlay.write_text(
        "video_out: {show_youtube: false, spare_rtmp_enabled: true, spare_rtmp_url: 'rtmp://explicit/spare'}\n"
    )
    initial = tmp_path / "initial-spare.json"
    initial.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": ["--config", overlay.name],
            }
        )
    )
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=initial)
    assert plan.config["video_out"]["spare_rtmp_enabled"] is True
    assert plan.config["video_out"]["show_youtube"] is False
    assert plan.sinks == {"RENDER", "RTMP"}
    assert "RTMP" in plan.launch_receipt()["sinks"]


def should_export_saved_recipe_and_execute_the_selected_backend(game, tmp_path, monkeypatch):
    output_root = tmp_path / "working output"
    monkeypatch.setenv("HM_OUTPUT_WORK_DIR", str(output_root))
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
    for directive in ("--nodes=1", "--ntasks=1", "--gres=gpu:1"):
        assert "#SBATCH " + directive in content
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
    environment_runner = tmp_path / "environment.sh"
    environment_runner.write_text("#!/bin/sh\nprintf '%s\\n' \"$HM_OUTPUT_WORK_DIR\"\nenv\n")
    environment_runner.chmod(0o755)
    result = subprocess.run(
        [str(script)],
        env={
            "HM_CLI_BIN": str(environment_runner),
            "HSTREAM_UI_PARENT_PID": "1",
            "HSTREAM_UI_BACKEND_SESSION": "/stale/session",
            "HM_UI_INITIAL_LAUNCH": "/stale/launch",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines()[0] == str(output_root)
    assert "HSTREAM_UI_PARENT_PID=" not in result.stdout
    assert "HSTREAM_UI_BACKEND_SESSION=" not in result.stdout
    assert "HM_UI_INITIAL_LAUNCH=" not in result.stdout
    with pytest.raises(FileExistsError):
        export_job(
            parser().parse_args(["--export-job", "--game-dir", str(game), "--output", str(script)])
        )


def should_launch_one_shared_desktop_with_private_initial_envelope(game, monkeypatch):
    from hmlib.ui import desktop

    monkeypatch.setattr(
        desktop, "executable", lambda explicit, name, candidates=(): Path("/bin") / name
    )
    monkeypatch.setattr(
        desktop,
        "_resolve_initial_sources",
        lambda runner, request, launch, environment: {
            "version": 1,
            "game_id": launch["game_id"],
            "sources": {"source": [str(game / "one.mp4")]},
            "arguments": launch["arguments"],
            "cwd": launch["cwd"],
        },
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
    assert envelope["source_ownership"]["sources"] == {"source": [str(game / "one.mp4")]}
    assert not path.exists()


def should_validate_anonymous_source_count_after_comma_expansion(tmp_path):
    sources = [tmp_path / f"{index}.mp4" for index in range(3)]
    for source in sources:
        source.touch()
    with pytest.raises(ValueError, match="left/right pair"):
        _anonymous_game(tmp_path, ["--input-video=" + ",".join(map(str, sources))])
    game_id = _anonymous_game(tmp_path, ["--input-video=" + ",".join(map(str, sources[:2]))])
    config = yaml.safe_load((tmp_path / game_id / "config.yaml").read_text())
    assert set(config["game"]["videos"]) == {"left", "right"}


@pytest.mark.parametrize("override", ["input", "config", "config-override"])
def should_resolve_initial_source_ownership_before_touching_saved_pair(
    game, tmp_path, monkeypatch, capsys, override
):
    from hmlib.cli.hstream_backend import main

    source = game / "prestitched.mp4"
    source.write_bytes(b"single recording")
    saved = {
        "game": {"videos": {"left": ["one.mp4"], "right": ["two.mp4"], "source": None}},
        "hstream_ui": {
            "stitching_calibration": {"status": "complete", "invalidation_id": "must-retain"},
            "hockeymon": {"detector": {"precision": "int8", "engine": "/missing/engine.onnx"}},
        },
    }
    config_path = game / "config.yaml"
    config_path.write_text(yaml.safe_dump(saved))
    before = config_path.read_bytes()
    videos = {"source": [str(source)], "left": None, "right": None}
    if override == "input":
        arguments = ["--input-video", str(source)]
    elif override == "config":
        config = tmp_path / "initial-sources.yaml"
        config.write_text(yaml.safe_dump({"game": {"videos": videos}}))
        arguments = ["--config", str(config)]
    else:
        arguments = [
            "--input-video",
            str(source),
            "--config-override",
            "rink.camera.color.exposure_ev=1",
        ]
    initial = tmp_path / "initial-sources.json"
    initial.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": arguments,
            }
        )
    )
    monkeypatch.setenv("HM_UI_INITIAL_LAUNCH", str(initial))
    assert main(["--resolve-initial-sources", "--game-id", game.name]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "version": 1,
        "game_id": game.name,
        "sources": {"source": [str(source)]},
    }
    assert config_path.read_bytes() == before


def should_bind_launcher_source_resolution_to_exact_initial_request(tmp_path, monkeypatch):
    from hmlib.ui import desktop

    source = tmp_path / "recording.mp4"
    source.touch()
    launch = {"game_id": "game", "arguments": ["--input-video", str(source)], "cwd": str(tmp_path)}
    request = tmp_path / "launch.json"
    request.write_text(json.dumps(launch))

    def run(command, **kwargs):
        assert command == ["/matching/hm-cli", "--resolve-initial-sources", "--game-id", "game"]
        assert kwargs["env"]["HM_GAME_DIR"] == str(tmp_path)
        assert kwargs["env"]["HM_UI_INITIAL_LAUNCH"] == str(request)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "version": 1,
                    "game_id": "game",
                    "sources": {"source": [str(source)]},
                }
            ),
            "",
        )

    monkeypatch.setattr(desktop.subprocess, "run", run)
    ownership = desktop._resolve_initial_sources(
        Path("/matching/hm-cli"), request, launch, {"HM_GAME_DIR": str(tmp_path)}
    )
    assert ownership["sources"] == {"source": [str(source)]}
    assert ownership["arguments"] == launch["arguments"]
    assert ownership["cwd"] == launch["cwd"]


@pytest.mark.parametrize("source_layer", ["saved", "overlay", "input", "override"])
def should_freeze_admitted_sources_across_first_launch_config_edits(game, tmp_path, source_layer):
    source = game / "admitted.mp4"
    source.touch()
    overlay = tmp_path / "initial-overlay.yaml"
    overlay.write_text(yaml.safe_dump({"game": {"videos": {"source": [str(source)]}}}))
    arguments = ["--game-id", game.name, "--config-override=rink.camera.color.exposure_ev=1"]
    if source_layer == "overlay":
        arguments += ["--config", str(overlay)]
    elif source_layer == "input":
        arguments += ["--input-video=" + str(source)]
    elif source_layer == "override":
        arguments += ["--config-override", "game.videos.source=" + str(source)]
    else:
        (game / "config.yaml").write_text(overlay.read_text())
    initial = tmp_path / "frozen-launch.json"
    request = {
        "version": 1,
        "module": "hmlib.cli.hmtrack",
        "game_id": game.name,
        "cwd": str(tmp_path),
        "arguments": arguments,
    }
    initial.write_text(json.dumps(request))
    preflight = make_plan(
        parser().parse_args(["-g", game.name]), initial=initial, validate_models=False
    )
    request["source_ownership"] = {
        "version": 1,
        "game_id": game.name,
        "cwd": request["cwd"],
        "arguments": arguments,
        "sources": {
            role: [str(path) for path in paths] for role, paths in preflight.sources.items()
        },
    }
    initial.write_text(json.dumps(request))
    replacement = {
        "game": {
            "videos": {"left": ["missing-left.mp4"], "right": ["missing-right.mp4"], "source": None}
        }
    }
    (game / "config.yaml").write_text(yaml.safe_dump(replacement))
    overlay.write_text(yaml.safe_dump(replacement))
    before = (game / "config.yaml").read_bytes()
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=initial)
    assert plan.sources == {"source": [source]}
    assert plan.config["game"]["videos"] == {"source": [str(source)]}
    assert plan.arguments[plan.arguments.index("--input-video") + 1] == str(source)
    assert not any(value.startswith("--input-video") for value in plan.explicit_arguments)
    assert not any("game.videos" in value for value in plan.explicit_arguments)
    assert "--config-override=rink.camera.color.exposure_ev=1" in plan.explicit_arguments
    command = plan.write(tmp_path)
    source_config = tmp_path / "initial-sources.yaml"
    assert command[-3:-1] == ["--config", str(source_config)]
    assert yaml.safe_load(source_config.read_text())["game"]["videos"] == {
        "source": [str(source)],
        "left": None,
        "right": None,
    }
    if source_layer == "overlay":
        assert str(overlay) in command and command.index(str(overlay)) < command.index(
            str(source_config)
        )
    assert (game / "config.yaml").read_bytes() == before


@pytest.mark.parametrize("explicit_root", [False, True])
def should_preserve_anonymous_games_and_outputs_after_desktop_closes(
    tmp_path, monkeypatch, explicit_root
):
    from hmlib.ui import desktop

    root = tmp_path / "recordings"
    monkeypatch.setenv("HM_GAME_DIR", str(root if not explicit_root else tmp_path / "unused"))
    monkeypatch.setattr(
        desktop, "executable", lambda explicit, name, candidates=(): Path("/bin") / name
    )
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        desktop,
        "_resolve_initial_sources",
        lambda runner, request, launch, environment: {
            "version": 1,
            "game_id": launch["game_id"],
            "sources": {"source": [str(source)]},
            "arguments": launch["arguments"],
            "cwd": launch["cwd"],
        },
    )
    launches = []

    def run(command, env):
        request = Path(
            next(item.split("=", 1)[1] for item in command if item.startswith("--backend-launch="))
        )
        envelope = json.loads(request.read_text())
        game = Path(env["HM_GAME_DIR"]) / envelope["game_id"]
        assert game.parent == root
        assert yaml.safe_load((game / "config.yaml").read_text())["game"]["videos"]["source"] == [
            str(source)
        ]
        for name in ("program-1.mkv", "recording_telemetry-1.db", "calibration.yaml"):
            (game / name).write_bytes(name.encode())
        launches.append((game, request))
        return 0

    monkeypatch.setattr(desktop.subprocess, "call", run)
    for _ in range(2):
        assert (
            launch_desktop(
                module="hmlib.cli.hmtrack",
                arguments=[f"--input-video={source}"],
                root_dir=root if explicit_root else None,
            )
            == 0
        )
    assert launches[0][0] != launches[1][0]
    for game, request in launches:
        assert not request.exists()
        assert {path.name for path in game.iterdir()} == {
            "config.yaml",
            "program-1.mkv",
            "recording_telemetry-1.db",
            "calibration.yaml",
        }


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
                    "--no-audio",
                ],
            }
        )
    )
    command, cwd = calibration_arguments(
        [
            "-g",
            game.name,
            "-c",
            "native.yaml",
            "--stitching-calibration-only",
            "--options=stitching.blend_mode=laplacian,stitching.sync_method=imu",
            "--options",
            "pipeline.hmaudio.enable=1",
            "--options=pipeline.hmstitcher.properties.high-bit-depth=auto",
            "--stitch-frame-time=00:00:09",
        ],
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
    assert overlay["stitching"]["sync_method"] == "manual"
    assert overlay["pipeline"]["hmaudio"]["enable"] is False
    assert not any(
        "stitching.blend_mode=" in arg or "stitching.sync_method=" in arg for arg in command
    )
    assert not any("pipeline.hmaudio.enable=" in arg for arg in command)
    assert "--options=pipeline.hmstitcher.properties.high-bit-depth=auto" in command
    assert "--stitch-frame-time=00:00:03" in command
    assert "--stitch-frame-time=00:00:09" not in command
    assert (game / "config.yaml").read_bytes() == before


def should_bind_shared_model_menu_arguments_to_exact_python_stages(game):
    from hmlib.aspen.plugins.reid_plugin import configure_reid
    from hmlib.ui.analytics_graph import NATIVE_POSE_CONFIG

    arguments = ["-g", game.name, "--enable-sinks=FAKE"]
    for feature, model in (
        ("pose", "rtmpose-m-coco17-256x192"),
        ("jersey", "parseq-hockey-cvprw2024"),
        ("action", "stgcnpp-coco2d-joint-ntu60"),
        ("team", "reidentificationnet-deployable-v1.2"),
    ):
        arguments += [
            f"--options=pipeline.player-analytics.{feature}.enable=true",
            f"--options=pipeline.player-analytics.{feature}.model={model}",
        ]
    arguments += [
        "--options=pipeline.tracker.reid-enable=true",
        "--options=pipeline.tracker.reid-model=deepstream",
    ]
    plan = make_plan(parser().parse_args(arguments))
    nodes = plan.config["aspen"]["plugins"]
    assert nodes["pose_factory"]["params"]["pose_config"] == NATIVE_POSE_CONFIG
    assert nodes["pose_factory"]["params"]["shared_model"]["model"] == "rtmpose-m-coco17-256x192"
    assert nodes["jersey_numbers"]["class"].endswith("ParseqJerseyPlugin")
    assert nodes["actions"]["class"].endswith("TemporalActionPlugin")
    assert plan.config["team"]["backend"] == "reidentificationnet"
    assert set(plan.config["hm_ui"]["shared_analytics"]) == {
        "pose",
        "jersey",
        "action",
        "team",
        "reid",
    }
    configure_reid(plan.config, game, validate_weights=False)
    assert nodes["player_reid"]["params"]["tracker_config"].endswith("tracker.yaml")


def should_preserve_shared_custom_bundles_and_rebind_them_after_seek(game, tmp_path):
    from hmlib.ui.shared_analytics import apply_prepared_shared_analytics

    bundle = tmp_path / "custom-pose"
    bundle.mkdir()
    plan = make_plan(
        parser().parse_args(
            [
                "-g",
                game.name,
                "--options=pipeline.player-analytics.pose.enable=true",
                "--options=pipeline.player-analytics.pose.model=custom",
                f"--options=pipeline.player-analytics.pose.bundle={bundle}",
            ]
        )
    )
    prepared = {
        "pipeline": {
            "player-analytics": {"gpu-id": 1, "pose": {"model": "custom", "bundle": str(bundle)}}
        }
    }
    apply_prepared_shared_analytics(plan.config, prepared)
    assert plan.config["aspen"]["plugins"]["pose_factory"]["params"]["shared_model"] == {
        "model": "custom",
        "bundle": str(bundle),
        "gpu": 1,
    }
    prepared["pipeline"]["player-analytics"]["pose"]["bundle"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="changed during the run"):
        apply_prepared_shared_analytics(plan.config, prepared)


def should_replace_legacy_analysis_preset_with_shared_independent_toggles(game):
    config = yaml.safe_load((game / "config.yaml").read_text())
    config["hm_ui"] = {"run": {"analytics": "tracking", "pose_model": "harpet"}}
    (game / "config.yaml").write_text(yaml.safe_dump(config))
    plan = make_plan(
        parser().parse_args(
            [
                "-g",
                game.name,
                "--options=pipeline.player-analytics.pose.enable=true",
                "--options=pipeline.player-analytics.pose.model=rtmpose-m-coco17-256x192",
            ]
        )
    )
    assert plan.config["hm_ui"]["run"]["analytics"] == "custom"
    assert plan.config["hm_ui"]["run"]["pose_model"] == "shared"
    assert "pose_factory" in plan.config["aspen"]["plugins"]


@pytest.mark.parametrize("team", [{"backend": "prtreid"}, {"enabled": False}])
def should_keep_initial_python_models_over_shared_menu_defaults(game, tmp_path, team):
    from hmlib.aspen.plugins.reid_plugin import configure_reid

    appearance = game / "operator-appearance.onnx"
    appearance.touch()
    initial_config = tmp_path / "initial-models.yaml"
    initial_config.write_text(
        yaml.safe_dump({"hm_ui": {"run": {"reid_onnx": str(appearance)}}, "team": team})
    )
    initial = tmp_path / "initial-models.json"
    initial.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": ["--game-id", game.name, "--config", str(initial_config)],
            }
        )
    )
    plan = make_plan(
        parser().parse_args(
            [
                "-g",
                game.name,
                "--options=pipeline.tracker.reid-enable=true",
                "--options=pipeline.tracker.reid-model=deepstream",
                "--options=pipeline.player-analytics.team.enable=true",
                "--options=pipeline.player-analytics.team.model=reidentificationnet-deployable-v1.2",
            ]
        ),
        initial=initial,
    )
    assert "reid" not in plan.config["hm_ui"]["shared_analytics"]
    assert "team" not in plan.config["hm_ui"]["shared_analytics"]
    assert all(plan.config["team"][key] == value for key, value in team.items())
    configure_reid(plan.config, game)
    assert plan.config["aspen"]["plugins"]["player_reid"]["params"] == {
        "onnx_path": str(appearance)
    }


def should_preserve_original_config_lists_and_package_relative_paths(game, tmp_path):
    from hmlib.config import load_yaml_files_ordered

    (tmp_path / "first.yaml").write_text("aspen:\n  pipeline:\n    queue_size: 3\n")
    (tmp_path / "last.yaml").write_text("aspen:\n  pipeline:\n    queue_size: 7\n")
    envelope = tmp_path / "initial-configs.json"
    request = {
        "version": 1,
        "module": "hmlib.cli.hmtrack",
        "game_id": game.name,
        "cwd": str(tmp_path),
        "arguments": ["--config", " first.yaml, last.yaml "],
    }
    envelope.write_text(json.dumps(request))
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=envelope)
    assert plan.initial_settings["aspen"]["pipeline"]["queue_size"] == 7
    assert plan.config["aspen"]["pipeline"]["queue_size"] == 7
    bundled = "config/aspen/tracking_pose.yaml"
    request["arguments"] = ["--config", bundled]
    # A colliding cwd filename must not override the bundled ROOT_DIR input.
    collision = tmp_path / bundled
    collision.parent.mkdir(parents=True)
    collision.write_text("wrong_origin: true\n")
    envelope.write_text(json.dumps(request))
    plan = make_plan(parser().parse_args(["-g", game.name]), initial=envelope)
    assert plan.initial_settings["aspen"] == load_yaml_files_ordered([bundled])["aspen"]
    assert "wrong_origin" not in plan.config


@pytest.mark.parametrize("backend", ["--detector-trt-enable", "--detector-onnx=selected.onnx"])
def should_ignore_unprepared_shared_int8_for_explicit_initial_backend(game, tmp_path, backend):
    envelope = tmp_path / "initial-backend.json"
    envelope.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": [backend, "--camera-ui=1"],
            }
        )
    )
    plan = make_plan(
        parser().parse_args(
            ["-g", game.name, "--options=hstream_ui.hockeymon.detector.precision=int8"]
        ),
        initial=envelope,
    )
    assert backend in plan.explicit_arguments
    assert not any("int8" in value or ".hm-ui/models" in value for value in plan.arguments)


@pytest.mark.parametrize("explicit", [False, True])
def should_resolve_packaged_native_wrapper_to_its_binary(tmp_path, monkeypatch, explicit):
    from hmlib.ui.hstream_launch import native_application_config
    from hmlib.ui.native_tools import native_runner

    prefix = tmp_path / "opt/hstream"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "configs").mkdir()
    binary = prefix / "bin/hstream-cli"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    wrapper = prefix / "run.sh"
    wrapper.write_text("#!/bin/sh\nexit 99\n")
    wrapper.chmod(0o755)
    public = tmp_path / "usr/bin"
    public.mkdir(parents=True)
    (public / "hstream-cli").symlink_to(wrapper)
    config = prefix / "configs/ds_hockey_app_config.yaml"
    config.write_text("tracker:\n  gpu-id: 0\n")
    monkeypatch.setenv("PATH", str(public))
    monkeypatch.delenv("HM_CONFIG_ROOT", raising=False)
    resolved = native_runner(str(public / "hstream-cli") if explicit else None)
    assert resolved == binary
    assert native_application_config(resolved) == config


@pytest.mark.parametrize(
    "backend",
    [
        "--pose-onnx=/operator/pose.onnx",
        "--pose-onnx-enable",
        "--pose-onnx-quantize-int8",
        "--pose-trt-enable",
        "--pose-trt-engine=/operator/pose.engine",
    ],
)
def should_keep_initial_pose_backend_out_of_shared_model_preparation(game, tmp_path, backend):
    from hmlib.cli.hmtrack import make_parser
    from hmlib.ui.runtime_setup import apply_engine_preferences

    config = yaml.safe_load((game / "config.yaml").read_text())
    config["hm_ui"] = {"run": {"analytics": "tracking"}}
    (game / "config.yaml").write_text(yaml.safe_dump(config))
    envelope = tmp_path / "initial-pose-backend.json"
    envelope.write_text(
        json.dumps(
            {
                "version": 1,
                "module": "hmlib.cli.hmtrack",
                "game_id": game.name,
                "cwd": str(tmp_path),
                "arguments": [backend],
            }
        )
    )
    plan = make_plan(
        parser().parse_args(
            [
                "-g",
                game.name,
                "--options=pipeline.player-analytics.pose.enable=true",
                "--options=pipeline.player-analytics.jersey.enable=true",
            ]
        ),
        initial=envelope,
    )
    assert "pose" not in plan.config["hm_ui"]["shared_analytics"]
    assert "jersey" in plan.config["hm_ui"]["shared_analytics"]
    assert plan.config["hm_ui"]["run"]["analytics"] == "custom"
    assert backend in plan.explicit_arguments
    command = plan.write(tmp_path)
    parsed = make_parser().parse_args(command[4:])
    option = backend.split("=", 1)[0]
    destination = make_parser()._option_string_actions[option].dest
    assert vars(parsed)[destination] == (backend.split("=", 1)[1] if "=" in backend else True)
    # Rebuilding engine preferences in the worker cannot reattach the shared
    # override that used to replace explicit ONNX/TensorRT settings at runtime.
    apply_engine_preferences(plan.config)
    assert "shared_model" not in plan.config["aspen"]["plugins"]["pose_factory"]["params"]
