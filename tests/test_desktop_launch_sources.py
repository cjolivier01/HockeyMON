"""Regressions for initial CLI launches and source-dependent saved preferences."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hmlib.ui.runtime_setup import (
    OperatorWorkspace,
    apply_pipeline_preferences,
    inspect_pipeline,
    launch_arguments,
    saved_pipeline_preferences,
)
from hmlib.ui.workspace import Workspace
from hmlib.ui.workspace.files import read_config, revision, write_config


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    import hmlib.config as configuration

    with Workspace(tmp_path / "games", tmp_path / "session") as instance:
        instance.create_game("launch-test")
        monkeypatch.setitem(
            configuration.get_game_config_private.__globals__,
            "GAME_DIR_BASE",
            str(instance.root_dir),
        )
        yield instance


def _initial_launch(monkeypatch, tmp_path, *, game_id, arguments, module="hmlib.cli.hmtrack"):
    from hmlib.ui import desktop

    calls = []

    def call(command, *, env):
        request = Path(
            next(
                value.split("=", 1)[1] for value in command if value.startswith("--backend-launch=")
            )
        )
        envelope = json.loads(request.read_text())
        captured = {
            "command": command,
            "environment": env,
            "envelope": envelope,
            "request": request,
        }
        if envelope.get("anonymous"):
            captured["sources"] = read_config(Path(env["HM_GAME_DIR"]) / envelope["game_id"])[
                "game"
            ]["videos"]
        calls.append(captured)
        return 0

    monkeypatch.setattr(
        desktop, "executable", lambda explicit, name, candidates=(): Path("/test/bin") / name
    )
    monkeypatch.setattr(desktop.subprocess, "call", call)
    assert (
        desktop.launch_desktop(
            game_id=game_id,
            module=module,
            arguments=arguments,
            root_dir=tmp_path / "games",
        )
        == 0
    )
    assert len(calls) == 1
    assert not calls[0]["request"].exists()
    return calls[0]


def should_launch_initial_configured_single_source_with_cli_overrides(
    workspace, tmp_path, monkeypatch
):
    game = workspace.game()
    source = game / "single.mp4"
    source.write_bytes(b"fixture")
    write_config(
        game,
        {
            "game": {"videos": {"source": [source.name]}},
            "hm_ui": {"run": {"output_fps": 15, "batch_size": 2}},
        },
    )
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    monkeypatch.chdir(invocation)
    arguments = [
        "--game-id=launch-test",
        "--camera-ui=1",
        "--output-fps",
        "30",
        "--batch-size",
        "1",
        "--deploy-dir",
        "relative-output",
    ]
    launch = _initial_launch(monkeypatch, tmp_path, game_id=game.name, arguments=arguments)
    assert launch["envelope"] == {
        "version": 1,
        "module": "hmlib.cli.hmtrack",
        "arguments": arguments,
        "cwd": str(invocation),
        "game_id": game.name,
    }
    assert launch["environment"]["HM_GAME_DIR"] == str(game.parent)
    assert launch["command"][:3] == [
        "/test/bin/hstream-ui",
        "--backend=hockeymon",
        "--backend-runner=/test/bin/hm-cli",
    ]
    assert "--play" in launch["command"] and "--game-id=launch-test" in launch["command"]


def should_keep_anonymous_game_after_the_desktop_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "relative.mp4").write_bytes(b"fixture")
    arguments = ["--input-video", "relative.mp4", "--camera-ui=1", "-t=2"]
    launch = _initial_launch(monkeypatch, tmp_path, game_id=None, arguments=arguments)
    assert launch["envelope"]["arguments"] == arguments
    assert launch["envelope"]["anonymous"] is True
    assert launch["envelope"]["cwd"] == str(tmp_path)
    assert launch["sources"] == {"source": [str(tmp_path / "relative.mp4")]}
    game_root = Path(launch["environment"]["HM_GAME_DIR"])
    assert game_root == tmp_path / "games"
    game = game_root / launch["envelope"]["game_id"]
    assert read_config(game)["game"]["videos"] == launch["sources"]


def should_preserve_explicit_initial_input_and_publication_choices(
    workspace, tmp_path, monkeypatch
):
    write_config(workspace.game(), {"game": {"videos": {"source": ["missing.mp4"]}}})
    arguments = ["--input-video=relative.mp4", "--camera-ui=1"]
    launch = _initial_launch(
        monkeypatch, tmp_path, game_id=workspace.game().name, arguments=arguments
    )
    assert launch["envelope"]["arguments"] == arguments
    assert "--deploy-dir" not in launch["command"]


def should_forward_original_stitch_invocation_to_the_native_calibration_trampoline(
    workspace, tmp_path, monkeypatch
):
    arguments = [
        "--game-id=launch-test",
        "--camera-ui=1",
        "--stitch-frame-time=00:00:03",
        "--config",
        "lens.yaml",
    ]
    launch = _initial_launch(
        monkeypatch,
        tmp_path,
        game_id=workspace.game().name,
        arguments=arguments,
        module="hmlib.cli.stitch",
    )
    assert "--stitching-calibration-only" in launch["command"]
    assert launch["envelope"]["module"] == "hmlib.cli.stitch"
    assert launch["envelope"]["arguments"] == arguments


@pytest.mark.parametrize("target", ["detector", "pose"])
@pytest.mark.parametrize("saved", ["onnx-int8", "trt-fp16"])
@pytest.mark.parametrize("backend", ["onnx", "trt"])
def should_replace_saved_engine_before_resolving_its_assets(workspace, target, saved, backend):
    from hmlib.cli.hmtrack import make_parser

    setting = "precision" if target == "detector" else "pose_precision"
    write_config(workspace.game(), {"hm_ui": {"run": {setting: saved}}})
    # No prepared INT8 assets exist. Explicit engines must not require them or
    # accidentally activate the saved competing backend.
    explicit = (
        [f"--{target}-onnx=/custom.onnx"] if backend == "onnx" else [f"--{target}-trt-enable"]
    )
    command = workspace.pipeline_command(workspace.game(), "track", explicit)
    parsed = vars(make_parser().parse_args(command[3:]))
    assert bool(parsed[f"{target}_trt_enable"]) is (backend == "trt")
    assert bool(parsed[f"{target}_onnx_path"]) is (backend == "onnx")
    if backend == "onnx":
        assert parsed[f"{target}_onnx_path"] == "/custom.onnx"
        assert f"--{target}-trt-fp16" not in command


@pytest.mark.parametrize("initial_launch", [True, False])
def should_override_saved_boolean_switches_without_argparse_conflicts(workspace, initial_launch):
    from hmlib.cli.hmtrack import make_parser

    write_config(
        workspace.game(),
        {"hm_ui": {"run": {"save_video": False, "precision": "trt-fp32", "output_fps": 15}}},
    )
    command = workspace.pipeline_command(
        workspace.game(),
        "track",
        ["--no_save_video", "--detector-trt-fp16", "--output-fps=30"],
        initial_launch=initial_launch,
    )
    parsed = make_parser().parse_args(command[3:])
    assert parsed.no_save_video is True
    assert parsed.detector_trt_fp16 is True
    assert parsed.output_fps == 30
    assert "--no-save-video" not in command
    assert "--no-detector-trt-fp16" not in command


def should_override_saved_stitch_boolean_switches(workspace):
    from hmlib.cli.stitch import make_parser
    from hmlib.hm_opts import hm_opts

    write_config(workspace.game(), {"hm_ui": {"run": {"save_video": False}}})
    command = workspace.pipeline_command(workspace.game(), "stitch", ["--no_save_video"])
    parsed = hm_opts.parser(parser=make_parser()).parse_args(command[3:])
    assert parsed.no_save_video is True


@pytest.mark.parametrize("explicit_dtype", [None, "float32"])
def should_preserve_distinct_generated_configuration_overrides(workspace, explicit_dtype):
    from hmlib.cli.hmtrack import make_parser
    from hmlib.hm_opts import hm_opts

    write_config(workspace.game(), {"hm_ui": {"run": {"stitch_precision": "float16"}}})
    arguments = ["--config-override", "rink.camera.color.gamma=1.2"]
    if explicit_dtype is not None:
        arguments += ["--config-override", f"stitching.dtype={explicit_dtype}"]
    command = workspace.pipeline_command(workspace.game(), "track", arguments)
    parsed = make_parser().parse_args(command[3:])
    config = OperatorWorkspace(workspace.game())._effective()
    hm_opts.apply_config_overrides(config, parsed.config_overrides)
    assert config["rink"]["camera"]["color"]["gamma"] == 1.2
    assert config["stitching"]["dtype"] == (explicit_dtype or "float16")


def _wait(workspace, job):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        workspace.poll()
        state = workspace.jobs.get(job["id"])
        if state["status"] not in {"running", "paused", "stopping"}:
            assert state["status"] == "completed", state
            return state
        time.sleep(0.02)
    raise AssertionError("Anonymous child did not exit")


def should_supervise_restart_and_export_anonymous_input_jobs(workspace, tmp_path):
    first = workspace.start_command(
        [sys.executable, "-c", "import os; print(os.getcwd()); print(os.environ['HM_GAME_DIR'])"],
        None,
    )
    assert first["game"] is None
    _wait(workspace, first)
    assert workspace.current_game is None
    second = workspace.restart_job({"job_id": first["id"]})
    _wait(workspace, second)
    assert second["game"] is None
    assert second["cwd"] == str(Path.cwd())
    text = workspace.jobs.log(second["id"])["text"]
    assert str(workspace.root_dir) in text
    script = tmp_path / "export.sh"
    workspace.export_job({"job_id": second["id"], "path": str(script)})
    assert f"export HM_GAME_DIR={workspace.root_dir}" in script.read_text()
    assert (
        json.loads((workspace.jobs.directory / first["id"] / "job.json").read_text())["game"]
        is None
    )


@pytest.mark.parametrize(
    "override", [[], ["--aspen-threaded"], ["--config-override", "aspen.pipeline.threaded=true"]]
)
def should_apply_saved_pipeline_choices_after_graph_before_explicit_cli(workspace, override):
    from hmlib.config import load_yaml_files_ordered, recursive_update
    from hmlib.hm_opts import hm_opts
    from hmlib.ui.analytics_graph import graph_paths

    editor = OperatorWorkspace(workspace.game())
    editor.update_settings(
        {
            "aspen.pipeline.threaded": False,
            "aspen.pipeline.queue_size": 7,
            "aspen.pipeline.max_concurrent": 5,
        }
    )
    saved = saved_pipeline_preferences(workspace.game().name)
    assert saved == {"threaded": False, "queue_size": 7, "max_concurrent": 5}
    config = recursive_update(editor._effective(), load_yaml_files_ordered(graph_paths("tracking")))
    assert config["aspen"]["pipeline"]["threaded"] is True
    apply_pipeline_preferences(config, saved)
    parser = hm_opts.parser(argparse.ArgumentParser())
    arguments = [*launch_arguments(workspace.game(), "track"), *override]
    args = parser.parse_args(arguments)
    hm_opts.apply_arg_config_overrides(
        config,
        args,
        parser=parser,
        explicit_arg_names=hm_opts.collect_explicit_arg_names(parser, argv=arguments),
    )
    hm_opts.apply_config_overrides(config, args.config_overrides)
    assert config["aspen"]["pipeline"]["threaded"] is bool(override)
    assert config["aspen"]["pipeline"]["queue_size"] == 7
    assert config["aspen"]["pipeline"]["max_concurrent"] == 5
    inspected = inspect_pipeline(workspace.game())
    assert inspected["mode"] == "sequential"
    assert inspected["queue_size"] == 7
    assert inspected["max_concurrent"] == 5


def should_preserve_unsaved_graph_choices_and_honor_private_config_opt_out(workspace, tmp_path):
    write_config(workspace.game(), {"aspen": {"pipeline": {"threaded": False}}})
    custom_root = tmp_path / "configuration"
    (custom_root / "config" / "games").mkdir(parents=True)
    (custom_root / "config" / "games" / "launch-test.yaml").write_text(
        "aspen:\n  pipeline:\n    queue_size: 9\n"
    )
    assert saved_pipeline_preferences("launch-test", root_dir=custom_root) == {
        "threaded": False,
        "queue_size": 9,
    }
    assert saved_pipeline_preferences(
        "launch-test", root_dir=custom_root, ignore_private_config=True
    ) == {
        "queue_size": 9,
    }
    config = {"aspen": {"pipeline": {"threaded": False, "queue_size": 11}}}
    apply_pipeline_preferences(config, saved_pipeline_preferences(None))
    assert config["aspen"]["pipeline"] == {"threaded": False, "queue_size": 11}


def _source_config(workspace, tmp_path, *, sync_method="audio"):
    for name in ("oldleft.mp4", "right.mp4", "newleft.mp4", "chapter2.mp4"):
        (tmp_path / name).write_bytes(name.encode())
    workspace.import_videos({"role": "left", "paths": [str(tmp_path / "oldleft.mp4")]})
    workspace.import_videos({"role": "right", "paths": [str(tmp_path / "right.mp4")]})
    config = read_config(workspace.game())
    config["stitching"] = {
        "sync_method": sync_method,
        "frame_offsets": {"left": 0, "right": 60},
        "auto_input_lens_frozen": True,
        "camera_fov": {"horizontal_fov": 122, "vertical_fov": 90},
        "camera_automation": {"fov": True, "lens": False},
        "input_lens": {"projection": "rectilinear", "distortion": [0.01, 0, 0]},
    }
    write_config(workspace.game(), config)
    archive = workspace.game() / "launch-test-stitched_output-1.mkv"
    archive.write_bytes(b"recording")
    return config, archive


def should_recompute_sync_and_automatic_optics_after_first_source_replacement(
    workspace, tmp_path, monkeypatch
):
    import hmlib.stitching.synchronize as sync
    import hmlib.stitching.video_lens as optics
    from hmlib.stitching.settings import read_stitching_settings

    original, archive = _source_config(workspace, tmp_path)
    workspace.remove_video({"role": "left", "path": "cam1/oldleft.mp4"})
    workspace.import_videos({"role": "left", "paths": [str(tmp_path / "newleft.mp4")]})
    changed = read_config(workspace.game())
    assert "frame_offsets" not in changed["stitching"]
    assert "auto_input_lens_frozen" not in changed["stitching"]
    assert changed["stitching"]["input_lens"] == original["stitching"]["input_lens"]
    assert changed["stitching"]["camera_automation"] == {"fov": True, "lens": False}
    assert archive.read_bytes() == b"recording"
    assert (workspace.game() / "cam1/oldleft.mp4").read_bytes() == b"oldleft.mp4"
    calls = []

    def correlate(**kwargs):
        calls.append(kwargs)
        return 12, 0

    monkeypatch.setattr(sync, "synchronize_by_audio", correlate)
    left, right = str(tmp_path / "newleft.mp4"), str(tmp_path / "right.mp4")
    assert sync.configure_synchronization(workspace.game().name, left, right) == {
        "left": 12,
        "right": 0,
    }
    assert len(calls) == 1 and calls[0]["file1_path"] == left
    assert read_config(workspace.game())["stitching"]["frame_offsets"] == {"left": 12, "right": 0}
    reads = []

    def metadata(video):
        reads.append(video)
        return optics.VideoLens(width=1920, height=1080, horizontal_fov=110, model="new camera")

    monkeypatch.setattr(optics, "read_video_lens", metadata)
    effective = OperatorWorkspace(workspace.game())._effective()
    before = read_stitching_settings(effective)
    resolved, notes = optics.resolve_video_lens(effective, before, [left, right])
    assert reads == [left, right]
    assert resolved.horizontal_fov != before.horizontal_fov
    assert resolved.input_lens == before.input_lens
    assert "saved experiment snapshot" not in " ".join(notes)


def should_keep_manual_timing_for_later_chapters_but_require_new_offsets_after_origin_changes(
    workspace, tmp_path
):
    original, _ = _source_config(workspace, tmp_path, sync_method="manual")
    workspace.import_videos({"role": "left", "paths": [str(tmp_path / "chapter2.mp4")]})
    changed = read_config(workspace.game())
    assert changed["stitching"]["frame_offsets"] == original["stitching"]["frame_offsets"]
    assert "auto_input_lens_frozen" not in changed["stitching"]
    workspace.remove_video({"role": "left", "path": "cam1/oldleft.mp4"})
    changed = read_config(workspace.game())
    assert changed["stitching"]["sync_method"] == "manual"
    assert "frame_offsets" not in changed["stitching"]
    from hmlib.stitching.synchronize import configure_synchronization

    with pytest.raises(ValueError, match="Manual synchronization requires"):
        configure_synchronization("launch-test", "/chapter2.mp4", "/right.mp4")


def should_keep_caches_when_reimporting_the_same_source(workspace, tmp_path):
    original, _ = _source_config(workspace, tmp_path)
    workspace.import_videos({"role": "left", "paths": [str(tmp_path / "oldleft.mp4")]})
    assert read_config(workspace.game()) == original


@pytest.mark.parametrize("role", ["cam1", "cam3", "source"])
def should_preserve_main_camera_preferences_for_unrelated_sources(workspace, tmp_path, role):
    original, _ = _source_config(workspace, tmp_path, sync_method="manual")
    result = workspace.import_videos({"role": role, "paths": [str(tmp_path / "chapter2.mp4")]})
    assert result["invalidated"] == []
    assert read_config(workspace.game())["stitching"] == original["stitching"]
    workspace.remove_video({"role": role, "path": result["paths"][0]})
    assert read_config(workspace.game())["stitching"] == original["stitching"]


def should_preserve_timing_and_frozen_optics_when_joining_equivalent_chapters(workspace, tmp_path):
    from hmlib.ui.workspace.concatenate import concatenate

    paths = []
    for index in range(2):
        path = tmp_path / f"chapter-{index}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=red:s=64x64:r=10:d=0.2",
                "-c:v",
                "libx264",
                str(path),
            ],
            check=True,
            timeout=30,
        )
        paths.append(path)
    workspace.import_videos({"role": "left", "paths": [str(path) for path in paths]})
    config = read_config(workspace.game())
    preferences = {
        "sync_method": "manual",
        "frame_offsets": {"left": 1, "right": 0},
        "auto_input_lens_frozen": True,
    }
    config["stitching"] = copy.deepcopy(preferences)
    write_config(workspace.game(), config)
    work = tmp_path / "join"
    work.mkdir()
    result = concatenate(
        {
            "game": str(workspace.game()),
            "role": "left",
            "paths": [str(path) for path in paths],
            "source_revision": revision(workspace.game()),
        },
        work,
    )
    assert Path(result["path"]).is_file()
    assert read_config(workspace.game())["stitching"] == preferences
    assert all(path.is_file() for path in paths)


def _alternate_configuration(tmp_path):
    import shutil

    from hmlib.config import ROOT_DIR

    root = tmp_path / "alternate-config"
    (root / "config" / "games").mkdir(parents=True)
    shutil.copyfile(Path(ROOT_DIR) / "config" / "baseline.yaml", root / "config" / "baseline.yaml")
    return root


@pytest.mark.parametrize("flag", ["--ignore-private-config=1", "--ignore_private_config=1"])
def should_ignore_private_launch_choices_sources_and_model_cache(workspace, monkeypatch, flag):
    from hmlib.cli.hmtrack import make_parser
    from hmlib.ui import model_preparation

    write_config(
        workspace.game(),
        {
            "game": {"videos": {"source": ["private-missing.mp4"]}},
            "hm_ui": {
                "run": {
                    "detector": "hockeyref",
                    "batch_size": 8,
                    "save_video": False,
                    "precision": "onnx-int8",
                }
            },
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Private INT8 cache must not be consulted")

    monkeypatch.setattr(model_preparation, "prepared_onnx_arguments", forbidden)
    for initial_launch in (False, True):
        command = workspace.pipeline_command(
            workspace.game(), "track", [flag], initial_launch=initial_launch
        )
        args = make_parser().parse_args(command[3:])
        assert args.input_video is None
        assert "--detector-model" not in command
        assert "--batch-size" not in command
        assert not args.no_save_video
        assert not args.detector_onnx_enable
    assert launch_arguments(workspace.game(), "track", explicit_arguments=[flag]) == []


@pytest.mark.parametrize("ignore_private", [True, False])
def should_resolve_launch_preferences_and_sources_from_the_chosen_configuration_root(
    workspace, tmp_path, ignore_private
):
    from hmlib.cli.hmtrack import make_parser

    root = _alternate_configuration(tmp_path)
    public_source = workspace.game() / "public.mp4"
    public_source.write_bytes(b"fixture")
    private_source = workspace.game() / "private.mp4"
    private_source.write_bytes(b"fixture")
    (root / "config" / "games" / "launch-test.yaml").write_text(
        "game:\n  videos:\n    source: [public.mp4]\nhm_ui:\n  run:\n    detector: distilled\n    batch_size: 3\n"
    )
    write_config(
        workspace.game(),
        {"game": {"videos": {"source": ["private.mp4"]}}, "hm_ui": {"run": {"batch_size": 8}}},
    )
    arguments = ["--root-dir", str(root), "--ignore-private-config", str(int(ignore_private))]
    command = workspace.pipeline_command(workspace.game(), "track", arguments, initial_launch=True)
    args = make_parser().parse_args(command[3:])
    assert args.detector_model == "distilled"
    assert args.batch_size == (3 if ignore_private else 8)
    assert args.input_video == str(public_source if ignore_private else private_source)
    assert args.root_dir == str(root)


def should_use_public_model_identity_for_private_opt_out(workspace, tmp_path, monkeypatch):
    from hmlib.ui import model_preparation

    root = _alternate_configuration(tmp_path)
    (root / "config" / "games" / "launch-test.yaml").write_text(
        "hm_ui:\n  run:\n    detector: distilled\n    precision: onnx-int8\n"
    )
    write_config(workspace.game(), {"hm_ui": {"run": {"detector": "hockeyref"}}})
    identities = []

    def status(game, target, *, verify_digest=False, config=None):
        assert verify_digest
        identities.append(config["hm_ui"]["run"]["detector"])
        return {"state": "ready", "path": "/prepared-public-model.onnx"}

    monkeypatch.setattr(model_preparation, "model_status", status)
    command = workspace.pipeline_command(
        workspace.game(),
        "track",
        [f"--root-dir={root}", "--ignore-private-config=1"],
        initial_launch=True,
    )
    assert identities == ["distilled"]
    assert command[command.index("--detector-onnx") + 1] == "/prepared-public-model.onnx"


@pytest.mark.parametrize(
    "order,expected",
    [
        ([], {"threaded": False, "queue_size": 7}),
        (["builtin"], {"threaded": False, "queue_size": 7}),
        (["builtin", "first"], {"threaded": True, "queue_size": 9}),
        (["first", "builtin"], {"threaded": True, "queue_size": 9}),
        (["builtin", "first", "second"], {"threaded": False, "queue_size": 11}),
        (["first"], {"threaded": True, "queue_size": 9}),
    ],
)
def should_honor_explicit_yaml_overlays_in_actual_tracking_main(
    workspace, tmp_path, monkeypatch, order, expected
):
    from hmlib.cli import hmtrack

    write_config(workspace.game(), {"aspen": {"pipeline": {"threaded": False, "queue_size": 7}}})
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("aspen:\n  pipeline:\n    threaded: true\n    queue_size: 9\n")
    second.write_text("aspen:\n  pipeline:\n    threaded: false\n    queue_size: 11\n")
    paths = {
        "builtin": Path(hmtrack.ROOT_DIR) / "config" / "aspen" / "tracking.yaml",
        "first": first,
        "second": second,
    }
    arguments = ["hmtrack", "--game-id", workspace.game().name, "--camera-ui=0", "--decoder=cv2"]
    for name in order:
        arguments.extend(["--config", str(paths[name])])
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(hmtrack, "configure_model", lambda *, config, args: args)
    captured = []
    monkeypatch.setattr(
        hmtrack,
        "_main",
        lambda args, num_gpus: captured.append(args.game_config["aspen"]["pipeline"].copy()),
    )
    hmtrack.main()
    assert len(captured) == 1
    assert {key: captured[0][key] for key in expected} == expected
