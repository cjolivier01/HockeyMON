from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
import yaml
from stitching_fixtures import write_generation

from hmlib import config as hmlib_config
from hmlib.stitching import artifacts, configure_stitching
from hmlib.stitching.artifact_validation import validate_artifact_generation, validate_mapping_tiff
from hmlib.stitching.calibration_leveling import CalibrationLevelingResult
from hmlib.stitching.seam import load_canvas_seam_mask, read_png_layout


def should_publish_all_files_and_allow_nested_readers(tmp_path):
    (tmp_path / "first").write_bytes(b"old")
    with artifacts.stitching_lock(tmp_path):
        with artifacts.artifact_stage(tmp_path) as stage:
            (stage / "first").write_bytes(b"new")
            (stage / "second").write_bytes(b"new")
            artifacts.publish_artifacts(tmp_path, stage, ["first", "second"])
    assert (tmp_path / "first").read_bytes() == (tmp_path / "second").read_bytes() == b"new"
    assert not list(tmp_path.glob(".stitching-stage-*"))
    assert not (tmp_path / artifacts._JOURNAL).exists()


@pytest.mark.parametrize("atomic_edit", [False, True])
def should_rollback_if_expected_config_changes_during_publication(
    tmp_path, monkeypatch, atomic_edit
):
    (tmp_path / "first").write_bytes(b"old artifact")
    (tmp_path / "config.yaml").write_bytes(b"old config")
    replace = os.replace
    changed = False

    def edit_config_after_first_artifact(source, destination):
        nonlocal changed
        replace(source, destination)
        if Path(destination).name == "first" and not changed:
            changed = True
            if atomic_edit:
                concurrent = tmp_path / ".config.concurrent"
                concurrent.write_bytes(b"concurrent config")
                replace(concurrent, tmp_path / "config.yaml")
            else:
                (tmp_path / "config.yaml").write_bytes(b"concurrent config")

    monkeypatch.setattr(artifacts.os, "replace", edit_config_after_first_artifact)
    with artifacts.artifact_stage(tmp_path) as stage:
        (stage / "first").write_bytes(b"new artifact")
        (stage / "config.yaml").write_bytes(b"new config")
        with pytest.raises(RuntimeError, match="changed while the generation was staged"):
            artifacts.publish_artifacts(
                tmp_path,
                stage,
                ["first", "config.yaml"],
                expected_old_contents={"config.yaml": b"old config"},
            )
    assert (tmp_path / "first").read_bytes() == b"old artifact"
    assert (tmp_path / "config.yaml").read_bytes() == b"concurrent config"
    assert not (tmp_path / artifacts._JOURNAL).exists()


def should_preserve_guarded_config_edited_after_its_replacement(tmp_path, monkeypatch):
    for name in ("first", "last"):
        (tmp_path / name).write_bytes(b"old artifact")
    (tmp_path / "config.yaml").write_bytes(b"old config")
    replace = os.replace

    def edit_config_then_fail(source, destination):
        if Path(destination).name == "last":
            raise OSError("injected publication failure")
        replace(source, destination)
        if Path(destination).name == "config.yaml":
            (tmp_path / "config.yaml").write_bytes(b"concurrent config")

    monkeypatch.setattr(artifacts.os, "replace", edit_config_then_fail)
    with artifacts.artifact_stage(tmp_path) as stage:
        for name in ("first", "last"):
            (stage / name).write_bytes(b"new artifact")
        (stage / "config.yaml").write_bytes(b"new config")
        with pytest.raises(OSError, match="injected publication failure"):
            artifacts.publish_artifacts(
                tmp_path,
                stage,
                ["first", "config.yaml", "last"],
                expected_old_contents={"config.yaml": b"old config"},
            )
    assert (tmp_path / "first").read_bytes() == b"old artifact"
    assert (tmp_path / "last").read_bytes() == b"old artifact"
    assert (tmp_path / "config.yaml").read_bytes() == b"concurrent config"
    assert not (tmp_path / artifacts._JOURNAL).exists()


def should_serialize_private_config_saves_with_artifact_publication(tmp_path, monkeypatch):
    monkeypatch.setitem(
        hmlib_config.save_private_config.__globals__, "GAME_DIR_BASE", str(tmp_path)
    )
    game = tmp_path / "demo"
    game.mkdir()
    started, saved = threading.Event(), threading.Event()

    def save():
        started.set()
        hmlib_config.save_private_config("demo", {"unrelated": "value"}, verbose=False)
        saved.set()

    with artifacts.stitching_lock(game):
        thread = threading.Thread(target=save)
        thread.start()
        assert started.wait(1)
        assert not saved.wait(0.1)
    thread.join(2)
    assert saved.is_set()
    assert yaml.safe_load((game / "config.yaml").read_text()) == {"unrelated": "value"}


@pytest.mark.parametrize("failed_replace", [1, 2, 3, 4])
def should_restore_previous_generation_when_publication_fails(
    tmp_path, monkeypatch, failed_replace
):
    (tmp_path / "first").write_bytes(b"old")
    (tmp_path / "second").write_bytes(b"old")
    replace = os.replace
    calls = 0

    def fail_once(source, destination):
        nonlocal calls
        calls += 1
        if calls == failed_replace:
            raise OSError("injected replacement failure")
        return replace(source, destination)

    with artifacts.artifact_stage(tmp_path) as stage:
        (stage / "first").write_bytes(b"new")
        (stage / "second").write_bytes(b"new")
        monkeypatch.setattr(artifacts.os, "replace", fail_once)
        with pytest.raises(OSError, match="injected replacement failure"):
            artifacts.publish_artifacts(tmp_path, stage, ["first", "second"])
    assert (tmp_path / "first").read_bytes() == (tmp_path / "second").read_bytes() == b"old"


@pytest.mark.parametrize("failed_sync", range(1, 13))
def should_never_leave_mixed_generation_after_sync_failure(tmp_path, monkeypatch, failed_sync):
    for name in ("first", "second"):
        (tmp_path / name).write_bytes(b"old")
    sync = os.fsync
    calls = 0

    def fail_once(descriptor):
        nonlocal calls
        calls += 1
        if calls == failed_sync:
            raise OSError("injected sync failure")
        return sync(descriptor)

    with artifacts.artifact_stage(tmp_path) as stage:
        for name in ("first", "second"):
            (stage / name).write_bytes(b"new")
        monkeypatch.setattr(artifacts.os, "fsync", fail_once)
        with pytest.raises(OSError, match="injected sync failure"):
            artifacts.publish_artifacts(tmp_path, stage, ["first", "second"])
    with artifacts.stitching_lock(tmp_path):
        assert (tmp_path / "first").read_bytes() == (tmp_path / "second").read_bytes()


def should_recover_after_process_exit_during_publication(tmp_path):
    for name in ("first", "second"):
        (tmp_path / name).write_bytes(b"old")
    script = """
import os, sys
from pathlib import Path
from hmlib.stitching import artifacts
root=Path(sys.argv[1])
replace=os.replace
def crash(source, target):
    replace(source, target)
    if Path(target).name == "first":
        os._exit(87)
with artifacts.artifact_stage(root) as stage:
    for name in ("first", "second"):
        (stage/name).write_bytes(b"new")
    artifacts.os.replace=crash
    artifacts.publish_artifacts(root, stage, ["first", "second"])
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False)
    assert result.returncode == 87
    assert (tmp_path / "first").read_bytes() == b"new"
    with artifacts.stitching_lock(tmp_path):
        assert (tmp_path / "first").read_bytes() == (tmp_path / "second").read_bytes() == b"old"
    assert not (tmp_path / artifacts._JOURNAL).exists()


def should_reject_busy_thread_without_releasing_its_lock(tmp_path):
    busy = []
    with artifacts.stitching_lock(tmp_path):

        def attempt():
            with pytest.raises(BlockingIOError):
                with artifacts.stitching_lock(tmp_path, blocking=False):
                    pytest.fail("acquired another thread's lock")
            busy.append(True)

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert busy == [True]


def should_preserve_symlink_targets_for_lock_and_journal(tmp_path):
    target = tmp_path / "keep"
    target.write_bytes(b"user data")
    (tmp_path / ".stitching.lock").symlink_to(target)
    with pytest.raises(OSError):
        with artifacts.stitching_lock(tmp_path):
            pytest.fail("symlink lock accepted")
    (tmp_path / ".stitching.lock").unlink()
    (tmp_path / (artifacts._JOURNAL + ".tmp")).symlink_to(target)
    with artifacts.artifact_stage(tmp_path) as stage:
        (stage / "output").write_bytes(b"new")
        artifacts.publish_artifacts(tmp_path, stage, ["output"])
    assert target.read_bytes() == b"user data"


def should_validate_real_artifacts_and_reject_mismatched_maps(tmp_path):
    write_generation(tmp_path)
    assert validate_artifact_generation(tmp_path, project_name="hm_project.pto") == (4, 3)
    tifffile.imwrite(tmp_path / "mapping_0000_x.tif", np.zeros((3, 3), np.uint16))
    with pytest.raises(ValueError, match="Mismatched"):
        validate_artifact_generation(tmp_path)


def should_reject_oversized_png_header_before_reading_its_payload(tmp_path):
    path = tmp_path / "seam.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I4s", 2**30, b"IHDR"))
    with pytest.raises(ValueError, match="chunk length"):
        read_png_layout(path)


def should_reject_huge_canvas_before_decoding(monkeypatch):
    monkeypatch.setattr(cv2, "imread", lambda *args: pytest.fail("decoded an unbounded canvas"))
    with pytest.raises(ValueError, match="bounds"):
        load_canvas_seam_mask("not-read.png", 65536, 65536)


def should_reject_huge_tiff_metadata_before_tifffile(tmp_path, monkeypatch):
    path = tmp_path / "map.tif"
    path.write_bytes(b"II" + struct.pack("<HIHHHII", 42, 8, 1, 270, 2, 2**30, 100))
    monkeypatch.setattr(tifffile, "TiffFile", lambda *args: pytest.fail("parsed unbounded IFD"))
    with pytest.raises(ValueError, match="Oversized.*tag"):
        validate_mapping_tiff(path)


def should_stage_images_and_publish_only_successful_builds(tmp_path, monkeypatch):
    game = tmp_path / "game"
    game.mkdir()
    images = []
    for name in ("source-left.png", "source-right.png"):
        path = tmp_path / name
        assert cv2.imwrite(str(path), np.zeros((3, 4, 3), np.uint8))
        images.append(str(path))
    write_generation(game)
    before = {path.name: path.read_bytes() for path in game.iterdir()}

    def failed_builder(**kwargs):
        stage = Path(kwargs["project_file_path"]).parent
        assert stage != game
        write_generation(stage)
        raise OSError("mapping command failed")

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", failed_builder)
    with pytest.raises(OSError, match="mapping command failed"):
        configure_stitching.build_stitching_project(str(game / "hm_project.pto"), images, 20)
    assert all((game / name).read_bytes() == data for name, data in before.items())

    def successful_builder(**kwargs):
        stage = Path(kwargs["project_file_path"]).parent
        write_generation(stage)
        for name in ("hm_project.pto", "autooptimiser_out.pto"):
            (stage / name).write_text("\n".join(f'i n"{image}"' for image in kwargs["image_files"]))
        return True

    monkeypatch.setattr(
        configure_stitching, "_build_stitching_project_in_place", successful_builder
    )
    assert configure_stitching.build_stitching_project(str(game / "hm_project.pto"), images, 20)
    assert (game / "s.png").is_file()
    assert (game / "left.png").is_file()
    assert str(game / "left.png") in (game / "autooptimiser_out.pto").read_text()
    assert ".stitching-stage-" not in (game / "autooptimiser_out.pto").read_text()
    assert "input_images" in json.loads((game / ".stitching_artifacts.json").read_text())


def should_reject_repeated_tiff_metadata_before_decoding(tmp_path, monkeypatch):
    path = tmp_path / "map.tif"
    path.write_bytes(
        b"II"
        + struct.pack("<HIH", 42, 8, 2)
        + struct.pack("<HHII", 270, 2, 1, 0) * 2
        + struct.pack("<I", 0)
    )
    monkeypatch.setattr(tifffile, "TiffFile", lambda *args: pytest.fail("parsed repeated tags"))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_mapping_tiff(path)


def should_reuse_published_game_local_input_images(tmp_path, monkeypatch):
    images = []
    for name in ("left.png", "right.png"):
        path = tmp_path / name
        assert cv2.imwrite(str(path), np.zeros((3, 4, 3), np.uint8))
        images.append(str(path))
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        write_generation(Path(kwargs["project_file_path"]).parent)
        return True

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    for _ in range(3):
        assert configure_stitching.build_stitching_project(
            str(tmp_path / "hm_project.pto"), images, 20
        )
    assert len(calls) == 1


def should_hold_lock_through_coordinate_decode(tmp_path, monkeypatch):
    from hmlib.stitching.artifact_validation import read_mapping_arrays

    write_generation(tmp_path)
    read = cv2.imread
    calls = []

    def verify_lock(path, flags):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import fcntl,sys; f=open(sys.argv[1], 'a+'); fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)",
                str(tmp_path / ".stitching.lock"),
            ],
            check=False,
            capture_output=True,
        )
        assert result.returncode != 0
        calls.append(path)
        return read(path, flags)

    monkeypatch.setattr(cv2, "imread", verify_lock)
    x, y, cols, rows = read_mapping_arrays(tmp_path, "mapping_0000")
    assert (x, y) == (0, 0)
    assert cols.shape == rows.shape == (3, 4)
    assert len(calls) == 2


def should_keep_primary_calibration_failure_when_stage_cleanup_fails(tmp_path, monkeypatch, caplog):
    failure = configure_stitching.CalibrationAlignmentError("rejected candidate")

    def fail_cleanup(path):
        raise OSError("cannot remove temporary files")

    with pytest.raises(configure_stitching.CalibrationAlignmentError) as caught:
        with artifacts.artifact_stage(tmp_path):
            monkeypatch.setattr(artifacts.shutil, "rmtree", fail_cleanup)
            raise failure
    assert caught.value is failure
    assert "retaining evidence" in caplog.text


def should_invalidate_only_derived_geometry_after_validating_replacement(tmp_path, monkeypatch):
    write_generation(tmp_path)
    images = []
    for name in ("left.png", "right.png"):
        path = tmp_path / name
        assert cv2.imwrite(str(path), np.zeros((3, 4, 3), np.uint8))
        images.append(str(path))
    mask = tmp_path / "rink_mask_1.png"
    mask.write_bytes(b"old mask")
    assert cv2.imwrite(str(tmp_path / "xor_file.png"), np.zeros((6, 8), np.uint8))
    config = {
        "stitching": {"stitch_frame_time": "5", "frame_offsets": {"left": 2}},
        "rink": {"scoreboard": {"perspective_polygon": [1, 2, 3]}, "ice_contours_mask_count": 1},
    }

    def fail(**kwargs):
        raise OSError("new calibration failed")

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", fail)
    with pytest.raises(OSError):
        configure_stitching.build_stitching_project(
            str(tmp_path / "hm_project.pto"), images, 20, game_config=config
        )
    assert mask.exists()
    assert config["rink"]["ice_contours_mask_count"] == 1

    def succeed(**kwargs):
        write_generation(Path(kwargs["project_file_path"]).parent)
        return True

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", succeed)
    configure_stitching.build_stitching_project(
        str(tmp_path / "hm_project.pto"), images, 20, game_config=config
    )
    assert not mask.exists()
    assert not (tmp_path / "xor_file.png").exists()
    assert "rink" not in config
    assert config["stitching"] == {"stitch_frame_time": "5", "frame_offsets": {"left": 2}}


def should_preserve_published_seam_when_legacy_regeneration_fails(tmp_path, monkeypatch):
    from hmlib.stitching import blender2

    write_generation(tmp_path)
    old_seam = (tmp_path / "seam_file.png").read_bytes()

    def fail(command, **kwargs):
        Path(
            next(value.split("=", 1)[1] for value in command if value.startswith("--save-masks="))
        ).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(blender2.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        blender2.make_seam_and_xor_masks(str(tmp_path), "mapping_", force=True)
    assert (tmp_path / "seam_file.png").read_bytes() == old_seam


def should_never_replace_the_game_lock_during_publication(tmp_path):
    with artifacts.artifact_stage(tmp_path) as stage:
        (stage / ".stitching.lock").write_bytes(b"not the held lock")
        with pytest.raises(ValueError, match="Invalid stitching artifact name"):
            artifacts.publish_artifacts(tmp_path, stage, [".stitching.lock"])


@pytest.mark.parametrize("byteorder", ["<", ">"])
def should_accept_bigtiff_coordinate_maps_without_native_struct_padding(tmp_path, byteorder):
    path = tmp_path / "coordinate.tif"
    tifffile.imwrite(path, np.zeros((3, 4), np.uint16), bigtiff=True, byteorder=byteorder)
    assert validate_mapping_tiff(path, coordinates=True) == (4, 3)


def should_accept_multiblend_palette_seams(tmp_path):
    from PIL import Image

    path = tmp_path / "seam.png"
    seam = Image.fromarray(np.array([[0, 1], [1, 0]], np.uint8)).convert("P")
    seam.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    seam.save(path)
    assert load_canvas_seam_mask(path, 2, 2).tolist() == [[0, 255], [255, 0]]


def should_rebind_quoted_and_backslash_pto_sources(tmp_path):
    import shlex

    game = tmp_path / 'quoted"game\\name'
    game.mkdir()
    stage = game / ".stitching-stage-example"
    stage.mkdir()
    path = stage / "project.pto"
    path.write_text("p f2 w4 h3 v180\ni w4 h3 n" + json.dumps(str(stage / "left.png")) + "\n")
    configure_stitching._rewrite_pto_sources(path, source_directory=stage, target_directory=game)
    token = next(
        value for value in shlex.split(path.read_text().splitlines()[1]) if value.startswith("n")
    )
    assert token[1:] == str(game / "left.png")


def should_reject_corrupt_staged_xor_before_publishing_a_new_seam(tmp_path, monkeypatch):
    from hmlib.stitching import blender2

    write_generation(tmp_path)
    old = (tmp_path / "seam_file.png").read_bytes()

    class BrokenBlender:
        def __init__(self, args):
            self.seam, self.xor = Path(args[1]), Path(args[3])

        def blend_images(self, **kwargs):
            self.seam.write_bytes(old)
            self.xor.write_bytes(b"not a PNG")

    from types import SimpleNamespace

    image = SimpleNamespace(image=np.zeros((3, 4, 4), np.uint8), xpos=0, ypos=0)
    monkeypatch.setattr(blender2, "EnBlender", BrokenBlender)
    monkeypatch.setattr(blender2, "make_cv_compatible_tensor", lambda array: array)
    with pytest.raises(ValueError):
        blender2.make_seam_and_xor_masks(
            str(tmp_path), "mapping_", [image, image], force=True, use_enblend_tool=False
        )
    assert (tmp_path / "seam_file.png").read_bytes() == old
    assert not (tmp_path / "xor_file.png").exists()


def should_reject_uniform_owner_seams_before_runtime_initialization(tmp_path):
    write_generation(tmp_path)
    assert cv2.imwrite(str(tmp_path / "seam_file.png"), np.zeros((3, 4), np.uint8))
    with pytest.raises(ValueError, match="uniform"):
        validate_artifact_generation(tmp_path)


def should_decode_lzw_panorama_without_optional_imagecodecs(tmp_path):
    from PIL import Image

    Image.new("RGB", (4, 3), (19, 83, 151)).save(tmp_path / "panorama.tif", compression="tiff_lzw")
    configure_stitching._save_stitched_reference_frame(tmp_path)
    with Image.open(tmp_path / "s.png") as image:
        assert image.size == (4, 3)
        assert image.getpixel((1, 1)) == (19, 83, 151)


def should_reject_uniform_staged_owner_seam_before_replacement(tmp_path, monkeypatch):
    from hmlib.stitching import blender2

    write_generation(tmp_path)
    old = (tmp_path / "seam_file.png").read_bytes()

    def uniform(command, **kwargs):
        path = next(
            value.split("=", 1)[1] for value in command if value.startswith("--save-masks=")
        )
        assert cv2.imwrite(path, np.zeros((3, 4), np.uint8))

    monkeypatch.setattr(blender2.subprocess, "run", uniform)
    with pytest.raises(ValueError, match="uniform"):
        blender2.make_seam_and_xor_masks(str(tmp_path), "mapping_", force=True)
    assert (tmp_path / "seam_file.png").read_bytes() == old


def _source_images(directory):
    paths = [directory / "left.png", directory / "right.png"]
    for image in paths:
        assert cv2.imwrite(str(image), np.zeros((3, 4, 3), np.uint8))
    return [str(path) for path in paths]


def _build_fake_generation(**kwargs):
    stage = Path(kwargs["project_file_path"]).parent
    retained = stage / "hm_project.pto"
    lines = retained.read_text().splitlines() if retained.exists() else []
    manual_points = [line for line in lines if line.startswith("c ")]
    write_generation(stage)
    for name in ("hm_project.pto", "autooptimiser_out.pto"):
        (stage / name).write_text(
            "p f2 w4 h3 v180\n"
            + "\n".join("i w4 h3 n" + json.dumps(image) for image in kwargs["image_files"])
            + "\n"
            + "\n".join(manual_points)
            + "\n"
        )
    return True


def should_publish_and_persist_selected_calibration_leveling_settings(tmp_path, monkeypatch):
    images = _source_images(tmp_path)
    game = tmp_path / "game"
    game.mkdir()
    config = {"stitching": {"mapping_backend": "nona", "run_autooptimizer": True}}
    private, selections = {}, []

    def select(**kwargs):
        selections.append(kwargs)
        return CalibrationLevelingResult(True, (11, -24, 3))

    def build(**kwargs):
        stage = Path(kwargs["project_file_path"]).parent
        effective = kwargs["calibration_leveling"](
            stage / ".autooptimiser_out.aligned.pto",
            stage / "autooptimiser_out.pto",
            kwargs["image_files"],
            kwargs["settings"],
        )
        assert effective.framing.rotation_degrees == (11, -24, 3)
        return _build_fake_generation(**kwargs)

    monkeypatch.setattr(configure_stitching, "select_calibration_leveling", select)
    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    monkeypatch.setattr(configure_stitching, "get_game_config_private", lambda **kwargs: private)
    monkeypatch.setattr(
        configure_stitching,
        "_private_config_path",
        lambda game_id: (game / "config.yaml").resolve(),
    )
    assert configure_stitching.build_stitching_project(
        str(game / "hm_project.pto"),
        images,
        20,
        game_id="demo",
        game_config=config,
    )
    assert len(selections) == 1
    assert selections[0]["game_id"] == "demo"
    assert config["stitching"]["projection_framing"]["rotation_degrees"] == [11, -24, 3]
    saved = yaml.safe_load((game / "config.yaml").read_text())
    assert saved["stitching"]["projection_framing"]["rotation_degrees"] == [11, -24, 3]
    manifest = json.loads((game / ".stitching_artifacts.json").read_text())
    assert json.loads(manifest["calibration_settings"])["framing"]["rotation_degrees"] == [
        11,
        -24,
        3,
    ]


def should_not_persist_leveling_when_downstream_generation_fails(tmp_path, monkeypatch):
    images = _source_images(tmp_path)
    game = tmp_path / "game"
    game.mkdir()
    write_generation(game)
    before = {path.name: path.read_bytes() for path in game.iterdir()}
    config = {"stitching": {"mapping_backend": "nona", "run_autooptimizer": True}}
    private = {"unrelated": "keep"}

    monkeypatch.setattr(
        configure_stitching,
        "select_calibration_leveling",
        lambda **kwargs: CalibrationLevelingResult(True, (11, -24, 3)),
    )
    monkeypatch.setattr(configure_stitching, "get_game_config_private", lambda **kwargs: private)

    def fail_after_selection(**kwargs):
        stage = Path(kwargs["project_file_path"]).parent
        kwargs["calibration_leveling"](
            stage / ".autooptimiser_out.aligned.pto",
            stage / "autooptimiser_out.pto",
            kwargs["image_files"],
            kwargs["settings"],
        )
        raise RuntimeError("downstream NONA failure")

    monkeypatch.setattr(
        configure_stitching, "_build_stitching_project_in_place", fail_after_selection
    )
    with pytest.raises(RuntimeError, match="downstream NONA"):
        configure_stitching.build_stitching_project(
            str(game / "hm_project.pto"),
            images,
            20,
            game_id="demo",
            game_config=config,
        )
    assert {
        path.name: path.read_bytes() for path in game.iterdir() if path.name != ".stitching.lock"
    } == before
    assert "projection_framing" not in config["stitching"]


@pytest.mark.parametrize(
    "change",
    [
        ("mapping_backend", "opencv-magsac"),
        ("projection", "equirectangular"),
        ("camera_fov", {"horizontal_fov": 105}),
        ("projection_framing", {"auto_crop": True}),
        ("projection_framing", {"rotation_degrees": [0, 3, -2]}),
        ("rink_config", "olympic"),
    ],
)
def should_reject_leveling_when_calibration_config_changes(tmp_path, monkeypatch, change):
    images = _source_images(tmp_path)
    game = tmp_path / "game"
    game.mkdir()
    write_generation(game)
    before = {path.name: path.read_bytes() for path in game.iterdir()}
    config = {"stitching": {"mapping_backend": "nona", "run_autooptimizer": True}}
    private = {}

    monkeypatch.setattr(
        configure_stitching,
        "select_calibration_leveling",
        lambda **kwargs: CalibrationLevelingResult(True, (11, -24, 3)),
    )
    monkeypatch.setattr(configure_stitching, "get_game_config_private", lambda **kwargs: private)

    def change_after_selection(**kwargs):
        stage = Path(kwargs["project_file_path"]).parent
        kwargs["calibration_leveling"](
            stage / ".autooptimiser_out.aligned.pto",
            stage / "autooptimiser_out.pto",
            kwargs["image_files"],
            kwargs["settings"],
        )
        config["stitching"][change[0]] = change[1]
        return _build_fake_generation(**kwargs)

    monkeypatch.setattr(
        configure_stitching, "_build_stitching_project_in_place", change_after_selection
    )
    with pytest.raises(ValueError, match="settings changed"):
        configure_stitching.build_stitching_project(
            str(game / "hm_project.pto"),
            images,
            20,
            game_id="demo",
            game_config=config,
        )
    assert {
        path.name: path.read_bytes() for path in game.iterdir() if path.name != ".stitching.lock"
    } == before
    assert not list(game.glob(".stitching-stage-*"))


def should_cancel_leveling_without_publishing_a_partial_generation(tmp_path, monkeypatch):
    images = _source_images(tmp_path)
    game = tmp_path / "game"
    game.mkdir()
    write_generation(game)
    before = {path.name: path.read_bytes() for path in game.iterdir()}
    config = {"stitching": {"mapping_backend": "nona", "run_autooptimizer": True}}

    monkeypatch.setattr(
        configure_stitching,
        "select_calibration_leveling",
        lambda **kwargs: CalibrationLevelingResult(False, (0, 0, 0), True),
    )

    def build(**kwargs):
        stage = Path(kwargs["project_file_path"]).parent
        kwargs["calibration_leveling"](
            stage / ".autooptimiser_out.aligned.pto",
            stage / "autooptimiser_out.pto",
            kwargs["image_files"],
            kwargs["settings"],
        )
        pytest.fail("cancelled selector returned to the builder")

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    with pytest.raises(configure_stitching.CalibrationLevelingCancelled):
        configure_stitching.build_stitching_project(
            str(game / "hm_project.pto"),
            images,
            20,
            game_id="demo",
            game_config=config,
        )
    assert {
        path.name: path.read_bytes() for path in game.iterdir() if path.name != ".stitching.lock"
    } == before
    assert not list(game.glob(".stitching-stage-*"))


def should_reuse_frame_content_before_running_matcher_or_invalidating_again(tmp_path, monkeypatch):
    import torch
    from hmlib.cli import create_control_points

    frame = np.zeros((3, 4, 3), np.uint8)
    matches, builds, invalidations = [], [], []
    points = {"m_kpts0": torch.zeros((4, 2)), "m_kpts1": torch.zeros((4, 2))}

    def match(left, right, **kwargs):
        matches.append((left, right))
        assert Path(left).is_file() and Path(right).is_file()
        return points

    def build(**kwargs):
        builds.append(kwargs)
        assert kwargs["control_points"] is points
        return _build_fake_generation(**kwargs)

    monkeypatch.setattr(create_control_points, "calculate_control_points", match)
    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    monkeypatch.setattr(configure_stitching, "get_game_config_private", lambda **kwargs: {})
    monkeypatch.setattr(
        configure_stitching,
        "invalidate_stitching_geometry",
        lambda directory, **kwargs: invalidations.append(kwargs),
    )
    config = {"rink": {"ice_contours_mask_count": 2}}
    for _ in range(3):
        assert create_control_points.configure_stitching(
            frame,
            frame,
            str(tmp_path),
            skip_if_exists=True,
            force=False,
            game_id="demo",
            game_config=config,
        )
    assert len(matches) == len(builds) == len(invalidations) == 1
    assert invalidations[0] == {}
    assert "ice_contours_mask_count" not in config.get("rink", {})
    assert not list(tmp_path.glob("hm-calibration-input-*"))
    # A real content change must rebuild even though temporary names are ignored.
    create_control_points.configure_stitching(
        frame + 1, frame, str(tmp_path), skip_if_exists=True, force=False
    )
    assert len(matches) == len(builds) == 2


def should_pin_original_game_lenses_across_staging_and_invalidate_profile_changes(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    images = _source_images(source)
    game = tmp_path / "game"
    game.mkdir()
    camera = {"width": 4, "height": 3, "fx": 2, "fy": 2, "cx": 2, "cy": 1, "d": [0, 0, 0, 0]}
    profile = {"left_uniforms": camera.copy(), "right_uniforms": camera.copy()}
    profile_path = game / "left_calibration.json"
    profile_path.write_text(json.dumps(profile))
    pairs = []

    def build(**kwargs):
        pair = kwargs["lens_calibration"]
        pairs.append(pair)
        assert kwargs["lens_calibration_resolved"] is True
        assert pair.fingerprint == kwargs["settings"].lens_profile_fingerprint
        assert not (Path(kwargs["project_file_path"]).parent / "left_calibration.json").exists()
        return _build_fake_generation(**kwargs)

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    for _ in range(2):
        configure_stitching.build_stitching_project(
            str(game / "hm_project.pto"), images, 20, control_point_matcher="akaze"
        )
    assert len(pairs) == 1
    profile["right_uniforms"]["fx"] = 3
    profile_path.write_text(json.dumps(profile))
    configure_stitching.build_stitching_project(
        str(game / "hm_project.pto"), images, 20, control_point_matcher="akaze"
    )
    assert len(pairs) == 2 and pairs[0].fingerprint != pairs[1].fingerprint
    assert pairs[0].right.fx == 2 and pairs[1].right.fx == 3


def should_freeze_missing_lens_profile_before_entering_private_builder(tmp_path, monkeypatch):
    images = _source_images(tmp_path)

    def build(**kwargs):
        assert kwargs["lens_calibration"] is None
        assert kwargs["lens_calibration_resolved"] is True
        (tmp_path / "left_calibration.json").write_text("malformed late profile")
        return _build_fake_generation(**kwargs)

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    configure_stitching.build_stitching_project(
        str(tmp_path / "hm_project.pto"), images, 20, control_point_matcher="akaze"
    )


def should_invalidate_direct_cache_when_effective_scale_changes(tmp_path, monkeypatch):
    images = _source_images(tmp_path)
    calls = []

    def build(**kwargs):
        calls.append(kwargs["scale"])
        return _build_fake_generation(**kwargs)

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    settings = configure_stitching.read_stitching_settings(
        {"stitching": {"mapping_backend": "nona", "run_autooptimizer": True}}
    )
    for scale in (None, 1, 0.5, 0.5, 2):
        configure_stitching.build_stitching_project(
            str(tmp_path / "hm_project.pto"), images, 20, settings=settings, scale=scale
        )
    assert calls == [None, 0.5, 2]
    assert json.loads((tmp_path / ".stitching_artifacts.json").read_text())["output_scale"] == "2"


@pytest.mark.parametrize("change", [None, "scale", "source", "settings", "force", "reference"])
def should_retain_edited_pto_only_for_current_video_generation(tmp_path, monkeypatch, change):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch

    images = _source_images(tmp_path)
    videos = [tmp_path / "left.mp4", tmp_path / "right.mp4"]
    for path in videos:
        path.write_bytes(b"video")
    settings = replace(
        configure_stitching.read_stitching_settings(
            {
                "stitching": {
                    "mapping_backend": "nona",
                    "run_autooptimizer": True,
                    "calibration_frame_count": 1,
                }
            }
        ),
        max_control_points=20,
    )
    provenance = {
        "source_videos": configure_stitching._file_provenance(videos),
        "source_frame_offsets": "[2, 1]",
        "stitch_frame_time": "",
    }
    calls, matches = [], []

    def build(**kwargs):
        calls.append(kwargs)
        return _build_fake_generation(**kwargs)

    monkeypatch.setattr(configure_stitching, "_build_stitching_project_in_place", build)
    configure_stitching.build_stitching_project(
        str(tmp_path / "hm_project.pto"),
        images,
        20,
        settings=settings,
        scale=0.5,
        provenance=provenance,
    )
    project = tmp_path / "hm_project.pto"
    project.write_text(project.read_text() + "c n0 N1 x1 y1 X1 Y1 t0\n")
    optimized_time = (tmp_path / "autooptimiser_out.pto").stat().st_mtime_ns
    os.utime(project, ns=(optimized_time + 1000000, optimized_time + 1000000))
    calls.clear()
    if change == "source":
        videos[0].write_bytes(b"new video")
    elif change == "settings":
        settings = replace(settings, max_output_dimension=400)
    elif change == "reference":
        assert cv2.imwrite(images[0], np.ones((3, 4, 3), np.uint8))
    monkeypatch.setattr(
        configure_stitching, "BasicVideoInfo", lambda video: SimpleNamespace(frame_count=20)
    )
    monkeypatch.setattr(
        configure_stitching,
        "extract_frame_image",
        lambda video, frame_number, dest_image: cv2.imwrite(
            dest_image, np.zeros((3, 4, 3), np.uint8)
        ),
    )

    def match(*args, **kwargs):
        matches.append(True)
        points = torch.tensor([[0, 0], [3, 0], [3, 2], [0, 2]], dtype=torch.float32)
        return {"m_kpts0": points, "m_kpts1": points}

    monkeypatch.setattr(configure_stitching, "calculate_control_points", match)
    configure_stitching.configure_video_stitching(
        str(tmp_path),
        str(videos[0]),
        str(videos[1]),
        20,
        left_frame_offset=2,
        right_frame_offset=1,
        settings=settings,
        scale=0.25 if change == "scale" else 0.5,
        force=change == "force",
        ignore_private_config=True,
    )
    assert len(calls) == 1
    if change is None:
        assert not matches
        assert calls[0]["force"] is False and calls[0]["control_points"] is None
        assert "c n0 N1 x1 y1 X1 Y1 t0" in project.read_text()
    else:
        assert matches and calls[0]["force"] is True and calls[0]["control_points"] is not None
        assert "c n0 N1 x1 y1 X1 Y1 t0" not in project.read_text()


def should_reject_changed_images_during_staging_without_replacing_generation(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    images = _source_images(source)
    game = tmp_path / "game"
    game.mkdir()
    write_generation(game)
    previous = (game / "hm_project.pto").read_bytes()
    copy = configure_stitching.shutil.copy2

    def changed(path, destination):
        Path(path).write_bytes(Path(path).read_bytes() + b"changed")
        return copy(path, destination)

    monkeypatch.setattr(configure_stitching.shutil, "copy2", changed)
    with pytest.raises(OSError, match="changed while staging"):
        configure_stitching.build_stitching_project(str(game / "hm_project.pto"), images, 20)
    assert (game / "hm_project.pto").read_bytes() == previous


def should_reject_nonregular_images_without_blocking(tmp_path):
    image = tmp_path / "left.png"
    os.mkfifo(image)
    with pytest.raises(ValueError, match="calibration image"):
        configure_stitching._image_content_provenance([image])


@pytest.mark.parametrize("clean_all", [False, True])
def should_preserve_archived_run_masks_when_invalidating_calibration(
    tmp_path, monkeypatch, clean_all
):
    for name in ["rink_mask_0.png", "rink_mask_1.png", "rink_mask_0-1.png", "rink_mask_0-7.png"]:
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(configure_stitching, "get_game_config_private", lambda **kwargs: {})
    if clean_all:
        configure_stitching.clean_stitch_game_artifacts("game", tmp_path)
    else:
        configure_stitching.invalidate_stitching_geometry(tmp_path)
    assert not (tmp_path / "rink_mask_0.png").exists()
    assert not (tmp_path / "rink_mask_1.png").exists()
    for name in ["rink_mask_0-1.png", "rink_mask_0-7.png"]:
        assert (tmp_path / name).read_bytes() == name.encode()
