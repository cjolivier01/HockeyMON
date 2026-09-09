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
from stitching_fixtures import write_generation

from hmlib.stitching import artifacts, configure_stitching
from hmlib.stitching.artifact_validation import validate_artifact_generation, validate_mapping_tiff
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
