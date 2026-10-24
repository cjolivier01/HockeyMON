import hashlib
from pathlib import Path

import pytest
import yaml

from hmlib.stitching import native_calibration as native


@pytest.fixture
def snapshot(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    videos = {}
    for role in ("left", "right"):
        source = game / f"{role}.mp4"
        source.write_bytes(b"source recording")
        videos[role] = [str(source)]
    directory = tmp_path / "snapshot"
    directory.mkdir()
    config = {"game": {"videos": videos}}
    manifest = {
        "version": 1,
        "game_directory": str(game),
        "generation": "native-hugin-generation",
        "output_generation": "native-output-generation",
        "invalidation_id": "owner",
        "frame_offsets": {"left": 0, "right": 0.885511},
        "canvas": {"width": 4, "height": 3},
        "post_stitch_rotate_degrees": 0,
        "configuration": config,
        "sources": {role: [native._identity(Path(videos[role][0]))] for role in videos},
        "artifacts": {},
    }
    for name in native._REQUIRED_ARTIFACTS:
        data = yaml.safe_dump(config).encode() if name == "config.yaml" else b"artifact"
        (directory / name).write_bytes(data)
        manifest["artifacts"][name] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (directory / "native-calibration.yaml").write_text(yaml.safe_dump(manifest))
    return directory, manifest


def should_preserve_native_fractional_sync_and_source_order(snapshot):
    directory, _ = snapshot
    result = native.load_native_calibration(directory)
    assert result.frame_offsets == (0.0, 0.885511)
    assert result.canvas_size == (4, 3)
    assert result.directory == directory
    swapped = {
        "left": result.config["game"]["videos"]["right"],
        "right": result.config["game"]["videos"]["left"],
    }
    with pytest.raises(ValueError, match="different or changed left"):
        result.verify_sources(result.game_directory, swapped)


def should_use_selected_game_for_explicit_native_inputs_in_distinct_directories(snapshot, tmp_path):
    from dataclasses import replace

    from hmlib.cli.hmtrack import _stitching_input_directory

    directory, _ = snapshot
    native_snapshot = native.load_native_calibration(directory)
    videos = {}
    for role in ("left", "right"):
        folder = tmp_path / (role + "-recordings")
        folder.mkdir()
        path = folder / "source.mp4"
        path.write_bytes(role.encode())
        videos[role] = [str(path)]
    native_snapshot = replace(
        native_snapshot,
        sources={role: [native._identity(Path(paths[0]))] for role, paths in videos.items()},
    )
    inputs = [videos[role][0] for role in ("left", "right")]
    game = _stitching_input_directory(inputs, native_snapshot.game_directory, native_snapshot)
    assert Path(game) == native_snapshot.game_directory
    native_snapshot.verify_sources(Path(game), videos)
    wrong_game = _stitching_input_directory(inputs, tmp_path, native_snapshot)
    with pytest.raises(ValueError, match="different game"):
        native_snapshot.verify_sources(Path(wrong_game), videos)
    Path(inputs[0]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="changed left"):
        native_snapshot.verify_sources(Path(game), videos)


def should_reject_calibration_for_a_different_game(snapshot, tmp_path):
    directory, _ = snapshot
    result = native.load_native_calibration(directory)
    with pytest.raises(ValueError, match="different game"):
        result.verify_sources(tmp_path, result.config["game"]["videos"])


def should_reject_replaced_source_recording(snapshot):
    directory, manifest = snapshot
    Path(manifest["sources"]["right"][0]["path"]).write_bytes(b"replacement recording")
    with pytest.raises(ValueError, match="different or changed right"):
        native.load_native_calibration(directory)


def should_reject_corrupted_mapping_even_at_the_same_size(snapshot):
    directory, _ = snapshot
    (directory / "mapping_0000_x.tif").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum changed"):
        native.load_native_calibration(directory)


def should_validate_snapshots_without_python_311_file_digest(snapshot, monkeypatch):
    directory, _ = snapshot
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    assert native.load_native_calibration(directory).directory == directory


def should_reject_artifact_symlinks(snapshot, tmp_path):
    directory, _ = snapshot
    path = directory / "mapping_0000_x.tif"
    other = tmp_path / "external"
    other.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match="not a regular file"):
        native.load_native_calibration(directory)


@pytest.mark.parametrize("change", ["version", "offset", "missing", "escape", "config"])
def should_reject_invalid_snapshot_contract(snapshot, change):
    directory, manifest = snapshot
    if change == "version":
        manifest["version"] = 2
    elif change == "offset":
        manifest["frame_offsets"]["left"] = float("nan")
    elif change == "missing":
        del manifest["artifacts"]["rink_mask_0.png"]
    elif change == "escape":
        manifest["artifacts"]["../outside"] = {"size": 1, "sha256": "invalid"}
    else:
        manifest["configuration"]["game"]["changed"] = True
    (directory / "native-calibration.yaml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError):
        native.load_native_calibration(directory)


def should_fail_when_explicit_native_exporter_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_HSTREAM_CALIBRATION_EXPORT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="not executable"):
        native.prepare_native_calibration(tmp_path, tmp_path)


def should_find_packaged_exporter_beside_the_native_wrapper(tmp_path, monkeypatch):
    prefix = tmp_path / "installation"
    (prefix / "bin").mkdir(parents=True)
    runner = prefix / "run.sh"
    runner.write_text("#!/bin/sh\n")
    exporter = prefix / "bin" / "hstream-calibration-export"
    exporter.write_text("#!/bin/sh\n")
    exporter.chmod(0o755)
    monkeypatch.delenv("HM_HSTREAM_CALIBRATION_EXPORT", raising=False)
    monkeypatch.setenv("HSTREAM_NATIVE_RUNNER", str(runner))
    assert native._exporter_path(None) == exporter


def should_reuse_native_mask_pixels_without_modifying_game(snapshot):
    import cv2
    import numpy as np

    directory, manifest = snapshot
    mask = np.array([[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0]], dtype=np.uint8)
    cv2.imwrite(str(directory / "rink_mask_0.png"), mask)
    data = (directory / "rink_mask_0.png").read_bytes()
    manifest["artifacts"]["rink_mask_0.png"] = {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (directory / "native-calibration.yaml").write_text(yaml.safe_dump(manifest))
    result = native.load_native_calibration(directory)
    profile = result.rink_profile((3, 4))
    assert np.array_equal(profile["combined_mask"].numpy(), mask != 0)
    assert profile["combined_bbox"] == [1, 1, 2, 2]
    assert sorted(p.name for p in result.game_directory.iterdir()) == ["left.mp4", "right.mp4"]
    with pytest.raises(ValueError, match="dimensions"):
        result.rink_profile((4, 3))
