"""Private, native-validated calibration snapshots for the shared HStream UI.

HStream owns calibration and its provenance. Its exporter holds the native
transaction locks and validates the selected generation; Python verifies the
exported bytes and uses its own copy for renderer caches.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from hmlib.utils.checksum import sha256_file

_REQUIRED_ARTIFACTS = {
    "hm_project.pto",
    "autooptimiser_out.pto",
    "mapping_0000.tif",
    "mapping_0000_x.tif",
    "mapping_0000_y.tif",
    "mapping_0001.tif",
    "mapping_0001_x.tif",
    "mapping_0001_y.tif",
    "seam_file.png",
    "stitching_generation_id",
    "config.yaml",
    "rink_mask_0.png",
}


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or stat.st_size <= 0:
        raise ValueError(f"Source recording is not a nonempty regular file: {path}")
    return {
        "path": str(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


@dataclass(frozen=True)
class NativeCalibration:
    directory: Path
    game_directory: Path
    config: dict[str, Any]
    generation: str
    output_generation: str
    invalidation_id: str
    frame_offsets: tuple[float, float]
    canvas_size: tuple[int, int]
    post_stitch_rotate_degrees: float
    sources: dict[str, list[dict[str, Any]]]

    def verify_sources(self, game_directory: Path, videos: dict[str, list[str]]) -> None:
        if Path(game_directory).resolve() != self.game_directory:
            raise ValueError("Native calibration belongs to a different game directory")
        for role in ("left", "right"):
            selected = []
            for name in videos[role]:
                path = Path(name)
                selected.append(
                    _identity(path if path.is_absolute() else self.game_directory / path)
                )
            if selected != self.sources[role]:
                raise ValueError(f"Native calibration has different or changed {role} recordings")

    def rink_profile(self, expected_shape: tuple[int, int]) -> dict[str, Any]:
        import cv2
        import torch

        image = cv2.imread(str(self.directory / "rink_mask_0.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Native rink mask could not be decoded")
        mask_array = image != 0
        if tuple(mask_array.shape) != tuple(expected_shape):
            raise ValueError("Native rink mask does not match the stitched output dimensions")
        mask = torch.from_numpy(mask_array.copy())
        ys, xs = torch.where(mask)
        if not xs.numel():
            raise ValueError("Native rink mask contains no rink pixels")
        return {
            "combined_mask": mask,
            "centroid": torch.stack((xs.float().mean(), ys.float().mean())),
            "combined_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "geometry_revision": self.output_generation,
        }


def load_native_calibration(directory: Path) -> NativeCalibration:
    directory = Path(directory).resolve(strict=True)
    manifest_path = directory / "native-calibration.yaml"
    if manifest_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Native calibration manifest exceeds the metadata limit")
    with manifest_path.open() as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported native calibration manifest version")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not _REQUIRED_ARTIFACTS.issubset(artifacts):
        raise ValueError("Native calibration snapshot is missing required artifacts")
    for name, expected in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid native calibration artifact name")
        artifact = directory / name
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError(f"Native calibration artifact is not a regular file: {name}")
        if artifact.stat().st_size != expected["size"]:
            raise ValueError(f"Native calibration artifact size changed: {name}")
        actual_hash = sha256_file(artifact)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"Native calibration artifact checksum changed: {name}")
    offsets = tuple(float(manifest["frame_offsets"][role]) for role in ("left", "right"))
    if any(not math.isfinite(value) or value < 0 for value in offsets):
        raise ValueError("Native synchronization offsets must be finite and nonnegative")
    canvas = tuple(int(manifest["canvas"][axis]) for axis in ("width", "height"))
    if any(value <= 0 for value in canvas):
        raise ValueError("Native calibration canvas must have positive dimensions")
    rotation = float(manifest["post_stitch_rotate_degrees"])
    if not math.isfinite(rotation):
        raise ValueError("Native calibration rotation must be finite")
    for key in ("generation", "output_generation", "invalidation_id"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"Native calibration has no {key}")
    config = manifest["configuration"]
    with (directory / "config.yaml").open() as stream:
        if yaml.safe_load(stream) != config:
            raise ValueError("Native calibration configuration differs from its snapshot")
    if not isinstance(config, dict):
        raise TypeError("Native calibration configuration must be a mapping")
    result = NativeCalibration(
        directory=directory,
        game_directory=Path(manifest["game_directory"]).resolve(strict=True),
        config=config,
        generation=manifest["generation"],
        output_generation=manifest["output_generation"],
        invalidation_id=manifest["invalidation_id"],
        frame_offsets=offsets,
        canvas_size=canvas,
        post_stitch_rotate_degrees=rotation,
        sources=manifest["sources"],
    )
    result.verify_sources(result.game_directory, config["game"]["videos"])
    return result


def _exporter_path(explicit: str | Path | None) -> Path:
    configured = explicit or os.environ.get("HM_HSTREAM_CALIBRATION_EXPORT")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"Native calibration exporter is not executable: {candidate}")
        return candidate.resolve()
    runner = os.environ.get("HSTREAM_NATIVE_RUNNER") or shutil.which("hstream-cli")
    candidates = []
    if runner:
        parent = Path(runner).resolve().parent
        candidates.extend(
            (
                parent / "hstream-calibration-export",
                parent / "bin" / "hstream-calibration-export",
                parent.parent / "hstream-calibration-export" / "hstream-calibration-export",
            )
        )
    executable = shutil.which("hstream-calibration-export")
    if executable:
        candidates.append(Path(executable))
    candidates.append(Path("/opt/hstream/bin/hstream-calibration-export"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "Install hstream-calibration-export or set HM_HSTREAM_CALIBRATION_EXPORT"
    )


def prepare_native_calibration(
    game_dir: Path,
    snapshot_parent: Path,
    *,
    exporter: str | Path | None = None,
    max_output_width: int = 0,
    effective_config: Path | None = None,
) -> NativeCalibration:
    if (
        isinstance(max_output_width, bool)
        or not isinstance(max_output_width, int)
        or max_output_width < 0
    ):
        raise ValueError("Maximum output width must be a nonnegative integer")
    Path(snapshot_parent).mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = Path(snapshot_parent) / f"native-calibration-{uuid.uuid4().hex}"
    subprocess.run(
        [
            str(_exporter_path(exporter)),
            "--game-dir",
            str(Path(game_dir).resolve()),
            "--destination",
            str(destination.resolve()),
            "--max-output-width",
            str(max_output_width),
            *(["--config", str(Path(effective_config).resolve())] if effective_config else []),
        ],
        check=True,
    )
    try:
        return load_native_calibration(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise


@lru_cache(maxsize=1)
def _load_shared_snapshot(directory: str) -> NativeCalibration:
    return load_native_calibration(Path(directory))


def shared_native_calibration() -> NativeCalibration | None:
    """Return the per-process startup snapshot, verifying every artifact once."""
    directory = os.environ.get("HM_SHARED_NATIVE_CALIBRATION")
    return _load_shared_snapshot(directory) if directory else None


def _native_operation(arguments: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(_exporter_path(None)), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"Native calibration operation failed ({completed.returncode}): {detail}"
        )
    result = yaml.safe_load(completed.stdout)
    if not isinstance(result, dict):
        raise TypeError("Native calibration operation returned invalid metadata")
    return result


def read_scoreboard_canvas(
    game_dir: Path, *, source_reference: Path | None = None, destination: Path | None = None
) -> dict[str, Any]:
    """Pin native scoreboard bytes and provenance. Caller removes image_path."""
    descriptor, image_path = tempfile.mkstemp(
        prefix="hm-scoreboard-", suffix=".png", dir=destination
    )
    os.close(descriptor)
    try:
        return _native_operation(
            [
                "--game-dir",
                str(Path(game_dir).resolve()),
                "--scoreboard-canvas",
                image_path,
                *(
                    ["--scoreboard-reference-game", str(source_reference.resolve())]
                    if source_reference
                    else []
                ),
            ]
        )
    except BaseException:
        Path(image_path).unlink(missing_ok=True)
        raise


def save_native_scoreboard(
    game_dir: Path, points: list[list[int]], canvas: dict[str, Any]
) -> list[list[int]]:
    """Persist a selection only while the pinned native canvas is still current."""
    with tempfile.TemporaryDirectory(prefix="hm-scoreboard-save-") as temporary:
        request_path = Path(temporary) / "request.yaml"
        request_path.write_text(yaml.safe_dump({"points": points, "canvas": canvas}))
        result = _native_operation(
            [
                "--game-dir",
                str(Path(game_dir).resolve()),
                "--save-scoreboard",
                str(request_path),
            ]
        )
    return result["points"]


def validate_live_output_epoch(game_dir: Path, serialized_epoch: str) -> dict[str, Any]:
    """Validate a UI-owned rotation/scoreboard epoch before the runtime applies it."""
    native = shared_native_calibration()
    if native is None or Path(game_dir).resolve() != native.game_directory:
        raise ValueError("Live output epoch requires this game's native calibration snapshot")
    with tempfile.TemporaryDirectory(prefix="hm-output-epoch-") as temporary:
        request_path = Path(temporary) / "request.yaml"
        request_path.write_text(
            yaml.safe_dump(
                {
                    "serialized_epoch": serialized_epoch,
                    "loaded_hugin_generation": native.generation,
                }
            )
        )
        return _native_operation(
            [
                "--game-dir",
                str(Path(game_dir).resolve()),
                "--validate-epoch",
                str(request_path),
            ]
        )


def prepare_native_rink_map(game_dir: Path) -> dict[str, Any]:
    """Adapt native fitted evidence and viewing rays without refitting in Python."""
    import base64
    import io

    import numpy as np

    native = shared_native_calibration()
    if native is None or Path(game_dir).resolve() != native.game_directory:
        raise ValueError("Native rink map requires this game's calibration snapshot")
    with tempfile.TemporaryDirectory(prefix="hm-rink-map-") as temporary:
        request_path = Path(temporary) / "request.yaml"
        request_path.write_text(
            yaml.safe_dump(
                {
                    "loaded_hugin_generation": native.generation,
                    "output_generation": native.output_generation,
                    "width": native.canvas_size[0],
                    "height": native.canvas_size[1],
                }
            )
        )
        exported = _native_operation(
            [
                "--game-dir",
                str(Path(game_dir).resolve()),
                "--rink-map",
                str(request_path),
            ]
        )
    if exported["version"] != 1 or exported["hugin_generation"] != native.generation:
        raise ValueError("Native rink map returned a different calibration")
    columns, rows = int(exported["columns"]), int(exported["rows"])
    if not 2 <= columns <= 512 or not 2 <= rows <= 256:
        raise ValueError("Native rink map projection grid exceeds its bounds")
    rays = np.asarray(exported["rays"], dtype=np.float32).reshape(rows, columns, 3)
    if not np.isfinite(rays).all():
        raise ValueError("Native rink map projection grid contains invalid rays")
    encoded = io.BytesIO()
    np.savez_compressed(encoded, rays=rays)
    size = [exported["width"], exported["height"]]
    return {
        "version": 2,
        "native": True,
        "coordinate_system": "corner-origin-metres",
        "geometry_revision": native.generation,
        "image_size": size,
        "rink_size": [exported["layout"]["length"], exported["layout"]["width"]],
        "layout": exported["layout"],
        "fit": exported["fit"],
        "goals": [],
        "observations": [],
        "landmarks": [],
        "projection": {
            "encoding": "hugin-unit-rays-v1",
            "native_bounds": True,
            "data": base64.b64encode(encoded.getvalue()).decode("ascii"),
            "image_size": size,
            "rotation_degrees": exported["rotation_degrees"],
        },
    }
