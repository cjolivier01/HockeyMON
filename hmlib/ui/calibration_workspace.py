"""Game-local calibration editors shared by the desktop workspace and jobs.

Edits validate the effective configuration but publish only touched private leaves.
Geometry edits invalidate derived annotations, while retaining the last complete
mapping generation until the existing calibrator can replace it transactionally.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from hmlib.config import (
    ROOT_DIR,
    load_yaml_preserving_player_frame_context,
    normalize_runtime_config,
    recursive_update,
    resolve_global_refs,
)
from hmlib.stitching.artifacts import artifact_stage, publish_artifacts, stitching_lock

_MAX_DOCUMENT = 16 * 1024 * 1024
_GEOMETRY_LEAVES = (
    "stitching.control_points",
    "rink.scoreboard.perspective_polygon",
    "rink.ice_contours_mask_count",
    "rink.ice_contours_mask_centroid",
    "rink.ice_contours_combined_bbox",
    "rink.ice_contours_geometry_revision",
    "rink.ice_contours_mask_file_prefix",
    "rink.workspace_map",
)
_MASK_NAME = re.compile(r"rink_mask_(?:[0-9a-f]{32}_)?\d+\.png")
_PTO_TOKEN = re.compile(r'(?:[^\s"]|"(?:\\.|[^"\\])*")+')


def _read(path: Path) -> bytes | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_DOCUMENT:
        raise ValueError(f"Expected a bounded regular document: {path}")
    return path.read_bytes()


def _yaml(path: Path) -> dict:
    data = _read(path)
    value = load_yaml_preserving_player_frame_context(data) if data else {}
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return value


def _get(config: Mapping, key: str, default=None):
    value = config
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return copy.deepcopy(value)


def _set(config: dict, key: str, value) -> None:
    parts = key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise TypeError(f"Cannot edit {key}: {part} is not a mapping")
        current = current[part]
    current[parts[-1]] = copy.deepcopy(value)


def _delete(config: dict, key: str) -> None:
    parts = key.split(".")
    current = config
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def _number(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{description} must be a finite number")
    return float(value)


def _point(value, description: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{description} must contain x and y")
    return [_number(item, description) for item in value]


def _image(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular calibration image: {path}")
    with Image.open(path) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > 256 * 1024 * 1024:
            raise ValueError(f"Calibration image exceeds the decode limit: {path}")
    return {"path": str(path), "width": width, "height": height}


def _bounded_point(value, size: Sequence[int], description: str) -> list[float]:
    point = _point(value, description)
    if not (0 <= point[0] < size[0] and 0 <= point[1] < size[1]):
        raise ValueError(f"{description} is outside the {size[0]} × {size[1]} image")
    return point


def _polygon(points, size: Sequence[int], description: str) -> list[list[float]]:
    import cv2

    if not isinstance(points, (tuple, list)) or len(points) != 4:
        raise ValueError(f"{description} requires exactly four corners")
    points = [_bounded_point(point, size, description) for point in points]
    array = np.asarray(points, dtype=np.float32)
    hull = cv2.convexHull(array).reshape(-1, 2)
    if len(hull) != 4 or abs(cv2.contourArea(hull)) < 1:
        raise ValueError(f"{description} must be a nondegenerate convex quadrilateral")
    # Top-left first, then clockwise. A hull avoids duplicated corners in the
    # old sum/difference ordering for a diamond-shaped selection.
    center = hull.mean(axis=0)
    hull = hull[np.argsort(np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0]))]
    hull = np.roll(hull, -int(np.argmin(hull.sum(axis=1))), axis=0)
    return hull.astype(float).tolist()


class CalibrationWorkspace:
    """Validated, explicitly rooted setup/editor operations for one existing game."""

    def __init__(self, game_dir: str | Path):
        self.game_dir = Path(game_dir).expanduser().resolve(strict=True)
        if not self.game_dir.is_dir():
            raise ValueError("The game directory must be a directory")
        self.config_path = self.game_dir / "config.yaml"

    @contextmanager
    def _mutation(self):
        path = self.game_dir / ".hm-workspace-config.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with stitching_lock(self.game_dir, blocking=False):
                yield

    def _effective(self, private: dict | None = None) -> dict:
        root = Path(ROOT_DIR) / "config"
        public = _yaml((root / "games" / f"{self.game_dir.name}.yaml").resolve())
        game = recursive_update(
            public, copy.deepcopy(private if private is not None else _yaml(self.config_path))
        )
        camera = game.get("camera")
        if isinstance(camera, dict):
            camera = camera.get("name")
        rink = _get(game, "game.rink")
        config = _yaml((root / "baseline.yaml").resolve(strict=True))
        for folder, name in (("camera", camera), ("rinks", rink)):
            if name:
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"Invalid {folder} profile name")
                config = recursive_update(
                    config,
                    _yaml(
                        (
                            root / folder / f"{name.lower() if folder == 'camera' else name}.yaml"
                        ).resolve()
                    ),
                )
        return resolve_global_refs(normalize_runtime_config(recursive_update(config, game)))

    def _revision(self) -> str:
        digest = hashlib.sha256()
        for name in (
            "config.yaml",
            ".stitching_artifacts.json",
            "hm_project.pto",
            "autooptimiser_out.pto",
            "left_calibration.json",
        ):
            digest.update(name.encode())
            digest.update(_read(self.game_dir / name) or b"<absent>")
        for name in ("left.png", "right.png", "s.png", "panorama.tif"):
            path = self.game_dir / name
            digest.update(name.encode())
            if path.exists():
                stat = path.stat()
                digest.update(
                    str(
                        (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                    ).encode()
                )
        source = self._scoreboard_source()
        if source is not None:
            from hmlib.ui.workspace.media import Media

            digest.update(Media.identity(source).encode())
        return digest.hexdigest()

    def _scoreboard_source(self) -> Path | None:
        sources = _get(self._effective(), "game.videos.source")
        if sources is None or sources == []:
            return None
        if not isinstance(sources, list) or not all(isinstance(path, str) for path in sources):
            raise ValueError("Single-source recordings must be a list of paths")
        return (self.game_dir / sources[0]).resolve(strict=True)

    def _scoreboard_reference(self) -> dict:
        source = self._scoreboard_source()
        if source is None:
            return self._panorama()
        from hmlib.ui.workspace.media import Media

        media = Media(self.game_dir / ".hm-workspace" / "scoreboard-reference")
        info = media.info(source)
        frame = media.frame(source)
        return {
            "image": frame["path"],
            "width": info["width"],
            "height": info["height"],
            "source": str(source),
        }

    def propose_scoreboard(self, *, expected_revision: str | None = None) -> dict:
        """Return an initial draft, leaving acceptance and persistence to the operator."""
        from hmlib.scoreboard.proposal import propose

        self._check_revision(expected_revision)
        revision = self._revision()
        saved = _get(self._effective(), "rink.scoreboard.perspective_polygon")
        if saved is not None:
            return {
                "revision": revision,
                "points": saved,
                "message": "A saved scoreboard selection already exists",
            }
        reference = self._scoreboard_reference()
        if not reference["image"]:
            raise ValueError(reference["unavailable"])
        with Image.open(reference["image"]) as image:
            points = propose(image, (reference["width"], reference["height"]))
        self._check_revision(revision)
        return {
            "revision": revision,
            "points": points or [],
            "message": (
                "Review the suggested corners before saving"
                if points
                else "No confident scoreboard found; select four corners manually"
            ),
        }

    def _require_current_geometry(self, *, panorama: bool = True) -> None:
        from dataclasses import replace

        from hmlib.stitching.akaze import load_lens_calibration
        from hmlib.stitching.configure_stitching import _file_provenance, _reference_images_match
        from hmlib.stitching.settings import read_stitching_settings

        payload = _read(self.game_dir / ".stitching_artifacts.json")
        if payload is None:
            raise ValueError("Calibrate this game before editing its reference geometry")
        manifest = json.loads(payload)
        settings = read_stitching_settings(self._effective())
        if (
            settings.control_point_matcher == "akaze-hamming"
            and (self.game_dir / "left_calibration.json").exists()
        ):
            settings = replace(
                settings, lens_profile_fingerprint=load_lens_calibration(self.game_dir).fingerprint
            )
        if "source_videos" in manifest:
            sources = json.loads(manifest["source_videos"])
            if manifest["source_videos"] != _file_provenance(
                [source["path"] for source in sources]
            ):
                raise ValueError("Source recordings changed; recalibrate before editing geometry")
        persisted = json.loads(manifest.get("calibration_settings", "{}"))
        # Imported automatic values are a generation result; its selected policy
        # and source identities still have to match the current configuration.
        for attribute, enabled in (
            ("horizontal_fov", settings.camera_auto_fov),
            ("vertical_fov", settings.camera_auto_fov),
        ):
            if enabled and attribute in persisted:
                settings = replace(settings, **{attribute: persisted[attribute]})
        if settings.camera_auto_lens and "input_lens" in persisted:
            from hmlib.stitching.input_lens import read_input_lens

            settings = replace(settings, input_lens=read_input_lens(persisted["input_lens"]))
        if any(
            manifest.get(key) != value for key, value in settings.manifest().items()
        ) or not _reference_images_match(self.game_dir, manifest):
            raise ValueError(
                "Calibration settings or source images changed; recalibrate before editing geometry"
            )
        if panorama:
            project, optimized = (
                self.game_dir / "hm_project.pto",
                self.game_dir / "autooptimiser_out.pto",
            )
            if not optimized.exists() or project.stat().st_mtime_ns > optimized.stat().st_mtime_ns:
                raise ValueError(
                    "Control-point edits await calibration; refresh the panorama before annotating it"
                )

    def _geometry_revision(self) -> str:
        from hmlib.stitching.settings import read_stitching_settings

        digest = hashlib.sha256()
        config = self._effective()
        geometry = {
            "settings": read_stitching_settings(config).manifest(),
            "post_stitch_rotate_degrees": _get(config, "stitching.post_stitch_rotate_degrees") or 0,
            "dimensions": {
                key: value for key, value in self._panorama().items() if key in ("width", "height")
            },
        }
        digest.update(json.dumps(geometry, sort_keys=True).encode())
        for name in (".stitching_artifacts.json", "autooptimiser_out.pto"):
            digest.update(name.encode())
            digest.update(_read(self.game_dir / name) or b"<absent>")
        return digest.hexdigest()

    def _check_revision(self, expected: str | None) -> None:
        if expected is not None and expected != self._revision():
            raise ValueError("The game or calibration changed; reload before applying this edit")

    def _catalog(self, config: dict) -> list[dict]:
        from hmlib.stitching.control_points import CONTROL_POINT_MATCHERS
        from hmlib.stitching.settings import MAPPING_BACKENDS, PROJECTION_PARAMETERS, PROJECTIONS

        def field(key, label, kind, **extra):
            return {
                "key": f"stitching.{key}",
                "label": label,
                "kind": kind,
                "apply_mode": "next-calibration",
                **extra,
            }

        fields = [
            field(
                "sync_method",
                "Synchronization",
                "choice",
                choices=["audio", "imu", "auto", "manual"],
            ),
            field("mapping_backend", "Mapping backend", "choice", choices=list(MAPPING_BACKENDS)),
            field(
                "control_point_matcher",
                "Control-point matcher",
                "choice",
                choices=list(CONTROL_POINT_MATCHERS),
            ),
            field("max_control_points", "Control points", "int", min=4, max=100000),
            field("calibration_frame_count", "Calibration frames", "int", min=1, max=64),
            field("stitch_frame_time", "Reference time (HH:MM:SS.mmm)", "text"),
            field("run_autooptimizer", "Run panorama optimizer", "bool"),
            field(
                "camera_config",
                "Camera FOV profile",
                "choice",
                choices=list(_get(config, "stitching.camera_configs", {})),
            ),
            field(
                "camera_fov.horizontal_fov", "Camera horizontal FOV", "number", min=0.01, max=359.99
            ),
            field(
                "camera_fov.vertical_fov",
                "Camera vertical FOV (provenance)",
                "number",
                min=0.01,
                max=180,
            ),
            field("projection", "Panorama projection", "choice", choices=list(PROJECTIONS)),
            field("projection_framing.auto_fov", "Fit panorama to coverage", "bool"),
            field(
                "projection_framing.horizontal_fov",
                "Panorama horizontal FOV",
                "number",
                min=0.01,
                max=360,
            ),
            field("projection_framing.auto_canvas", "Automatic canvas", "bool"),
            field("projection_framing.auto_crop", "Crop valid pixels automatically", "bool"),
            field(
                "max_output_width",
                "Maximum stitched width (0 = unbounded)",
                "int",
                min=0,
                max=65534,
            ),
            field(
                "max_output_dimension",
                "Maximum canvas dimension (0 = unbounded)",
                "int",
                min=0,
                max=65534,
            ),
            field(
                "rink_config",
                "Rink profile",
                "choice",
                choices=["", *list(_get(config, "stitching.rink_configs", {}))],
            ),
            field(
                "projection_framing.rotation_degrees",
                "Yaw, pitch, roll (degrees)",
                "array",
                length=3,
            ),
            field("projection_framing.crop", "Crop: left, right, top, bottom", "array", length=4),
            field(
                "frame_offsets.left",
                "Left synchronization offset (frames)",
                "number",
                min=0,
                max=100000000,
            ),
            field(
                "frame_offsets.right",
                "Right synchronization offset (frames)",
                "number",
                min=0,
                max=100000000,
            ),
        ]
        fields.extend(
            [
                field(
                    "rink_mask_frame_time",
                    "Rink mask reference time (auto or signed time)",
                    "text",
                    apply_mode="next-run",
                ),
                field(
                    "control_point_resolution",
                    "SuperPoint image size",
                    "choice",
                    choices=["2k", "1k", "native"],
                ),
                field(
                    "control_point_filtering.epipolar_ransac", "Epipolar outlier rejection", "bool"
                ),
                field(
                    "control_point_filtering.homography_consistency",
                    "Homography consistency",
                    "bool",
                ),
                field("camera_automation.fov", "Import camera FOV from original video", "bool"),
                field(
                    "camera_automation.lens", "Import lens correction from original video", "bool"
                ),
                field(
                    "input_lens.projection",
                    "Source lens projection",
                    "choice",
                    choices=["rectilinear", "full-frame-fisheye"],
                ),
                field(
                    "input_lens.optimization",
                    "Lens optimization",
                    "choice",
                    choices=["auto", "fixed", "refine"],
                ),
                field("input_lens.distortion", "Lens distortion a, b, c", "array", length=3),
                field(
                    "input_lens.center_offset",
                    "Optical center offsets (image fractions)",
                    "array",
                    length=2,
                ),
            ]
        )
        for key, label in (
            ("enabled", "Enable rink-aware camera"),
            ("anticipate", "Anticipate play near rink boundaries"),
            ("frame_goals", "Keep goals in view"),
            ("adaptive_zoom", "Adapt zoom to rink position"),
            ("show_map", "Show rink mini-map"),
        ):
            fields.append(
                {
                    "key": f"rink.camera.rink_framing.{key}",
                    "label": label,
                    "kind": "bool",
                    "apply_mode": "next-run",
                }
            )
        for projection, definitions in PROJECTION_PARAMETERS.items():
            fields.append(
                field(
                    f"projection_parameters.{projection}",
                    f"{projection} parameters",
                    "array",
                    length=len(definitions),
                )
            )
        return fields

    def _settings(self, config: dict, catalog: list[dict]) -> dict:
        from hmlib.stitching.settings import read_stitching_settings

        settings = read_stitching_settings(config)
        values = {field["key"]: _get(config, field["key"]) for field in catalog}
        values.update(
            {
                "stitching.camera_fov.horizontal_fov": settings.horizontal_fov,
                "stitching.camera_fov.vertical_fov": settings.vertical_fov,
                "stitching.camera_automation.fov": settings.camera_auto_fov,
                "stitching.camera_automation.lens": settings.camera_auto_lens,
                "stitching.input_lens.projection": settings.input_lens.projection,
                "stitching.input_lens.optimization": settings.input_lens.optimization,
                "stitching.input_lens.distortion": list(settings.input_lens.distortion),
                "stitching.input_lens.center_offset": list(settings.input_lens.center_offset),
                "stitching.control_point_resolution": settings.control_point_resolution,
                "stitching.control_point_filtering.epipolar_ransac": settings.control_point_filtering.epipolar_ransac,
                "stitching.control_point_filtering.homography_consistency": settings.control_point_filtering.homography_consistency,
                "rink.camera.rink_framing.anticipate": _get(
                    config, "rink.camera.rink_framing.anticipate", True
                ),
                "stitching.projection_framing.rotation_degrees": list(
                    settings.framing.rotation_degrees
                ),
                "stitching.projection_framing.crop": list(settings.framing.crop),
                "stitching.max_output_dimension": settings.max_output_dimension or 0,
                "stitching.max_output_width": settings.max_output_width or 0,
                "stitching.max_control_points": settings.max_control_points,
                "stitching.rink_config": _get(config, "stitching.rink_config") or "",
            }
        )
        from hmlib.stitching.rink_mask_sampling import read_mask_frame_time

        values["stitching.rink_mask_frame_time"] = read_mask_frame_time(config)
        return values

    def _panorama(self) -> dict:
        for name in ("s.png", "panorama.tif"):
            path = self.game_dir / name
            if path.exists():
                image = _image(path)
                return {"image": image["path"], "width": image["width"], "height": image["height"]}
        return {
            "image": None,
            "width": 0,
            "height": 0,
            "unavailable": "Calibrate this game to create a stitched reference image",
        }

    def snapshot(self) -> dict:
        with stitching_lock(self.game_dir, blocking=False):
            config = self._effective()
            catalog = self._catalog(config)
            from hmlib.stitching.rink_mask_sampling import read_mask_frame_time

            mask_selection = _get(config, "stitching.rink_mask_frame_time")
            rink = _get(config, "stitching.rink_config")
            profiles = _get(config, "stitching.rink_configs", {}) or {}
            rink_time = (profiles.get(rink) or {}).get("rink_mask_frame_time")
            return {
                "revision": self._revision(),
                "geometry_revision": self._geometry_revision(),
                "settings": self._settings(config, catalog),
                "catalog": catalog,
                "rink_mask_frame_time": {
                    "selection": mask_selection,
                    "effective": read_mask_frame_time(config),
                    "source": (
                        "custom"
                        if mask_selection is not None
                        else "rink" if rink_time else "default"
                    ),
                },
                "artifacts": {
                    name: str(self.game_dir / name)
                    for name in (
                        "hm_project.pto",
                        "autooptimiser_out.pto",
                        "panorama.tif",
                        "left.png",
                        "right.png",
                    )
                    if (self.game_dir / name).is_file()
                },
                "scoreboard": {
                    **self._scoreboard_reference(),
                    "points": _get(config, "rink.scoreboard.perspective_polygon"),
                },
                "control_points": self.read_control_points(),
                "rink_map": self.read_rink_map(),
                "notes": [
                    "Setup edits take effect on the next calibration.",
                    "KB4 lens calibration uses paired left_calibration.json with AKAZE and an OpenCV backend.",
                    "IMU synchronization reads GoPro GPMF or Insta360 gyro data; Auto reports failures before using audio. Manual offsets are source frames.",
                ],
            }

    def stage_settings(
        self, changes: Mapping[str, Any], *, expected_revision: str | None = None
    ) -> dict:
        """Return a validated reviewable draft without modifying any file."""
        with stitching_lock(self.game_dir, blocking=False):
            self._check_revision(expected_revision)
            private = _yaml(self.config_path)
            effective = self._effective(private)
            if not isinstance(changes, Mapping) or not changes:
                raise ValueError("Settings changes must be a nonempty mapping")
            changes = copy.deepcopy(dict(changes))
            catalog = {field["key"]: field for field in self._catalog(effective)}
            for key, value in changes.items():
                if key not in catalog:
                    raise ValueError(f"Unsupported calibration setting: {key}")
                spec = catalog[key]
                if spec["kind"] == "bool" and not isinstance(value, bool):
                    raise ValueError(f"{key} must be true or false")
                if spec["kind"] in ("int", "number"):
                    number = _number(value, key)
                    if spec["kind"] == "int" and not number.is_integer():
                        raise ValueError(f"{key} must be an integer")
                    if not spec["min"] <= number <= spec["max"]:
                        raise ValueError(f"{key} must be between {spec['min']} and {spec['max']}")
                if spec["kind"] == "choice" and value not in spec["choices"]:
                    raise ValueError(f"Unsupported {key}: {value!r}")
                if spec["kind"] == "array":
                    if not isinstance(value, (tuple, list)) or len(value) != spec["length"]:
                        raise ValueError(f"{key} requires {spec['length']} values")
                    for item in value:
                        _number(item, key)
                if key == "stitching.stitch_frame_time" and (
                    not isinstance(value, str)
                    or not re.fullmatch(r"\d{2,}:[0-5]\d:[0-5]\d(?:\.\d{1,3})?", value)
                ):
                    raise ValueError("Reference time must be HH:MM:SS or HH:MM:SS.mmm")
                if key == "stitching.rink_mask_frame_time" and value is not None:
                    from hmlib.stitching.rink_mask_sampling import read_mask_frame_time

                    value = read_mask_frame_time({"stitching": {"rink_mask_frame_time": value}})
                    changes[key] = value
                _set(private, key, value)
            from hmlib.stitching.settings import read_stitching_settings

            candidate = self._effective(private)
            from hmlib.stitching.rink_mask_sampling import read_mask_frame_time

            read_mask_frame_time(candidate)
            read_stitching_settings(candidate)
            if _get(candidate, "stitching.sync_method") == "manual":
                for side in ("left", "right"):
                    if _get(candidate, f"stitching.frame_offsets.{side}") is None:
                        raise ValueError(
                            "Manual synchronization requires both camera frame offsets"
                        )
            guide = _get(candidate, "rink.camera.rink_framing", {})
            if guide.get("enabled") and not any(
                guide.get(key, key == "anticipate")
                for key in ("anticipate", "frame_goals", "adaptive_zoom")
            ):
                raise ValueError(
                    "Enable at least one rink camera behavior before enabling rink framing"
                )
            if guide.get("enabled") or guide.get("show_map"):
                mapping = _get(candidate, "rink.workspace_map")
                if not mapping or mapping.get("geometry_revision") != self._geometry_revision():
                    raise ValueError(
                        "Rink framing requires a reviewed map for the current calibration"
                    )
                if any(key.startswith("stitching.") for key in changes):
                    raise ValueError(
                        "Disable rink framing before changing calibration, then review its map"
                    )
            return {
                "revision": self._revision(),
                "changes": copy.deepcopy(dict(changes)),
                "apply_mode": (
                    "next-run"
                    if set(changes) == {"stitching.rink_mask_frame_time"}
                    else "next-calibration"
                ),
            }

    def _publish(
        self,
        private: dict,
        files: dict[str, bytes] | None = None,
        *,
        geometry: bool = False,
        mask_geometry: bool = False,
    ) -> None:
        if geometry or mask_geometry:
            for key in _GEOMETRY_LEAVES:
                if mask_geometry and not geometry and key == "stitching.control_points":
                    continue
                _delete(private, key)
        payloads = {
            "config.yaml": yaml.safe_dump(private, sort_keys=False).encode(),
            **(files or {}),
        }
        expected = {name: _read(self.game_dir / name) for name in payloads}
        with artifact_stage(self.game_dir) as stage:
            for name, data in payloads.items():
                target = stage / name
                target.write_bytes(data)
                old = self.game_dir / name
                if old.exists():
                    target.chmod(old.stat().st_mode & 0o777)
            publish_artifacts(self.game_dir, stage, list(payloads), expected_old_contents=expected)
        if geometry or mask_geometry:
            # These are cheap derived masks, never published output recordings.
            for path in self.game_dir.iterdir():
                if _MASK_NAME.fullmatch(path.name) or (geometry and path.name == "xor_file.png"):
                    path.unlink()

    def apply(self, draft: Mapping[str, Any]) -> dict:
        with self._mutation():
            checked = self.stage_settings(draft["changes"], expected_revision=draft["revision"])
            private = _yaml(self.config_path)
            effective = self._effective(private)
            changes = {
                key: value
                for key, value in checked["changes"].items()
                if _get(effective, key) != value
            }
            if changes:
                for key, value in changes.items():
                    _set(private, key, value)
                if (
                    "stitching.sync_method" in changes
                    and changes["stitching.sync_method"] != "manual"
                ):
                    _set(private, "stitching.frame_offsets", None)
                self._publish(
                    private,
                    geometry=any(
                        key.startswith("stitching.") and key != "stitching.rink_mask_frame_time"
                        for key in changes
                    ),
                    mask_geometry="stitching.rink_mask_frame_time" in changes,
                )
        return self.snapshot()

    def update_settings(
        self, changes: Mapping[str, Any], *, expected_revision: str | None = None
    ) -> dict:
        return self.apply(self.stage_settings(changes, expected_revision=expected_revision))

    def set_crop(
        self,
        crop: Sequence[float],
        *,
        auto_crop: bool = False,
        expected_revision: str | None = None,
    ) -> dict:
        return self.update_settings(
            {
                "stitching.projection_framing.crop": list(crop),
                "stitching.projection_framing.auto_crop": auto_crop,
            },
            expected_revision=expected_revision,
        )

    def set_leveling(
        self, rotation_degrees: Sequence[float], *, expected_revision: str | None = None
    ) -> dict:
        return self.update_settings(
            {"stitching.projection_framing.rotation_degrees": list(rotation_degrees)},
            expected_revision=expected_revision,
        )

    def set_scoreboard(self, points: Sequence, *, expected_revision: str | None = None) -> dict:
        with self._mutation():
            self._check_revision(expected_revision)
            if not isinstance(points, (list, tuple)):
                raise TypeError("Scoreboard points must be a list of corners")
            if not points:
                selected = [[0, 0] for _ in range(4)]
            else:
                if self._scoreboard_source() is None:
                    self._require_current_geometry()
                image = self._scoreboard_reference()
                if image["image"] is None:
                    raise ValueError(image["unavailable"])
                selected = _polygon(points, [image["width"], image["height"]], "Scoreboard")
                selected = [[round(x), round(y)] for x, y in selected]
                _polygon(selected, [image["width"], image["height"]], "Rounded scoreboard")
            private = _yaml(self.config_path)
            _set(private, "rink.scoreboard.perspective_polygon", selected)
            self._publish(private)
        return self.snapshot()

    def read_control_points(self) -> dict:
        from hmlib.stitching.hugin import parse_hugin_control_points

        path = self.game_dir / "hm_project.pto"
        if not path.exists():
            return {
                "pairs": [],
                "images": [],
                "unavailable": "Calibrate this game to create a control-point project",
            }
        payload = (_read(path) or b"").decode()
        images = []
        for line in payload.splitlines():
            if line.lstrip().startswith(("i ", "i\t")):
                tokens = [token.group() for token in _PTO_TOKEN.finditer(line)]
                names = [shlex.split(token)[0][1:] for token in tokens[1:] if token.startswith("n")]
                if len(names) != 1:
                    raise ValueError("Each PTO camera requires one source image")
                source = (self.game_dir / names[0]).resolve(strict=True)
                if source.parent != self.game_dir:
                    raise ValueError(
                        "Editable control-point sources must be in the selected game directory"
                    )
                images.append(_image(source))
        if len(images) != 2:
            raise ValueError("Manual control points require exactly two calibrated cameras")
        matched = parse_hugin_control_points(payload.splitlines())
        pairs = (
            []
            if matched is None
            else [
                list(left) + list(right)
                for left, right in zip(matched[0].tolist(), matched[1].tolist(), strict=True)
            ]
        )
        return {"pairs": pairs, "images": images, "project": str(path)}

    def save_control_points(self, pairs: Sequence, *, expected_revision: str | None = None) -> dict:
        from hmlib.stitching.hugin import remove_control_points

        with self._mutation():
            self._check_revision(expected_revision)
            self._require_current_geometry(panorama=False)
            current = self.read_control_points()
            if len(current["images"]) != 2:
                raise ValueError(current["unavailable"])
            if not isinstance(pairs, (list, tuple)) or not 4 <= len(pairs) <= 100000:
                raise ValueError(
                    "Manual calibration requires between four and 100000 paired points"
                )
            normalized = []
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 4:
                    raise ValueError("Each match requires x0,y0,x1,y1")
                left, right = current["images"]
                normalized.append(
                    _bounded_point(pair[:2], [left["width"], left["height"]], "Left control point")
                    + _bounded_point(
                        pair[2:], [right["width"], right["height"]], "Right control point"
                    )
                )
            for offset in (0, 2):
                points = np.asarray(normalized)[:, offset : offset + 2]
                if np.linalg.matrix_rank(np.column_stack((points, np.ones(len(points))))) < 3:
                    raise ValueError(
                        "Control points must span a two-dimensional region in both cameras"
                    )
            # Preserve provenance: the runtime recognizes a newer edited PTO,
            # rebuilds its maps and consumes these points instead of rematching.
            path = self.game_dir / "hm_project.pto"
            lines, _ = remove_control_points((_read(path) or b"").decode().splitlines())
            lines += ["# control points"] + [
                f"c n0 N1 x{pair[0]:.12g} y{pair[1]:.12g} X{pair[2]:.12g} Y{pair[3]:.12g} t0"
                for pair in normalized
            ]
            self._publish(
                _yaml(self.config_path),
                {path.name: ("\n".join(lines) + "\n").encode()},
                geometry=True,
            )
            optimized = self.game_dir / "autooptimiser_out.pto"
            if optimized.exists() and path.stat().st_mtime_ns <= optimized.stat().st_mtime_ns:
                os.utime(path, ns=(path.stat().st_atime_ns, optimized.stat().st_mtime_ns + 1000000))
        return self.snapshot()

    def read_rink_map(self) -> dict:
        saved = _get(self._effective(), "rink.workspace_map")
        result = {
            **self._panorama(),
            "landmarks": [],
            "goals": [],
            "rink_size": [60.96, 25.908],
            "mapping": saved,
        }
        if saved:
            result.update(
                {
                    key: copy.deepcopy(saved[key])
                    for key in (
                        "landmarks",
                        "goals",
                        "rink_size",
                        "homography",
                        "reprojection_error",
                    )
                }
            )
            result["stale"] = saved.get("geometry_revision") != self._geometry_revision()
        return result

    def save_rink_map(
        self,
        landmarks: Sequence,
        goals: Sequence | None = None,
        *,
        rink_size: Sequence[float] = (60.96, 25.908),
        expected_revision: str | None = None,
    ) -> dict:
        import cv2

        with self._mutation():
            self._check_revision(expected_revision)
            self._require_current_geometry()
            image = self._panorama()
            if image["image"] is None:
                raise ValueError(image["unavailable"])
            size = [image["width"], image["height"]]
            rink_size = _point(rink_size, "Rink dimensions")
            if any(not 0 < value <= 1000 for value in rink_size):
                raise ValueError("Rink dimensions must be positive and at most 1000")
            if not isinstance(landmarks, (list, tuple)) or not 4 <= len(landmarks) <= 512:
                raise ValueError("A rink map requires between four and 512 paired landmarks")
            normalized = []
            for item in landmarks:
                if not isinstance(item, dict) or set(item) != {"image", "rink"}:
                    raise ValueError("Each landmark requires image and rink coordinates")
                destination = _point(item["rink"], "Rink landmark")
                if any(not 0 <= value <= limit for value, limit in zip(destination, rink_size)):
                    raise ValueError("Rink landmark is outside the rink dimensions")
                normalized.append(
                    {
                        "image": _bounded_point(item["image"], size, "Image landmark"),
                        "rink": destination,
                    }
                )
            source = np.asarray([item["image"] for item in normalized], dtype=np.float64)
            target = np.asarray([item["rink"] for item in normalized], dtype=np.float64)
            matrix, _ = cv2.findHomography(source, target, method=0)
            if (
                matrix is None
                or not np.isfinite(matrix).all()
                or abs(np.linalg.det(matrix)) < 1e-12
            ):
                raise ValueError("Landmarks do not determine an invertible rink projection")
            projected = cv2.perspectiveTransform(source[None], matrix)[0]
            error = float(np.sqrt(np.mean(np.sum((projected - target) ** 2, axis=1))))
            if error > max(rink_size) * 0.05:
                raise ValueError(
                    f"Rink landmarks disagree (RMS error {error:.2f}); correct the marked pairs"
                )
            if goals is None:
                goals = []
            if not isinstance(goals, (tuple, list)) or len(goals) > 2:
                raise ValueError("A rink map supports zero, one or two goal polygons")
            normalized_goals = [_polygon(goal, size, "Goal") for goal in goals]
            mapping = {
                "version": 1,
                "coordinate_system": "corner-origin-metres",
                "geometry_revision": self._geometry_revision(),
                "image_size": size,
                "rink_size": rink_size,
                "landmarks": normalized,
                "goals": normalized_goals,
                "homography": matrix.tolist(),
                "reprojection_error": error,
            }
            private = _yaml(self.config_path)
            _set(private, "rink.workspace_map", mapping)
            self._publish(private)
        return self.snapshot()

    def set_lens_calibration(
        self, calibration: Mapping, *, expected_revision: str | None = None
    ) -> dict:
        from hmlib.stitching.akaze import _parse_camera
        from hmlib.stitching.settings import read_stitching_settings

        with self._mutation():
            self._check_revision(expected_revision)
            settings = read_stitching_settings(self._effective())
            if (
                settings.control_point_matcher != "akaze-hamming"
                or settings.mapping_backend == "nona"
            ):
                raise ValueError("KB4 lenses require AKAZE and an OpenCV mapping backend")
            for key in ("left_uniforms", "right_uniforms"):
                _parse_camera(calibration, key)
            self._publish(
                _yaml(self.config_path),
                {
                    "left_calibration.json": json.dumps(
                        calibration, allow_nan=False, indent=2
                    ).encode()
                },
                geometry=True,
            )
        return self.snapshot()

    def reset_synchronization(self, *, expected_revision: str | None = None) -> dict:
        """Request a fresh audio correlation on the next calibration."""
        with self._mutation():
            self._check_revision(expected_revision)
            private = _yaml(self.config_path)
            _set(private, "stitching.frame_offsets", None)
            _set(private, "stitching.sync_method", "audio")
            self._publish(private, geometry=True)
        return self.snapshot()

    def leveling_session(self):
        """Open the real bounded Hugin preview/post estimator with stale-save fencing."""
        from hmlib.stitching.leveling_editor import LevelingSession

        return LevelingSession(self.game_dir, self._effective)


def prepare_rink_map(game_dir: str | Path) -> dict | None:
    """Load and validate the mapping once, before processing any video frames."""
    workspace = CalibrationWorkspace(game_dir)
    with stitching_lock(workspace.game_dir, blocking=False):
        state = workspace.read_rink_map()
        if state["mapping"] is None:
            return None
        if state.get("stale"):
            raise ValueError(
                "The saved rink map belongs to a different calibration; review its landmarks"
            )
        mapping = state["mapping"]
        project_rink_points([[0, 0]], mapping)
        return copy.deepcopy(mapping)


def project_rink_points(
    points: Sequence, mapping: Mapping, *, image_size: Sequence[int] | None = None
) -> list[list[float]]:
    """Project stitched image coordinates without file access or model inference."""
    matrix = np.asarray(mapping["homography"], dtype=np.float64)
    if mapping.get("version") != 1 or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid saved rink projection")
    values = np.asarray([_point(point, "Player point") for point in points], dtype=np.float64)
    if not len(values):
        return []
    if image_size is not None:
        size = np.asarray(_point(image_size, "Stitched image dimensions"))
        if (size <= 0).any():
            raise ValueError("Stitched image dimensions must be positive")
        values = values * np.asarray(mapping["image_size"], dtype=np.float64) / size
    transformed = np.column_stack((values, np.ones(len(values)))) @ matrix.T
    if not np.isfinite(transformed).all() or (np.abs(transformed[:, 2]) < 1e-12).any():
        raise ValueError("A projected point lies on the rink horizon")
    return (transformed[:, :2] / transformed[:, 2:]).tolist()


def project_goal_polygons(mapping: Mapping) -> list[list[list[float]]]:
    return [project_rink_points(goal, mapping) for goal in mapping["goals"]]


def unproject_rink_points(
    points: Sequence, mapping: Mapping, *, image_size: Sequence[int] | None = None
) -> list[list[float]]:
    """Map physical rink metres back into the original or resized stitched image."""
    matrix = np.asarray(mapping["homography"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid saved rink projection")
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("The saved rink projection is singular") from error
    result = project_rink_points(points, {**mapping, "homography": inverse.tolist()})
    if image_size is not None:
        size = np.asarray(_point(image_size, "Stitched image dimensions"))
        if (size <= 0).any():
            raise ValueError("Stitched image dimensions must be positive")
        result = (
            np.asarray(result).reshape(-1, 2) * size / np.asarray(mapping["image_size"])
        ).tolist()
    return result
