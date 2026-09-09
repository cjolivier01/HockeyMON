"""Validated calibration choices shared by the stitching entry points.

Projection definitions follow hstream's StitchingAlgorithms and Hugin/libpano.
Keep requested config separate from resolved generation provenance: in
particular, resolving a rink must not turn inheritance into a game override.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from hmlib.stitching.control_points import normalize_control_point_matcher

MAX_CANVAS_DIMENSION = 65534
MAX_CANVAS_PIXELS = 256 * 1024 * 1024
PROJECTIONS = (
    "rectilinear",
    "cylindrical",
    "equirectangular",
    "full-frame-fisheye",
    "stereographic",
    "mercator",
    "transverse-mercator",
    "sinusoidal",
    "lambert-cylindrical-equal-area",
    "lambert-azimuthal-equal-area",
    "albers-equal-area-conic",
    "miller-cylindrical",
    "panini",
    "architectural",
    "orthographic",
    "equisolid",
    "equirectangular-panini",
    "biplane",
    "triplane",
    "general-panini",
    "thoby",
    "hammer-aitoff",
)
# Parameter tuples: (minimum, maximum, default).
PROJECTION_PARAMETERS = {
    "albers-equal-area-conic": ((-90, 90, 0), (-90, 90, 60)),
    "biplane": ((1, 179, 45), (0, 1, 0)),
    "triplane": ((1, 120, 60),),
    "general-panini": ((0, 150, 100), (-100, 100, 0), (-100, 100, 0)),
}
OPENCV_MAPPING_BACKENDS = ("opencv-magsac", "opencv-affine-ransac")
MAPPING_BACKENDS = ("nona", *OPENCV_MAPPING_BACKENDS)


def normalize_mapping_backend(value: str) -> str:
    name = str(value).strip().lower().replace("_", "-")
    name = {
        "magsac": "opencv-magsac",
        "magsac++": "opencv-magsac",
        "affine-ransac": "opencv-affine-ransac",
        "ransac": "opencv-affine-ransac",
    }.get(name, name)
    if name not in MAPPING_BACKENDS:
        raise ValueError(
            f"Unsupported mapping backend {value!r}; choose {', '.join(MAPPING_BACKENDS)}"
        )
    return name


def normalize_projection(value: str) -> str:
    name = str(value).strip().lower().replace("_", "-")
    name = {
        "planar": "rectilinear",
        "panini-general": "general-panini",
        "panini-generalized": "general-panini",
        "hammer": "hammer-aitoff",
        "fullframe-fisheye": "full-frame-fisheye",
    }.get(name, name)
    if name not in PROJECTIONS:
        raise ValueError(f"Unsupported stitching projection {value!r}")
    return name


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{path} must be a finite number")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _boolean(value: Any, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{path} must be true or false")


def _array(value: Any, size: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{path} must contain {size} numbers")
    return tuple(_number(item, path) for item in value)


def _defaulted(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    value = mapping.get(key)
    return default if value is None else value


def normalize_max_output_dimension(value: Any) -> int | None:
    if value is None:
        return None
    dimension = _number(value, "max_output_dimension")
    if not dimension.is_integer() or not 0 < dimension <= MAX_CANVAS_DIMENSION:
        raise ValueError(f"max_output_dimension must be between 1 and {MAX_CANVAS_DIMENSION}")
    return int(dimension)


def projection_parameters(projection: str, values: Any = None) -> tuple[float, ...]:
    definitions = PROJECTION_PARAMETERS.get(projection, ())
    if values is None:
        return tuple(float(definition[2]) for definition in definitions)
    parameters = _array(values, len(definitions), f"projection_parameters.{projection}")
    for value, (minimum, maximum, _) in zip(parameters, definitions):
        if not minimum <= value <= maximum:
            raise ValueError(f"{projection} parameter must be between {minimum} and {maximum}")
        if abs(value * 100 - round(value * 100)) > 1e-7:
            raise ValueError(
                f"{projection} parameters must use increments of 0.01 (Hugin precision)"
            )
    if projection == "biplane" and parameters[1] not in (0, 1):
        raise ValueError("biplane corners parameter must be exactly 0 or 1")
    return parameters


def maximum_projection_fov(projection: str, parameters: tuple[float, ...]) -> float:
    if projection in ("rectilinear", "transverse-mercator"):
        return 179.0
    if projection in ("stereographic", "panini", "equirectangular-panini"):
        return 359.0
    if projection == "orthographic":
        return 180.0
    if projection == "biplane":
        return min(360.0, parameters[0] + 179.0)
    if projection == "triplane":
        return min(360.0, 2 * parameters[0] + 179.0)
    if projection == "general-panini":
        angle = math.radians(80)
        compression = 1.5 / ((150 - parameters[0]) / 50 + 0.0001) - 1.5 / 3.0001
        half_fov = math.acos(-1 / compression if compression > 1 else -compression)
        argument = compression * math.sin(angle)
        if argument <= 1:
            half_fov = min(half_fov, math.asin(max(-1, argument)) + angle)
        return math.degrees(2 * half_fov)
    return 360.0


@dataclass(frozen=True)
class ProjectionFraming:
    auto_fov: bool = False
    horizontal_fov: float = 180.0
    auto_canvas: bool = True
    auto_crop: bool = False
    rotation_degrees: tuple[float, ...] = (0.0, 0.0, 0.0)
    crop: tuple[float, ...] = (0.0, 1.0, 0.0, 1.0)


@dataclass(frozen=True)
class StitchingSettings:
    control_point_matcher: str
    mapping_backend: str
    projection: str
    parameters: tuple[float, ...]
    framing: ProjectionFraming
    run_autooptimizer: bool
    camera_config: str
    horizontal_fov: float
    vertical_fov: float
    max_output_dimension: int | None = None
    max_output_width: int | None = None

    def manifest(self) -> dict[str, str]:
        """Stable resolved provenance, independent of config spelling/inheritance."""
        return {
            "control_point_matcher": self.control_point_matcher,
            "mapping_backend": self.mapping_backend,
            "max_output_dimension": str(self.max_output_dimension or 0),
            "calibration_settings": json.dumps(asdict(self), sort_keys=True, separators=(",", ":")),
        }


def read_stitching_settings(
    config: Mapping[str, Any] | None = None, **overrides: Any
) -> StitchingSettings:
    """Resolve effective config without mutating it or materializing inherited angles.

    Keyword overrides are explicit legacy API values; None means unspecified.
    Preset definitions come from the bundled baseline when not supplied.
    """
    config = _mapping(config, "config")
    stitch = dict(_mapping(config.get("stitching"), "stitching"))
    stitch.update({key: value for key, value in overrides.items() if value is not None})
    baseline_path = Path(__file__).resolve().parents[1] / "config" / "baseline.yaml"
    with baseline_path.open(encoding="utf-8") as stream:
        baseline = yaml.safe_load(stream)["stitching"]
    matcher = normalize_control_point_matcher(
        _defaulted(stitch, "control_point_matcher", "superpoint-lightglue")
    )
    backend = normalize_mapping_backend(_defaulted(stitch, "mapping_backend", "opencv-magsac"))
    projection = normalize_projection(
        _defaulted(stitch, "projection", "general-panini" if backend == "nona" else "rectilinear")
    )
    if backend != "nona" and projection != "rectilinear":
        raise ValueError(f"{backend} supports only rectilinear output; {projection} requires nona")
    optimizer = _boolean(
        _defaulted(stitch, "run_autooptimizer", False), "stitching.run_autooptimizer"
    )
    if backend == "nona" and not optimizer:
        raise ValueError(
            "NONA requires stitching.run_autooptimizer=true; choose an OpenCV backend to disable optimization"
        )
    parameter_map = _mapping(stitch.get("projection_parameters"), "stitching.projection_parameters")
    parameters = projection_parameters(projection, parameter_map.get(projection))
    definitions = _mapping(
        _defaulted(stitch, "camera_configs", baseline["camera_configs"]), "stitching.camera_configs"
    )
    camera = _defaulted(stitch, "camera_config", baseline["camera_config"])
    if not isinstance(camera, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", camera):
        raise ValueError("stitching.camera_config must be a lowercase kebab-case identifier")
    preset = _mapping(definitions.get(camera), f"stitching.camera_configs.{camera}")
    fov = _mapping(stitch.get("camera_fov"), "stitching.camera_fov")
    if set(fov) - {"horizontal_fov", "vertical_fov"}:
        raise ValueError("stitching.camera_fov contains an unsupported key")
    horizontal = _number(
        _defaulted(fov, "horizontal_fov", preset.get("horizontal_fov")), "camera_fov.horizontal_fov"
    )
    vertical = _number(
        _defaulted(fov, "vertical_fov", preset.get("vertical_fov")), "camera_fov.vertical_fov"
    )
    if not 0 < horizontal < 360 or not 0 < vertical <= 180:
        raise ValueError(
            "Camera FOV must be between 0 and 360 horizontally, and 0 and 180 vertically"
        )
    framing_config = _mapping(stitch.get("projection_framing"), "stitching.projection_framing")
    if set(framing_config) - set(ProjectionFraming.__dataclass_fields__):
        raise ValueError("stitching.projection_framing contains an unsupported key")
    rink = stitch.get("rink_config")
    rotation = (0.0, 0.0, 0.0)
    if rink is not None and rink != "":
        rinks = _mapping(
            _defaulted(stitch, "rink_configs", baseline.get("rink_configs")),
            "stitching.rink_configs",
        )
        if not isinstance(rink, str) or rink not in rinks:
            raise ValueError(f"Unknown stitching.rink_config {rink!r}")
        profile = _mapping(rinks[rink], f"stitching.rink_configs.{rink}")
        rotation = _array(
            profile.get("rotation_degrees"), 3, f"rink_configs.{rink}.rotation_degrees"
        )
    rotation = _array(
        _defaulted(framing_config, "rotation_degrees", rotation),
        3,
        "projection_framing.rotation_degrees",
    )
    if any(abs(angle) > 180 for angle in rotation):
        raise ValueError("projection_framing.rotation_degrees must be between -180 and 180")
    crop = _array(_defaulted(framing_config, "crop", (0, 1, 0, 1)), 4, "projection_framing.crop")
    if not (0 <= crop[0] < crop[1] <= 1 and 0 <= crop[2] < crop[3] <= 1):
        raise ValueError("projection_framing.crop must be normalized left,right,top,bottom bounds")
    framing = ProjectionFraming(
        auto_fov=_boolean(
            _defaulted(framing_config, "auto_fov", False), "projection_framing.auto_fov"
        ),
        horizontal_fov=_number(
            _defaulted(framing_config, "horizontal_fov", 180), "projection_framing.horizontal_fov"
        ),
        auto_canvas=_boolean(
            _defaulted(framing_config, "auto_canvas", True), "projection_framing.auto_canvas"
        ),
        auto_crop=_boolean(
            _defaulted(framing_config, "auto_crop", False), "projection_framing.auto_crop"
        ),
        rotation_degrees=rotation,
        crop=crop,
    )
    if framing.auto_crop and crop != (0, 1, 0, 1):
        raise ValueError("projection_framing.crop and auto_crop are mutually exclusive")
    if not 0 < framing.horizontal_fov <= 360:
        raise ValueError("projection_framing.horizontal_fov must be between 0 and 360")
    if (
        backend == "nona"
        and not framing.auto_fov
        and framing.horizontal_fov > maximum_projection_fov(projection, parameters) + 1e-9
    ):
        raise ValueError(
            f"{projection} horizontal FOV exceeds projection limit {maximum_projection_fov(projection, parameters):.6g}"
        )
    if backend != "nona" and framing != ProjectionFraming():
        raise ValueError(
            "Non-default projection framing (including rink rotation and crop) requires "
            "mapping_backend=nona and run_autooptimizer=true"
        )
    width = normalize_max_output_dimension(stitch.get("max_output_width"))
    return StitchingSettings(
        matcher,
        backend,
        projection,
        parameters,
        framing,
        optimizer,
        camera,
        horizontal,
        vertical,
        normalize_max_output_dimension(stitch.get("max_output_dimension")),
        width,
    )
