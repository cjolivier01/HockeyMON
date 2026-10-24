"""Source camera optics and Hugin optimization policy shared with native HStream."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InputLens:
    projection: str = "rectilinear"
    optimization: str = "auto"
    distortion: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_offset: tuple[float, float] = (0.0, 0.0)


def read_input_lens(value=None, defaults: InputLens | None = None) -> InputLens:
    if defaults is None:
        defaults = InputLens()
    if value is None:
        return defaults
    if not isinstance(value, Mapping) or set(value) - set(InputLens.__dataclass_fields__):
        raise ValueError("stitching.input_lens contains an unsupported key or is not a mapping")
    fields = asdict(defaults)
    fields.update({key: item for key, item in value.items() if item is not None})
    if fields["projection"] not in ("rectilinear", "full-frame-fisheye"):
        raise ValueError("Input lens projection must be rectilinear or full-frame-fisheye")
    if fields["optimization"] not in ("auto", "fixed", "refine"):
        raise ValueError("Input lens optimization must be auto, fixed or refine")
    for key, count, limit in (("distortion", 3, 1.0), ("center_offset", 2, 0.5)):
        array = fields[key]
        if (
            not isinstance(array, (tuple, list))
            or len(array) != count
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or abs(item) > limit
                for item in array
            )
        ):
            raise ValueError(f"Input lens {key} must contain {count} finite values within ±{limit}")
        fields[key] = tuple(map(float, array))
    return InputLens(**fields)


def read_camera_automation(stitch: Mapping, camera: str) -> tuple[bool, bool]:
    value = stitch.get("camera_automation") or {}
    if not isinstance(value, Mapping) or set(value) - {"fov", "lens"}:
        raise ValueError("stitching.camera_automation accepts only fov and lens")
    if any(item is not None and not isinstance(item, bool) for item in value.values()):
        raise ValueError("Camera automation values must be booleans or null")
    # Explicit legacy overrides predate the Auto selector. Preserve them until
    # the operator explicitly enables automatic import for that component.
    fov_override = any(item is not None for item in (stitch.get("camera_fov") or {}).values())
    lens_override = any(item is not None for item in (stitch.get("input_lens") or {}).values())
    return tuple(
        value.get(key) if value.get(key) is not None else camera == "auto" and not overridden
        for key, overridden in (("fov", fov_override), ("lens", lens_override))
    )


def apply_input_lens(pto: str, lens: InputLens) -> str:
    """Apply coefficients, linked source optics and fixed/refinable variables."""
    if lens == InputLens():
        return pto
    lines, dimensions = [], []
    for line in pto.splitlines():
        if lens.optimization != "auto" and (
            line.startswith(("#hugin_optimizerMasterSwitch", "v ")) or line == "v"
        ):
            continue
        if line.startswith("i "):
            size = []
            for key in ("w", "h"):
                match = re.search(rf"\b{key}([0-9]+)\b", line)
                if not match or not 0 < int(match[1]) <= 32768:
                    raise ValueError("Input lens requires valid source dimensions in the PTO")
                size.append(int(match[1]))
            if dimensions and dimensions[0] != size:
                raise ValueError("Shared input lens correction requires equal source dimensions")
            values = {
                "f": 3 if lens.projection == "full-frame-fisheye" else 0,
                **dict(zip("abc", lens.distortion, strict=True)),
                "d": lens.center_offset[0] * size[0],
                "e": lens.center_offset[1] * size[1],
            }
            for key, value in values.items():
                token = f" {key}" + ("=0" if dimensions and key != "f" else format(value, ".17g"))
                pattern = rf" {key}=?[-+0-9.eE]+(?=\s|$)"
                line = re.sub(pattern, token, line) if re.search(pattern, line) else line + token
            dimensions.append(size)
        lines.append(line)
    if len(dimensions) != 2:
        raise ValueError("Input lens correction requires exactly two source images")
    if lens.optimization != "auto":
        variables = "v r0 p0 r1 p1 y1" + (" v0 b0 d0 e0" if lens.optimization == "refine" else "")
        lines.extend(("#hugin_optimizerMasterSwitch 0", variables, "v"))
    return "\n".join(lines) + "\n"
