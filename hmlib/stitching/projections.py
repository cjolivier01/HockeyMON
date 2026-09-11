"""Apply and verify Hugin projection framing before generating mapping TIFFs."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from hmlib.stitching.settings import (
    MAX_CANVAS_DIMENSION,
    MAX_CANVAS_PIXELS,
    PROJECTIONS,
    StitchingSettings,
    validate_output_scale,
)


@dataclass(frozen=True)
class PanoramaGeometry:
    projection: int
    width: int
    height: int
    horizontal_fov: float
    parameters: tuple[float, ...]
    crop: tuple[int, ...]

    @property
    def effective_size(self) -> tuple[int, int]:
        return self.crop[1] - self.crop[0], self.crop[3] - self.crop[2]


def read_panorama_geometry(path: str | Path) -> PanoramaGeometry:
    """Parse panorama tokens without confusing output-format strings with P/S."""
    lines = [
        line
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.startswith("p ")
    ]
    if len(lines) != 1:
        raise ValueError(f"PTO must contain exactly one panorama line: {path}")
    values: dict[str, str] = {}
    for token in shlex.split(lines[0])[1:]:
        if token and token[0] in "fwhvPS":
            key = token[0]
            if key in values:
                raise ValueError(f"PTO contains duplicate panorama {key} token: {path}")
            values[key] = token[1:]
    try:
        projection, width, height = (int(values[key]) for key in ("f", "w", "h"))
        fov = float(values["v"])
        parameters = tuple(float(value) for value in values.get("P", "").split())
        crop = (
            tuple(int(value) for value in values["S"].split(","))
            if "S" in values
            else (0, width, 0, height)
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"PTO has malformed panorama geometry: {path}") from exc
    if width <= 0 or height <= 0 or not math.isfinite(fov) or not 0 < fov <= 360:
        raise ValueError(f"PTO has invalid canvas dimensions or field of view: {path}")
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError(f"PTO has non-finite projection parameters: {path}")
    if len(crop) != 4 or not (0 <= crop[0] < crop[1] <= width and 0 <= crop[2] < crop[3] <= height):
        raise ValueError(f"PTO has invalid crop bounds: {path}")
    return PanoramaGeometry(projection, width, height, fov, parameters, crop)


def _verify_projection(geometry: PanoramaGeometry, settings: StitchingSettings) -> None:
    if geometry.projection != PROJECTIONS.index(settings.projection):
        raise ValueError("pano_modify did not select the requested projection")
    if geometry.parameters != settings.parameters:
        raise ValueError("pano_modify did not preserve requested projection parameters")
    framing = settings.framing
    if not framing.auto_fov and abs(geometry.horizontal_fov - framing.horizontal_fov) > 0.000500001:
        raise ValueError("pano_modify clamped or changed the requested horizontal FOV")
    if not framing.auto_crop:
        expected = tuple(
            value * (geometry.width if index < 2 else geometry.height)
            for index, value in enumerate(framing.crop)
        )
        if any(abs(value - actual) > 1.01 for value, actual in zip(expected, geometry.crop)):
            raise ValueError("pano_modify did not preserve the requested crop")


def validate_canvas_size(width: int, height: int, settings: StitchingSettings) -> None:
    maximum = settings.max_output_dimension or MAX_CANVAS_DIMENSION
    if (
        width <= 0
        or height <= 0
        or max(width, height) > maximum
        or width * height > MAX_CANVAS_PIXELS
    ):
        raise ValueError(f"Stitched canvas {width}x{height} exceeds the permitted canvas limits")
    if settings.max_output_width is not None and width > settings.max_output_width:
        raise ValueError(
            f"Stitched canvas width {width} exceeds max_output_width={settings.max_output_width}"
        )


def _cap_panorama(
    project: Path,
    temporary: Path,
    settings: StitchingSettings,
    run: Callable[[Sequence[str]], None],
    binary: str,
    scale: float | None,
) -> None:
    """Scale the canvas and its crop together, validating Hugin's even rounding."""
    geometry = read_panorama_geometry(project)
    # Limit the full PTO canvas as well as the emitted crop: nona may allocate
    # working buffers for the full projection before producing cropped images.
    maximum = settings.max_output_dimension or MAX_CANVAS_DIMENSION
    ratio = min(
        1.0 if scale is None else scale,
        maximum / geometry.width,
        maximum / geometry.height,
        math.sqrt(MAX_CANVAS_PIXELS / (geometry.width * geometry.height)),
    )
    if settings.max_output_width is not None:
        ratio = min(ratio, settings.max_output_width / geometry.width)
    if ratio != 1:
        # Hugin rounds odd dimensions upwards; request even values inside caps.
        width = max(2, int(geometry.width * ratio) // 2 * 2)
        height = max(2, int(geometry.height * ratio) // 2 * 2)
        run([binary, f"--canvas={width}x{height}", "-o", str(temporary), str(project)])
        scaled = read_panorama_geometry(temporary)
        if (
            scaled.projection != geometry.projection
            or scaled.parameters != geometry.parameters
            or abs(scaled.horizontal_fov - geometry.horizontal_fov) > 0.000500001
        ):
            raise ValueError("pano_modify changed projection geometry while capping canvas")
        if (scaled.width, scaled.height) != (width, height):
            raise ValueError("pano_modify produced an unexpected capped canvas size")
        for index, edge in enumerate(scaled.crop):
            scale = width / geometry.width if index < 2 else height / geometry.height
            if abs(edge - geometry.crop[index] * scale) > 1.01:
                raise ValueError("pano_modify changed framing while capping canvas")
        temporary.replace(project)
        geometry = scaled
    validate_canvas_size(geometry.width, geometry.height, settings)


def apply_projection_framing(
    project: str | Path,
    settings: StitchingSettings,
    run: Callable[[Sequence[str]], None],
    binary: str = "pano_modify",
) -> PanoramaGeometry:
    """Apply exact projection/framing rules without limiting the output canvas."""
    project = Path(project)
    read_panorama_geometry(project)
    temporary = project.with_name(f".{project.stem}.projection.pto")
    framing = settings.framing
    command = [binary, f"--projection={PROJECTIONS.index(settings.projection)}"]
    if settings.parameters:
        command.append(
            "--projection-parameter="
            + " ".join(format(value, ".12g") for value in settings.parameters)
        )
    if any(framing.rotation_degrees):
        command.append(
            "--rotate=" + ",".join(format(value, ".12g") for value in framing.rotation_degrees)
        )
    command.append("--fov=AUTO" if framing.auto_fov else f"--fov={framing.horizontal_fov:.12g}")
    if framing.auto_canvas:
        command.append("--canvas=AUTO")
    command.append(
        "--crop=AUTO"
        if framing.auto_crop
        else "--crop=" + ",".join(format(100 * value, ".12g") for value in framing.crop) + "%"
    )
    command.extend(["-o", str(temporary), str(project)])
    try:
        run(command)
        geometry = read_panorama_geometry(temporary)
        _verify_projection(geometry, settings)
        temporary.replace(project)
        return geometry
    finally:
        temporary.unlink(missing_ok=True)


def cap_projection_canvas(
    project: str | Path,
    settings: StitchingSettings,
    run: Callable[[Sequence[str]], None],
    binary: str = "pano_modify",
    scale: float | None = None,
) -> PanoramaGeometry:
    """Limit an already framed PTO while preserving its projection and crop."""
    validate_output_scale(scale, settings.mapping_backend)
    project = Path(project)
    temporary = project.with_name(f".{project.stem}.capped.pto")
    try:
        _cap_panorama(project, temporary, settings, run, binary, scale)
        return read_panorama_geometry(project)
    finally:
        temporary.unlink(missing_ok=True)


def apply_projection(
    project: str | Path,
    settings: StitchingSettings,
    run: Callable[[Sequence[str]], None],
    binary: str = "pano_modify",
    scale: float | None = None,
) -> PanoramaGeometry:
    """Apply projection framing and then cap its canvas for compatibility."""
    validate_output_scale(scale, settings.mapping_backend)
    apply_projection_framing(project, settings, run, binary)
    return cap_projection_canvas(project, settings, run, binary, scale)


def set_source_horizontal_fov(project: str | Path, horizontal_fov: float) -> None:
    """Update retained source PTOs too when a user changes the camera preset."""
    path = Path(project)
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("i "):
            # Linked v=0 tokens must remain linked to the first camera.
            lines[index] = re.sub(r"(?<!\S)v(?![=])[^\s]+", f"v{horizontal_fov:.12g}", line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
