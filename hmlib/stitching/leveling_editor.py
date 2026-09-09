"""Private, bounded previews and fenced config saves for the rink editor."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image

from hmlib.stitching.projections import apply_projection, read_panorama_geometry
from hmlib.stitching.rink_leveling import (
    estimate_leveling,
    format_points,
    parse_rays,
    prepare_project,
    rotation_delta,
)
from hmlib.stitching.settings import read_stitching_settings

_TOKENS = re.compile(r'(?:[^\s"]|"(?:\\.|[^"\\])*")+')
_MANIFEST = ".stitching_artifacts.json"


def _absolute_sources(pto: str, directory: Path) -> tuple[str, tuple[Path, ...]]:
    paths, lines = [], []
    for line in pto.splitlines():
        if line.lstrip().startswith(("i ", "i\t")):
            tokens = list(_TOKENS.finditer(line))
            names = [token for token in tokens[1:] if token.group().startswith("n")]
            if len(names) != 1:
                raise ValueError("Each source image in the project must have exactly one filename")
            token = names[0]
            name = shlex.split(token.group())[0][1:]
            source = (directory / name).resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"Source image is not a regular file: {source}")
            paths.append(source)
            line = line[: token.start()] + "n" + json.dumps(str(source)) + line[token.end() :]
        lines.append(line)
    return "\n".join(lines) + "\n", tuple(paths)


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_hugin(command, *, input=None):
    """Resolve bundled Hugin tools and propagate execution errors with stderr."""
    from hmlib.stitching.configure_stitching import _resolve_local_binary

    binary = _resolve_local_binary(command[0]) or shutil.which(command[0])
    if binary is None:
        raise RuntimeError(f"Required Hugin tool is unavailable: {command[0]}")
    command = [binary, *command[1:]]
    try:
        result = subprocess.run(command, input=input, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run {command[0]}: {exc}") from exc
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed: {result.stderr[-4000:] or result.stdout[-4000:]}")
    return result.stdout


class LevelingSession:
    """An editor owns immutable calibration provenance until it saves or closes.

    Hold the calibration lock whenever reading sources or replacing the private
    config. File digests and fresh effective settings prevent stale dialogs
    overwriting a newer calibration/config. No active mapping file is edited.
    """

    def __init__(self, game_dir: Path, config_loader: Callable[[], dict], *, run=run_hugin):
        self.game_dir = Path(game_dir).resolve(strict=True)
        self.config_loader = config_loader
        self.run = run
        self._temporary = tempfile.TemporaryDirectory(prefix="hm-level-")
        self.directory = Path(self._temporary.name)
        self.saved = False
        self.preview_token = None
        self.preview_state = None
        self.preview_image = None
        try:
            with self._lock():
                self.config = copy.deepcopy(config_loader())
                self.settings = read_stitching_settings(self.config)
                if self.settings.mapping_backend != "nona":
                    raise ValueError(
                        "Rink leveling and manual crop require the NONA mapping backend; recalibrate with NONA first"
                    )
                self.project_path = self.game_dir / "autooptimiser_out.pto"
                self.manifest_path = self.game_dir / _MANIFEST
                self.config_path = self.game_dir / "config.yaml"
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("mapping_backend") != "nona":
                    raise ValueError("Recalibrate with NONA before opening the rink editor")
                try:
                    published = json.loads(manifest["calibration_settings"])
                    self.published_rotation = tuple(published["framing"]["rotation_degrees"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "The calibration has no valid rotation provenance; recalibrate first"
                    ) from exc
                rotation_delta(self.published_rotation, self.settings.framing.rotation_degrees)
                current = asdict(self.settings)
                # Framing can be previewed from registered poses; different
                # source lenses, matchers or optimizers require fresh poses.
                framing_fields = {
                    "framing",
                    "projection",
                    "parameters",
                    "max_output_dimension",
                    "max_output_width",
                }
                if any(
                    published.get(key) != value
                    for key, value in current.items()
                    if key not in framing_fields
                ):
                    raise ValueError(
                        "Camera or calibration settings changed; recalibrate before opening the rink editor"
                    )
                if self.project_path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("The stitching project is too large")
                self.pto, self.sources = _absolute_sources(
                    self.project_path.read_text(encoding="utf-8"), self.game_dir
                )
                self.leveling_project = prepare_project(self.pto)
                self._paths = (
                    self.project_path,
                    self.manifest_path,
                    self.config_path,
                    *self.sources,
                )
                self._fingerprint = self._snapshot()
                self.source_images = []
                for source, size in zip(self.sources, self.leveling_project.image_sizes):
                    if max(size) > 65535 or size[0] * size[1] > 256 * 1024 * 1024:
                        raise ValueError(
                            f"Source image exceeds the editor's decode limits: {source}"
                        )
                    with Image.open(source) as image:
                        if image.size != size:
                            raise ValueError(f"Source dimensions differ from calibration: {source}")
                        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                        buffer = io.BytesIO()
                        image.convert("RGB").save(buffer, format="JPEG", quality=90)
                        self.source_images.append(buffer.getvalue())
                self._check_fresh()
        except BaseException:
            self.close()
            raise

    def _lock(self):
        from hmlib.stitching.artifacts import stitching_lock

        return stitching_lock(self.game_dir, blocking=False)

    def _snapshot(self):
        config = self.config_loader()
        return (
            tuple(_digest(path) for path in self._paths),
            read_stitching_settings(config).manifest(),
            # Include inherited settings outside StitchingSettings too (for
            # example frame offsets and source-video selection).
            yaml.safe_dump(config, sort_keys=True),
        )

    def _check_fresh(self):
        if self.saved:
            raise ValueError("This editor has already saved; close it and recalibrate")
        if self._snapshot() != self._fingerprint:
            raise ValueError(
                "Calibration, source images or config changed; reopen the editor before continuing"
            )

    def info(self) -> dict:
        return {
            "rotation_degrees": list(self.settings.framing.rotation_degrees),
            "crop": list(self.settings.framing.crop),
            "auto_crop": self.settings.framing.auto_crop,
            "sources": [
                {"name": path.name, "width": size[0], "height": size[1]}
                for path, size in zip(self.sources, self.leveling_project.image_sizes)
            ],
        }

    def estimate(self, posts: list[dict], yaw: float) -> dict:
        points = format_points(posts, self.leveling_project.image_sizes)
        with self._lock():
            self._check_fresh()
            project = self.directory / "rays.pto"
            project.write_text(self.leveling_project.pto, encoding="utf-8")
            output = self.run(["pano_trafo", str(project)], input=points)
            estimate = estimate_leveling(
                parse_rays(output, len(posts)), self.published_rotation, yaw
            )
            self._check_fresh()
        return asdict(estimate)

    def _validate_state(self, state: dict):
        if not isinstance(state, dict) or set(state) != {
            "rotation_degrees",
            "crop",
            "auto_crop",
            "posts",
        }:
            raise ValueError("Preview requires rotation_degrees, crop, auto_crop and posts")
        posts = state["posts"]
        if not isinstance(posts, list):
            raise ValueError("Posts must be a list")
        if posts:
            format_points(posts, self.leveling_project.image_sizes)
        config = copy.deepcopy(self.config)
        framing = config.setdefault("stitching", {}).setdefault("projection_framing", {})
        framing.update({key: state[key] for key in ("rotation_degrees", "crop", "auto_crop")})
        return read_stitching_settings(config)

    def preview(self, state: dict) -> dict:
        settings = self._validate_state(state)
        with self._lock():
            self._check_fresh()
            # Invalidate the prior save token before starting a new render.
            self.preview_token = None
            framing = replace(
                settings.framing,
                rotation_degrees=rotation_delta(
                    self.published_rotation, settings.framing.rotation_degrees
                ),
                crop=(0, 1, 0, 1),
                auto_crop=False,
            )
            bounded = replace(
                settings,
                framing=framing,
                max_output_dimension=min(settings.max_output_dimension or 1920, 1920),
                max_output_width=min(settings.max_output_width or 1920, 1920),
            )
            project = self.directory / "preview.pto"
            project.write_text(self.pto, encoding="utf-8")
            geometry = apply_projection(project, bounded, self.run)
            crop = settings.framing.crop
            if settings.framing.auto_crop:
                auto_project = self.directory / "auto.pto"
                auto_project.write_text(project.read_text(encoding="utf-8"), encoding="utf-8")
                self.run(["pano_modify", "--crop=AUTO", "-o", str(auto_project), str(project)])
                automatic = read_panorama_geometry(auto_project)
                if (automatic.width, automatic.height) != (geometry.width, geometry.height):
                    raise ValueError("Automatic crop unexpectedly changed the preview canvas")
                crop = tuple(
                    value / (geometry.width if index < 2 else geometry.height)
                    for index, value in enumerate(automatic.crop)
                )
            output = self.directory / "preview.png"
            output.unlink(missing_ok=True)
            self.run(
                [
                    "nona",
                    "-m",
                    "PNG",
                    "-p",
                    "UINT8",
                    "--seam=blend",
                    "-o",
                    str(output),
                    str(project),
                ]
            )
            with Image.open(output) as image:
                if image.size != (geometry.width, geometry.height):
                    raise ValueError(
                        "Rendered preview dimensions differ from the full panorama canvas"
                    )
                image.verify()
            image_bytes = output.read_bytes()
            self._check_fresh()
            self.preview_token = secrets.token_urlsafe(24)
            self.preview_state = copy.deepcopy(state)
            self.preview_image = image_bytes
            return {
                "token": self.preview_token,
                "crop": list(crop),
                "width": geometry.width,
                "height": geometry.height,
            }

    def save(self, state: dict, token: str) -> None:
        settings = self._validate_state(state)
        if not self.preview_token or token != self.preview_token or state != self.preview_state:
            raise ValueError("Preview the current marks, rotation and crop before saving")
        with self._lock():
            self._check_fresh()
            config = (
                yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
                if self.config_path.exists()
                else {}
            )
            if config is None:
                config = {}
            if not isinstance(config, dict):
                raise ValueError("The private game config must be a YAML mapping")
            stitch = config.setdefault("stitching", {})
            if not isinstance(stitch, dict):
                raise ValueError("Private stitching config must be a mapping")
            framing = stitch.setdefault("projection_framing", {})
            if not isinstance(framing, dict):
                raise ValueError("Private projection_framing config must be a mapping")
            # Preserve rink inheritance when the user only adjusted the crop.
            if settings.framing.rotation_degrees != self.settings.framing.rotation_degrees:
                framing["rotation_degrees"] = list(settings.framing.rotation_degrees)
            framing["crop"] = list(settings.framing.crop)
            framing["auto_crop"] = settings.framing.auto_crop
            fd, temporary_name = tempfile.mkstemp(
                prefix=".config.leveling-", suffix=".yaml", dir=self.game_dir
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    if self.config_path.exists():
                        os.fchmod(stream.fileno(), self.config_path.stat().st_mode & 0o777)
                    yaml.safe_dump(config, stream, sort_keys=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._check_fresh()
                os.replace(temporary, self.config_path)
                self.saved = True
                self.preview_token = None
                directory_fd = os.open(self.game_dir, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)

    def close(self):
        self._temporary.cleanup()
