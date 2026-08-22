"""High-level helpers for configuring two-camera stitching projects.

This module wraps Hugin PTO generation, control-point creation, seam
estimation and per-game synchronization into reusable functions.

@see @ref hmlib.stitching.control_points.calculate_control_points "calculate_control_points"
@see @ref hmlib.stitching.hugin.configure_control_points "configure_control_points"
"""

import json
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image

from hmlib.config import (
    get_game_config_private,
    get_nested_value,
    normalize_runtime_config,
    save_private_config,
    set_nested_value,
)
from hmlib.stitching.control_points import (
    calculate_control_points,
    normalize_control_point_matcher,
)
from hmlib.stitching.hugin import configure_control_points
from hmlib.stitching.homography_maps import (
    create_opencv_affine_ransac_mapping_files,
    create_opencv_magsac_mapping_files,
)
from hmlib.video.video_stream import extract_frame_image

from .synchronize import configure_synchronization

logger = logging.getLogger(__name__)

_STITCH_FRAME_TIME_PATH = ("stitching", "stitch_frame_time")
_STITCH_FRAME_TIME_ALT_PATH = ("stitching", "stitch-frame-time")
OPENCV_MAPPING_BACKENDS = ("opencv-magsac", "opencv-affine-ransac")
MAPPING_BACKENDS = ("nona", *OPENCV_MAPPING_BACKENDS)
_STITCH_ARTIFACT_MANIFEST = ".stitching_artifacts.json"


@contextmanager
def _stitch_game_lock(game_dir: Union[str, Path]) -> Iterator[None]:
    """Serialize mutation of shared stitching artifacts for one game."""
    lock_path = Path(game_dir) / ".stitching.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _resolve_local_binary(executable: str) -> Optional[str]:
    """Return a package-local binary path if available.

    Prefers `<hmlib_root>/bin/<executable>` both for Bazel runfiles
    and for installed wheels. When running from a source checkout, also
    recognizes Bazel's local output tree after `//hmlib:embed_*` has been built.
    """
    base_dir = Path(__file__).resolve().parent.parent
    workspace_dir = base_dir.parent
    candidates = [
        base_dir / "bin" / executable,
        workspace_dir / "bazel-bin" / "hmlib" / "bin" / executable,
    ]
    candidates.extend(workspace_dir.glob(f"bazel-out/*/bin/hmlib/bin/{executable}"))
    for bin_path in candidates:
        if bin_path.is_file() and os.access(bin_path, os.X_OK):
            return str(bin_path)
    return None


def _save_stitched_reference_frame(dir_name: Union[str, Path]) -> None:
    """Refresh ``s.png`` from the stitched reference panorama, when available."""
    panorama_file = Path(dir_name) / "panorama.tif"
    if not panorama_file.exists():
        return
    frame_file = panorama_file.with_name("s.png")
    try:
        panorama = np.asarray(tifffile.imread(str(panorama_file)))
        if panorama.ndim == 4:
            panorama = panorama[0]
        if panorama.ndim == 3 and panorama.shape[0] in (3, 4) and panorama.shape[-1] not in (3, 4):
            panorama = np.moveaxis(panorama, 0, -1)
        if panorama.ndim == 3 and panorama.shape[-1] > 3:
            panorama = panorama[:, :, :3]
        if panorama.dtype != np.uint8:
            panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        image = Image.fromarray(panorama)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(frame_file)
    except Exception:
        logger.debug("Failed to refresh stitched reference frame under %s", dir_name, exc_info=True)


def get_multiblend_bin() -> str:
    """Return the path to the `multiblend` binary, preferring a workspace-local build."""
    resolved = _resolve_local_binary("multiblend")
    if resolved is not None:
        return resolved
    return "multiblend"


def get_enblend_bin() -> str:
    """Return the path to the `enblend` binary, preferring a workspace-local build."""
    resolved = _resolve_local_binary("enblend")
    if resolved is not None:
        return resolved
    return "enblend"


def _run_stitching_command(cmd: Sequence[str]) -> None:
    """Run an external stitching command and fail if it does not complete."""
    logger.info("Running stitching command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def get_tiff_tag_value(tiff_tag):
    """Decode a TIFF rational tag into a Python scalar."""
    if len(tiff_tag.value) == 1:
        return tiff_tag.value
    assert len(tiff_tag.value) == 2
    numerator, denominator = tiff_tag.value
    return float(numerator) / denominator


def is_older_than(file1: str, file2: str):
    """Return True if `file2` is older than `file1`, or None if missing."""
    try:
        mtime1 = os.path.getmtime(file1)
        mtime2 = os.path.getmtime(file2)
        return mtime2 < mtime1
    except OSError:
        return None


def _stitch_project_is_complete(
    project_file_path: Union[str, Path],
    autooptimiser_path: Union[str, Path],
    control_point_matcher: Optional[str] = None,
    mapping_backend: Optional[str] = None,
) -> bool:
    """Return whether every artifact required to initialize stitching exists."""
    project_path = Path(project_file_path)
    game_dir = project_path.parent
    required_paths = (
        project_path,
        Path(autooptimiser_path),
        game_dir / "mapping_0000.tif",
        game_dir / "mapping_0000_x.tif",
        game_dir / "mapping_0000_y.tif",
        game_dir / "mapping_0001.tif",
        game_dir / "mapping_0001_x.tif",
        game_dir / "mapping_0001_y.tif",
        game_dir / "seam_file.png",
    )
    if not all(path.is_file() for path in required_paths):
        return False
    if control_point_matcher is None and mapping_backend is None:
        return True

    manifest = _read_stitch_artifact_manifest(game_dir)
    if manifest is None:
        return False
    return manifest == {
        "control_point_matcher": control_point_matcher,
        "mapping_backend": mapping_backend,
    }


def _read_stitch_artifact_manifest(game_dir: Union[str, Path]) -> Optional[Dict[str, str]]:
    """Read the backend choices used to build cached stitching artifacts."""
    manifest_path = Path(game_dir) / _STITCH_ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in manifest.items()
    ):
        return None
    return manifest


def get_image_geo_position(tiff_image_file: str):
    """Return integer pixel position (x, y) from a mapping TIFF file."""
    xpos, ypos = 0, 0
    with tifffile.TiffFile(tiff_image_file) as tif:
        tags = tif.pages[0].tags
        # Access the TIFFTAG_XPOSITION
        x_position = get_tiff_tag_value(tags.get("XPosition"))
        y_position = get_tiff_tag_value(tags.get("YPosition"))
        x_resolution = get_tiff_tag_value(tags.get("XResolution"))
        y_resolution = get_tiff_tag_value(tags.get("YResolution"))
        xpos = int(x_position * x_resolution + 0.5)
        ypos = int(y_position * y_resolution + 0.5)
    return xpos, ypos


def get_extracted_frame_image_file(video_name: str, dir_name: Optional[str] = None):
    """Return the PNG path used when extracting a frame from a video file."""
    if dir_name:
        video_name = os.path.join(dir_name, video_name)
    file_name_without_extension, _ = os.path.splitext(video_name)
    return file_name_without_extension + ".png"


def extract_frames(
    video_left: str,
    left_frame_number: int,
    video_right: str,
    right_frame_number: int,
    force: Optional[bool] = False,
):
    """Extract one frame from each side video to PNGs on disk.

    @param video_left: Absolute path to left video.
    @param left_frame_number: Frame index to extract from left video.
    @param video_right: Absolute path to right video.
    @param right_frame_number: Frame index to extract from right video.
    @return: Tuple of paths ``(left_png, right_png)``.
    """
    # Absolute paths
    assert "/" in video_left
    assert "/" in video_right
    left_output_image_file = get_extracted_frame_image_file(video_left)

    right_output_image_file = get_extracted_frame_image_file(video_right)

    if force:
        if os.path.exists(left_output_image_file):
            os.unlink(left_output_image_file)
        if os.path.exists(right_output_image_file):
            os.unlink(right_output_image_file)

    if force or not os.path.exists(left_output_image_file):
        extract_frame_image(
            video_left,
            frame_number=left_frame_number,
            dest_image=left_output_image_file,
        )
    if force or not os.path.exists(right_output_image_file):
        extract_frame_image(
            video_right,
            frame_number=right_frame_number,
            dest_image=right_output_image_file,
        )

    return left_output_image_file, right_output_image_file


def _unlink_best_effort(path: Path) -> bool:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
            return True
    except FileNotFoundError:
        return False
    except Exception as exc:
        logger.debug("Failed to unlink path %s: %s", path, exc, exc_info=True)
        return False
    return False


def _delete_globs(game_dir: Path, patterns: Sequence[str]) -> int:
    removed = 0
    for pat in patterns:
        for match in game_dir.glob(pat):
            if _unlink_best_effort(match):
                removed += 1
    return removed


def _set_hugin_optimization_variables(
    project_file_path: Union[str, Path], variables: Sequence[str]
) -> None:
    """Restrict Hugin optimization to the geometry variables used by learned matches."""
    path = Path(project_file_path)
    lines = path.read_text().splitlines()
    updated: List[str] = []
    in_variables = False
    replaced = False

    def write_variable_block() -> None:
        updated.extend(f"v {variable}" for variable in variables)
        updated.append("v")

    for line in lines:
        if line.startswith("# specify variables"):
            updated.append(line)
            write_variable_block()
            in_variables = True
            replaced = True
            continue
        if in_variables:
            if line.startswith("v"):
                continue
            if not line.strip():
                in_variables = False
                updated.append(line)
                continue
            in_variables = False
        updated.append(line)

    if not replaced:
        insertion = next(
            (index for index, line in enumerate(updated) if line.startswith("# control points")),
            len(updated),
        )
        block = [
            "# specify variables",
            *[f"v {variable}" for variable in variables],
            "v",
            "",
        ]
        updated[insertion:insertion] = block

    path.write_text("\n".join(updated) + "\n")


def _delete_extracted_frames(game_dir: Path) -> int:
    """Remove extracted frame PNGs that share a stem with a video file."""
    removed = 0
    video_exts = {".mp4", ".mkv", ".m4v", ".mov", ".avi"}
    for vid in game_dir.rglob("*"):
        if not vid.is_file():
            continue
        if vid.suffix.lower() not in video_exts:
            continue
        png = vid.with_suffix(".png")
        if _unlink_best_effort(png):
            removed += 1
    return removed


def _delete_nested_key(cfg: Dict[str, Any], path: Sequence[str]) -> bool:
    """Delete a dotted path from a nested dict. Returns True if deleted."""
    if not path:
        return False
    cur: Any = cfg
    parents: List[Tuple[Dict[str, Any], str]] = []
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return False
        parents.append((cur, key))
        cur = cur[key]
    leaf = path[-1]
    if not isinstance(cur, dict) or leaf not in cur:
        return False
    try:
        del cur[leaf]
    except Exception:
        return False
    # Prune empty dicts up the chain
    for parent, key in reversed(parents):
        try:
            node = parent.get(key)
            if isinstance(node, dict) and not node:
                del parent[key]
        except Exception:
            pass
    return True


def _parse_stitch_frame_time_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    time_str = str(value).strip()
    if not time_str:
        return None
    tokens = time_str.split(":")
    if not 1 <= len(tokens) <= 3:
        return None
    try:
        seconds = float(tokens[-1])
        minutes = int(tokens[-2]) if len(tokens) >= 2 else 0
        hours = int(tokens[-3]) if len(tokens) == 3 else 0
    except Exception:
        return None
    return hours * 3600.0 + minutes * 60.0 + seconds


def _normalize_stitch_frame_time_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    time_str = str(value).strip()
    if not time_str:
        return None
    seconds = _parse_stitch_frame_time_seconds(time_str)
    if seconds is not None and seconds <= 0:
        return None
    return time_str


def _stitch_frame_time_values_equal(lhs: Any, rhs: Any) -> bool:
    lhs_norm = _normalize_stitch_frame_time_value(lhs)
    rhs_norm = _normalize_stitch_frame_time_value(rhs)
    if lhs_norm is None or rhs_norm is None:
        return lhs_norm is None and rhs_norm is None
    lhs_seconds = _parse_stitch_frame_time_seconds(lhs_norm)
    rhs_seconds = _parse_stitch_frame_time_seconds(rhs_norm)
    if lhs_seconds is not None and rhs_seconds is not None:
        return abs(lhs_seconds - rhs_seconds) < 1e-6
    return lhs_norm == rhs_norm


def _get_stitch_frame_time_stamp(cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(cfg, dict):
        return None
    normalize_runtime_config(cfg)
    value = get_nested_value(cfg, ".".join(_STITCH_FRAME_TIME_PATH), None)
    if value is None:
        value = get_nested_value(cfg, ".".join(_STITCH_FRAME_TIME_ALT_PATH), None)
    return _normalize_stitch_frame_time_value(value)


def _set_stitch_frame_time_stamp(cfg: Dict[str, Any], stitch_frame_time: Optional[str]) -> bool:
    if not isinstance(cfg, dict):
        return False

    normalize_runtime_config(cfg)
    normalized = _normalize_stitch_frame_time_value(stitch_frame_time)
    current_primary = get_nested_value(cfg, ".".join(_STITCH_FRAME_TIME_PATH), None)
    current_alt = get_nested_value(cfg, ".".join(_STITCH_FRAME_TIME_ALT_PATH), None)
    dirty = False

    if normalized is None:
        dirty |= _delete_nested_key(cfg, _STITCH_FRAME_TIME_PATH)
        dirty |= _delete_nested_key(cfg, _STITCH_FRAME_TIME_ALT_PATH)
        return dirty

    if not _stitch_frame_time_values_equal(current_primary, normalized):
        dirty = True
    if current_alt is not None:
        dirty = True

    set_nested_value(cfg, ".".join(_STITCH_FRAME_TIME_PATH), normalized)
    _delete_nested_key(cfg, _STITCH_FRAME_TIME_ALT_PATH)
    return dirty


def sync_stitch_frame_time_state(
    game_id: Optional[str],
    game_dir: Union[str, Path],
    stitch_frame_time: Optional[str],
    *,
    force: bool = False,
    ignore_private_config: bool = False,
    game_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Persist stitch-frame-time and clean cached stitch artifacts when it changes.

    The persisted value lives under ``stitching.stitch_frame_time`` in the
    private game config. We keep it separate from rebuildable stitch artifacts
    so `stitch --clean` can remove caches without dropping the manually entered
    stitch timestamp.

    Returns ``True`` when the effective stitch-frame-time changed for this run.
    """
    normalized = _normalize_stitch_frame_time_value(stitch_frame_time)
    runtime_previous = _get_stitch_frame_time_stamp(game_config)
    runtime_changed = not _stitch_frame_time_values_equal(runtime_previous, normalized)

    if isinstance(game_config, dict):
        _set_stitch_frame_time_stamp(game_config, normalized)

    if not game_id or ignore_private_config:
        return runtime_changed

    try:
        private_cfg = get_game_config_private(game_id=game_id) or {}
        normalize_runtime_config(private_cfg)
    except Exception as ex:
        logger.warning(
            "Failed to load private config for stitch-frame-time sync on %s: %s", game_id, ex
        )
        private_cfg = {}

    previous = _get_stitch_frame_time_stamp(private_cfg)
    changed = not _stitch_frame_time_values_equal(previous, normalized)

    if changed and not force:
        logger.info(
            "stitch_frame_time changed for %s (%r -> %r); cleaning cached stitch artifacts",
            game_id,
            previous,
            normalized,
        )
        try:
            clean_stitch_game_artifacts(game_id=game_id, game_dir=game_dir)
        except Exception as ex:
            logger.warning("Failed to clean stitch artifacts for %s: %s", game_id, ex)
        try:
            private_cfg = get_game_config_private(game_id=game_id) or {}
            normalize_runtime_config(private_cfg)
        except Exception:
            private_cfg = {}

    dirty = _set_stitch_frame_time_stamp(private_cfg, normalized)
    if dirty:
        try:
            save_private_config(game_id=game_id, data=private_cfg, verbose=True)
        except Exception as ex:
            logger.warning("Failed to save stitch-frame-time stamp for %s: %s", game_id, ex)

    return changed or runtime_changed


def clean_stitch_game_artifacts(game_id: str, game_dir: Union[str, Path]) -> int:
    """Delete rebuildable stitching / seam / mask outputs for a game.

    Does not delete config.yaml, but removes cached stitching/rink entries
    from the private config so they will be recomputed (e.g. audio sync
    offsets, scoreboard selection, rink-mask metadata). Manual stitch inputs
    such as ``stitching.stitch_frame_time`` are preserved.
    """
    game_dir = Path(game_dir)

    removed_files = 0
    removed_files += _delete_globs(
        game_dir,
        patterns=[
            "hm_project.pto",
            "autooptimiser_out.pto",
            "*.pto",
            "mapping_*.tif",
            "mapping_*.tiff",
            "panorama.tif",
            "seam_file.png",
            "matches.png",
            "keypoints.png",
            "s.png",
            _STITCH_ARTIFACT_MANIFEST,
        ],
    )
    removed_files += _delete_globs(game_dir, patterns=["rink_mask_*.png"])
    removed_files += _delete_extracted_frames(game_dir)

    try:
        cfg = get_game_config_private(game_id=game_id)
        normalize_runtime_config(cfg)
    except Exception as ex:
        logger.warning("Failed to load private config while cleaning %s: %s", game_id, ex)
        cfg = None

    if isinstance(cfg, dict) and cfg:
        changed = False
        changed |= _delete_nested_key(cfg, ["stitching", "frame_offsets"])
        changed |= _delete_nested_key(cfg, ["stitching", "control_points"])
        changed |= _delete_nested_key(cfg, ["rink", "scoreboard", "perspective_polygon"])
        changed |= _delete_nested_key(cfg, ["rink", "ice_contours_mask_count"])
        changed |= _delete_nested_key(cfg, ["rink", "ice_contours_mask_centroid"])
        changed |= _delete_nested_key(cfg, ["rink", "ice_contours_combined_bbox"])
        if changed:
            try:
                save_private_config(game_id=game_id, data=cfg, verbose=True)
            except Exception as ex:
                logger.warning("Failed to save private config while cleaning %s: %s", game_id, ex)

    if removed_files:
        logger.info("Cleaned %d rebuildable files under %s", removed_files, str(game_dir))

    return removed_files


def build_stitching_project(
    project_file_path: str,
    image_files: List[str],
    max_control_points: int,
    skip_if_exists: bool = True,
    test_blend: bool = True,
    fov: int = 108,
    scale: Optional[float] = None,
    force: bool = False,
    control_point_matcher: str = "superpoint-lightglue",
    mapping_backend: str = "nona",
    max_output_dimension: Optional[int] = None,
):
    """Create or update a Hugin PTO project and seam masks for two images.

    @param project_file_path: Output PTO project path.
    @param image_files: List of two input image paths (left, right).
    @param max_control_points: Maximum number of control points to use.
    @param skip_if_exists: If True, reuse existing project when up-to-date.
    @param test_blend: Whether to create/test seam masks using `enblend`.
    @param fov: Horizontal field-of-view in degrees.
    @param scale: Optional scale factor passed to `autooptimiser`.
    @param force: If True, always rebuild, ignoring mtimes.
    @param control_point_matcher: Feature matcher used to find control points.
    @param mapping_backend: ``nona`` or a native OpenCV remapping backend.
    @param max_output_dimension: Optional maximum mapping canvas dimension.
    @return: True on success, False if seam quality tests fail.
    """
    pto_path = Path(project_file_path)
    control_point_matcher = normalize_control_point_matcher(control_point_matcher)
    mapping_backend = str(mapping_backend).strip().lower().replace("_", "-")
    if mapping_backend not in MAPPING_BACKENDS:
        choices = ", ".join(MAPPING_BACKENDS)
        raise ValueError(
            f"Unsupported mapping backend {mapping_backend!r}; choose one of: {choices}"
        )
    if mapping_backend in OPENCV_MAPPING_BACKENDS and scale not in (None, 1.0):
        raise ValueError(
            f"The {mapping_backend} backend does not accept Hugin's relative scale; "
            "use max_output_dimension instead"
        )
    dir_name = pto_path.parent
    previous_manifest = _read_stitch_artifact_manifest(dir_name)
    previous_control_point_matcher = (
        previous_manifest.get("control_point_matcher")
        if previous_manifest is not None
        else "superpoint-lightglue"
    )
    hm_project = project_file_path
    autooptimiser_out = os.path.join(dir_name, "autooptimiser_out.pto")
    assert autooptimiser_out != hm_project
    if (
        skip_if_exists
        and _stitch_project_is_complete(
            project_file_path,
            autooptimiser_out,
            control_point_matcher=control_point_matcher,
            mapping_backend=mapping_backend,
        )
        and not is_older_than(project_file_path, autooptimiser_out)
    ):
        print(f"Project file already exists (skipping project creation): {autooptimiser_out}")
        return True
    assert len(image_files) == 2
    left_image_file = image_files[0]
    right_image_file = image_files[1]

    curr_dir = os.getcwd()
    os.chdir(dir_name)
    try:

        def generate_pto() -> None:
            cmd = [
                "pto_gen",
                "-p",
                "0",
                "-o",
                hm_project,
                "-f",
                str(fov),
                left_image_file,
                right_image_file,
            ]
            _run_stitching_command(cmd)

        def remove_remap_outputs() -> None:
            _delete_globs(
                Path(dir_name),
                patterns=[
                    "autooptimiser_out.pto",
                    "mapping_*.tif",
                    "mapping_*.tiff",
                    "panorama.tif",
                    "seam_file.png",
                ],
            )

        def run_remap_pipeline(control_points: Dict[str, torch.Tensor]) -> bool:
            remove_remap_outputs()
            if mapping_backend == "nona":
                cmd = [
                    "autooptimiser",
                    "-n",
                    "-l",
                    "-s",
                    "-q",
                    "-o",
                    autooptimiser_out,
                    hm_project,
                ]
                if scale and scale != 1.0:
                    cmd += [
                        "-x",
                        str(scale),
                    ]
                _run_stitching_command(cmd)
                _set_hugin_optimization_variables(autooptimiser_out, ("r1", "p1", "y1"))

                cmd = [
                    "nona",
                    "-m",
                    "TIFF_m",
                    "-z",
                    "NONE",
                    "--bigtiff",
                    "-c",
                    "-o",
                    "mapping_",
                    autooptimiser_out,
                ]
                _run_stitching_command(cmd)
                mapping_files = sorted(
                    str(path) for path in Path(dir_name).glob("mapping_????.tif")
                )
                if not mapping_files:
                    raise FileNotFoundError(f"No Hugin mapping TIFFs were generated in {dir_name}")
            elif mapping_backend == "opencv-magsac":
                shutil.copyfile(hm_project, autooptimiser_out)
                mapping_files = create_opencv_magsac_mapping_files(
                    [left_image_file, right_image_file],
                    control_points,
                    dir_name,
                    max_output_dimension=max_output_dimension,
                )
            else:
                shutil.copyfile(hm_project, autooptimiser_out)
                mapping_files = create_opencv_affine_ransac_mapping_files(
                    [left_image_file, right_image_file],
                    control_points,
                    dir_name,
                    max_output_dimension=max_output_dimension,
                )

            seam_file: str = os.path.join(dir_name, "seam_file.png")
            cmd = [
                get_enblend_bin(),
                f"--save-masks={seam_file}",
                "-o",
                os.path.join(dir_name, "panorama.tif"),
                *mapping_files,
            ]
            try:
                _run_stitching_command(cmd)
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "enblend failed with exit code %s; trying multiblend",
                    exc.returncode,
                )
            # See if it came out with a reasonable seam file
            distribution: Dict[int, float] = get_pixel_value_percentages(seam_file)
            # Really, it should be way above this number for a good seam, but so far
            # the "broken case" is much below this number (like 0.5%).
            kMinAllowableSeamPercent: float = 10.0
            if not distribution or any(
                pct < kMinAllowableSeamPercent for pct in distribution.values()
            ):
                print(f"Warning: seam file {seam_file} has low seam values, indicating a bad seam.")
                for val, pct in sorted(distribution.items()):
                    print(f"Seam value {val:3d}: {pct:5.2f}%")
                # Delete the seam file so that it doesn't get used accidentally
                if os.path.exists(seam_file):
                    os.remove(seam_file)
                # If the seam is bad, try using multiblend instead
                cmd = [
                    get_multiblend_bin(),
                    f"--save-seams={seam_file}",
                    "-o",
                    os.path.join(dir_name, "panorama.tif"),
                    *mapping_files,
                ]
                _run_stitching_command(cmd)
                # Check again (should be ok now unless the stitch is really bad)
                distribution = get_pixel_value_percentages(seam_file)
                if not distribution or any(
                    pct < kMinAllowableSeamPercent for pct in distribution.values()
                ):
                    print(
                        f"Warning: seam file {seam_file} has low seam values, indicating a bad seam."
                    )
                    for val, pct in sorted(distribution.items()):
                        print(f"Seam value {val:3d}: {pct:5.2f}%")
                    return False
            return True

        use_hugin = False
        if not os.path.exists(hm_project) or force:
            generate_pto()
        elif previous_control_point_matcher == control_point_matcher:
            use_hugin = True

        control_points = configure_control_points(
            output_directory=str(dir_name),
            project_file_path=hm_project,
            image0=left_image_file,
            image1=right_image_file,
            max_control_points=max_control_points,
            force=True,
            use_hugin=use_hugin,
            matcher=control_point_matcher,
        )
        _set_hugin_optimization_variables(hm_project, ("r1", "p1", "y1"))
        try:
            remap_ok = run_remap_pipeline(control_points)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"{control_point_matcher} control points did not produce "
                f"remappable {mapping_backend} outputs"
            ) from exc
        if not remap_ok:
            raise RuntimeError(
                f"{control_point_matcher} control points produced low-quality seam masks"
            )
        manifest = {
            "control_point_matcher": control_point_matcher,
            "mapping_backend": mapping_backend,
        }
        (Path(dir_name) / _STITCH_ARTIFACT_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return True

    finally:
        os.chdir(curr_dir)


def get_pixel_value_percentages(image_path: str) -> Dict[int, float]:
    """
    Opens a grayscale image and computes the percentage of each pixel value.

    Args:
        image_path (str): Path to the input PNG.

    Returns:
        Dict[int, float]: Mapping from pixel value (0–+5) to percentage of image,
                          as a float in [0.0, 100.0].
    """
    # Open image and ensure it's in 8-bit grayscale mode
    arr: np.ndarray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return {}

    # Total number of pixels
    total: int = arr.size

    # Count occurrences of each value
    counts: np.ndarray = np.bincount(arr.flatten(), minlength=256)

    # Build percentage dict, omitting values with zero count
    percentages: Dict[int, float] = {
        value: (count / total) * 100.0 for value, count in enumerate(counts) if count > 0
    }

    return percentages


def load_or_calculate_control_points(
    game_id: str,
    image0: Union[str, Path, torch.Tensor],
    image1: Union[str, Path, torch.Tensor],
    force: bool = False,
    device: Optional[torch.device] = None,
    save: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load game-specific control points or compute them with a learned matcher.

    @param game_id: Game identifier used to resolve private config.
    @param image0: First image (path or tensor).
    @param image1: Second image (path or tensor).
    @param force: If True, ignore cached control points and recompute.
    @param device: Optional device for LightGlue/SuperPoint.
    @param max_control_points: Maximum number of points to keep.
    @param output_directory: Optional directory for debug visualizations.
    @param save: If True, persist control points into game config.
    @return: Dict with at least ``m_kpts0`` and ``m_kpts1`` tensors.
    """
    config = get_game_config_private(game_id=game_id) or {}
    normalize_runtime_config(config)
    control_points = get_nested_value(config, "stitching.control_points") if not force else {}
    if force or not control_points:
        # Calculate them...
        control_points = calculate_control_points(image=image0, image1=image1, device=device)
        assert "m_kpts0" in control_points and "m_kpts1" in control_points
        # Remove stuff we don't want
        control_points.pop("kpts0")
        control_points.pop("kpts1")

        if save:
            config = set_nested_value(config, "stitching.control_points", control_points)
            save_private_config(game_id=game_id, data=config)


def configure_video_stitching(
    dir_name: str,
    video_left: str,
    video_right: str,
    max_control_points: int,
    project_file_name: str = "hm_project.pto",
    left_frame_offset: int = None,
    right_frame_offset: int = None,
    base_frame_offset: int = 0,
    audio_sync_seconds: int = 15,
    force: bool = False,
    game_id: Optional[str] = None,
    stitch_frame_time: Optional[str] = None,
    ignore_private_config: bool = False,
    game_config: Optional[Dict[str, Any]] = None,
    control_point_matcher: str = "superpoint-lightglue",
    mapping_backend: str = "nona",
):
    """Configure stitching while serializing shared artifacts per game."""
    with _stitch_game_lock(dir_name):
        return _configure_video_stitching_locked(
            dir_name=dir_name,
            video_left=video_left,
            video_right=video_right,
            max_control_points=max_control_points,
            project_file_name=project_file_name,
            left_frame_offset=left_frame_offset,
            right_frame_offset=right_frame_offset,
            base_frame_offset=base_frame_offset,
            audio_sync_seconds=audio_sync_seconds,
            force=force,
            game_id=game_id,
            stitch_frame_time=stitch_frame_time,
            ignore_private_config=ignore_private_config,
            game_config=game_config,
            control_point_matcher=control_point_matcher,
            mapping_backend=mapping_backend,
        )


def _configure_video_stitching_locked(
    dir_name: str,
    video_left: str,
    video_right: str,
    max_control_points: int,
    project_file_name: str = "hm_project.pto",
    left_frame_offset: int = None,
    right_frame_offset: int = None,
    base_frame_offset: int = 0,
    audio_sync_seconds: int = 15,
    force: bool = False,
    game_id: Optional[str] = None,
    stitch_frame_time: Optional[str] = None,
    ignore_private_config: bool = False,
    game_config: Optional[Dict[str, Any]] = None,
    control_point_matcher: str = "superpoint-lightglue",
    mapping_backend: str = "nona",
):
    """Configure a two-camera stitching project from game videos.

    Uses audio-based synchronization, frame extraction, optional per-side
    color adjustment and PTO generation to produce mapping TIFFs with either
    nona, native OpenCV MAGSAC++ homography maps, or affine RANSAC maps.

    @param dir_name: Game directory containing videos and config.
    @param video_left: Left-side video filename or path.
    @param video_right: Right-side video filename or path.
    @param max_control_points: Max number of control points to search.
    @param project_file_name: PTO filename inside ``dir_name``.
    @param left_frame_offset: Manually specified left frame offset (or None).
    @param right_frame_offset: Manually specified right frame offset (or None).
    @param base_frame_offset: Global offset added to both sides.
    @param audio_sync_seconds: Seconds of audio used for synchronization.
    @param force: If True, recompute PTO and seam even if up-to-date.
    @param game_id: Optional game identifier used to persist stitch state.
    @param stitch_frame_time: Effective stitch-frame-time used to build the PTO.
    @param ignore_private_config: If True, do not read/write the private config stamp.
    @param game_config: Optional in-memory game config to update with the effective stamp.
    @param control_point_matcher: Learned feature matcher backend.
    @param mapping_backend: Mapping generator backend.
    @return: Tuple ``(pto_project_file, left_frame_offset, right_frame_offset)``.
    """
    stitch_frame_time_changed = sync_stitch_frame_time_state(
        game_id=game_id,
        game_dir=dir_name,
        stitch_frame_time=stitch_frame_time,
        force=force,
        ignore_private_config=ignore_private_config,
        game_config=game_config,
    )
    force = bool(force or stitch_frame_time_changed)

    if left_frame_offset is None or right_frame_offset is None:
        frame_offsets = configure_synchronization(
            game_id=dir_name.split("/")[-1],
            video_left=video_left,
            video_right=video_right,
            audio_sync_seconds=audio_sync_seconds,
            force=force,
        )
        left_frame_offset = float(frame_offsets["left"])
        right_frame_offset = float(frame_offsets["right"])

    # PTO Project File
    pto_project_file: str = os.path.join(dir_name, project_file_name)
    autooptimiser_out: str = os.path.join(dir_name, "autooptimiser_out.pto")
    if (
        force
        or not _stitch_project_is_complete(
            pto_project_file,
            autooptimiser_out,
            control_point_matcher=control_point_matcher,
            mapping_backend=mapping_backend,
        )
        or (os.path.exists(pto_project_file) and is_older_than(pto_project_file, autooptimiser_out))
    ):
        left_image_file, right_image_file = extract_frames(
            video_left,
            base_frame_offset + left_frame_offset,
            video_right,
            base_frame_offset + right_frame_offset,
            force=True,
        )

        project_built = build_stitching_project(
            project_file_path=pto_project_file,
            image_files=[left_image_file, right_image_file],
            max_control_points=max_control_points,
            force=force,
            skip_if_exists=not force,
            control_point_matcher=control_point_matcher,
            mapping_backend=mapping_backend,
        )
        if not project_built:
            raise RuntimeError("Failed to build stitching project")

    _save_stitched_reference_frame(dir_name)

    return pto_project_file, left_frame_offset, right_frame_offset
