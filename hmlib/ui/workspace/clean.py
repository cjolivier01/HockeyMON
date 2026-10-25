"""Retire managed calibration products without removing source media or runs."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from hmlib.stitching.artifacts import stitching_lock

from .files import config_lock, read_config, write_config, write_json

_DERIVED_CONFIG = (
    "stitching.frame_offsets",
    "stitching.control_points",
    "rink.scoreboard.perspective_polygon",
    "rink.workspace_map",
    "rink.ice_contours_mask_count",
    "rink.ice_contours_mask_centroid",
    "rink.ice_contours_combined_bbox",
    "rink.ice_contours_geometry_revision",
    "rink.ice_contours_mask_file_prefix",
)
_DERIVED_FILES = {
    ".stitching_artifacts.json",
    "hm_project.pto",
    "autooptimiser_out.pto",
    "panorama.tif",
    "seam_file.png",
    "matches.png",
    "keypoints.png",
    "s.png",
    "left.png",
    "right.png",
}


def clean_stitching(game: Path) -> dict:
    with stitching_lock(game), config_lock(game):
        config = read_config(game)
        paths = [
            path
            for path in game.iterdir()
            if path.name in _DERIVED_FILES
            or re.fullmatch(r"mapping_[0-9]+\.tiff?", path.name)
            or re.fullmatch(r"rink_mask_(?:[0-9a-f]{32}_)?[0-9]+\.png", path.name)
        ]
        for path in paths:
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Managed calibration product must be a regular file: {path}")
        for dotted in _DERIVED_CONFIG:
            parts = dotted.split(".")
            parent = config
            for part in parts[:-1]:
                parent = parent.get(part, {}) if isinstance(parent, dict) else {}
            if isinstance(parent, dict):
                parent.pop(parts[-1], None)
        # Keep a recovery copy in the game filesystem; only derived paths leave
        # the active generation. Named run snapshots and original clips stay put.
        backup = game / ".hm-workspace" / "cleaned" / uuid.uuid4().hex
        backup.mkdir(parents=True)
        config_path = game / "config.yaml"
        if config_path.exists():
            (backup / "config.before.yaml").write_bytes(config_path.read_bytes())
        write_json(backup / "receipt.json", {"game": str(game), "files": [p.name for p in paths]})
        moved = []
        try:
            for path in sorted(paths, key=lambda value: value.name != ".stitching_artifacts.json"):
                os.rename(path, backup / path.name)
                moved.append(path)
            write_config(game, config)
        except BaseException:
            for path in reversed(moved):
                os.rename(backup / path.name, path)
            raise
        return {
            "game": str(game),
            "removed": [p.name for p in paths],
            "recovery_directory": str(backup),
        }
