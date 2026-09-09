"""Tiny real artifacts for calibration and publication integration tests."""

from pathlib import Path

import cv2
import numpy as np
import tifffile


def write_mapping_files(directory):
    directory = Path(directory)
    outputs = []
    for index in range(2):
        path = directory / f"mapping_{index:04d}.tif"
        tifffile.imwrite(
            path,
            np.zeros((3, 4, 4), np.uint8),
            photometric="rgb",
            resolution=(1, 1),
            extratags=[(286, 5, 1, (0, 1), False), (287, 5, 1, (0, 1), False)],
        )
        for axis in ("x", "y"):
            tifffile.imwrite(path.with_name(f"{path.stem}_{axis}.tif"), np.zeros((3, 4), np.uint16))
        outputs.append(str(path))
    return outputs


def write_seam(directory):
    directory = Path(directory)
    assert cv2.imwrite(str(directory / "seam_file.png"), np.array([[0, 0, 255, 255]] * 3, np.uint8))
    tifffile.imwrite(directory / "panorama.tif", np.zeros((3, 4, 3), np.uint8), photometric="rgb")


def write_generation(directory):
    directory = Path(directory)
    write_mapping_files(directory)
    write_seam(directory)
    for name in ("hm_project.pto", "autooptimiser_out.pto"):
        (directory / name).write_text("p f2 w4 h3 v180\n", encoding="utf-8")
