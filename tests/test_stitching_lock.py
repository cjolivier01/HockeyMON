from __future__ import annotations

import subprocess
import sys

from hmlib.stitching.configure_stitching import (
    _stitch_game_lock,
    _stitch_project_is_complete,
)


def _can_lock(lock_path: str) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, pathlib, sys; "
                "f = pathlib.Path(sys.argv[1]).open('a+'); "
                "fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)"
            ),
            lock_path,
        ],
        check=False,
    )
    return result.returncode == 0


def should_exclusively_lock_stitching_artifacts_per_game(tmp_path):
    lock_path = str(tmp_path / ".stitching.lock")

    with _stitch_game_lock(tmp_path):
        assert not _can_lock(lock_path)

    assert _can_lock(lock_path)


def should_reject_partial_stitching_project_cache(tmp_path):
    project_path = tmp_path / "hm_project.pto"
    autooptimiser_path = tmp_path / "autooptimiser_out.pto"
    project_path.touch()
    autooptimiser_path.touch()

    assert not _stitch_project_is_complete(project_path, autooptimiser_path)

    for index in range(2):
        (tmp_path / f"mapping_{index:04d}.tif").touch()
    (tmp_path / "seam_file.png").touch()

    assert not _stitch_project_is_complete(project_path, autooptimiser_path)

    for index in range(2):
        (tmp_path / f"mapping_{index:04d}_x.tif").touch()
        (tmp_path / f"mapping_{index:04d}_y.tif").touch()

    assert _stitch_project_is_complete(project_path, autooptimiser_path)
