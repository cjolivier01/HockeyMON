"""Launch the shared HStream desktop with HockeyMON's processing backend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def executable(explicit: str | None, name: str, candidates: tuple[Path, ...] = ()) -> Path:
    """Resolve an explicitly selected executable without silently replacing it."""
    choices = [explicit] if explicit else [shutil.which(name), *candidates]
    for choice in choices:
        if choice is None:
            continue
        path = Path(choice).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    detail = (
        f"Requested executable is unavailable: {explicit}" if explicit else f"Cannot find {name}"
    )
    raise FileNotFoundError(
        f"{detail}. Install the shared HStream desktop and HockeyMON backend, "
        "or select their executable paths with HSTREAM_UI_BIN and HM_CLI_BIN."
    )


def launch_desktop(
    *,
    game_id: str | None = None,
    module: str | None = None,
    arguments: list[str] | None = None,
    root_dir: str | Path | None = None,
) -> int:
    source_root = Path(__file__).resolve().parents[2]
    ui = executable(
        os.environ.get("HSTREAM_UI_BIN"),
        "hstream-ui",
        (Path("/opt/hstream/bin/hstream-ui"),),
    )
    runner = executable(
        os.environ.get("HM_CLI_BIN"),
        "hm-cli",
        (source_root / "hm-ui" / "hm-cli",),
    )
    environment = os.environ.copy()
    if root_dir is not None:
        environment["HM_GAME_DIR"] = str(Path(root_dir).expanduser().resolve())
    command = [str(ui), "--backend=hockeymon", f"--backend-runner={runner}"]
    if module not in {None, "hmlib.cli.hmtrack", "hmlib.cli.stitch"}:
        raise ValueError(f"Unsupported desktop launch module: {module}")
    with tempfile.TemporaryDirectory(prefix="hm-desktop-launch-") as temporary:
        directory = Path(temporary)
        if module:
            launch = {
                "version": 1,
                "module": module,
                "arguments": arguments or [],
                "cwd": str(Path.cwd()),
                "game_id": game_id,
            }
            if not game_id:
                game_root = Path(environment.get("HM_GAME_DIR", str(Path.home() / "Videos")))
                game_root = game_root.expanduser().resolve()
                game_id = _anonymous_game(game_root, arguments or [])
                environment["HM_GAME_DIR"] = str(game_root)
                launch["game_id"] = game_id
                launch["anonymous"] = True
            request = directory / "launch.json"
            request.write_text(json.dumps(launch), encoding="utf-8")
            command += [f"--backend-launch={request}", "--play"]
            if module == "hmlib.cli.stitch":
                command.append("--stitching-calibration-only")
        if game_id:
            command.append(f"--game-id={game_id}")
        # The desktop owns every processing child; closing it waits for shutdown.
        return subprocess.call(command, env=environment)


def _anonymous_game(game_root: Path, arguments: list[str]) -> str:
    import yaml

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--input-video", "--input_video", nargs="+")
    parser.add_argument("--input-videos", "--input_videos", nargs="+")
    options, _ = parser.parse_known_args(arguments)
    sources = options.input_video or options.input_videos
    if not sources:
        raise ValueError("Launching the desktop without --game-id requires --input-video")
    sources = [part for source in sources for part in source.split(",")]
    if len(sources) not in {1, 2}:
        raise ValueError("Anonymous desktop input requires one source or a left/right pair")
    paths = [Path(source).expanduser().resolve() for source in sources]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    game_root.mkdir(parents=True, exist_ok=True)
    # Recordings, calibration and telemetry belong to a durable game. Only the
    # initial launch envelope is temporary; closing the UI never removes a run.
    game = Path(tempfile.mkdtemp(prefix="session-input-", dir=game_root))
    names = ["source"] if len(paths) == 1 else ["left", "right"]
    videos = {name: [str(path)] for name, path in zip(names, paths)}
    (game / "config.yaml").write_text(
        yaml.safe_dump({"game": {"videos": videos}, "stitching": {"enabled": len(paths) == 2}}),
        encoding="utf-8",
    )
    return game.name


def use_desktop(camera_ui: int) -> bool:
    return bool(camera_ui) and "HM_UI_SESSION_DIR" not in os.environ


def main() -> None:
    parser = argparse.ArgumentParser(description="HockeyMON using the shared HStream desktop")
    parser.add_argument("--game-id")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    raise SystemExit(launch_desktop(game_id=args.game_id, root_dir=args.root))


if __name__ == "__main__":
    main()
