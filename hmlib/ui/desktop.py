"""Own the desktop and its Python pipeline jobs for the whole operator session."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path


class WorkspaceEditors:
    """Bind game editors to one supervisor, including their temporary previews."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.leveling = {}
        from hmlib.ui.runtime_setup import inspect_pipeline

        workspace.register_operation(
            "pipeline.inspect", lambda args: inspect_pipeline(workspace.game(args.get("game_id")))
        )
        workspace.register_operation("models.status", self._model_status)
        workspace.register_operation("models.prepare", self._prepare_model)
        operations = {
            "snapshot": lambda editor, args: editor.snapshot(),
            "update": lambda editor, args: editor.update_settings(
                args["changes"], expected_revision=args.get("expected_revision")
            ),
            "crop": lambda editor, args: editor.set_crop(
                args["crop"],
                auto_crop=args.get("auto_crop", False),
                expected_revision=args.get("expected_revision"),
            ),
            "leveling": lambda editor, args: editor.set_leveling(
                args["rotation_degrees"], expected_revision=args.get("expected_revision")
            ),
            "scoreboard": lambda editor, args: editor.set_scoreboard(
                args["points"], expected_revision=args.get("expected_revision")
            ),
            "scoreboard-proposal": lambda editor, args: editor.propose_scoreboard(
                expected_revision=args.get("expected_revision")
            ),
            "control-points": lambda editor, args: editor.save_control_points(
                args["pairs"], expected_revision=args.get("expected_revision")
            ),
            "rink-map": lambda editor, args: editor.save_rink_map(
                args.get("landmarks", []),
                args.get("goals"),
                rink_size=args.get("rink_size", [60.96, 25.908]),
                observations=args.get("observations"),
                layout=args.get("layout"),
                allow_partial=args.get("allow_partial", False),
                expected_revision=args.get("expected_revision"),
            ),
            "rink-assist": lambda editor, args: editor.assist_rink_map(
                args["observations"],
                detect=args.get("detect", False),
                use_mask=args.get("use_mask", False),
                layout=args.get("layout"),
                expected_revision=args.get("expected_revision"),
            ),
            "lens": lambda editor, args: editor.set_lens_calibration(
                args["calibration"], expected_revision=args.get("expected_revision")
            ),
            "reset_sync": lambda editor, args: editor.reset_synchronization(
                expected_revision=args.get("expected_revision")
            ),
        }
        for name, operation in operations.items():
            workspace.register_operation(f"calibration.{name}", self._operation(name, operation))
        for name in ("open", "estimate", "preview", "save", "close"):
            workspace.register_operation(f"leveling.{name}", self._level_operation(name))

    def _model_status(self, args):
        from hmlib.ui.model_preparation import readiness

        return readiness(self.workspace.game(args.get("game_id")))

    def _prepare_model(self, args):
        from hmlib.ui.model_preparation import preparation_command

        game = self.workspace.game(args.get("game_id"))
        self.workspace.require_idle()
        target = args["target"]
        command = preparation_command(game, target, args.get("frames", 200))
        return self.workspace.start_command(command, game.name, f"prepare-{target}")

    def _operation(self, name, operation):
        def dispatch(args):
            from hmlib.ui.runtime_setup import OperatorWorkspace

            game = self.workspace.game(args.get("game_id"))
            if name != "snapshot":
                self.workspace.require_idle(game)
            return operation(OperatorWorkspace(game), args)

        return dispatch

    def _level_operation(self, name):
        def dispatch(args):
            from hmlib.ui.runtime_setup import OperatorWorkspace

            game = self.workspace.game(args.get("game_id"))
            self.workspace.require_idle(game)
            if name == "open":
                previous = self.leveling.pop(game, None)
                if previous:
                    previous.close()
                editor = OperatorWorkspace(game).leveling_session()
                try:
                    info = editor.info()
                    for source, path in zip(info["sources"], editor.sources):
                        source["path"] = str(path)
                    if args.get("from_rink_map"):
                        estimate = OperatorWorkspace(game).rink_map_leveling(
                            yaw=info["rotation_degrees"][0]
                        )
                        info["rotation_degrees"] = estimate["rotation_degrees"]
                        info["rink_leveling"] = estimate
                    editor.annotate_sources(info)
                except BaseException:
                    editor.close()
                    raise
                self.leveling[game] = editor
                return info
            editor = self.leveling.get(game)
            if editor is None:
                raise ValueError("Open the leveling editor first")
            if name == "estimate":
                return editor.estimate(args["posts"], args.get("yaw", 0))
            if name == "preview":
                result = editor.preview(args["state"])
                result["image"] = str(editor.directory / "preview.png")
                return result
            if name == "save":
                editor.save(args["state"], args["token"])
                editor.close()
                del self.leveling[game]
                return OperatorWorkspace(game).snapshot()
            editor.close()
            del self.leveling[game]
            return {"closed": True}

        return dispatch

    def close(self):
        for editor in self.leveling.values():
            editor.close()
        self.leveling.clear()


def launch_desktop(
    *,
    game_id: str | None = None,
    module: str | None = None,
    arguments: list[str] | None = None,
    root_dir: str | Path | None = None,
) -> int:
    from hmlib.camera.hm_ui_bridge import HmUiProcess
    from hmlib.config import GAME_DIR_BASE
    from hmlib.ui.workspace import Workspace

    command = HmUiProcess._resolve_command()
    if command is None:
        raise RuntimeError("Build hm-ui with `bazelisk build //hm-ui:hm-ui` or set HM_UI_BIN")
    root = Path(root_dir or GAME_DIR_BASE).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep logs/results after shutdown so failures remain inspectable.
    session = Path(tempfile.mkdtemp(prefix="hm-workspace-"))
    live = session / "live"
    live.mkdir()
    HmUiProcess._write_json_atomic(
        live / "spec.json",
        {
            "version": 3,
            "session_id": "workspace-idle",
            "title": "HockeyMON",
            "subtitle": "Select a game and start a run",
            "windows": [],
            "previews": [],
            "runtime_path": str(live / "runtime.json"),
        },
    )
    HmUiProcess._write_json_atomic(
        live / "runtime.json",
        {
            "state": "finished",
            "frame": None,
            "fps": None,
            "message": "Select a game and start a run",
            "updated_ms": int(time.time() * 1000),
            "capabilities": [],
        },
    )
    workspace = Workspace(root, session)
    editors = WorkspaceEditors(workspace)
    ui = None
    try:
        if game_id:
            workspace.dispatch("games.select", {"game_id": game_id})
        workspace.poll()
        with (session / "desktop.log").open("ab") as log:
            ui = subprocess.Popen(
                [
                    *command,
                    "--workspace",
                    str(session),
                    "--spec",
                    str(live / "spec.json"),
                    "--state",
                    str(live / "state.json"),
                    "--title",
                    "HockeyMON",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if module:
            kind = "stitch" if module.endswith(".stitch") else "track"
            workspace.start_command(
                workspace.pipeline_command(
                    workspace.game(game_id) if game_id else None,
                    kind,
                    arguments or [],
                    initial_launch=True,
                ),
                game_id,
                kind,
            )
        while ui.poll() is None:
            workspace.poll()
            time.sleep(0.05)
        if ui.returncode:
            raise RuntimeError(
                f"Desktop exited with status {ui.returncode}; see {session / 'desktop.log'}"
            )
        return 0
    finally:
        workspace.close()
        editors.close()
        if ui is not None and ui.poll() is None:
            ui.terminate()
            try:
                ui.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ui.kill()
                ui.wait(timeout=5)


def use_desktop(camera_ui: int) -> bool:
    return bool(camera_ui) and "HM_UI_SESSION_DIR" not in os.environ


def main() -> None:
    parser = argparse.ArgumentParser(description="HockeyMON operator desktop")
    parser.add_argument("--game-id")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    raise SystemExit(launch_desktop(game_id=args.game_id, root_dir=args.root))


if __name__ == "__main__":
    main()
