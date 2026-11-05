"""Keep accepted scoreboard corners attached to this run's native output canvas."""

from __future__ import annotations

import copy
import os
from pathlib import Path

from hmlib.ui.hstream_runtime import read_document, write_document

_NAME = "native-scoreboard.json"


def _read(session: Path, game: Path) -> dict | None:
    path = session / _NAME
    if not path.exists():
        return None
    state = read_document(path)
    if (
        state.get("version") != 1
        or state.get("session") != str(session.resolve())
        or state.get("game_directory") != str(game.resolve())
        or not state.get("hugin_generation")
        or not state.get("output_generation")
    ):
        raise ValueError("Native scoreboard selection belongs to a different run or game")
    return state


def _apply(config: dict, state: dict) -> None:
    config.setdefault("rink", {}).setdefault("scoreboard", {})["perspective_polygon"] = (
        copy.deepcopy(state["points"])
    )
    config.setdefault("stitching", {})["post_stitch_rotate_degrees"] = state["rotation"]


def _epoch_points(epoch: dict) -> list[list[int]]:
    coordinates = [int(value) for value in epoch["scoreboard_property_value"].split(",")]
    if len(coordinates) != 8:
        raise ValueError("Native output epoch has an invalid scoreboard polygon")
    return [coordinates[index : index + 2] for index in range(0, 8, 2)]


def prepare_native_scoreboard_export(config: dict, session: Path, game: Path) -> None:
    """Override original launch choices before exporting; verify the export afterward."""
    state = _read(session, game)
    if state is not None:
        _apply(config, state)


def restore_native_scoreboard(config: dict, session: Path, native, *, initialize=False) -> None:
    """Replay only corners for the exact validated native snapshot being loaded."""
    state = _read(session, native.game_directory)
    if state is None:
        if not initialize:
            return
        state = {
            "version": 1,
            "session": str(session.resolve()),
            "game_directory": str(native.game_directory.resolve()),
            "hugin_generation": native.generation,
            "output_generation": native.output_generation,
            "base_output_generation": native.output_generation,
            "rotation": native.post_stitch_rotate_degrees,
            "points": copy.deepcopy(
                config.get("rink", {}).get("scoreboard", {}).get("perspective_polygon")
            ),
        }
        write_document(session / _NAME, state)
    if state["hugin_generation"] != native.generation:
        raise ValueError("Native scoreboard selection belongs to a different output generation")
    if state["output_generation"] == native.output_generation:
        if state["rotation"] != native.post_stitch_rotate_degrees:
            raise ValueError("Native scoreboard selection belongs to a different output generation")
        config.pop("hm_shared_runtime", None)
    elif state.get("base_output_generation") == native.output_generation and state.get(
        "serialized_epoch"
    ):
        from hmlib.stitching.native_calibration import validate_live_output_epoch

        # Python rotates frames privately; the saved native canvas can still
        # name the completed base output. Revalidate its pending authorization
        # rather than pretending the old PNG represents the active rotation.
        epoch = validate_live_output_epoch(
            native.game_directory, state["serialized_epoch"], native=native
        )
        if (
            epoch["output_generation"] != state["output_generation"]
            or epoch["post_stitch_rotate_degrees"] != state["rotation"]
            or _epoch_points(epoch) != state["points"]
        ):
            raise ValueError("Native scoreboard selection differs from its authorized output epoch")
        config["hm_shared_runtime"] = {
            "output_generation": epoch["output_generation"],
            "authorization_id": epoch["authorization_id"],
            "post_stitch_rotate_degrees": epoch["post_stitch_rotate_degrees"],
        }
    else:
        raise ValueError("Native scoreboard selection belongs to a different output generation")
    _apply(config, state)


def restore_worker_native_scoreboard(config: dict, game_id: str) -> None:
    """Run after original worker YAML/CLI overrides, before resolving graph globals."""
    if os.environ.get("HM_SHARED_UI_BACKEND") != "1":
        return
    from hmlib.config import get_game_dir
    from hmlib.stitching.native_calibration import shared_native_calibration

    native = shared_native_calibration()
    if native is None:
        return
    if native.game_directory != Path(get_game_dir(game_id)).resolve():
        raise ValueError("Native scoreboard calibration belongs to a different game")
    restore_native_scoreboard(
        config, Path(os.environ["HM_UI_SESSION_DIR"]).resolve(), native, initialize=True
    )


def native_selection_for_canvas(session: Path, game: Path, canvas: dict) -> dict | None:
    """Reject a pinned native dialog once its canvas differs from the active run."""
    state = _read(session, game)
    if state is not None:
        expected = canvas["expected_generation"]
        if (
            state["hugin_generation"] != expected["hugin_generation"]
            or state["output_generation"] != expected["stitched_output_generation"]
        ):
            raise ValueError("Scoreboard dialog belongs to a different active output generation")
    return state


def save_native_selection(session: Path, state: dict, points: list) -> None:
    write_document(session / _NAME, {**state, "points": points})


def publish_native_scoreboard_epoch(
    game_id: str, previous_config: dict, epoch: dict, serialized: str
) -> None:
    """Commit only an epoch already accepted by native validation and the frame owner."""
    if os.environ.get("HM_SHARED_UI_BACKEND") != "1":
        return
    from hmlib.config import get_game_dir
    from hmlib.stitching.native_calibration import shared_native_calibration

    native = shared_native_calibration()
    if native is None:
        raise ValueError("Native scoreboard epoch has no loaded calibration")
    session = Path(os.environ["HM_UI_SESSION_DIR"]).resolve()
    state = _read(session, Path(get_game_dir(game_id)))
    current_output = previous_config.get("hm_shared_runtime", {}).get(
        "output_generation", native.output_generation
    )
    if (
        state is None
        or state["hugin_generation"] != native.generation
        or state["output_generation"] != current_output
    ):
        raise ValueError("Native scoreboard epoch does not match the active run generation")
    write_document(
        session / _NAME,
        {
            **state,
            "output_generation": epoch["output_generation"],
            "base_output_generation": native.output_generation,
            "serialized_epoch": serialized,
            "rotation": epoch["post_stitch_rotate_degrees"],
            "points": _epoch_points(epoch),
        },
    )
