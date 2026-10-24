"""Authenticated shared-desktop scoreboard selection and frame-boundary saving."""

from __future__ import annotations

import copy
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from hmlib.ui.hstream_runtime import read_document, write_document

logger = logging.getLogger(__name__)
_DISABLED = [[0, 0], [0, 0], [0, 0], [0, 0]]


def _points(value, width: int, height: int) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("Select four scoreboard corners")
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair):
            raise ValueError("Scoreboard corners require integer x and y coordinates")
        if not 0 <= pair[0] < width or not 0 <= pair[1] < height:
            raise ValueError("Scoreboard corner is outside the pinned canvas")
    return value


def _source_paths(config: dict, game_dir: Path) -> list[str]:
    videos = config.get("game", {}).get("videos", {})
    playlists = [value for value in videos.values() if value] if isinstance(videos, dict) else []
    if len(playlists) != 1:
        return []
    values = playlists[0] if isinstance(playlists[0], list) else [playlists[0]]
    return [str((game_dir / path).resolve()) for path in values]


def _accepted_source_paths() -> list[str]:
    sources = json.loads(os.environ["HM_SHARED_SOURCE_CHAPTERS"])
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(path, str) or not Path(path).is_absolute() for path in sources)
    ):
        raise ValueError("Shared source processing requires its accepted source playlist")
    return [str(Path(path).resolve()) for path in sources]


def shared_source_owns_saved_game(game_id: str) -> bool:
    """Whether a shared single-source run may reuse or update saved source geometry."""
    from hmlib.config import get_game_dir

    game_dir = Path(get_game_dir(game_id)).resolve()
    saved = read_document(game_dir / "config.yaml", yaml_document=True)
    return _source_paths(saved, game_dir) == _accepted_source_paths()


def publish_source_canvas(game_id: str, config: dict, image) -> None:
    """Publish one private source canvas, never the saved native panorama.

    The source snapshot lives for the supervised run, including seeks. A saved
    pair's inherited polygon does not describe an explicitly substituted source.
    """
    import cv2

    from hmlib.config import get_game_dir
    from hmlib.utils.image import make_channels_last, to_uint8_image

    session = Path(os.environ["HM_UI_SESSION_DIR"]).resolve()
    game_dir = Path(get_game_dir(game_id)).resolve()
    sources = _accepted_source_paths()
    directory = session / "scoreboard-source"
    directory.mkdir(exist_ok=True)
    owner = {"version": 1, "game_directory": str(game_dir), "sources": sources}
    owner_path = directory / "owner.json"
    if owner_path.exists():
        if read_document(owner_path) != owner:
            raise ValueError("Shared scoreboard reference belongs to a different run source")
        private = read_document(directory / "config.yaml", yaml_document=True)
        # For saved single-source playback the native transaction persisted to
        # the original game; for an override it persisted only to this run.
        saved = read_document(game_dir / "config.yaml", yaml_document=True)
        if _source_paths(saved, game_dir) == sources:
            private["rink"]["scoreboard"] = copy.deepcopy(
                saved.get("rink", {}).get("scoreboard", {})
            )
    else:
        saved = read_document(game_dir / "config.yaml", yaml_document=True)
        preferences = copy.deepcopy(config.get("rink", {}).get("scoreboard", {}))
        if _source_paths(saved, game_dir) != sources:
            explicit_polygon = os.environ.get("HM_SHARED_SOURCE_SCOREBOARD_POLYGON")
            preferences["perspective_polygon"] = (
                json.loads(explicit_polygon) if explicit_polygon is not None else None
            )
        private = {"game": {"videos": {"source": sources}}, "rink": {"scoreboard": preferences}}
        pixels = make_channels_last(to_uint8_image(image))
        if pixels.ndim == 4:
            pixels = pixels[0]
        valid, encoded = cv2.imencode(".png", pixels.detach().cpu().numpy())
        if not valid:
            raise OSError("Cannot encode the shared scoreboard reference canvas")
        temporary = directory / f".s-{os.getpid()}.png"
        try:
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, directory / "s.png")
        finally:
            temporary.unlink(missing_ok=True)
        write_document(directory / "config.yaml", private)
        write_document(owner_path, owner)
    config.setdefault("rink", {}).setdefault("scoreboard", {})["perspective_polygon"] = (
        private.get("rink", {}).get("scoreboard", {}).get("perspective_polygon")
    )


def apply_scoreboard_selection(game_id: str, config: dict, points: list, canvas: dict) -> None:
    """Persist through the native transaction, then publish to Python transforms."""
    from hmlib.config import get_game_dir
    from hmlib.stitching.native_calibration import save_native_scoreboard

    points = _points(points, canvas["width"], canvas["height"])
    if "source_game_directory" in canvas:
        reference = Path(os.environ["HM_UI_SESSION_DIR"]).resolve() / "scoreboard-source"
        if Path(canvas["source_game_directory"]) != reference or canvas["game_directory"] != str(
            Path(get_game_dir(game_id)).resolve()
        ):
            raise ValueError("Scoreboard canvas belongs to a different run or game")
    try:
        ordered = save_native_scoreboard(Path(get_game_dir(game_id)), points, canvas)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Native scoreboard save failed: {error.stderr or error}") from error
    rink = config.setdefault("rink", {})
    scoreboard = rink.setdefault("scoreboard", {})
    scoreboard["perspective_polygon"] = ordered


class ScoreboardSession:
    """Serve the native dialog without blocking the supervisor's event loop.

    Native calibration pins the original PNG. The pipeline acknowledges a save
    only after its frame owner commits native provenance and updates its live
    configuration. Proposal inference runs in a separate Python process so the
    supervisor never loads Torch or inherits a CUDA context.
    """

    def __init__(self, game_dir: Path, session: Path, session_id: str, emit) -> None:
        self.game_dir, self.session = Path(game_dir), Path(session)
        self.session_id, self.emit = session_id, emit
        self.token = secrets.token_hex(32)
        self.server = None
        self.canvas = None
        self.initial = {"points": [], "message": "Select the four scoreboard corners."}
        self._closed = threading.Event()
        self._submission = threading.Lock()
        self._completed = False
        self._proposal = None
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, name="shared-scoreboard", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        from hmlib.stitching.native_calibration import read_scoreboard_canvas

        try:
            reference = self.session / "scoreboard-source"
            if reference.is_dir():
                owner = read_document(reference / "owner.json")
                if owner.get("version") != 1 or owner.get("game_directory") != str(
                    self.game_dir.resolve()
                ):
                    raise ValueError("Scoreboard source reference belongs to a different game")
                self.canvas = read_scoreboard_canvas(
                    self.game_dir, source_reference=reference, destination=self.session
                )
            else:
                self.canvas = read_scoreboard_canvas(self.game_dir, destination=self.session)
            self._prepare_initial()
            if self._closed.is_set():
                return
            owner = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, message, *args):
                    logger.debug("Shared scoreboard: " + message, *args)

                def _send(self, status: int, body: dict):
                    encoded = json.dumps(body, allow_nan=False).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        self.wfile.write(encoded)
                    except (BrokenPipeError, ConnectionResetError) as error:
                        logger.info("Scoreboard dialog disconnected: %s", error)

                def _authorize(self, *, post: bool) -> str:
                    parsed = urlsplit(self.path)
                    query = parse_qs(parsed.query)
                    if query.get("token") != [owner.token]:
                        raise PermissionError("Invalid scoreboard session token")
                    if self.headers.get("Host") != owner.authority:
                        raise PermissionError("Invalid scoreboard host")
                    if post and self.headers.get("Origin") != owner.origin:
                        raise PermissionError("Invalid scoreboard request origin")
                    return parsed.path

                def do_GET(self):
                    try:
                        path = self._authorize(post=False)
                        if path != "/initial":
                            self._send(404, {"error": "Unknown scoreboard endpoint"})
                            return
                        self._send(200, owner.initial)
                    except PermissionError as error:
                        self._send(403, {"error": str(error)})

                def do_POST(self):
                    try:
                        path = self._authorize(post=True)
                        if path not in {"/save", "/none", "/cancel"}:
                            self._send(404, {"error": "Unknown scoreboard endpoint"})
                            return
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 4096:
                            raise ValueError("Scoreboard request must be a bounded JSON object")
                        if self.headers.get_content_type() != "application/json":
                            raise ValueError("Scoreboard request must use application/json")
                        self.connection.settimeout(5)
                        body = json.loads(self.rfile.read(length))
                        if not isinstance(body, dict):
                            raise TypeError("Scoreboard request must be a JSON object")
                        if not owner._submission.acquire(blocking=False):
                            raise ValueError("Another scoreboard submission is in progress")
                        try:
                            if owner._completed or owner._closed.is_set():
                                raise ValueError("Scoreboard session is already complete")
                            if path == "/cancel":
                                message = (
                                    "Scoreboard selection cancelled; keeping the previous selection"
                                )
                            else:
                                points = (
                                    _DISABLED
                                    if path == "/none"
                                    else _points(
                                        body.get("points"),
                                        owner.canvas["width"],
                                        owner.canvas["height"],
                                    )
                                )
                                owner._save(points)
                                message = (
                                    "Scoreboard overlay disabled"
                                    if points == _DISABLED
                                    else "Loaded scoreboard perspective polygon"
                                )
                            owner._completed = True
                            self._send(200, {"message": message})
                            owner.emit(message)
                        finally:
                            owner._submission.release()
                    except PermissionError as error:
                        self._send(403, {"error": str(error)})
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        self._send(409, {"error": str(error)})

            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.authority = f"127.0.0.1:{self.server.server_port}"
            self.origin = f"http://{self.authority}"
            selector_url = f"{self.origin}/?token={self.token}"
            write_document(
                self.session / "scoreboard-selection.json",
                {
                    "version": 1,
                    "game_id": self.game_dir.name,
                    "session_id": self.session_id,
                    "url": selector_url,
                    "image_path": self.canvas["image_path"],
                    "points": self.initial["points"],
                    "scope": "run" if self.canvas.get("persist_to_game") is False else "game",
                },
            )
            self.emit(f"Scoreboard selector: {selector_url}")
            self.server.timeout = 0.05
            while not self._closed.is_set():
                self.server.handle_request()
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            subprocess.SubprocessError,
        ) as error:
            self.emit(f"runtime command failed: playcropper0.scoreboard-reselect: {error}")
        finally:
            if self.server is not None:
                self.server.server_close()
            if self.canvas is not None:
                Path(self.canvas["image_path"]).unlink(missing_ok=True)

    def _prepare_initial(self) -> None:
        reference = self.canvas.get("source_game_directory")
        config_dir = (
            Path(reference) if reference and not self.canvas["persist_to_game"] else self.game_dir
        )
        config = read_document(config_dir / "config.yaml", yaml_document=True)
        saved = config.get("rink", {}).get("scoreboard", {}).get("perspective_polygon")
        if saved is not None:
            points = _points(saved, self.canvas["width"], self.canvas["height"])
            self.initial = {
                "points": [] if points == _DISABLED else points,
                "message": (
                    "Run scoreboard selection loaded. This selection applies only to the current source override."
                    if self.canvas.get("persist_to_game") is False
                    else "Saved scoreboard selection loaded."
                ),
            }
            return
        if self._closed.is_set():
            return
        self._proposal = subprocess.Popen(
            [sys.executable, "-m", __name__, "--propose", self.canvas["image_path"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = self._proposal.communicate(timeout=120)
            if self._proposal.returncode:
                raise ValueError(stderr.strip() or "Scoreboard proposal process failed")
            proposal = json.loads(stdout)
            points = proposal["points"]
            self.initial = {
                "points": (
                    _points(points, self.canvas["width"], self.canvas["height"]) if points else []
                ),
                "message": (
                    "Review the suggested corners before saving."
                    if points
                    else "No scoreboard was detected. Select the four corners manually."
                ),
            }
        except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
            self._proposal.kill()
            self._proposal.communicate()
            self.emit(
                f"Scoreboard proposal unavailable; manual selection remains available: {error}"
            )
            self.initial = {
                "points": [],
                "message": "Scoreboard proposal unavailable. Select the four corners manually.",
            }
        finally:
            self._proposal = None
        if self.canvas.get("persist_to_game") is False:
            self.initial[
                "message"
            ] += " This selection applies only to the current source override."

    def _save(self, points: list) -> None:
        identifier = uuid.uuid4().hex
        request_path = self.session / "runtime-requests" / f"{identifier}.json"
        result_path = self.session / "runtime-results" / f"{identifier}.json"
        deadline = time.monotonic() + 8
        write_document(
            request_path,
            {
                "session_id": self.session_id,
                "deadline": deadline,
                "kind": "operation",
                "operation": {"kind": "scoreboard", "points": points, "canvas": self.canvas},
            },
        )
        try:
            while not self._closed.wait(0.02):
                try:
                    result = read_document(result_path)
                except FileNotFoundError:
                    if time.monotonic() >= deadline:
                        raise ValueError("Pipeline did not acknowledge the scoreboard selection")
                    continue
                if result.get("session_id") != self.session_id or result.get("ok") is not True:
                    raise ValueError(result.get("error", "Stale scoreboard acknowledgement"))
                return
            raise ValueError("Pipeline stopped before applying the scoreboard selection")
        finally:
            request_path.unlink(missing_ok=True)
            result_path.unlink(missing_ok=True)

    def close(self) -> None:
        self._closed.set()
        proposal = self._proposal
        if proposal is not None:
            proposal.terminate()
        if self._thread is not None:
            self._thread.join(timeout=1)


if __name__ == "__main__":
    from PIL import Image

    from hmlib.scoreboard.proposal import propose

    if len(sys.argv) != 3 or sys.argv[1] != "--propose":
        raise SystemExit("Usage: python -m hmlib.ui.hstream_scoreboard --propose IMAGE")
    with Image.open(sys.argv[2]) as image:
        print(json.dumps({"points": propose(image, image.size)}))
