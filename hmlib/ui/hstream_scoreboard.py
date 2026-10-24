"""Authenticated shared-desktop scoreboard selection and frame-boundary saving."""

from __future__ import annotations

import json
import logging
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


def apply_scoreboard_selection(game_id: str, config: dict, points: list, canvas: dict) -> None:
    """Persist through the native transaction, then publish to Python transforms."""
    from hmlib.config import get_game_dir
    from hmlib.stitching.native_calibration import save_native_scoreboard

    points = _points(points, canvas["width"], canvas["height"])
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
            self.canvas = read_scoreboard_canvas(self.game_dir)
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
            self.emit(f"Scoreboard selector: {self.origin}/?token={self.token}")
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
        config = read_document(self.game_dir / "config.yaml", yaml_document=True)
        saved = config.get("rink", {}).get("scoreboard", {}).get("perspective_polygon")
        if saved is not None:
            points = _points(saved, self.canvas["width"], self.canvas["height"])
            self.initial = {
                "points": [] if points == _DISABLED else points,
                "message": "Saved scoreboard selection loaded.",
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
