"""Open the local browser editor for NONA rink leveling and normalized crop."""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from hmlib.stitching.leveling_editor import LevelingSession
from hmlib.stitching.leveling_page import PAGE

logger = logging.getLogger(__name__)


def create_server(session: LevelingSession, port: int = 0) -> tuple[HTTPServer, str]:
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.info(format, *args)

        def _send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            )
            try:
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                logger.info("Browser disconnected before the %s response was delivered", self.path)

        def _local_request(self):
            # Restrict Host as well as bind address to reject DNS rebinding.
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self._local_request():
                self._send(403, b'{"error":"Invalid editor host"}')
                return
            if self.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            for index, data in enumerate(session.source_images):
                if self.path == f"/source/{index}.jpg":
                    self._send(200, data, "image/jpeg")
                    return
            if session.preview_token and self.path == f"/preview/{session.preview_token}.png":
                self._send(200, session.preview_image, "image/png")
                return
            self._send(404, b'{"error":"No such image"}')

        def do_POST(self):
            if not self._local_request() or not secrets.compare_digest(
                self.headers.get("X-Editor-Token", ""), token
            ):
                self._send(403, b'{"error":"Invalid editor token"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Editor request is empty or too large")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Editor request must be an object")
                if self.path == "/info":
                    result = session.info()
                elif self.path == "/estimate":
                    result = session.estimate(data["posts"], data["yaw"])
                elif self.path == "/preview":
                    result = session.preview(data)
                elif self.path == "/save":
                    session.save(data["state"], data["token"])
                    result = {"saved": True}
                elif self.path == "/close":
                    result = {"closed": True}
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    raise ValueError("Unknown editor action")
                self._send(200, json.dumps(result, allow_nan=False).encode("utf-8"))
            except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
                logger.warning("Rink editor %s failed: %s", self.path, exc)
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"))

    server = HTTPServer(("127.0.0.1", port), Handler)
    return server, f"http://127.0.0.1:{server.server_port}/#{token}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    from hmlib.config import get_config, get_game_dir

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    session = LevelingSession(
        Path(get_game_dir(args.game_id)), lambda: get_config(game_id=args.game_id)
    )
    try:
        server, url = create_server(session, args.port)
        try:
            print(f"Rink leveling and crop: {url}", flush=True)
            if not args.no_browser and not webbrowser.open(url):
                logger.warning("Could not launch a browser; open the URL above manually")
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            logger.info("Editor closed")
        finally:
            server.server_close()
    finally:
        session.close()


if __name__ == "__main__":
    main()
