"""Exercise the native dialog's real HTTP requests and pipeline receipts."""

import json
import shutil
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml
from PIL import Image

from hmlib.ui.hstream_runtime import read_document, write_document
from hmlib.ui.hstream_scoreboard import ScoreboardSession


@pytest.fixture
def selector(tmp_path, monkeypatch):
    from hmlib.stitching import native_calibration

    game = tmp_path / "game"
    game.mkdir()
    session = tmp_path / "session"
    session.mkdir()
    points = [[2, 2], [30, 2], [30, 10], [2, 10]]
    (game / "config.yaml").write_text(
        yaml.safe_dump({"rink": {"scoreboard": {"perspective_polygon": points}}})
    )
    Image.new("RGB", (64, 40), "white").save(game / "s.png")

    def canvas(_game, **kwargs):
        image = tmp_path / "pinned.png"
        shutil.copyfile(game / "s.png", image)
        return {
            "width": 64,
            "height": 40,
            "image_path": str(image),
            "expected_generation": {
                "hugin_generation": "",
                "stitched_output_generation": "",
                "snapshot_identity": "pinned",
            },
        }

    monkeypatch.setattr(native_calibration, "read_scoreboard_canvas", canvas)
    messages = []
    server = ScoreboardSession(game, session, "session-1", messages.append)
    server.start()
    deadline = time.monotonic() + 3
    while server.server is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.server is not None, messages
    yield server, session, messages, points
    server.close()
    assert not (tmp_path / "pinned.png").exists()


def request(server, path, body=None, *, token=None, origin=True):
    url = server.origin + path + "?token=" + (server.token if token is None else token)
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
        if origin:
            headers["Origin"] = server.origin
    return urlopen(Request(url, data=data, headers=headers), timeout=3)


def should_authenticate_requests_and_load_saved_selection_without_inference(selector):
    server, _session, _messages, points = selector
    descriptor = read_document(_session / "scoreboard-selection.json")
    assert descriptor["url"] == server.origin + "/?token=" + server.token
    assert descriptor["session_id"] == "session-1"
    assert descriptor["points"] == points
    assert descriptor["scope"] == "game"
    with request(server, "/initial") as response:
        assert json.load(response)["points"] == points
    with pytest.raises(HTTPError) as denied:
        request(server, "/initial", token="bad")
    assert denied.value.code == 403
    with pytest.raises(HTTPError) as denied:
        request(server, "/save", {"points": points}, origin=False)
    assert denied.value.code == 403


def should_wait_for_frame_owner_before_success_and_send_native_completion(selector):
    server, session, messages, points = selector
    results = []
    thread = threading.Thread(
        target=lambda: results.append(request(server, "/save", {"points": points}).status)
    )
    thread.start()
    deadline = time.monotonic() + 2
    requests = []
    while not requests and time.monotonic() < deadline:
        requests = list((session / "runtime-requests").glob("*.json"))
        time.sleep(0.01)
    assert len(requests) == 1
    assert results == []
    payload = read_document(requests[0])
    assert payload["operation"]["canvas"]["expected_generation"]["snapshot_identity"] == "pinned"
    assert payload["operation"]["points"] == points
    write_document(
        session / "runtime-results" / requests[0].name, {"ok": True, "session_id": "session-1"}
    )
    thread.join(timeout=2)
    assert results == [200]
    deadline = time.monotonic() + 1
    while "Loaded scoreboard perspective polygon" not in messages and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "Loaded scoreboard perspective polygon" in messages


def should_reject_bad_corners_and_cancel_without_mutation(selector):
    server, session, messages, points = selector
    points[0][0] = 64
    with pytest.raises(HTTPError) as rejected:
        request(server, "/save", {"points": points})
    assert rejected.value.code == 409
    assert not (session / "runtime-requests").exists()
    with request(server, "/cancel", {}) as response:
        assert response.status == 200
    deadline = time.monotonic() + 1
    while (
        not any(message.startswith("Scoreboard selection cancelled;") for message in messages)
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert any(message.startswith("Scoreboard selection cancelled;") for message in messages)
    assert not (session / "runtime-requests").exists()
