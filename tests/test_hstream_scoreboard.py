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


@pytest.mark.parametrize("owns_saved_source", [False, True])
@pytest.mark.parametrize("initial_disabled", [False, True])
@pytest.mark.parametrize("selected_disabled", [False, True])
def should_retain_source_selection_through_initial_launch_save_seek_and_reopen(
    tmp_path, monkeypatch, owns_saved_source, initial_disabled, selected_disabled
):
    import copy

    import torch

    import hmlib.config as configuration
    from hmlib.stitching import native_calibration
    from hmlib.transforms.scoreboard_transforms import HmConfigureScoreboard
    from hmlib.ui.hstream_scoreboard import apply_scoreboard_selection

    game, session = tmp_path / "game", tmp_path / "session"
    game.mkdir()
    session.mkdir()
    source = tmp_path / "source.mp4"
    source.touch()
    saved = [[2, 2], [30, 2], [30, 10], [2, 10]]
    initial = [[0, 0]] * 4 if initial_disabled else [[10, 4], [40, 4], [40, 20], [10, 20]]
    selected = [[0, 0]] * 4 if selected_disabled else [[4, 8], [44, 8], [44, 24], [4, 24]]
    config = {
        "game": {
            "videos": (
                {"source": [str(source)]}
                if owns_saved_source
                else {"left": ["left.mp4"], "right": ["right.mp4"]}
            )
        },
        "rink": {
            "scoreboard": {
                "perspective_polygon": saved,
                "projected_width": 16,
                "projected_height": 8,
            }
        },
    }
    write_document(game / "config.yaml", config)
    Image.new("RGB", (64, 40), "blue").save(game / "s.png")
    saved_image = (game / "s.png").read_bytes()
    monkeypatch.setattr(configuration, "get_game_dir", lambda game_id: str(game))
    monkeypatch.setenv("HM_SHARED_UI_BACKEND", "1")
    monkeypatch.setenv("HM_UI_SESSION_DIR", str(session))
    monkeypatch.setenv("HM_SHARED_SOURCE_CHAPTERS", json.dumps([str(source)]))
    monkeypatch.setenv("HM_SHARED_SOURCE_SCOREBOARD_POLYGON", json.dumps(initial))
    monkeypatch.delenv("HM_SHARED_NATIVE_CALIBRATION", raising=False)
    monkeypatch.setattr(
        "hmlib.ui.hstream_scoreboard.subprocess.Popen",
        lambda *a, **kw: pytest.fail("Saved or disabled corners must bypass proposal inference"),
    )
    private = session / "scoreboard-source"
    canvas = {
        "width": 64,
        "height": 40,
        "game_directory": str(game),
        "source_game_directory": str(private),
        "persist_to_game": owns_saved_source,
    }

    def polygon(directory):
        return read_document(directory / "config.yaml", yaml_document=True)["rink"]["scoreboard"][
            "perspective_polygon"
        ]

    def rebuild(expected):
        runtime = copy.deepcopy(config)
        runtime["rink"]["scoreboard"]["perspective_polygon"] = copy.deepcopy(initial)
        transform = HmConfigureScoreboard(game_id="game", allow_interactive_setup=False)
        result = transform(
            {"game_config": runtime, "img": torch.full((1, 40, 64, 3), 71, dtype=torch.uint8)}
        )
        assert runtime["rink"]["scoreboard"]["perspective_polygon"] == expected
        if expected == [[0, 0]] * 4:
            assert "scoreboard_cfg" not in result
        else:
            assert result["scoreboard_cfg"]["scoreboard_points"] == expected
        dialog = ScoreboardSession(game, session, "session-1", lambda message: None)
        dialog.canvas = canvas
        dialog._prepare_initial()
        assert dialog.initial["points"] == ([] if expected == [[0, 0]] * 4 else expected)
        return runtime

    runtime = rebuild(initial)
    assert polygon(game) == saved
    rebuild(initial)
    assert polygon(game) == saved
    assert polygon(private) == initial
    pinned_image = (private / "s.png").read_bytes()

    def native_save(directory, points, pinned):
        assert directory == game and pinned == canvas
        assert polygon(private) == initial
        # The native helper commits to exactly one destination: a same-source
        # save updates the game; a source override updates the private run.
        destination = game if owns_saved_source else private
        document = read_document(destination / "config.yaml", yaml_document=True)
        document["rink"]["scoreboard"]["perspective_polygon"] = points
        write_document(destination / "config.yaml", document)
        return points

    monkeypatch.setattr(native_calibration, "save_native_scoreboard", native_save)
    apply_scoreboard_selection("game", runtime, selected, canvas)
    assert runtime["rink"]["scoreboard"]["perspective_polygon"] == selected
    assert polygon(private) == selected
    assert polygon(game) == (selected if owns_saved_source else saved)
    rebuild(selected)
    assert (game / "s.png").read_bytes() == saved_image
    assert (private / "s.png").read_bytes() == pinned_image
