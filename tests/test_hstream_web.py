"""Opt-in Program producer: portable admission, local pins and confined receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import selectors
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import test_hstream_supervisor as supervisor_tests
import yaml

from hmlib.cli.hstream_backend import Supervisor, main
from hmlib.ui import hstream_web as web
from hmlib.utils.output_publication import publish_artifacts

process_tree = supervisor_tests.process_tree


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    root = tmp_path / "run"
    (root / "game").mkdir(parents=True)
    native = tmp_path / "native"
    for key, relative in web.NATIVE_FILES.items():
        path = native / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"baseline: true\n" if key == "baseline" else b"#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
    media = tmp_path / "media.mp4"
    media.write_bytes(b"immutable uploaded source")
    (root / "game/source.mp4").symlink_to(media)
    (root / "game/config.yaml").write_text(
        yaml.safe_dump(
            {
                "game": {"videos": {"source": [str(media)]}},
                "preview_overlays": {"players": False, "play": False, "rink": False},
            }
        )
    )
    runtime = {
        "content_sha256": "1" * 64,
        "native_extension": True,
        "file_count": 3,
        "baseline": web.file_content(native / web.NATIVE_FILES["baseline"]),
        "python": {
            "implementation": "cpython",
            "version": list(sys.version_info[:3]),
            "size": 1,
            "sha256": "2" * 64,
        },
    }
    monkeypatch.setattr(web, "runtime_provenance", lambda: copy.deepcopy(runtime))
    document = {
        "version": 1,
        "run_id": str(uuid4()),
        "deployment_id": "3" * 64,
        "runtime": runtime,
        "native_root": str(native),
        "native": {
            key: web.file_content(native / relative) for key, relative in web.NATIVE_FILES.items()
        },
        "game_config": web.file_content(root / "game/config.yaml"),
        "sources": {
            "source": [
                {"alias": "game/source.mp4", "path": str(media), "content": web.file_content(media)}
            ]
        },
        "launch": {
            "sinks": ["FAKE"],
            "telemetry": False,
            "start_seconds": 0,
            "duration_seconds": 2,
            "realtime": False,
            "preview": "none",
        },
        "calibration": None,
    }
    request = root / web.REQUEST_NAME

    def write():
        document["request_digest"] = web.request_digest(document)
        request.write_bytes(web.canonical_bytes(document))

    write()
    return SimpleNamespace(
        root=root,
        native=native,
        media=media,
        runtime=runtime,
        document=document,
        request=request,
        write=write,
    )


def open_admitted(fixture, monkeypatch):
    context = web.WebProgram(fixture.request, environment={})
    environment = context.environment({"PATH": os.environ["PATH"]})
    environment.update(
        HM_UI_SESSION_DIR=str(context.session),
        HM_SHARED_WORK_DIR=str(context.output),
        HM_BACKEND_PUBLICATION_FILE=str(context.session / "publication.json"),
    )
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    context.write_admission()
    return context


def should_have_a_portable_explicit_canonical_request_digest(admitted):
    document = {"z": "é", "a": [1, True, None, 1.25], "request_digest": "ignored"}
    assert (
        web.canonical_bytes({"z": "é", "a": [1, True, None, 1.25]})
        == b'{"a":[1,true,null,1.25],"z":"\xc3\xa9"}'
    )
    assert (
        web.request_digest(document)
        == hashlib.sha256(b'{"a":[1,true,null,1.25],"z":"\xc3\xa9"}').hexdigest()
    )
    assert set(admitted.document["native"]["runner"]) == {"size", "sha256"}
    context = web.WebProgram(admitted.request, environment={})
    try:
        assert "device" in context.pins["native_runner"].identity
        assert "device" not in admitted.document["runtime"]
    finally:
        context.close()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(version=True),
        lambda d: d.update(run_id="not-a-uuid"),
        lambda d: d.update(deployment_id="not-a-deployment"),
        lambda d: d.update(extra="untrusted"),
        lambda d: d["native"].update(extra={"path": "/etc/passwd"}),
        lambda d: d["launch"].update(argv=["--dry-run"]),
        lambda d: d["launch"].update(telemetry=1),
        lambda d: d["launch"].update(start_seconds=-1),
        lambda d: d["launch"].update(sinks=["WEBRTC"]),
        lambda d: d["launch"].update(sinks=["FAKE", "FAKE"]),
        lambda d: d["sources"]["source"][0].update(alias="../outside"),
    ],
)
def should_reject_untyped_or_unconfined_requests(admitted, mutation):
    mutation(admitted.document)
    admitted.write()
    with pytest.raises((ValueError, TypeError)):
        web.WebProgram(admitted.request, environment={})


def should_reject_digest_runtime_and_desktop_launch_replacements(admitted):
    with pytest.raises(ValueError, match="initial-launch"):
        web.WebProgram(
            admitted.request, environment={"HM_UI_INITIAL_LAUNCH": "/private/desktop.json"}
        )
    admitted.document["request_digest"] = "0" * 64
    admitted.request.write_bytes(web.canonical_bytes(admitted.document))
    with pytest.raises(ValueError, match="digest"):
        web.WebProgram(admitted.request, environment={})
    admitted.document["runtime"] = {**admitted.runtime, "content_sha256": "0" * 64}
    admitted.write()
    with pytest.raises(ValueError, match="runtime"):
        web.WebProgram(admitted.request, environment={})


def should_reject_runtime_boolean_integer_equivalence(admitted):
    admitted.document["runtime"] = {**admitted.runtime, "native_extension": 1}
    admitted.write()
    with pytest.raises(ValueError, match="runtime"):
        web.WebProgram(admitted.request, environment={})


@pytest.mark.parametrize("replacement", ["symlink", "fifo", "oversized"])
def should_read_request_only_as_bounded_regular_metadata(admitted, replacement):
    original = admitted.request.read_bytes()
    admitted.request.unlink()
    if replacement == "symlink":
        other = admitted.root / "other.json"
        other.write_bytes(original)
        admitted.request.symlink_to(other)
    elif replacement == "fifo":
        os.mkfifo(admitted.request)
    else:
        admitted.request.write_bytes(b"x" * (web.MAX_REQUEST_BYTES + 1))
    with pytest.raises((ValueError, OSError)):
        web.WebProgram(admitted.request, environment={})


def should_pin_helpers_and_detect_mutation_even_with_restored_mtime(admitted):
    context = web.WebProgram(admitted.request, environment={})
    try:
        helper = admitted.native / web.NATIVE_FILES["model_prepare"]
        info = helper.stat()
        helper.write_bytes(b"#!/bin/sh\nexit 9\n")
        os.utime(helper, ns=(info.st_atime_ns, info.st_mtime_ns))
        with pytest.raises(ValueError, match="changed"):
            context.check()
    finally:
        context.close()


def should_reject_replaced_private_directories_and_source_aliases(admitted):
    context = web.WebProgram(admitted.request, environment={})
    try:
        alias = admitted.root / "game/source.mp4"
        alias.unlink()
        alias.symlink_to(admitted.native / web.NATIVE_FILES["runner"])
        with pytest.raises(ValueError, match="alias"):
            context.check()
        alias.unlink()
        alias.symlink_to(admitted.media)
        context.home.rename(admitted.root / "old-home")
        context.home.symlink_to(admitted.root / "old-home", target_is_directory=True)
        with pytest.raises(OSError):
            context.check()
    finally:
        context.close()


def should_keep_child_environment_private_and_recheck_worker_pins(admitted, monkeypatch):
    context = open_admitted(admitted, monkeypatch)
    try:
        environment = context.environment(
            {
                "PATH": "/bin",
                "PYTHONPATH": "/untrusted",
                "HOME": "/other",
                "HM_DB_PASSWORD": "not-a-real-password",
                "HSTREAM_UI_BACKEND_SESSION": "/other",
            }
        )
        assert set(environment) & {"PYTHONPATH", "HM_DB_PASSWORD"} == set()
        assert environment["HOME"] == str(context.home)
        assert environment["HSTREAM_UI_PARENT_PID"] == str(os.getppid())
        web.verify_worker_environment()
        helper = admitted.native / web.NATIVE_FILES["model_infer"]
        original = helper.read_bytes()
        helper.unlink()
        helper.write_bytes(original)
        with pytest.raises(ValueError, match="after producer admission"):
            web.verify_worker_environment()
    finally:
        context.close()


@pytest.mark.parametrize(
    "selection,expected",
    [
        ({"players": False, "play": False, "rink": False}, ""),
        ({"players": True, "play": False, "rink": True}, "players,rink"),
        ({"players": True, "play": True, "rink": True}, "players,play,rink"),
    ],
)
def should_initialize_overlays_only_from_the_pinned_config(admitted, selection, expected):
    path = admitted.root / "game/config.yaml"
    config = yaml.safe_load(path.read_text())
    config["preview_overlays"] = selection
    path.write_text(yaml.safe_dump(config))
    admitted.document["game_config"] = web.file_content(path)
    admitted.write()
    context = web.WebProgram(admitted.request, environment={})
    try:
        result = context.environment({"HSTREAM_UI_PREVIEW_OVERLAYS": "untrusted,play"})
        assert result["HSTREAM_UI_PREVIEW_OVERLAYS"] == expected
        path.write_text(yaml.safe_dump({**config, "preview_overlays": {"play": True}}))
        with pytest.raises(ValueError, match="changed"):
            context.environment({})
    finally:
        context.close()


@pytest.mark.parametrize(
    "selection",
    [
        None,
        {},
        {"players": True},
        {"players": 1, "play": False, "rink": False},
        {"players": "false", "play": False, "rink": False},
        {"players": False, "play": False, "rink": False, "extra": True},
    ],
)
def should_require_frozen_complete_boolean_overlay_selection(admitted, selection):
    path = admitted.root / "game/config.yaml"
    config = yaml.safe_load(path.read_text())
    config["preview_overlays"] = selection
    path.write_text(yaml.safe_dump(config))
    admitted.document["game_config"] = web.file_content(path)
    admitted.write()
    context = web.WebProgram(admitted.request, environment={})
    try:
        with pytest.raises(ValueError, match="preview_overlays"):
            context.environment({"HSTREAM_UI_PREVIEW_OVERLAYS": "players,play,rink"})
    finally:
        context.close()


def should_refuse_a_second_producer_and_replay_of_an_admitted_request(admitted):
    context = web.WebProgram(admitted.request, environment={})
    try:
        with pytest.raises(BlockingIOError):
            web.WebProgram(admitted.request, environment={})
        context.write_admission()
    finally:
        context.close()
    with pytest.raises(ValueError, match="already admitted"):
        web.WebProgram(admitted.request, environment={})


def should_require_completed_calibration_for_paired_inputs_before_any_spawn(admitted):
    source = admitted.document["sources"].pop("source")
    admitted.document["sources"] = {"left": source, "right": copy.deepcopy(source)}
    admitted.write()
    with pytest.raises(ValueError, match="Prepare stitching"):
        web.WebProgram(admitted.request, environment={})


def should_skip_native_calibration_and_local_audio_in_web_program():
    accepted = []
    snapshot = object()
    owner = SimpleNamespace(
        web=SimpleNamespace(check=lambda: accepted.append("checked"), calibration=snapshot),
        stop_requested=False,
        plan=SimpleNamespace(sources={"left": [], "right": []}),
        _prepare_models=lambda: None,
        _native_runner=lambda: pytest.fail("hidden native calibration"),
        _accept_calibration=lambda: accepted.append("accepted"),
    )
    Supervisor._prepare(owner)
    assert owner.snapshot is snapshot and accepted == ["checked", "accepted"]
    Supervisor._prepare_audio(owner)


def should_publish_only_exact_confined_outputs_after_private_work_cleanup(admitted, monkeypatch):
    admitted.document["launch"]["sinks"] = ["ENCODE_FILE"]
    admitted.write()
    context = open_admitted(admitted, monkeypatch)
    try:
        source = context.output / "video.mp4"
        source.write_bytes(b"closed video")
        metadata = context.output / "video.json"
        metadata.write_text('{"complete":true}')
        result = publish_artifacts(
            {"video.mp4": source, "video.json": metadata},
            context.game,
            kinds={"video.mp4": "program"},
        )
        records = web.read_publications(context.root)
        assert {row["kind"] for row in records} == {"program", "metadata"}
        source.unlink()
        metadata.unlink()
        receipt = context.receipt("complete", records)
        assert receipt["request_digest"] == admitted.document["request_digest"]
        assert {row["path"] for row in receipt["artifacts"]} == {
            "game/" + path.name for path in result.files.values()
        }
        assert all(
            {"device", "inode", "size", "mtime_ns", "ctime_ns"} <= row.keys()
            for row in receipt["artifacts"]
        )
        assert json.loads((context.root / web.RECEIPT_NAME).read_bytes()) == receipt
    finally:
        context.close()


@pytest.mark.parametrize("bad", ["escaped", "missing", "replaced", "extra_kind", "cancelled"])
def should_reject_unsafe_or_incomplete_terminal_publication(admitted, monkeypatch, bad):
    admitted.document["launch"]["sinks"] = ["ENCODE_FILE"]
    admitted.write()
    context = open_admitted(admitted, monkeypatch)
    try:
        source = context.output / "video.mp4"
        source.write_bytes(b"closed video")
        result = publish_artifacts(
            {"video.mp4": source}, context.game, kinds={"video.mp4": "program"}
        )
        records = web.read_publications(context.root)
        if bad == "escaped":
            records[0]["path"] = str(source)
        elif bad == "missing":
            records = []
        elif bad == "replaced":
            target = result.files["video.mp4"]
            data = target.read_bytes()
            target.unlink()
            target.write_bytes(data)
        elif bad == "extra_kind":
            records[0]["kind"] = "stitched"
        with pytest.raises(ValueError):
            context.receipt("cancelled" if bad == "cancelled" else "complete", records)
        assert not (context.root / web.RECEIPT_NAME).exists()
    finally:
        context.close()


def should_refuse_publication_outside_game_before_copying(admitted, monkeypatch):
    context = open_admitted(admitted, monkeypatch)
    try:
        source = context.output / "video.mp4"
        source.write_bytes(b"closed video")
        with pytest.raises(ValueError, match="private game"):
            publish_artifacts(
                {"video.mp4": source}, context.root.parent, kinds={"video.mp4": "program"}
            )
        assert not (context.root.parent / "video-1.mp4").exists()
    finally:
        context.close()


@pytest.mark.parametrize("bad", ["duplicate_kind", "different_generation"])
def should_require_one_common_generation_and_unique_output_kinds(admitted, monkeypatch, bad):
    admitted.document["launch"]["sinks"] = ["ENCODE_FILE", "ENCODE_STITCHED_FILE"]
    admitted.write()
    context = open_admitted(admitted, monkeypatch)
    try:
        sources = {}
        for name in ("video.mp4", "stitched.mp4"):
            sources[name] = context.output / name
            sources[name].write_bytes(b"closed video")
        publish_artifacts(
            sources, context.game, kinds={"video.mp4": "program", "stitched.mp4": "stitched"}
        )
        records = web.read_publications(context.root)
        if bad == "duplicate_kind":
            records[1]["kind"] = records[0]["kind"]
            message = "repeated"
        else:
            records[1]["version"] += 1
            message = "generation"
        with pytest.raises(ValueError, match=message):
            context.receipt("complete", records)
        assert not (context.root / web.RECEIPT_NAME).exists()
    finally:
        context.close()


def should_refuse_changed_helper_before_spawning_any_child(admitted, monkeypatch):
    context = web.WebProgram(admitted.request, environment={})
    try:
        helper = admitted.native / web.NATIVE_FILES["runner"]
        helper.unlink()
        helper.write_bytes(b"replacement executable")
        owner = SimpleNamespace(web=context)
        monkeypatch.setattr(
            subprocess, "Popen", lambda *args, **kwargs: pytest.fail("spawned replaced helper")
        )
        with pytest.raises(ValueError, match="changed"):
            Supervisor._spawn(owner, [str(helper)], phase="calibration")
    finally:
        context.close()


def should_keep_desktop_handshake_and_reject_mixed_web_arguments(capsys):
    assert main(["--ui-backend-info"]) == 0
    assert json.loads(capsys.readouterr().out)["protocol"] == "hstream-ui"
    assert main(["--web-program-request", "/run/request.json", "--dry-run"]) == 1
    assert "cannot be combined" in capsys.readouterr().out


def should_hash_installed_native_libraries_without_importing_gpu_modules(tmp_path):
    """A synthetic installed package exercises the real lightweight CLI path."""
    installed = tmp_path / "installed"
    package = installed / "hmlib"
    for folder in (
        package,
        package / "ui",
        package / "cli",
        package / "config",
        installed / "hockeymon",
    ):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    shutil.copyfile(web.__file__, package / "ui/hstream_web.py")
    shutil.copyfile(
        Path(web.__file__).parent / "hstream_web_control.py", package / "ui/hstream_web_control.py"
    )
    shutil.copyfile(
        Path(web.__file__).parents[1] / "cli/hstream_backend.py", package / "cli/hstream_backend.py"
    )
    (package / "config/baseline.yaml").write_text("baseline: true\n")
    extension = installed / "hockeymon/_hockeymon.fixture.so"
    extension.write_bytes(b"synthetic native extension bytes")
    library = package / "owned-library.so"
    library.write_bytes(b"synthetic owned shared library")
    script = """
import importlib.abc, sys
class BlockGpuImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'cv2', 'yaml', 'hockeymon'}:
            raise AssertionError('Heavy import during handshake: '+fullname)
sys.meta_path.insert(0, BlockGpuImports())
sys.path.insert(0, sys.argv[1])
from hmlib.cli.hstream_backend import main
raise SystemExit(main(['--web-backend-info']))
"""

    def handshake():
        result = subprocess.run(
            [sys.executable, "-I", "-c", script, str(installed)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return json.loads(result.stdout)

    first = handshake()
    assert first["protocol"] == "hstream-web-program" and first["qualified"] is False
    assert first["runtime"]["native_extension"] is True
    extension.write_bytes(b"different native extension bytes")
    second = handshake()
    assert first["runtime"]["content_sha256"] != second["runtime"]["content_sha256"]
    library.write_bytes(b"different owned shared library")
    assert handshake()["runtime"]["content_sha256"] != second["runtime"]["content_sha256"]


def should_reap_guarded_web_worker_and_stubborn_descendants(admitted, process_tree):
    context = web.WebProgram(admitted.request, environment={})
    owner = SimpleNamespace(
        web=context,
        environment=context.environment(os.environ),
        plan=SimpleNamespace(cwd=context.root),
        selector=selectors.DefaultSelector(),
        child=None,
    )
    try:
        Supervisor._spawn(
            owner,
            [sys.executable, str(process_tree), str(context.root), "stubborn"],
            phase="python",
        )
        supervisor_tests.wait_for(lambda: (context.root / "ready").exists())
        Supervisor._terminate_child(owner)
        supervisor_tests.assert_reaped(context.root)
    finally:
        if owner.child is not None and owner.child.poll() is None:
            Supervisor._terminate_child(owner)
        owner.selector.close()
        context.close()


def should_retire_resources_and_work_before_cancellation_receipt(admitted, monkeypatch):
    context = open_admitted(admitted, monkeypatch)
    work = context.output / "private-run"
    work.mkdir()
    context.working_directory = work
    actions = []
    owner = SimpleNamespace(
        web=context,
        session=context.session,
        plan=SimpleNamespace(launch_receipt=dict),
        stop_requested=False,
    )
    owner._prepare = lambda: setattr(owner, "stop_requested", True)
    owner._close_resources = lambda: actions.append("closed")

    def cleanup():
        assert actions == ["closed"]
        work.rmdir()
        actions.append("cleaned")

    owner._cleanup_working_directory = cleanup

    def receipt(status, artifacts):
        assert actions == ["closed", "cleaned"]
        assert status == "cancelled" and artifacts == []
        context.receipt(status, artifacts)
        actions.append("receipt")

    owner._web_publication = receipt
    try:
        with pytest.raises(ValueError, match="cleanup"):
            context.receipt("cancelled", [])
        assert Supervisor.run(owner) == 0
        assert actions == ["closed", "cleaned", "receipt"]
    finally:
        context.close()


def should_not_publish_receipt_when_resource_cleanup_is_unconfirmed(admitted, monkeypatch):
    context = open_admitted(admitted, monkeypatch)
    owner = SimpleNamespace(
        web=context,
        session=context.session,
        plan=SimpleNamespace(launch_receipt=dict),
        stop_requested=True,
    )
    owner._prepare = lambda: None
    owner._close_resources = lambda: (_ for _ in ()).throw(
        RuntimeError("owned helper still running")
    )
    owner._cleanup_working_directory = lambda: pytest.fail("cleanup preceded resource retirement")
    owner._web_publication = lambda *args: pytest.fail("receipt preceded resource retirement")
    try:
        with pytest.raises(RuntimeError, match="still running"):
            Supervisor.run(owner)
        assert not (context.root / web.RECEIPT_NAME).exists()
    finally:
        context.close()


def paired_snapshot(fixture):
    from hmlib.stitching.native_calibration import _REQUIRED_ARTIFACTS

    source = fixture.document["sources"].pop("source")
    fixture.document["sources"] = {"left": source, "right": copy.deepcopy(source)}
    videos = {role: [str(fixture.media)] for role in ("left", "right")}
    config = {
        "game": {"videos": videos},
        "preview_overlays": {"players": False, "play": False, "rink": False},
        "hstream_ui": {"stitching_calibration": {"status": "complete"}},
    }
    (fixture.root / "game/config.yaml").write_text(yaml.safe_dump(config))
    fixture.document["game_config"] = web.file_content(fixture.root / "game/config.yaml")
    directory = fixture.root / "calibration/selected"
    directory.mkdir(parents=True)
    info = fixture.media.stat()
    source_identity = {
        "path": str(fixture.media),
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }
    manifest = {
        "version": 1,
        "game_directory": str(fixture.root / "game"),
        "generation": "hugin-generation",
        "output_generation": "output-generation",
        "invalidation_id": "owned-completed-calibration",
        "frame_offsets": {"left": 0, "right": 0.75},
        "canvas": {"width": 16, "height": 8},
        "post_stitch_rotate_degrees": 0,
        "configuration": config,
        "sources": {role: [source_identity] for role in videos},
        "artifacts": {},
    }
    for name in _REQUIRED_ARTIFACTS:
        (directory / name).write_bytes(
            yaml.safe_dump(config).encode()
            if name == "config.yaml"
            else b"native snapshot artifact"
        )
        manifest["artifacts"][name] = web.file_content(directory / name)
    path = directory / "native-calibration.yaml"
    path.write_text(yaml.safe_dump(manifest))
    fixture.document["calibration"] = {
        "directory": "calibration/selected",
        "manifest": web.file_content(path),
        **{key: manifest[key] for key in ("generation", "output_generation", "invalidation_id")},
    }
    fixture.write()
    return directory, manifest


def should_accept_only_the_exact_completed_native_pair_and_pin_every_map(admitted):
    directory, _ = paired_snapshot(admitted)
    context = web.WebProgram(admitted.request, environment={})
    try:
        assert context.calibration.frame_offsets == (0, 0.75)
        assert context.calibration.output_generation == "output-generation"
        mapping = directory / "mapping_0000.tif"
        info = mapping.stat()
        mapping.write_bytes(b"changed snapshot data!!!")
        os.utime(mapping, ns=(info.st_atime_ns, info.st_mtime_ns))
        with pytest.raises(ValueError, match="changed"):
            context.check()
    finally:
        context.close()


@pytest.mark.parametrize("change", ["generation", "pending", "corrupt", "game", "source"])
def should_reject_changed_or_pending_native_calibration(admitted, change):
    directory, manifest = paired_snapshot(admitted)
    if change == "generation":
        admitted.document["calibration"]["output_generation"] = "different-generation"
    elif change == "pending":
        manifest["configuration"]["hstream_ui"]["stitching_calibration"]["status"] = "pending"
        (directory / "config.yaml").write_text(yaml.safe_dump(manifest["configuration"]))
        manifest["artifacts"]["config.yaml"] = web.file_content(directory / "config.yaml")
    elif change == "corrupt":
        (directory / "mapping_0000.tif").write_bytes(b"bad")
    elif change == "game":
        manifest["game_directory"] = str(admitted.root)
    else:
        manifest["sources"]["left"][0] = {**manifest["sources"]["left"][0], "inode": 0}
    path = directory / "native-calibration.yaml"
    path.write_text(yaml.safe_dump(manifest))
    admitted.document["calibration"]["manifest"] = web.file_content(path)
    admitted.write()
    with pytest.raises(ValueError):
        web.WebProgram(admitted.request, environment={})


@pytest.mark.parametrize("kind", ["symlink", "fifo", "oversized"])
def should_reject_nonregular_or_unbounded_publication_journals(admitted, monkeypatch, kind):
    context = open_admitted(admitted, monkeypatch)
    try:
        journal = context.session / "publication.json"
        if kind == "symlink":
            journal.symlink_to(admitted.request)
        elif kind == "fifo":
            os.mkfifo(journal)
        else:
            journal.write_bytes(b"x" * (web.MAX_RECEIPT_BYTES + 1))
        with pytest.raises((ValueError, OSError)):
            web.read_publications(context.root)
    finally:
        context.close()
