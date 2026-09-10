import importlib.util
import json
import logging
import os
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch


def _load_hmlib_config_light():
    """Load an isolated config module without leaking dependency stubs."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hm_mod = ModuleType("hmlib")
    hm_mod.__file__ = os.path.join(repo_root, "hmlib", "__init__.py")
    log_mod = ModuleType("hmlib.log")
    log_mod.get_logger = logging.getLogger
    yaml_stub = ModuleType("yaml")
    yaml_stub.safe_load = json.load
    yaml_stub.YAMLError = ValueError

    def _dump(data, stream=None, sort_keys=False):
        text = json.dumps(data, sort_keys=sort_keys)
        if stream is None:
            return text
        stream.write(text)

    yaml_stub.dump = _dump
    config_path = os.path.join(repo_root, "hmlib", "config.py")
    spec = importlib.util.spec_from_file_location("_hm_config_test", config_path)
    assert spec and spec.loader, "Failed to create import spec for hmlib.config"
    mod = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"hmlib": hm_mod, "hmlib.log": log_mod, "yaml": yaml_stub, spec.name: mod},
    ):
        spec.loader.exec_module(mod)
    return mod


def should_merge_yaml_files_ordered():
    cfg = _load_hmlib_config_light()

    a = {
        "camera": {"name": "CamA"},
        "game": {"name": "G1"},
        "aspen": {
            "inference_pipeline": [{"type": "LoadImageFromFile"}],
        },
    }
    b = {
        "camera": {"name": "CamB"},
        "game": {"phase": "regular"},
        "aspen": {
            "video_out_pipeline": [{"type": "HmImageOverlays"}],
        },
    }

    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "a.yaml")
        p2 = os.path.join(td, "b.yaml")
        with open(p1, "w") as f:
            f.write(json.dumps(a))
        with open(p2, "w") as f:
            f.write(json.dumps(b))

        merged = cfg.load_yaml_files_ordered([p1, p2])

    assert merged["camera"]["name"] == "CamB"
    assert merged["game"]["name"] == "G1"
    assert merged["game"]["phase"] == "regular"
    assert merged["aspen"]["inference_pipeline"][0]["type"] == "LoadImageFromFile"
    assert merged["video_out_pipeline"][0]["type"] == "HmImageOverlays"
    assert "video_out_pipeline" not in merged["aspen"]


def should_merge_aspen_namespace_mock():
    cfg = _load_hmlib_config_light()
    # Minimal base + aspen graph as JSON-y YAML files
    base = {"camera": {"name": "BaseCam"}, "game": {"name": "Base"}}
    graph = {"aspen": {"trunks": {"image_prep": {"class": "X", "depends": [], "params": {}}}}}

    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "base.yaml")
        p2 = os.path.join(td, "graph.yaml")
        with open(p1, "w") as f:
            f.write(json.dumps(base))
        with open(p2, "w") as f:
            f.write(json.dumps(graph))
        merged = cfg.load_yaml_files_ordered([p1, p2])

    assert merged["camera"]["name"] == "BaseCam"
    assert merged["game"]["name"] == "Base"
    assert isinstance(merged.get("aspen"), dict)
    assert "trunks" in merged["aspen"]


def should_skip_private_game_config_when_requested():
    cfg = _load_hmlib_config_light()

    cfg.baseline_config = lambda root_dir=None: {}
    cfg.get_camera_config = lambda camera=None, root_dir=None: {}
    cfg.get_rink_config = lambda rink=None, root_dir=None: {}
    cfg.resolve_global_refs = lambda d: d

    private_calls = {"count": 0}

    def _load_config_file(
        root_dir=None, config_type=None, config_name=None, merge_into_config=None
    ):
        if config_type == "games":
            return {"game": {"name": "public-game"}}
        return {}

    def _get_game_config(game_id=None, root_dir=None):
        return {
            "game": {"name": "public-game"},
            "private_only": {"enabled": True},
        }

    def _get_game_config_private(game_id=None, merge_into_config=None):
        private_calls["count"] += 1
        return {"private_only": {"enabled": True}}

    cfg.load_config_file = _load_config_file
    cfg.get_game_config = _get_game_config
    cfg.get_game_config_private = _get_game_config_private

    merged = cfg.get_config(
        game_id="example-game",
        ignore_private_config=True,
        resolve_globals=False,
    )

    assert private_calls["count"] == 0
    assert merged.get("game", {}).get("name") == "public-game"
    assert "private_only" not in merged
