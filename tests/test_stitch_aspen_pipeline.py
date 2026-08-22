from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - Bazel Python toolchain lacks torch
    torch = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(torch is None, reason="requires torch")

if torch is not None:
    import hmlib.cli.stitch as stitch_cli
else:
    stitch_cli = None  # type: ignore[assignment]

_TORCH_MODULE_BASE = torch.nn.Module if torch is not None else object


class _DummyAspenNet(_TORCH_MODULE_BASE):
    """Lightweight AspenNet stub to capture graph config and contexts."""

    def __init__(
        self,
        name: str,
        graph_cfg: Dict[str, Any],
        shared: Dict[str, Any] | None = None,
        **_: Any,
    ):  # type: ignore[override]
        super().__init__()
        self.name = name
        self.graph_cfg = graph_cfg
        self.shared = dict(shared or {})
        self.calls: List[Dict[str, Any]] = []

    def to(self, *args: Any, **kwargs: Any):  # pragma: no cover - trivial passthrough
        return self

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        # Capture the per-frame context passed from stitch_videos.
        self.calls.append(dict(context))
        return context

    def finalize(self):  # pragma: no cover - not exercised deeply here
        pass


class _DummyVideoInfo:
    """Minimal BasicVideoInfo stub to avoid real ffprobe/IO."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self.frame_count = 10
        self.fps = 30.0


class _DummyStitchDataset:
    """Small iterable that yields a couple of dummy frames."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - init is trivial
        self.batch_size = int(kwargs.get("batch_size", 1) or 1)
        self._len = 2
        self._closed = False
        self._fps = 30.0

    @property
    def fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        for _ in range(self._len):
            # Simple uint8 frame tensor [B, H, W, C]
            yield torch.zeros((self.batch_size, 4, 4, 3), dtype=torch.uint8)

    def close(self) -> None:
        self._closed = True


def should_build_aspen_pipeline_for_stitching(monkeypatch, tmp_path):
    captured_net: Dict[str, _DummyAspenNet] = {}

    def _make_dummy_aspen(name: str, graph_cfg: Dict[str, Any], shared: Dict[str, Any] | None = None, **kwargs: Any):  # type: ignore[override]
        net = _DummyAspenNet(name=name, graph_cfg=graph_cfg, shared=shared, **kwargs)
        captured_net["net"] = net
        return net

    # Patch heavy dependencies in stitch_videos.
    monkeypatch.setattr(stitch_cli, "AspenNet", _make_dummy_aspen, raising=False)
    monkeypatch.setattr(stitch_cli, "BasicVideoInfo", _DummyVideoInfo, raising=False)
    monkeypatch.setattr(stitch_cli, "StitchDataset", _DummyStitchDataset, raising=False)

    def _fake_configure_video_stitching(
        dir_name: str,
        video_left: str,
        video_right: str,
        project_file_name: str,
        left_frame_offset: int,
        right_frame_offset: int,
        base_frame_offset: int,
        max_control_points: int,
        force: bool,
        game_id: str,
        stitch_frame_time: str | None,
        ignore_private_config: bool,
        game_config: Dict[str, Any] | None,
        control_point_matcher: str,
        mapping_backend: str,
        max_output_dimension: int | None,
    ):
        # Return a dummy project path and zero offsets.
        captured_net["configure_game_id"] = game_id
        captured_net["configure_stitch_frame_time"] = stitch_frame_time
        captured_net["configure_ignore_private_config"] = ignore_private_config
        captured_net["configure_game_config"] = game_config
        captured_net["configure_control_point_matcher"] = control_point_matcher
        captured_net["configure_mapping_backend"] = mapping_backend
        captured_net["configure_max_output_dimension"] = max_output_dimension
        pto_path = str(tmp_path / "dummy.pto")
        Path(pto_path).touch()
        return pto_path, 0, 0

    monkeypatch.setattr(
        stitch_cli, "configure_video_stitching", _fake_configure_video_stitching, raising=False
    )

    args = types.SimpleNamespace(
        profiler=None,
        max_blend_levels=None,
        skip_final_video_save=False,
        save_frame_dir=None,
        dataset_prefetch_batches=1,
        no_cuda_streams=True,
        no_progress_bar=True,
        serial=False,
        checkerboard_input=False,
        show_youtube=False,
        ignore_private_config=False,
        config_overrides=["stitching.enabled=false"],
    )

    out_path = tmp_path / "stitched.mkv"

    lfo, rfo = stitch_cli.stitch_videos(
        dir_name=str(tmp_path),
        videos={"left": ["left.mp4"], "right": ["right.mp4"]},
        max_control_points=10,
        game_id="test-game",
        output_stitched_video_file=str(out_path),
        args=args,
    )

    # Offsets should come back from the fake configure_video_stitching.
    assert lfo == 0
    assert rfo == 0

    net = captured_net.get("net")
    assert isinstance(net, _DummyAspenNet)

    # Ensure the stitching Aspen graph has a video_out plugin with the
    # CLI-provided output path wired in.
    plugins = net.graph_cfg.get("plugins", {})
    assert "stitch_ui" not in plugins
    assert "video_out" in plugins
    vo_spec = plugins["video_out"]
    params = vo_spec.get("params", {})
    assert params.get("output_video_path") == str(out_path)

    # --config-override should have been applied to the loaded config.
    # Disabling stitching.enabled should disable the StitchingPlugin trunk.
    assert plugins.get("stitching", {}).get("enabled") is False

    # At least one frame should have been forwarded into AspenNet with
    # img, frame_ids, fps and game_id populated.
    assert net.calls, "Expected AspenNet.forward to be called at least once"
    ctx0 = net.calls[0]
    assert "img" in ctx0
    assert "frame_ids" in ctx0
    assert ctx0.get("game_id") == "test-game"
    assert ctx0.get("fps") == 30.0
    assert captured_net.get("configure_game_id") == "test-game"
    assert captured_net.get("configure_ignore_private_config") is False
    assert isinstance(captured_net.get("configure_game_config"), dict)
    assert captured_net.get("configure_control_point_matcher") == "superpoint-lightglue"
    assert captured_net.get("configure_mapping_backend") == "nona"
    assert captured_net.get("configure_max_output_dimension") is None


def should_keep_stitch_ui_terminal_after_post_processing() -> None:
    config_path = Path(stitch_cli.__file__).resolve().parents[1] / "config/aspen/stitching.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plugins = config["aspen"]["plugins"]

    assert plugins["apply_camera"]["depends"] == ["stitching"]
    assert plugins["video_out_prep"]["depends"] == ["apply_camera"]
    assert plugins["stitch_ui"]["depends"] == ["video_out_prep"]
    assert all(
        "stitch_ui" not in (spec.get("depends") or [])
        for name, spec in plugins.items()
        if name != "stitch_ui"
    )


def should_use_configured_stitch_frame_time_for_base_offset(monkeypatch, tmp_path):
    captured: Dict[str, Any] = {}

    monkeypatch.setattr(stitch_cli, "AspenNet", _DummyAspenNet, raising=False)
    monkeypatch.setattr(stitch_cli, "BasicVideoInfo", _DummyVideoInfo, raising=False)
    monkeypatch.setattr(stitch_cli, "StitchDataset", _DummyStitchDataset, raising=False)

    def _fake_configure_video_stitching(
        dir_name: str,
        video_left: str,
        video_right: str,
        project_file_name: str,
        left_frame_offset: int,
        right_frame_offset: int,
        base_frame_offset: int,
        max_control_points: int,
        force: bool,
        game_id: str,
        stitch_frame_time: str | None,
        ignore_private_config: bool,
        game_config: Dict[str, Any] | None,
        control_point_matcher: str,
        mapping_backend: str,
        max_output_dimension: int | None,
    ):
        captured["base_frame_offset"] = base_frame_offset
        captured["game_id"] = game_id
        captured["stitch_frame_time"] = stitch_frame_time
        captured["ignore_private_config"] = ignore_private_config
        captured["game_config"] = game_config
        captured["control_point_matcher"] = control_point_matcher
        captured["mapping_backend"] = mapping_backend
        captured["max_output_dimension"] = max_output_dimension
        pto_path = str(tmp_path / "dummy.pto")
        Path(pto_path).touch()
        return pto_path, 0, 0

    monkeypatch.setattr(
        stitch_cli, "configure_video_stitching", _fake_configure_video_stitching, raising=False
    )

    args = types.SimpleNamespace(
        profiler=None,
        max_blend_levels=None,
        skip_final_video_save=False,
        save_frame_dir=None,
        dataset_prefetch_batches=1,
        no_cuda_streams=True,
        no_progress_bar=True,
        serial=False,
        checkerboard_input=False,
        show_youtube=False,
        ignore_private_config=False,
        config_overrides=[
            "stitching.enabled=false",
            "stitching.stitch_frame_time=00:00:02",
            "stitching.max_output_dimension=4096",
        ],
    )

    stitch_cli.stitch_videos(
        dir_name=str(tmp_path),
        videos={"left": ["left.mp4"], "right": ["right.mp4"]},
        max_control_points=10,
        game_id="test-game",
        output_stitched_video_file=str(tmp_path / "stitched.mkv"),
        args=args,
    )

    # 2 seconds at 30 FPS from _DummyVideoInfo.
    assert captured.get("base_frame_offset") == 60
    assert captured.get("game_id") == "test-game"
    assert captured.get("stitch_frame_time") == "00:00:02"
    assert captured.get("ignore_private_config") is False
    assert isinstance(captured.get("game_config"), dict)
    assert captured.get("max_output_dimension") == 4096


def should_apply_conservative_stitch_buffering_defaults_when_not_explicit():
    cfg = {
        "aspen": {
            "pipeline": {
                "threaded": True,
                "graph": True,
                "queue_size": 2,
            }
        }
    }
    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[],
        game_id=None,
        ignore_private_config=False,
    )

    stitch_cli._apply_stitch_buffering_defaults(cfg, args)

    pipeline = cfg["aspen"]["pipeline"]
    assert pipeline["queue_size"] == 1
    assert pipeline["max_concurrent"] == 1


def should_preserve_explicit_stitch_buffering_settings():
    cfg = {
        "aspen": {
            "pipeline": {
                "threaded": True,
                "graph": True,
                "queue_size": 4,
                "max_concurrent": 3,
            }
        }
    }
    args = types.SimpleNamespace(
        explicit_arg_names={"aspen_thread_queue_size"},
        config_overrides=["aspen.pipeline.max_concurrent=3"],
        game_id=None,
        ignore_private_config=False,
    )

    stitch_cli._apply_stitch_buffering_defaults(cfg, args)

    pipeline = cfg["aspen"]["pipeline"]
    assert pipeline["queue_size"] == 4
    assert pipeline["max_concurrent"] == 3


def should_preserve_configured_stitch_buffering_settings():
    cfg = {
        "aspen": {
            "pipeline": {
                "threaded": True,
                "graph": True,
                "queue_size": 4,
                "max_concurrent": 4,
            }
        }
    }
    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[],
        game_id=None,
        ignore_private_config=False,
    )

    stitch_cli._apply_stitch_buffering_defaults(cfg, args)

    pipeline = cfg["aspen"]["pipeline"]
    assert pipeline["queue_size"] == 4
    assert pipeline["max_concurrent"] == 4


def should_apply_lowmem_stitch_runtime_overrides_without_marking_args_explicit():
    cfg = {
        "stitching": {
            "dtype": "float32",
            "max_blend_levels": 11,
            "minimize_blend": False,
            "max_output_width": None,
            "cache_rotation_grid": True,
        },
        "video_out": {
            "output_width": "auto",
            "output_height": None,
        },
        "aspen": {
            "pipeline": {
                "max_concurrent": 3,
            },
            "plugins": {
                "stitching": {
                    "params": {
                        "dtype": "GLOBAL.stitching.dtype",
                        "max_blend_levels": "GLOBAL.stitching.max_blend_levels",
                        "minimize_blend": "GLOBAL.stitching.minimize_blend",
                        "max_output_width": "GLOBAL.stitching.max_output_width",
                        "cache_rotation_grid": "GLOBAL.stitching.cache_rotation_grid",
                    }
                },
                "video_out_prep": {
                    "params": {
                        "output_width": "GLOBAL.video_out.output_width",
                    }
                },
            },
        },
    }
    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[],
        fp16_stitch=False,
        output_width=None,
        max_blend_levels=11,
        minimize_blend=0,
        no_minimize_blend=False,
        aspen_max_concurrent=None,
        game_id=None,
        ignore_private_config=False,
    )

    use_half_dtype = stitch_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert use_half_dtype is True
    assert args.explicit_arg_names == set()
    assert args.fp16_stitch is True
    assert cfg["stitching"]["dtype"] == "float16"
    assert cfg["stitching"]["max_blend_levels"] == 5
    assert cfg["stitching"]["minimize_blend"] is True
    assert cfg["stitching"]["max_output_width"] == 1920
    assert cfg["stitching"]["cache_rotation_grid"] is False
    assert cfg["video_out"]["output_width"] == 1920
    assert cfg["aspen"]["pipeline"]["max_concurrent"] == 1
    assert cfg["aspen"]["plugins"]["stitching"]["params"]["dtype"] == "float16"
    assert cfg["aspen"]["plugins"]["stitching"]["params"]["max_output_width"] == 1920
    assert cfg["aspen"]["plugins"]["stitching"]["params"]["cache_rotation_grid"] is False
    assert cfg["aspen"]["plugins"]["video_out_prep"]["params"]["output_width"] == 1920


def should_respect_config_override_opt_outs_for_lowmem_stitch_overrides():
    cfg = {
        "stitching": {
            "dtype": "float32",
            "max_blend_levels": 11,
            "minimize_blend": False,
            "max_output_width": None,
            "cache_rotation_grid": True,
        },
        "video_out": {
            "output_width": "auto",
            "output_height": None,
        },
        "aspen": {
            "pipeline": {
                "max_concurrent": 3,
            },
            "plugins": {
                "stitching": {
                    "params": {
                        "dtype": "GLOBAL.stitching.dtype",
                        "max_blend_levels": "GLOBAL.stitching.max_blend_levels",
                        "minimize_blend": "GLOBAL.stitching.minimize_blend",
                        "max_output_width": "GLOBAL.stitching.max_output_width",
                        "cache_rotation_grid": "GLOBAL.stitching.cache_rotation_grid",
                    }
                },
                "video_out_prep": {
                    "params": {
                        "output_width": "GLOBAL.video_out.output_width",
                    }
                },
            },
        },
    }
    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[
            "stitching.dtype=float32",
            "video_out.output_width=auto",
            "stitching.cache_rotation_grid=true",
            "aspen.pipeline.max_concurrent=3",
        ],
        fp16_stitch=False,
        output_width=None,
        max_blend_levels=11,
        minimize_blend=0,
        no_minimize_blend=False,
        aspen_max_concurrent=None,
        game_id=None,
        ignore_private_config=False,
    )

    use_half_dtype = stitch_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert use_half_dtype is False
    assert cfg["stitching"]["dtype"] == "float32"
    assert cfg["video_out"]["output_width"] == "auto"
    assert cfg["stitching"]["max_output_width"] is None
    assert cfg["stitching"]["cache_rotation_grid"] is True
    assert cfg["aspen"]["pipeline"]["max_concurrent"] == 3


def should_respect_plugin_config_override_opt_outs_for_lowmem_stitch_overrides():
    cfg = {
        "stitching": {
            "dtype": "float32",
            "max_blend_levels": 11,
            "minimize_blend": False,
            "max_output_width": None,
        },
        "video_out": {
            "output_width": "auto",
            "output_height": None,
        },
        "aspen": {
            "plugins": {
                "stitching": {
                    "params": {
                        "dtype": "float32",
                        "max_blend_levels": 11,
                        "minimize_blend": False,
                        "max_output_width": None,
                    }
                },
                "video_out_prep": {
                    "params": {
                        "output_width": "auto",
                    }
                },
            }
        },
    }
    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[
            "aspen.plugins.stitching.params.dtype=float32",
            "aspen.plugins.video_out_prep.params.output_width=auto",
        ],
        fp16_stitch=False,
        output_width=None,
        max_blend_levels=11,
        minimize_blend=0,
        no_minimize_blend=False,
        game_id=None,
        ignore_private_config=False,
    )

    use_half_dtype = stitch_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert use_half_dtype is False
    assert cfg["stitching"]["dtype"] == "float32"
    assert cfg["aspen"]["plugins"]["stitching"]["params"]["dtype"] == "float32"
    assert cfg["video_out"]["output_width"] == "auto"
    assert cfg["aspen"]["plugins"]["video_out_prep"]["params"]["output_width"] == "auto"


def should_ignore_global_plugin_config_overrides_when_applying_lowmem_stitch_overrides():
    cfg = {
        "stitching": {
            "dtype": "float32",
            "max_blend_levels": 11,
            "minimize_blend": False,
            "max_output_width": None,
        },
        "video_out": {
            "output_width": "auto",
            "output_height": None,
        },
        "aspen": {
            "plugins": {
                "stitching": {
                    "params": {
                        "dtype": "GLOBAL.stitching.dtype",
                        "max_blend_levels": "GLOBAL.stitching.max_blend_levels",
                        "minimize_blend": "GLOBAL.stitching.minimize_blend",
                        "max_output_width": "GLOBAL.stitching.max_output_width",
                    }
                },
                "video_out_prep": {
                    "params": {
                        "output_width": "GLOBAL.video_out.output_width",
                    }
                },
            }
        },
    }
    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[
            "aspen.plugins.stitching.params.dtype=GLOBAL.stitching.dtype",
            "aspen.plugins.video_out_prep.params.output_width=GLOBAL.video_out.output_width",
        ],
        fp16_stitch=False,
        output_width=None,
        max_blend_levels=11,
        minimize_blend=0,
        no_minimize_blend=False,
        game_id=None,
        ignore_private_config=False,
    )

    use_half_dtype = stitch_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert use_half_dtype is True
    assert cfg["stitching"]["dtype"] == "float16"
    assert cfg["stitching"]["max_output_width"] == 1920
    assert cfg["video_out"]["output_width"] == 1920


def should_resolve_legacy_stitch_dataset_dtype_from_config():
    assert (
        stitch_cli._resolve_stitch_tensor_dtype(torch.float32, {"dtype": "float16"})
        == torch.float16
    )
    assert (
        stitch_cli._resolve_stitch_tensor_dtype(torch.float16, {"dtype": "float32"})
        == torch.float32
    )
    assert (
        stitch_cli._resolve_stitch_tensor_dtype(torch.float32, {"dtype": "fp16"}) == torch.float16
    )
    assert (
        stitch_cli._resolve_stitch_tensor_dtype(torch.float16, {"dtype": "fp32"}) == torch.float32
    )
    assert stitch_cli._resolve_stitch_tensor_dtype(torch.float32, {"dtype": "uint8"}) == torch.uint8

    try:
        stitch_cli._resolve_stitch_tensor_dtype(torch.float32, {"dtype": "bogus"})
    except ValueError as exc:
        assert "Unsupported stitch dtype" in str(exc)
    else:  # pragma: no cover - defensive failure path
        raise AssertionError("Expected unsupported stitch dtype to raise ValueError")


def should_respect_game_or_private_config_opt_outs_for_lowmem_stitch_overrides(monkeypatch):
    cfg = {
        "stitching": {
            "dtype": "float32",
            "max_blend_levels": 11,
            "minimize_blend": False,
            "max_output_width": None,
        },
        "video_out": {
            "output_width": "auto",
            "output_height": None,
        },
        "aspen": {
            "plugins": {
                "stitching": {
                    "params": {
                        "dtype": "GLOBAL.stitching.dtype",
                        "max_blend_levels": "GLOBAL.stitching.max_blend_levels",
                        "minimize_blend": "GLOBAL.stitching.minimize_blend",
                        "max_output_width": "GLOBAL.stitching.max_output_width",
                    }
                },
                "video_out_prep": {
                    "params": {
                        "output_width": "GLOBAL.video_out.output_width",
                    }
                },
            }
        },
    }

    monkeypatch.setattr(
        stitch_cli,
        "load_config_file",
        lambda *args, **kwargs: {
            "aspen": {
                "plugins": {
                    "stitching": {
                        "params": {
                            "dtype": "float32",
                            "max_blend_levels": 11,
                            "minimize_blend": False,
                        }
                    }
                }
            }
        },
        raising=False,
    )
    monkeypatch.setattr(
        stitch_cli,
        "get_game_config_private",
        lambda *args, **kwargs: {
            "aspen": {
                "plugins": {
                    "video_out_prep": {
                        "params": {
                            "output_width": "auto",
                        }
                    }
                }
            }
        },
        raising=False,
    )

    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[],
        fp16_stitch=False,
        output_width=None,
        max_blend_levels=11,
        minimize_blend=0,
        no_minimize_blend=False,
        game_id="test-game",
        ignore_private_config=False,
    )

    use_half_dtype = stitch_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert use_half_dtype is False
    assert cfg["stitching"]["dtype"] == "float32"
    assert cfg["stitching"]["max_blend_levels"] == 11
    assert cfg["stitching"]["minimize_blend"] is False
    assert cfg["stitching"]["max_output_width"] is None
    assert cfg["video_out"]["output_width"] == "auto"
    assert cfg["aspen"]["plugins"]["stitching"]["params"]["dtype"] == "GLOBAL.stitching.dtype"
    assert (
        cfg["aspen"]["plugins"]["video_out_prep"]["params"]["output_width"]
        == "GLOBAL.video_out.output_width"
    )


def should_ignore_global_plugin_wiring_in_game_or_private_config_when_applying_lowmem_overrides(
    monkeypatch,
):
    cfg = {
        "stitching": {
            "dtype": "float32",
            "max_blend_levels": 11,
            "minimize_blend": False,
            "max_output_width": None,
        },
        "video_out": {
            "output_width": "auto",
            "output_height": None,
        },
        "aspen": {
            "plugins": {
                "stitching": {
                    "params": {
                        "dtype": "GLOBAL.stitching.dtype",
                        "max_blend_levels": "GLOBAL.stitching.max_blend_levels",
                        "minimize_blend": "GLOBAL.stitching.minimize_blend",
                        "max_output_width": "GLOBAL.stitching.max_output_width",
                    }
                },
                "video_out_prep": {
                    "params": {
                        "output_width": "GLOBAL.video_out.output_width",
                    }
                },
            }
        },
    }

    monkeypatch.setattr(
        stitch_cli,
        "load_config_file",
        lambda *args, **kwargs: {
            "aspen": {
                "plugins": {
                    "stitching": {
                        "params": {
                            "dtype": "GLOBAL.stitching.dtype",
                            "max_blend_levels": "GLOBAL.stitching.max_blend_levels",
                            "minimize_blend": "GLOBAL.stitching.minimize_blend",
                        }
                    }
                }
            }
        },
        raising=False,
    )
    monkeypatch.setattr(
        stitch_cli,
        "get_game_config_private",
        lambda *args, **kwargs: {
            "aspen": {
                "plugins": {
                    "video_out_prep": {
                        "params": {
                            "output_width": "GLOBAL.video_out.output_width",
                        }
                    }
                }
            }
        },
        raising=False,
    )

    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[],
        fp16_stitch=False,
        output_width=None,
        max_blend_levels=11,
        minimize_blend=0,
        no_minimize_blend=False,
        game_id="test-game",
        ignore_private_config=False,
    )

    use_half_dtype = stitch_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert use_half_dtype is True
    assert cfg["stitching"]["dtype"] == "float16"
    assert cfg["stitching"]["max_output_width"] == 1920
    assert cfg["video_out"]["output_width"] == 1920
