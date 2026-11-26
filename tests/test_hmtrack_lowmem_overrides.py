from __future__ import annotations

import importlib.util
import types

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch is not available")

if _HAS_TORCH:
    import hmlib.cli.hmtrack as hmtrack_cli


def should_reject_independent_scoreboard_before_loading_processing_modules():
    args = types.SimpleNamespace(
        game_config={"rink": {"scoreboard": {"camera": {"enabled": True}}}}
    )
    with pytest.raises(ValueError, match="scoreboard camera requires the native"):
        hmtrack_cli._main(args, 0)


def _make_cfg():
    return {
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


def _make_args(**overrides):
    args = types.SimpleNamespace(
        explicit_arg_names=set(),
        config_overrides=[],
        fp16_stitch=False,
        output_width=None,
        max_blend_levels=11,
        minimize_blend=0,
        no_minimize_blend=False,
        cache_size=4,
        aspen_max_concurrent=None,
        game_id=None,
        ignore_private_config=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def should_apply_lowmem_hmtrack_runtime_overrides_without_marking_args_explicit():
    cfg = _make_cfg()
    args = _make_args()

    hmtrack_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert args.explicit_arg_names == set()
    assert args.cache_size == 0
    assert args.fp16_stitch is True
    assert cfg["stitching"]["dtype"] == "float16"
    assert cfg["stitching"]["max_blend_levels"] == 5
    assert cfg["stitching"]["minimize_blend"] is True
    assert cfg["stitching"]["max_output_width"] == 1920
    assert cfg["stitching"]["cache_rotation_grid"] is False
    assert cfg["video_out"]["output_width"] == 1920
    assert cfg["aspen"]["pipeline"]["max_concurrent"] == 1


def should_respect_source_config_override_opt_outs_for_lowmem_hmtrack_overrides():
    cfg = _make_cfg()
    args = _make_args(
        config_overrides=[
            "stitching.dtype=float32",
            "video_out.output_width=auto",
            "stitching.cache_rotation_grid=true",
            "aspen.pipeline.max_concurrent=3",
        ]
    )

    hmtrack_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert args.fp16_stitch is False
    assert cfg["stitching"]["dtype"] == "float32"
    assert cfg["video_out"]["output_width"] == "auto"
    assert cfg["stitching"]["max_output_width"] is None
    assert cfg["stitching"]["cache_rotation_grid"] is True
    assert cfg["aspen"]["pipeline"]["max_concurrent"] == 3


def should_respect_plugin_config_override_opt_outs_for_lowmem_hmtrack_overrides():
    cfg = _make_cfg()
    cfg["aspen"]["plugins"]["stitching"]["params"]["dtype"] = "float32"
    cfg["aspen"]["plugins"]["video_out_prep"]["params"]["output_width"] = "auto"
    args = _make_args(
        config_overrides=[
            "aspen.plugins.stitching.params.dtype=float32",
            "aspen.plugins.video_out_prep.params.output_width=auto",
        ]
    )

    hmtrack_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert args.fp16_stitch is False
    assert cfg["stitching"]["dtype"] == "float32"
    assert cfg["aspen"]["plugins"]["stitching"]["params"]["dtype"] == "float32"
    assert cfg["video_out"]["output_width"] == "auto"
    assert cfg["aspen"]["plugins"]["video_out_prep"]["params"]["output_width"] == "auto"


def should_ignore_global_plugin_config_overrides_when_applying_lowmem_hmtrack_overrides():
    cfg = _make_cfg()
    args = _make_args(
        config_overrides=[
            "aspen.plugins.stitching.params.dtype=GLOBAL.stitching.dtype",
            "aspen.plugins.video_out_prep.params.output_width=GLOBAL.video_out.output_width",
        ]
    )

    hmtrack_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert args.fp16_stitch is True
    assert cfg["stitching"]["dtype"] == "float16"
    assert cfg["stitching"]["max_output_width"] == 1920
    assert cfg["video_out"]["output_width"] == 1920


def should_respect_game_or_private_config_opt_outs_for_lowmem_hmtrack_overrides(monkeypatch):
    cfg = _make_cfg()
    monkeypatch.setattr(
        hmtrack_cli,
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
        hmtrack_cli,
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
    args = _make_args(game_id="test-game")

    hmtrack_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert args.fp16_stitch is False
    assert cfg["stitching"]["dtype"] == "float32"
    assert cfg["stitching"]["max_blend_levels"] == 11
    assert cfg["stitching"]["minimize_blend"] is False
    assert cfg["video_out"]["output_width"] == "auto"


def should_ignore_global_plugin_wiring_in_game_or_private_config_when_applying_lowmem_hmtrack_overrides(
    monkeypatch,
):
    cfg = _make_cfg()
    monkeypatch.setattr(
        hmtrack_cli,
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
        hmtrack_cli,
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
    args = _make_args(game_id="test-game")

    hmtrack_cli._apply_single_lowmem_gpu_overrides(args, cfg)

    assert args.fp16_stitch is True
    assert cfg["stitching"]["dtype"] == "float16"
    assert cfg["stitching"]["max_output_width"] == 1920
    assert cfg["video_out"]["output_width"] == 1920
