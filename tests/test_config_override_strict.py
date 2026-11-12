import pytest

from hmlib.hm_opts import hm_opts


def should_apply_config_override_when_key_exists() -> None:
    cfg = {"stitching": {"enabled": True}}
    hm_opts.apply_config_overrides(cfg, ["stitching.enabled=false"])
    assert cfg["stitching"]["enabled"] is False


def should_error_when_config_override_key_missing() -> None:
    cfg = {"stitching": {"enabled": True}}
    with pytest.raises(KeyError):
        hm_opts.apply_config_overrides(cfg, ["stitching.does_not_exist=true"])


def should_normalize_python_color_cli_overrides_as_a_separate_layer():
    from hmlib.runtime_controls import normalize_control_layer

    cfg = normalize_control_layer({"video_out": {"tone": {"exposure": 0.2}}})
    hm_opts.apply_config_overrides(cfg, ["rink.camera.color.exposure_ev=0.4"])
    assert cfg["video_out"]["tone"]["exposure"] == 0.8
    hm_opts.apply_config_overrides(
        cfg, ["video_out.tone.exposure=0.6", "rink.camera.color.exposure_ev=0.1"]
    )
    assert cfg["rink"]["camera"]["color"]["exposure_ev"] == 0.3
