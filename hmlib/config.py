"""Game- and camera-specific YAML configuration helpers.

This module loads, merges and saves configuration files used throughout
HockeyMON pipelines (games, rinks, cameras and private overrides).

@see @ref hmlib.hm_opts.hm_opts "hm_opts" for CLI flags that drive these configs.
@see @ref hmlib.game_audio.transfer_audio "transfer_audio" for one consumer.
"""

import argparse
import copy
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

import hmlib
from hmlib.log import get_logger

_HOME_DIR: str = os.environ.get("HOME") or str(Path.home())
GAME_DIR_BASE: str = os.environ.get("HM_GAME_DIR") or os.path.join(_HOME_DIR, "Videos")
ROOT_DIR: str = os.path.dirname(os.path.abspath(hmlib.__file__))


@dataclass
class Game:
    game_id: Optional[str] = None
    season: Optional[str] = None
    team: Optional[str] = None


def get_root_dir() -> str:
    """Return the root directory of the hmlib installation."""
    return ROOT_DIR


def prepend_root_dir(path: str) -> str:
    """Join a relative path against :data:`ROOT_DIR` if needed."""
    if not path:
        return ROOT_DIR
    if "://" in path:
        # Likely a URL
        return path
    if path[0] != "/":
        # Many configs historically used paths like "hmlib/config/...".
        # When running from an installed wheel, ROOT_DIR is ".../site-packages/hmlib",
        # so joining ROOT_DIR + "hmlib/..." would incorrectly duplicate the segment.
        if path.startswith("hmlib/") and Path(ROOT_DIR).name == "hmlib":
            return str(Path(ROOT_DIR).parent / path)
        return os.path.join(ROOT_DIR, path)
    return path


def get_game_dir(game_id: str, assert_exists: bool = True) -> Optional[str]:
    """Return the video directory for a given game id.

    @param game_id: Game identifier string.
    @param assert_exists: If True, raise if the directory does not exist.
    @return: Absolute path to ``$HOME/Videos/<game_id>`` or ``None``.
    """
    if not game_id:
        raise AttributeError("No valid Game ID specified")
    game_video_dir = os.path.join(GAME_DIR_BASE, game_id)
    if os.path.isdir(game_video_dir):
        return game_video_dir
    if assert_exists:
        raise AssertionError(f"No game directory found for game id: {game_id}")
    return None


# TODO: implement passing all of this in cleanly somehow
def adjusted_config_path(path: str, team: str, season: str, args: argparse.Namespace):
    if team:
        path = os.path.join(path, args.team)
    if season:
        path = os.path.join(path, args.season)
    return path


def get_game_config_file_name(game: Game, root_dir: Optional[str] = None) -> Path:
    # Our first try is in the game dir
    game_dir = Path(GAME_DIR_BASE) / game.game_id / "config.yaml"
    if os.path.exists(game_dir) and not os.path.isdir(game_dir):
        return game_dir
    return Path(root_dir) / "config" / "games" / game.game_id


def load_config_file_yaml(yaml_file_path: str, merge_into_config: dict = None):
    """Load a YAML config from disk and optionally merge into a base dict."""
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as file:
            try:
                yaml_content = yaml.safe_load(file)
                if yaml_content is None:
                    # Empty file
                    return {}
                if isinstance(yaml_content, dict):
                    yaml_content = normalize_runtime_config(yaml_content)
                if merge_into_config:
                    yaml_content = recursive_update(merge_into_config, yaml_content)
                return yaml_content
            except yaml.YAMLError as exc:
                get_logger(__name__).exception(
                    "Failed to parse YAML config %s: %s", yaml_file_path, exc
                )
                raise
    return {} if not merge_into_config else merge_into_config


def load_yaml_files_ordered(
    paths: Sequence[str], base: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load multiple YAML files in order and merge them into a dictionary.

    Later files override earlier values and add new fields.

    @param paths: Sequence of YAML file paths (absolute or ROOT_DIR-relative).
    @param base: Optional starting dictionary to merge into.
    @return: Merged configuration dictionary.
    """
    merged: Dict[str, Any] = {} if base is None else dict(base)
    for p in paths:
        if not p:
            continue
        try:
            # Allow both absolute and ROOT_DIR-relative paths
            yaml_path = p
            if not os.path.isabs(yaml_path):
                candidate = os.path.join(ROOT_DIR, yaml_path)
                if os.path.exists(candidate):
                    yaml_path = candidate
            y = load_config_file_yaml(yaml_path)
            if y:
                merged = recursive_update(merged, y)
        except Exception:
            # Re-raise with additional context
            raise
    return merged


def load_config_file(
    config_type: str,
    config_name: str,
    merge_into_config: Optional[Dict[str, Any]] = None,
    root_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if root_dir is None:
        root_dir = ROOT_DIR
    return load_config_file_yaml(
        os.path.join(root_dir, "config", config_type, config_name + ".yaml"),
        merge_into_config=merge_into_config,
    )


def save_config_file(root_dir: str, config_type: str, config_name: str, data: dict):
    if root_dir is None:
        root_dir = ROOT_DIR
    yaml_file_path = os.path.join(root_dir, "config", config_type, config_name + ".yaml")
    with open(yaml_file_path, "w") as file:
        yaml.dump(data, file, sort_keys=False)


def baseline_config(root_dir: str) -> Dict[str, Any]:
    """Load the baseline configuration from ``config/baseline.yaml``."""
    return load_config_file(root_dir=root_dir, config_type=".", config_name="baseline")


def get_game_config_private(
    game_id: str,
    merge_into_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return load_config_file_yaml(
        yaml_file_path=os.path.join(GAME_DIR_BASE, game_id, "config.yaml"),
        merge_into_config=merge_into_config,
    )


def get_game_config(game_id: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    config_public = load_config_file(root_dir=root_dir, config_type="games", config_name=game_id)
    config_private = get_game_config_private(game_id=game_id)
    consolidated_config = recursive_update(config_public, config_private)
    return consolidated_config


def save_private_config(game_id: str, data: Dict[str, Any], verbose: bool = True):
    yaml_file_path = os.path.join(GAME_DIR_BASE, game_id, "config.yaml")
    data_to_save = copy.deepcopy(data) if isinstance(data, dict) else data
    if isinstance(data_to_save, dict):
        normalize_runtime_config(data_to_save)
    with open(yaml_file_path, "w") as file:
        yaml.dump(data_to_save, stream=file, sort_keys=False)
    if verbose:
        get_logger(__name__).info("Saved private config to %s", yaml_file_path)


def save_game_config(game_id: str, data: dict, root_dir: Optional[str] = None):
    return save_config_file(root_dir=root_dir, config_type="games", config_name=game_id, data=data)


def get_rink_config(rink: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    return load_config_file(root_dir=root_dir, config_type="rinks", config_name=rink)


def get_camera_config(camera: str, root_dir: Optional[str] = None) -> Dict[str, Any]:
    return load_config_file(root_dir=root_dir, config_type="camera", config_name=camera.lower())


def get_item(key: str, maps: List[Dict]):
    for map in maps:
        if map is not None and key in map:
            return map[key]
    return None


def get_config(
    root_dir: Optional[str] = None,
    game_id: Optional[str] = None,
    rink: Optional[str] = None,
    camera: Optional[str] = None,
    ignore_private_config: bool = False,
    resolve_globals: bool = True,
):
    """Return a consolidated configuration from baseline + rink + camera + game.

    Direct parameters override higher-level YAML (e.g. an explicit ``rink``
    argument overrides the rink specified in the game config).
    When ``ignore_private_config`` is true, per-game private config
    (``$HOME/Videos/<game_id>/config.yaml``) is not merged.
    """
    consolidated_config: Dict[str, Any] = baseline_config(root_dir=root_dir)
    game_config: Dict[str, Any] = dict()
    rink_config: Dict[str, Any] = dict()
    camera_config: Dict[str, Any] = dict()
    private_config: Dict[str, Any] = dict()
    if camera is not None:
        camera_config = get_camera_config(camera=camera, root_dir=root_dir)
    if rink is not None:
        rink_config = get_rink_config(rink=rink, root_dir=root_dir)
    if game_id is not None:
        if ignore_private_config:
            game_config = load_config_file(
                root_dir=root_dir, config_type="games", config_name=game_id
            )
        else:
            game_config = get_game_config(game_id=game_id, root_dir=root_dir)
            private_config = get_game_config_private(game_id=game_id)
    if camera is None:
        camera = get_item("camera", [game_config, rink_config])
        if isinstance(camera, str):
            camera_config = get_camera_config(camera=camera, root_dir=root_dir)
        elif camera and isinstance(camera, dict) and "name" in camera:
            camera_config = get_camera_config(camera=camera["name"], root_dir=root_dir)
    if rink is None:
        rink = get_nested_value(game_config, "game.rink")
        if rink:
            rink_config = get_rink_config(rink=rink, root_dir=root_dir)
    consolidated_config = recursive_update(consolidated_config, camera_config)
    consolidated_config = recursive_update(consolidated_config, rink_config)
    consolidated_config = recursive_update(consolidated_config, game_config)
    consolidated_config = recursive_update(consolidated_config, private_config)
    consolidated_config = normalize_runtime_config(consolidated_config)
    if resolve_globals:
        consolidated_config = resolve_global_refs(consolidated_config)
    return consolidated_config


def update_config(
    baseline_config: dict, config_type: str, config_name: str, root_dir: Optional[str] = None
):
    yaml_file_path = os.path.join(root_dir, "config", config_type, config_name + ".yaml")
    if not os.path.exists(yaml_file_path):
        return baseline_config
    config = load_config_file(root_dir=root_dir, config_type=config_type, config_name=config_name)
    return recursive_update(baseline_config, config)


@lru_cache
def get_clip_box(game_id: str, root_dir: Optional[str] = None, use_rink_boundary: bool = False):
    """Return the configured clip box for a game, optionally derived from rink.

    @param game_id: Game identifier.
    @param root_dir: Optional config root; defaults to :data:`ROOT_DIR`.
    @param use_rink_boundary: If True, fall back to rink boundary bbox.
    @return: Clip box as ``[x1, y1, x2, y2]`` or ``None``.
    """
    game_config = get_game_config(game_id=game_id, root_dir=root_dir)
    if game_config:
        game = game_config.get("game", None)
        if game and "clip_box" in game:
            return game["clip_box"]
        if use_rink_boundary:
            # Alternatively, use the rink boundary box
            rink_combined_bbox = get_nested_value(game_config, "rink.ice_contours_combined_bbox")
            if rink_combined_bbox:
                # Import lazily to avoid importing torch/numpy-heavy dependencies
                # during lightweight config/CLI usage.
                from hmlib.bbox.box_functions import scale_bbox_with_constraints

                rink_scaled_bbox = scale_bbox_with_constraints(
                    bbox=rink_combined_bbox,
                    ratio_x=1.1,
                    ratio_y=1.1,
                    min_x=0,
                    min_y=0,
                    max_x=float("inf"),
                    max_y=float("inf"),
                )
                rink_scaled_bbox = [int(i) for i in rink_scaled_bbox]
                return rink_scaled_bbox
    return None


#
# Dict utilities
#
def recursive_update(original, update):
    """
    Recursively update the original dictionary with the update dictionary.
    If a key in the original dictionary is not present in the update dictionary,
    its value is preserved.
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in original:
            recursive_update(original[key], value)
        else:
            original[key] = value
    return original


def _delete_nested_key(dct: Dict[str, Any], key_str: str) -> bool:
    keys = [k for k in str(key_str).split(".") if k]
    if not keys or not isinstance(dct, dict):
        return False
    cur: Any = dct
    parents: List[Tuple[Dict[str, Any], str]] = []
    for key in keys[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return False
        parents.append((cur, key))
        cur = cur[key]
    if not isinstance(cur, dict) or keys[-1] not in cur:
        return False
    del cur[keys[-1]]
    while parents:
        parent, key = parents.pop()
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            del parent[key]
        else:
            break
    return True


_LEGACY_RUNTIME_KEY_MAP: Tuple[Tuple[str, str], ...] = (
    ("aspen.stitching", "stitching"),
    ("aspen.video_out", "video_out"),
    ("aspen.apply_camera", "apply_camera"),
    ("aspen.play_tracker", "play_tracker"),
    ("aspen.ice_boundaries", "ice_boundaries"),
    ("aspen.left_stitch_pipeline", "stitching.left_stitch_pipeline"),
    ("aspen.right_stitch_pipeline", "stitching.right_stitch_pipeline"),
    ("aspen.video_out_pipeline", "video_out_pipeline"),
)

_LEGACY_STITCHING_KEY_MAP: Tuple[Tuple[str, str], ...] = (
    ("stitch-frame-time", "stitch_frame_time"),
    ("stitch_rotate_degrees", "post_stitch_rotate_degrees"),
    ("stitch-rotate-degrees", "post_stitch_rotate_degrees"),
)

_LEGACY_GAME_STITCHING_KEY_MAP: Tuple[Tuple[str, str], ...] = ()


def _merge_missing_nested_values(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if key not in dst or dst[key] is None:
            dst[key] = copy.deepcopy(value)
            continue
        if isinstance(dst[key], dict) and isinstance(value, dict):
            _merge_missing_nested_values(dst[key], value)
    return dst


def _canonicalize_stitching_config(stitching_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(stitching_cfg, dict):
        return stitching_cfg

    for legacy_key, canonical_key in _LEGACY_STITCHING_KEY_MAP:
        if legacy_key not in stitching_cfg:
            continue
        legacy_value = stitching_cfg.pop(legacy_key)
        if canonical_key not in stitching_cfg or stitching_cfg[canonical_key] is None:
            stitching_cfg[canonical_key] = legacy_value
    return stitching_cfg


def _migrate_game_stitching_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return config

    stitching_cfg = config.get("stitching")
    game_cfg = config.get("game")
    legacy_stitching_cfg = game_cfg.get("stitching") if isinstance(game_cfg, dict) else None
    has_legacy_game_stitching = isinstance(legacy_stitching_cfg, dict) or any(
        isinstance(game_cfg, dict) and key in game_cfg for key, _ in _LEGACY_GAME_STITCHING_KEY_MAP
    )

    if not isinstance(stitching_cfg, dict):
        if not has_legacy_game_stitching:
            return config
        stitching_cfg = {}
        config["stitching"] = stitching_cfg
    _canonicalize_stitching_config(stitching_cfg)

    if not isinstance(game_cfg, dict):
        return config

    if isinstance(legacy_stitching_cfg, dict):
        legacy_copy = copy.deepcopy(legacy_stitching_cfg)
        _canonicalize_stitching_config(legacy_copy)
        _merge_missing_nested_values(stitching_cfg, legacy_copy)
        del game_cfg["stitching"]

    for legacy_key, canonical_key in _LEGACY_GAME_STITCHING_KEY_MAP:
        if legacy_key not in game_cfg:
            continue
        legacy_value = game_cfg.pop(legacy_key)
        current_value = stitching_cfg.get(canonical_key)
        if isinstance(current_value, dict) and isinstance(legacy_value, dict):
            _merge_missing_nested_values(current_value, legacy_value)
        elif canonical_key not in stitching_cfg or stitching_cfg[canonical_key] is None:
            stitching_cfg[canonical_key] = copy.deepcopy(legacy_value)

    if not game_cfg:
        del config["game"]
    return config


def normalize_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize legacy runtime config keys.

    The Aspen namespace now owns graph structure (`plugins`, `pipeline`,
    `inference_pipeline`, etc.). Runtime values consumed by plugins live at
    top-level keys such as ``stitching`` and ``video_out``. Legacy nested
    stitching layouts are also migrated into the root-level ``stitching`` block.

    Legacy configs are still accepted by moving known ``aspen.*`` runtime keys
    into their new locations before CLI overrides and GLOBAL substitution.
    """
    if not isinstance(config, dict):
        return config

    for legacy_path, new_path in _LEGACY_RUNTIME_KEY_MAP:
        legacy_value = get_nested_value(config, legacy_path, default_value=None)
        if legacy_value is None:
            continue
        current_value = get_nested_value(config, new_path, default_value=None)
        if isinstance(current_value, dict) and isinstance(legacy_value, dict):
            recursive_update(current_value, copy.deepcopy(legacy_value))
        elif current_value is None:
            set_nested_value(config, new_path, copy.deepcopy(legacy_value))
        _delete_nested_key(config, legacy_path)
    _migrate_game_stitching_config(config)
    return config


def get_nested_value(dct, key_str, default_value=None):
    """
    Retrieve a value from a nested dictionary using a dot-separated key string.

    Parameters:
    - dct (dict): The dictionary to search.
    - key_str (str): The dot-separated key string.

    Returns:
    - The value if found, otherwise None.
    """
    keys = key_str.split(".")
    current = dct

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default_value

    return current


def set_nested_value(dct, key_str, set_to, noset_value=None, *, create_missing: bool = True):
    if noset_value is None and set_to is None:
        return get_nested_value(dct, key_str, set_to)
    if set_to == noset_value:
        return get_nested_value(dct, key_str, noset_value)

    keys = key_str.split(".")
    current = dct

    for i, key in enumerate(keys):
        if not isinstance(current, dict):
            raise TypeError(
                f"Expected dict at {'.'.join(keys[:i]) or '<root>'}, got {type(current)}"
            )
        if key in current:
            if i == len(keys) - 1:
                current[key] = set_to
            else:
                current = current[key]
            continue

        if not create_missing:
            prefix = ".".join(keys[: i + 1])
            raise KeyError(f"Key path not found: {key_str!r} (missing {prefix!r})")

        if i == len(keys) - 1:
            current[key] = set_to
        else:
            current[key] = dict()
            current = current[key]
    return get_nested_value(dct, key_str)


GLOBAL_REF_PREFIX = "GLOBAL."


def _lookup_path(config: Dict[str, Any], path_parts: Sequence[str]) -> Tuple[bool, Any]:
    cur: Any = config
    for key in path_parts:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return False, None
    return True, cur


def _resolve_global_value(
    config: Dict[str, Any], value: Any, seen: Optional[Set[str]] = None
) -> Any:
    if seen is None:
        seen = set()
    if isinstance(value, str) and value.startswith(GLOBAL_REF_PREFIX):
        path_str = value[len(GLOBAL_REF_PREFIX) :]
        if path_str in seen:
            return value
        seen.add(path_str)
        ok, resolved = _lookup_path(config, [p for p in path_str.split(".") if p])
        if not ok:
            return value
        return _resolve_global_value(config, resolved, seen)
    return value


def resolve_global_refs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace ``GLOBAL.*`` string references with values from the merged config.

    Example::
        brightness: GLOBAL.camera.color.brightness

    Args:
        config: Merged configuration dictionary.
    Returns:
        The same dict with references resolved in-place.
    """

    normalize_runtime_config(config)

    def _walk(node: Any, root: Dict[str, Any]) -> Any:
        if isinstance(node, dict):
            for k, v in node.items():
                node[k] = _walk(v, root)
            return node
        if isinstance(node, list):
            for i, v in enumerate(node):
                node[i] = _walk(v, root)
            return node
        return _resolve_global_value(root, node)

    _walk(config, config)
    return config
