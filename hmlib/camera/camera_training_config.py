"""Strict YAML configuration and immutable catalog selection for camera training."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import random
from pathlib import Path

import yaml

from hmlib.camera.camera_gpt_dataset import GameCsvPaths


def read_mapping(path: Path) -> dict:
    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in result:
                    raise ValueError(f"Duplicate YAML key {key!r} in {path}")
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    value = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def expand_training_config(parser: argparse.ArgumentParser, argv: list[str]) -> list[str]:
    """Translate YAML defaults to CLI arguments, so CLI overrides keep normal validation."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    options, _ = bootstrap.parse_known_args(argv)
    if options.config is None:
        return argv
    path = options.config.expanduser().resolve()
    config = read_mapping(path)
    if config.pop("schema", None) != "hockey-drivegpt-training-v1":
        raise ValueError(f"Unsupported training configuration schema: {path}")
    actions = {a.dest: a for a in parser._actions}
    option_actions = {
        option: action for action in parser._actions for option in action.option_strings
    }
    explicit = {
        option_actions[arg.split("=", 1)[0]].dest
        for arg in argv
        if arg.split("=", 1)[0] in option_actions
    }
    if explicit.intersection({"resume", "no_resume"}):
        explicit.update({"resume", "no_resume"})
    if explicit.intersection({"steps", "max_iters"}):
        explicit.update({"steps", "max_iters"})
    groups = {
        "dataset",
        "model",
        "features",
        "sampling",
        "optimization",
        "validation",
        "checkpoint",
        "distributed",
    }
    args, seen = [], set()
    for group, mapping in config.items():
        if group not in groups or not isinstance(mapping, dict):
            raise ValueError(f"Unknown/non-mapping training group {group!r} in {path}")
        for key, value in mapping.items():
            if key not in actions or key in {"help", "config"} or key in seen:
                raise ValueError(f"Unknown or repeated training option {key!r} in {path}")
            seen.add(key)
            if key in explicit:
                continue
            action = actions[key]
            if value is None:
                continue
            if key == "dataset_config":
                dataset_path = Path(value).expanduser()
                value = str(
                    dataset_path if dataset_path.is_absolute() else path.parent / dataset_path
                )
            if isinstance(action, argparse.BooleanOptionalAction):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a YAML boolean")
                positive = next(
                    s
                    for s in action.option_strings
                    if s.startswith("--") and not s.startswith("--no-")
                )
                args.append(positive if value else "--no-" + positive[2:])
            elif isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    raise ValueError(f"{key} must be a YAML boolean")
                if value:
                    args.append(action.option_strings[0])
            elif isinstance(action, argparse._AppendAction):
                if not isinstance(value, list):
                    raise ValueError(f"{key} must be a YAML list")
                args.extend(f"{action.option_strings[0]}={item}" for item in value)
            else:
                if isinstance(value, (dict, list, bool)):
                    raise ValueError(f"Invalid scalar value for {key}")
                args.append(f"{action.option_strings[0]}={value}")
    return args + argv


def catalog_split(
    config_path: str,
    root_override: str | None = None,
    min_train_frames: int = 2,
    min_val_frames: int = 2,
) -> tuple[list[GameCsvPaths], list[GameCsvPaths], dict]:
    path = Path(config_path).expanduser().resolve()
    config = read_mapping(path)
    allowed = {"schema", "root", "catalog", "include", "exclude", "split"}
    if set(config) - allowed or config.get("schema") != "hockey-drivegpt-dataset-v1":
        raise ValueError(f"Invalid dataset schema/keys in {path}")
    root = Path(root_override or config.get("root", ".")).expanduser()
    root = (root if root.is_absolute() else path.parent / root).resolve()
    catalog_path = root / config.get("catalog", "catalog.json")
    raw = catalog_path.read_bytes()
    catalog = json.loads(raw)
    if catalog.get("schema") != "hockey-drivegpt-catalog-v1":
        raise ValueError(f"Unsupported catalog: {catalog_path}")
    include, exclude = config.get("include", ["*"]), config.get("exclude", [])
    for name, patterns in (("include", include), ("exclude", exclude)):
        if not isinstance(patterns, list) or not all(isinstance(s, str) for s in patterns):
            raise ValueError(f"Dataset {name} must be a list of game ID glob patterns")
    entries = []
    identities, ids = set(), set()
    for game in catalog["games"]:
        gid = game["game_id"]
        if not any(fnmatch.fnmatchcase(gid, p) for p in include) or any(
            fnmatch.fnmatchcase(gid, p) for p in exclude
        ):
            continue
        identity = game["files"]["tracking"]["sha256"]
        if gid in ids or identity in identities:
            raise ValueError(
                f"Duplicate game/tracking identity selected: {gid}; exclude the duplicate in dataset YAML"
            )
        ids.add(gid)
        identities.add(identity)
        entries.append(game)
    if not entries:
        raise ValueError("Dataset selectors matched no games")
    split = config.get("split", {})
    if not isinstance(split, dict) or set(split) - {
        "seed",
        "validation_fraction",
        "validation_games",
        "groups",
    }:
        raise ValueError("Invalid dataset split options")
    groups = split.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError("split.groups must map source-game group names to game ID lists")
    group_for = {gid: gid for gid in ids}
    grouped = set()
    all_ids = {g["game_id"] for g in catalog["games"]}
    for group, members in groups.items():
        if (
            not isinstance(group, str)
            or not isinstance(members, list)
            or not all(isinstance(gid, str) for gid in members)
        ):
            raise ValueError("Invalid source-game group")
        if set(members) - all_ids or grouped.intersection(members):
            raise ValueError(f"Unknown/repeated source-game group members: {group}")
        grouped.update(members)
        for gid in set(members).intersection(ids):
            group_for[gid] = "group:" + group
    if "validation_games" in split:
        if not isinstance(split["validation_games"], list) or not all(
            isinstance(gid, str) for gid in split["validation_games"]
        ):
            raise ValueError("validation_games must be a list of game IDs")
        val_ids = set(split["validation_games"])
        if val_ids - ids:
            raise ValueError(f"Validation games are not selected: {sorted(val_ids - ids)}")
        val_groups = {group_for[gid] for gid in val_ids}
        val_ids = {gid for gid in ids if group_for[gid] in val_groups}
    else:
        fraction = float(split.get("validation_fraction", 0.1))
        if not 0 <= fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        shuffled = sorted(set(group_for.values()))
        random.Random(int(split.get("seed", 0))).shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * fraction)) if fraction else 0
        val_groups = set(shuffled[:n_val])
        val_ids = {gid for gid in ids if group_for[gid] in val_groups}
    train, val = [], []
    for entry in sorted(entries, key=lambda g: g["game_id"]):
        required = min_val_frames if entry["game_id"] in val_ids else min_train_frames
        if entry["longest_run"] < required:
            raise ValueError(f"{entry['game_id']} has no contiguous run of {required} frames")
        files = {}
        for role, artifact in entry["files"].items():
            artifact_path = (root / artifact["path"]).resolve()
            if not artifact_path.is_relative_to(root):
                raise ValueError(f"Catalog artifact escapes dataset root: {artifact_path}")
            if not artifact_path.is_file() or artifact_path.stat().st_size != artifact["bytes"]:
                raise ValueError(f"Missing/changed catalog artifact: {artifact_path}")
            digest = hashlib.sha256()
            with artifact_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != artifact["sha256"]:
                raise ValueError(f"Catalog artifact checksum mismatch: {artifact_path}")
            files[role] = str(artifact_path)
        paths = GameCsvPaths(
            entry["game_id"], files["tracking"], files["camera"], files["camera_fast"]
        )
        (val if entry["game_id"] in val_ids else train).append(paths)
    if not train:
        raise ValueError("Dataset split contains no training games")
    identity = {
        "catalog_sha256": hashlib.sha256(raw).hexdigest(),
        "train_games": [p.game_id for p in train],
        "validation_games": [p.game_id for p in val],
        "source_groups": group_for,
    }
    return train, val, identity
