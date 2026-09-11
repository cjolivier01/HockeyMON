import hashlib
import json
import shutil
import sqlite3
import uuid

import cv2
import numpy as np
import pytest

from hmlib.camera.camera_database import (
    discover_database_games,
    load_database_rink,
    publish_database_dataset,
    split_database_games,
)
from hmlib.camera.camera_gpt_dataset import _load_game
from hmlib.camera.camera_training_config import catalog_split
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.camera.rink_context import mask_to_grid
from hmlib.telemetry.database import (
    create_database,
    discover_runs,
    merge_databases,
    read_database,
)


def recording(path, game="game-a", run_id=None, *, width=200, height=100):
    run_id = run_id or str(uuid.uuid4())
    connection = create_database(path)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 10 : -height // 10, width // 10 : -width // 10] = 255
    ok, png = cv2.imencode(".png", mask)
    assert ok
    png = png.tobytes()
    connection.execute(
        "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            game,
            "2026-09-11T00:00:00Z",
            "hstream",
            "source: {}",
            "effective: {}",
            1,
            "intentional-stop",
            80,
        ),
    )
    connection.execute(
        "INSERT INTO geometries VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            1,
            width,
            height,
            "original_stitched_pixels",
            "revision-1",
            "png",
            png,
            hashlib.sha256(png).hexdigest(),
            json.dumps([[1, 0, 0], [0, 1, 0]]),
        ),
    )
    for sample in range(1, 81):
        count = 0 if sample == 10 else 2
        connection.execute(
            "INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                sample,
                0,
                sample - 1,
                0,
                sample - 1,
                sample * 16666667,
                None,
                0,
                0,
                1,
                0,
                count,
            ),
        )
        for ordinal in range(count):
            connection.execute(
                "INSERT INTO tracks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    sample,
                    ordinal,
                    str(2**64 - 1 - ordinal),
                    30 + ordinal * 10,
                    20,
                    5,
                    10,
                    0.9,
                    0,
                    "{}",
                ),
            )
        for role in ("program", "fast"):
            connection.execute(
                "INSERT INTO cameras VALUES(?,?,?,?,?,?,?)", (run_id, sample, role, 10, 10, 100, 60)
            )
    connection.execute(
        "INSERT INTO config_events VALUES(?,?,?,?,?,?,?,?)",
        (run_id, 1, 41, "runtime-tuning", "speed", "2", "config", "speed: 2"),
    )
    connection.commit()
    connection.close()
    return run_id, mask


def should_mixed_single_and_merged_databases_deduplicate_and_group_games(tmp_path):
    first, second, third = (tmp_path / name for name in ("one.db", "two.db", "three.db"))
    first_id, _ = recording(first)
    second_id, _ = recording(second)
    third_id, _ = recording(third, "game-b")
    merged = tmp_path / "merged.db"
    assert set(merge_databases(merged, [first, second])["added"]) == {first_id, second_id}
    assert merge_databases(merged, [first])["already_present"] == [first_id]
    games, identity = discover_database_games([first, merged, third])
    assert {game.run_id for game in games} == {first_id, second_id, third_id}
    assert len(identity["runs"]) == 3
    train, val = split_database_games(games, 0.5, 4, ["game-b"])
    assert {game.run_id for game in train} == {first_id, second_id}
    assert {game.run_id for game in val} == {third_id}


def should_conflicting_guid_aborts_the_whole_merge(tmp_path):
    original, conflict, new = (tmp_path / name for name in ("a.db", "z-conflict.db", "b-new.db"))
    run_id, _ = recording(original)
    new_id, _ = recording(new, "game-b")
    shutil.copy2(original, conflict)
    with sqlite3.connect(conflict) as connection:
        connection.execute("UPDATE cameras SET left=left+1")
    merged = tmp_path / "merged.db"
    merge_databases(merged, [original])
    with pytest.raises(ValueError, match="Conflicting content"):
        merge_databases(merged, [new, conflict])
    assert [run["run_id"] for run in discover_runs([merged])] == [run_id]
    assert new_id != run_id
    with pytest.raises(ValueError, match="Conflicting content"):
        discover_runs([original, conflict])


def should_merge_preserve_full_run_configuration_archive(tmp_path):
    source, merged = tmp_path / "game.db", tmp_path / "merged.db"
    run_id, _ = recording(source)
    configuration = (
        "schema: hstream-run-configuration-v1\n"
        "resolved: {pipeline: {detector: example}, stitching: {projection: panini}}\n"
        "input-layers: {baseline: {version: 1}, user: {}, game: {}}\n"
    )
    with sqlite3.connect(source) as connection:
        connection.execute(
            "INSERT INTO config_events VALUES(?,?,?,?,?,?,?,?)",
            (
                run_id,
                2,
                1,
                "run-configuration",
                "startup",
                "hstream-run-configuration-v1",
                "run-config.yaml",
                configuration,
            ),
        )
    merge_databases(merged, [source])
    with sqlite3.connect(merged) as connection:
        assert connection.execute(
            "SELECT run_id,sample_boundary,artifact_contents FROM config_events WHERE kind='run-configuration'"
        ).fetchone() == (run_id, 1, configuration)
    assert len(discover_runs([source, merged])) == 1


def should_database_training_retains_empty_frames_order_and_policy_boundaries(tmp_path):
    path = tmp_path / "hstream_telemetry-2.db"
    _, mask = recording(path)
    games, _ = discover_database_games([path])
    norm = CameraNorm(scale_x=200, scale_y=100, max_players=8)
    loaded = _load_game(
        games[0],
        norm=norm,
        target_mode="slow_fast_tlwh",
        include_rink=True,
        include_pose=False,
        rink_input="grid",
    )
    assert loaded.frames == list(range(1, 81))
    assert [len(run) for run in loaded.frame_runs] == [40, 40]
    assert 10 not in loaded.tracks_by_frame
    assert loaded.tracks_by_frame[1][:, 0].tolist() == [30, 40]
    np.testing.assert_array_equal(
        loaded.rink_feat, mask_to_grid(mask, norm, 32, 64, frame_size=(200, 100))
    )
    assert list(tmp_path.iterdir()) == [path]


def should_mask_hash_and_completion_are_required(tmp_path):
    path = tmp_path / "run.db"
    recording(path)
    games, _ = discover_database_games([path])
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE geometries SET mask_sha256='bad'")
    with pytest.raises(ValueError, match="checksum"):
        load_database_rink(games[0], CameraNorm(200, 100, 8))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE runs SET completed=0")
    with pytest.raises(ValueError, match="incomplete"):
        discover_runs([path])


def should_geometry_revisions_split_a_run_into_independent_training_sequences(tmp_path):
    path = tmp_path / "run.db"
    run_id, _ = recording(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO geometries SELECT run_id,2,width,height,coordinate_space,'revision-2',mask_codec,mask,mask_sha256,mask_to_tracking FROM geometries"
        )
        connection.execute("UPDATE frames SET geometry_id=2 WHERE sample_id>=41")
    games, _ = discover_database_games([path])
    assert [(game.run_id, game.geometry_id) for game in games] == [(run_id, 1), (run_id, 2)]
    norm = CameraNorm(200, 100, 8)
    first = _load_game(
        games[0],
        norm=norm,
        target_mode="slow_fast_tlwh",
        include_rink=True,
        include_pose=False,
        rink_input="grid",
    )
    second = _load_game(
        games[1],
        norm=norm,
        target_mode="slow_fast_tlwh",
        include_rink=True,
        include_pose=False,
        rink_input="grid",
    )
    assert first.frames[-1] == 40
    assert second.frames[0] == 41


def should_published_dataset_preserves_suffixes_and_trains_from_databases(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    recording(source / "hstream_telemetry-2.db")
    recording(source / "hstream_telemetry-4.db", "game-b")
    destination = tmp_path / "published"
    catalog = publish_database_dataset([source], destination)
    assert catalog["schema"] == "hockey-drivegpt-catalog-v2"
    assert sorted(path.name for path in destination.rglob("*.db")) == [
        "hstream_telemetry-2.db",
        "hstream_telemetry-4.db",
    ]
    assert not list(destination.rglob("*.csv"))
    train, val, identity = catalog_split(
        str(destination / "dataset.yaml"),
        min_train_frames=32,
        min_val_frames=32,
        require_rink_grid=True,
    )
    assert len(train) == len(val) == 1
    assert len(identity["runs"]) == 2
    with read_database(train[0].database_path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchone() is None


def should_preserve_dataset_identity_when_database_packing_changes(tmp_path):
    a, b, combined = (tmp_path / name for name in ("a.db", "b.db", "combined.db"))
    recording(a, run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    recording(b, run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    merge_databases(combined, [a, b])
    original, original_identity = discover_database_games([a, b])
    merged, merged_identity = discover_database_games([combined])
    assert [game.game_id for game in original] == [game.game_id for game in merged]
    assert original_identity == merged_identity


def should_verify_publication_and_omit_short_geometry_selections(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    first, second = source / "a.db", source / "b.db"
    run_id, _ = recording(first)
    recording(second, "game-b")
    with sqlite3.connect(first) as connection:
        connection.execute(
            "INSERT INTO geometries SELECT run_id,2,width,height,coordinate_space,'short',mask_codec,mask,mask_sha256,mask_to_tracking FROM geometries"
        )
        connection.execute("UPDATE frames SET geometry_id=2 WHERE sample_id=80")
    destination = tmp_path / "published"
    catalog = publish_database_dataset([source], destination, min_frames=32)
    assert [(item["run_id"], item["geometry_id"]) for item in catalog["rejected"]] == [(run_id, 2)]
    train, val, _ = catalog_split(
        str(destination / "dataset.yaml"), min_train_frames=32, min_val_frames=32
    )
    assert len(train) == len(val) == 1
    assert all(game.geometry_id == 1 for game in train + val)
    with sqlite3.connect(train[0].database_path) as connection:
        connection.execute("UPDATE cameras SET left=left+10")
    with pytest.raises(ValueError, match="published catalog"):
        catalog_split(str(destination / "dataset.yaml"), min_train_frames=32, min_val_frames=32)


def should_filter_short_geometries_before_worker_sharding(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hmlib.camera import camera_gpt_dataset as dataset_module

    path = tmp_path / "run.db"
    recording(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO geometries SELECT run_id,2,width,height,coordinate_space,'short',mask_codec,mask,mask_sha256,mask_to_tracking FROM geometries"
        )
        connection.execute("UPDATE frames SET geometry_id=2 WHERE sample_id=80")
    games, _ = discover_database_games([path])
    dataset = dataset_module.CameraPanZoomGPTIterableDataset(
        games,
        CameraNorm(200, 100, 8),
        seq_len=32,
        target_mode="slow_fast_tlwh",
        include_pose=False,
        include_rink=False,
        shard_games_by_worker=True,
    )
    assert len(dataset._games) == 1
    for worker_id in range(2):
        monkeypatch.setattr(
            dataset_module, "get_worker_info", lambda: SimpleNamespace(id=worker_id, num_workers=2)
        )
        assert next(iter(dataset))["y"].shape[0] == 32
