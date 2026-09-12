from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from hmlib.aspen.plugins.save_plugins import (
    SaveCameraPlugin,
    SaveDetectionsPlugin,
    SaveTrackingPlugin,
)
from hmlib.camera.camera_database import (
    discover_database_games,
    load_database_frames,
    load_database_rink,
)
from hmlib.camera.camera_transformer import CameraNorm
from hmlib.telemetry.database import discover_runs, read_database
from hmlib.telemetry.recorder import TelemetryRecorder, complete_recording, configuration_snapshot
from hmlib.telemetry.tracking import DatabaseTrackingDataFrame


def context(frames=(0, 1), mask=None):
    if mask is None:
        mask = torch.ones((40, 80), dtype=torch.bool)
    samples = []
    for frame in frames:
        inst = SimpleNamespace(
            bboxes=torch.tensor([[1.0, 2.0, 8.0, 12.0], [0.0, 0.0, 0.0, 0.0]]),
            scores=torch.tensor([0.9, 0.0]),
            labels=torch.tensor([0, -1]),
            num_detections=torch.tensor(1),
            num_tracks=torch.tensor(1),
            instances_id=np.array([2**63 + 5, 0], dtype=np.uint64),
        )
        samples.append(
            SimpleNamespace(
                metainfo={"frame_id": frame}, pred_instances=inst, pred_track_instances=inst
            )
        )
    return {
        "data_samples": [samples],
        "original_images": torch.zeros(len(frames), 3, 40, 80),
        "frame_ids": torch.tensor(frames),
        "frame_id": frames[0],
        "fps": 30.0,
        "rink_profile": {"combined_mask": mask},
        "current_box": torch.tensor([[0.0, 0.0, 80.0, 40.0]] * len(frames)),
        "current_fast_box_list": torch.tensor([[2.0, 3.0, 72.0, 38.0]] * len(frames)),
        "camera_policy_events": [],
    }


def capture(recorder, ctx, *, stages=("detections", "tracks", "cameras")):
    batch = recorder.new_batch()
    ctx["telemetry_batch"] = batch
    for kind in stages:
        batch.capture(kind, ctx)
    return batch


def should_write_compressed_mask_once_and_read_training_data(tmp_path, monkeypatch):
    encodes = []
    original = cv2.imencode
    monkeypatch.setattr(cv2, "imencode", lambda *args: (encodes.append(args[0]) or original(*args)))
    recorder = TelemetryRecorder(tmp_path, "game-a", {}, {"detections", "tracks", "cameras"})
    ctx = context()
    ctx["camera_policy_events"] = [{"frame": 0, "kind": "startup", "policy": {"zoom": 1}}]
    capture(recorder, ctx)
    ctx["frame_ids"] = torch.tensor([2, 3])
    ctx["camera_policy_events"] = [{"frame": 3, "kind": "change", "policy": {"zoom": 2}}]
    # Include a genuine zero-track sample, separately from static padded allocations.
    ctx["data_samples"][0][1].pred_track_instances = None
    capture(recorder, ctx)
    recorder.close()
    with pytest.raises(ValueError, match="incomplete"):
        discover_runs([recorder.path])
    complete_recording(recorder.path)
    assert encodes == [".png"]
    games, _ = discover_database_games([recorder.path])
    grid = load_database_rink(games[0], CameraNorm(80, 40, 50), height=4, width=8)
    assert grid.any()
    tracks, program, fast, frames, boundaries = load_database_frames(games[0])
    assert frames == {1, 2, 3, 4}
    assert len(tracks) == 3 and len(program) == len(fast) == 4
    assert {1, 4} <= boundaries
    with read_database(recorder.path) as db:
        assert db.execute("SELECT tracking_id FROM tracks LIMIT 1").fetchone()[0] == str(2**63 + 5)
        geometry = db.execute("SELECT * FROM geometries").fetchone()
        assert len(geometry["mask"]) < 40 * 80
        decoded = cv2.imdecode(np.frombuffer(geometry["mask"], np.uint8), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(decoded, ctx["rink_profile"]["combined_mask"].numpy() * 255)
        assert db.execute("SELECT sum(detection_count) FROM frames").fetchone()[0] == 4


def should_snapshot_at_save_stages_and_emit_no_csv(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "game", {}, {"detections", "tracks", "cameras"})
    ctx = context((5,))
    ctx["telemetry_batch"] = recorder.new_batch()
    ctx["work_dir"] = str(tmp_path)
    SaveDetectionsPlugin().forward(ctx)
    # Mutating the shared detection object after its save stage cannot alter the archive.
    ctx["data_samples"][0][0].pred_instances.bboxes[0, 0] = 3
    SaveTrackingPlugin().forward(ctx)
    SaveCameraPlugin().forward(ctx)
    recorder.close()
    complete_recording(recorder.path)
    with read_database(recorder.path) as db:
        assert db.execute("SELECT left FROM detections").fetchone()[0] == 1
        assert db.execute("SELECT left FROM tracks").fetchone()[0] == 3
    assert not list(tmp_path.glob("*.csv"))


def should_order_parallel_batches_and_archive_geometry_changes(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "game", {}, {"tracks", "cameras"})
    first, second = recorder.new_batch(), recorder.new_batch()
    a, b = context((8, 9)), context((3, 4), torch.eye(40, 80, dtype=torch.bool))
    second.capture("cameras", b)
    second.capture("tracks", b)
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(first.capture, kind, a) for kind in ("cameras", "tracks")]
        for future in futures:
            future.result()
    recorder.close()
    complete_recording(recorder.path)
    with read_database(recorder.path) as db:
        assert [
            tuple(row)
            for row in db.execute("SELECT source_frame,seek_epoch FROM frames ORDER BY sample_id")
        ] == [(8, 0), (9, 0), (3, 1), (4, 1)]
        assert db.execute("SELECT count(*) FROM geometries").fetchone()[0] == 2


def should_fail_missing_stages_and_never_complete(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "game", {}, {"tracks", "cameras"})
    capture(recorder, context(), stages=("tracks",))
    with pytest.raises(RuntimeError, match="missing"):
        recorder.close()
    with pytest.raises(ValueError, match="empty|incomplete"):
        complete_recording(recorder.path)


def should_propagate_late_writer_errors(tmp_path, monkeypatch):
    recorder = TelemetryRecorder(tmp_path, "game", {}, {"tracks"}, capacity=1)
    monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    capture(recorder, context(), stages=("tracks",))
    with pytest.raises(RuntimeError, match="encode"):
        recorder.close()
    with read_database(recorder.path) as db:
        assert db.execute("SELECT completed FROM runs").fetchone()[0] == 0


def should_reject_mismatched_frames_and_duplicate_stages(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "game", {}, {"tracks", "cameras"})
    batch = recorder.new_batch()
    batch.capture("tracks", context((1,)))
    with pytest.raises(ValueError, match="duplicate"):
        batch.capture("tracks", context((1,)))
    batch.capture("cameras", context((2,)))
    with pytest.raises(RuntimeError, match="frame IDs"):
        recorder.close()


def should_reuse_recorded_tracks_and_preserve_empty_frames(tmp_path):
    recorder = TelemetryRecorder(tmp_path, "game", {}, {"tracks"})
    ctx = context()
    ctx["data_samples"][0][0].pred_track_instances.instances_id = torch.tensor([77, 0])
    ctx["data_samples"][0][1].pred_track_instances = None
    capture(recorder, ctx, stages=("tracks",))
    recorder.close()
    complete_recording(recorder.path)
    tracks = DatabaseTrackingDataFrame(input_file=recorder.path, input_batch_size=1)
    assert tracks.get_data_dict_by_frame(0)["tracking_ids"].tolist() == [77]
    assert tracks.get_data_dict_by_frame(1)["tracking_ids"].tolist() == []
    with pytest.raises(ValueError, match="absent"):
        tracks.get_data_dict_by_frame(2)


def should_preserve_prior_working_recordings_and_config_cycles(tmp_path):
    config = {"resolution": [80, 40]}
    config["initial_args"] = {"game_config": config, "runtime": object()}
    snapshot = configuration_snapshot(config)
    assert snapshot["initial_args"]["game_config"] == {"reference": "root"}
    paths = []
    for _ in range(2):
        recorder = TelemetryRecorder(tmp_path, "game", config, {"tracks"})
        capture(recorder, context(), stages=("tracks",))
        recorder.close()
        complete_recording(recorder.path)
        paths.append(recorder.path)
    assert paths[0] != paths[1]
    assert len(discover_runs(paths)) == 2


def should_publish_only_current_database_and_keep_calibration_png(tmp_path):
    from hmlib.cli.hmtrack import _deploy_output_artifacts

    work = tmp_path / "work"
    game = tmp_path / "game"
    game.mkdir()
    (game / "rink_mask_0.png").write_bytes(b"calibration")
    (game / "hstream_telemetry-5.db").write_bytes(b"old generation")
    recorder = TelemetryRecorder(work, "game", {}, {"tracks"})
    capture(recorder, context(), stages=("tracks",))
    recorder.close()
    complete_recording(recorder.path)
    (work / "tracking.csv").write_text("stale CSV")
    (work / "rink_mask_0.png").write_bytes(b"stale snapshot")
    (work / "hm_telemetry-999.db").write_bytes(b"unrelated working run")
    _deploy_output_artifacts(
        output_video_path=None,
        output_video=None,
        results_folder=str(work),
        target_deploy_dir=str(game),
        game_id="game",
        telemetry_path=recorder.path,
    )
    assert (game / "rink_mask_0.png").read_bytes() == b"calibration"
    assert (game / "hm_telemetry-6.db").is_file()
    assert not list(game.glob("*.csv"))
    assert [p.name for p in game.glob("rink_mask_*.png")] == ["rink_mask_0.png"]


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("fail_shutdown", [False, True])
def should_integrate_with_pipeline_and_keep_failed_shutdown_incomplete(
    tmp_path, monkeypatch, threaded, fail_shutdown
):
    from hmlib.aspen import AspenNet
    from hmlib.tasks.tracking import run_mmtrack

    class Loader:
        batch_size = 2
        fps = 30.0

        def __len__(self):
            return 2

        def __iter__(self):
            for frames in ((0, 1), (2, 3)):
                ctx = context(frames)
                yield {
                    "pano": {
                        "original_images": ctx["original_images"],
                        "img_info": {"frame_id": frames[0]},
                        "ids": ctx["frame_ids"],
                        "data_samples": ctx["data_samples"],
                    }
                }

    if fail_shutdown:
        original = AspenNet.finalize

        def fail(net):
            original(net)
            raise OSError("late encoder shutdown failure")

        monkeypatch.setattr(AspenNet, "finalize", fail)
    device = torch.device("cuda" if threaded and torch.cuda.is_available() else "cpu")
    cfg = {
        "game_id": "test",
        "game_dir": str(tmp_path),
        "work_dir": str(tmp_path),
        "output_label": "variant",
        "aspen": {
            "threaded_trunks": threaded,
            "pipeline": {
                "threaded": threaded,
                "graph": device.type == "cuda",
                "cuda_streams": device.type == "cuda",
                "queue_size": 1,
            },
            "graph": {"minimal_context": True},
            "plugins": {
                "save_tracking": {
                    "class": "hmlib.aspen.plugins.save_plugins.SaveTrackingPlugin",
                    "depends": [],
                    "params": {},
                },
                "save_pose": {
                    "class": "hmlib.aspen.plugins.save_plugins.SavePosePlugin",
                    "depends": ["save_tracking"],
                    "params": {},
                },
            },
        },
    }
    if fail_shutdown:
        with pytest.raises(Exception, match="late encoder"):
            run_mmtrack(None, cfg, Loader(), None, device=device, no_cuda_streams=True)
        path = tmp_path / "hm_telemetry.db"
    else:
        artifacts = run_mmtrack(None, cfg, Loader(), None, device=device, no_cuda_streams=True)
        path = artifacts.telemetry_path
        assert artifacts.supplementary_paths == (tmp_path / "variant_pose.csv",)
    with read_database(path) as db:
        assert db.execute("SELECT completed FROM runs").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM frames").fetchone()[0] == 4
    if not fail_shutdown:
        complete_recording(path)
        assert len(discover_runs([path])) == 1


def should_keep_loaded_tracks_in_experiment_recording_graph():
    from hmlib.cli.hmtrack import _enable_load_tracking_plugin

    cfg = {
        "aspen": {
            "plugins": {
                name: {"enabled": True}
                for name in (
                    "tracker",
                    "save_tracking",
                    "save_detections",
                    "image_prep",
                    "camera_controller",
                    "ice_config",
                )
            }
        }
    }
    _enable_load_tracking_plugin(cfg, disable_detector=True)
    plugins = cfg["aspen"]["plugins"]
    assert plugins["save_tracking"]["enabled"]
    assert plugins["save_tracking"]["depends"] == ["load_tracking"]
    assert not plugins["tracker"]["enabled"]


def should_publish_exact_labeled_supplements_without_stale_outputs(tmp_path):
    from hmlib.cli.hmtrack import _deploy_output_artifacts

    work, game = tmp_path / "work", tmp_path / "game"
    writer = TelemetryRecorder(work, "game", {}, {"tracks"})
    capture(writer, context(), stages=("tracks",))
    writer.close()
    complete_recording(writer.path)
    pose = work / "variant_pose.csv"
    action = work / "variant_actions.csv"
    pose.write_text("current pose")
    action.write_text("current actions")
    (work / "pose.csv").write_text("stale pose")
    (work / "actions.csv").write_text("stale actions")
    _deploy_output_artifacts(
        output_video_path=None,
        output_video=None,
        results_folder=str(work),
        target_deploy_dir=str(game),
        game_id="game",
        telemetry_path=writer.path,
        supplementary_paths=(pose, action),
    )
    assert (game / "variant_pose-1.csv").read_text() == "current pose"
    assert (game / "variant_actions-1.csv").read_text() == "current actions"
    assert not (game / "pose-1.csv").exists()
    assert not (game / "actions-1.csv").exists()


def should_capture_real_instance_metadata_and_mask(tmp_path):
    from mmdet.structures import DetDataSample, TrackDataSample
    from mmengine.structures import InstanceData

    writer = TelemetryRecorder(tmp_path, "game", {}, {"detections", "tracks"})
    ctx = context((0,))
    sample = DetDataSample()
    detection = InstanceData()
    detection.bboxes = torch.tensor([[1.0, 2.0, 8.0, 12.0], [0.0, 0.0, 0.0, 0.0]])
    detection.scores = torch.tensor([0.9, 0.0])
    detection.labels = torch.tensor([0, -1])
    detection.set_metainfo({"num_valid_after_nms": torch.tensor(1)})
    tracks = InstanceData()
    tracks.bboxes = detection.bboxes.clone()
    tracks.scores = detection.scores.clone()
    tracks.labels = detection.labels.clone()
    tracks.instances_id = torch.tensor([99, -1])
    tracks.set_metainfo({"num_tracks": torch.tensor(1)})
    sample.pred_instances = detection
    sample.pred_track_instances = tracks
    sample.set_metainfo({"frame_id": 0})
    clip = TrackDataSample()
    clip.video_data_samples = [sample]
    ctx["data_samples"] = [clip]
    capture(writer, ctx, stages=("detections", "tracks"))
    writer.close()
    complete_recording(writer.path)
    with read_database(writer.path) as db:
        assert tuple(db.execute("SELECT detection_count,track_count FROM frames").fetchone()) == (
            1,
            1,
        )
        assert db.execute("SELECT tracking_id FROM tracks").fetchone()[0] == "99"
