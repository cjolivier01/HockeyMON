import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from mmengine.structures import InstanceData

from hmlib.aspen.plugins.base import Plugin

REPO_ROOT = Path(__file__).absolute().parents[1]


class _FixturePoseFactory(Plugin):
    """Provide deterministic pose inference through the real PosePlugin."""

    @staticmethod
    def _identity(value):
        return value

    @staticmethod
    def _infer(batch, *, merge_results, bbox_thr, pose_based_nms):
        from mmpose.structures import PoseDataSample

        predictions = []
        for item in batch:
            bbox = item["bbox"]
            center = (bbox[:, :2] + bbox[:, 2:]) / 2
            result = PoseDataSample(metainfo={"hm_frame_index": item["hm_frame_index"]})
            result.pred_instances = InstanceData(
                bboxes=bbox,
                bbox_scores=item["bbox_score"],
                keypoints=center[:, None, :].repeat(1, 17, 1),
                keypoint_scores=torch.ones((1, 17), device=bbox.device),
            )
            predictions.append(result)
        return predictions

    def forward(self, context):
        pose_impl = SimpleNamespace(
            cfg=SimpleNamespace(data_mode="topdown"),
            model=SimpleNamespace(dataset_meta={}),
            pipeline=self._identity,
            collate_fn=self._identity,
            forward=self._infer,
        )
        return {"pose_inferencer": SimpleNamespace(inferencer=pose_impl, filter_args={})}

    def output_keys(self):
        return {"pose_inferencer"}


def should_cli_load_tracking_smoke_runs(tmp_path):
    from mmdet.structures import DetDataSample

    from hmlib.tracking_utils.pose_dataframe import PoseDataFrame
    from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame

    game_id = "devtest"
    game_dir_base = tmp_path / "games"
    (game_dir_base / game_id).mkdir(parents=True)
    env = os.environ.copy()
    env["HM_GAME_DIR"] = str(game_dir_base)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "tests"), env.get("PYTHONPATH", "")]
    )

    tracking_csv = tmp_path / "tracking.csv"
    df = TrackingDataFrame(output_file=tracking_csv, input_batch_size=1)
    for frame in range(1, 6):
        inst = InstanceData(
            instances_id=torch.tensor([1], dtype=torch.long),
            bboxes=torch.tensor([[10.0 + frame, 20.0, 40.0, 60.0]], dtype=torch.float32),
            scores=torch.tensor([0.9], dtype=torch.float32),
            labels=torch.tensor([0], dtype=torch.long),
        )
        ds = DetDataSample()
        ds.pred_track_instances = inst
        df.add_frame_sample(frame, ds)
    df.flush()

    video = tmp_path / "black.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:r=5",
            "-t",
            "1",
            str(video),
        ],
        check=True,
        timeout=20,
    )

    # Keep the actual load-tracking -> pose -> save pipeline, with a small
    # inferencer fixture and unrelated video/rink processing explicitly disabled.
    source_config = REPO_ROOT / "hmlib/config/aspen/tracking_pose_load_tracking.yaml"
    config = yaml.safe_load(source_config.read_text())
    plugins = config["aspen"]["plugins"]
    plugins["pose_factory"] = {
        "class": "test_tracking_pose_load_tracking._FixturePoseFactory",
        "depends": [],
        "params": {},
    }
    for name in ("stitching", "ice_config", "ice_boundaries", "postprocess", "overlays"):
        plugins[name]["enabled"] = False
    plugins["pose"]["params"]["plot_pose"] = False
    config_file = tmp_path / "tracking_pose_fixture.yaml"
    config_file.write_text(yaml.safe_dump(config))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hmlib.cli.hmtrack",
            "-b=1",
            f"--input-video={video}",
            f"--game-id={game_id}",
            f"--config={config_file}",
            "--ignore-private-config=1",
            "--no-wide-start",
            "--no-crop",
            "--skip-final-video-save",
            "--no-play-tracking",
            f"--input-tracking-data={tracking_csv}",
            "-t",
            "0.5",
            "--no-save-video",
            "--save-pose-data",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    out_pose = tmp_path / "output_workdirs" / game_id / "pose.csv"
    pose_df = PoseDataFrame(input_file=out_pose, input_batch_size=1)
    assert pose_df.data["Frame"].tolist() == [1, 2]
    for frame in (1, 2):
        sample = pose_df.get_sample_by_frame(frame)
        assert sample is not None
        expected_bbox = torch.tensor([[10.0 + frame, 20.0, 40.0, 60.0]])
        expected_center = (expected_bbox[:, :2] + expected_bbox[:, 2:]) / 2
        torch.testing.assert_close(sample.pred_instances.bboxes, expected_bbox)
        torch.testing.assert_close(
            sample.pred_instances.keypoints, expected_center[:, None, :].repeat(1, 17, 1)
        )
        torch.testing.assert_close(sample.pred_instances.keypoint_scores, torch.ones((1, 17)))
