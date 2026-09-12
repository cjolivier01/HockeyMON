from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from mmengine.structures import InstanceData

from hmlib.bbox.box_functions import tlwh_to_tlbr_multiple
from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.datasets.dataframe import find_latest_dataframe_file
from hmlib.log import logger
from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
from hmlib.tracking_utils.pose_dataframe import PoseDataFrame
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.tracking_utils.utils import get_track_mask

from .base import Plugin


def _ctx_value(context: Dict[str, Any], key: str) -> Optional[Any]:
    if not key:
        return None
    if key in context:
        return context[key]
    shared = context.get("shared")
    if isinstance(shared, dict):
        return shared.get(key)
    return None


class LoadDetectionsPlugin(Plugin):
    """
    Loads detections from `detection_dataframe` and attaches them to data_samples
    as `pred_instances`, emulating DetectorInferencePlugin.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - frame_id: first frame id in batch
      - detection_dataframe: DetectionDataFrame (input_file provided)
    """

    def __init__(
        self,
        enabled: bool = True,
        input_path_key: str = "detection_data_path",
        game_dir_key: str = "game_dir",
        file_stem: str = "detections",
        input_batch_size: int = 1,
    ):
        super().__init__(enabled=enabled)
        self._input_path_key = input_path_key
        self._game_dir_key = game_dir_key
        self._file_stem = file_stem
        self._input_batch_size = input_batch_size
        self._detection_dataframe: Optional[DetectionDataFrame] = None
        self._warned_missing = False

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[DetectionDataFrame]:
        if self._detection_dataframe is not None:
            return self._detection_dataframe
        path = _ctx_value(context, self._input_path_key)
        if not path:
            path = find_latest_dataframe_file(
                _ctx_value(context, self._game_dir_key), self._file_stem
            )
        if not path or not os.path.exists(path):
            if not self._warned_missing:
                logger.info("LoadDetectionsPlugin: no %s CSV found; skipping load", self._file_stem)
                self._warned_missing = True
            return None
        self._detection_dataframe = DetectionDataFrame(
            input_file=path,
            input_batch_size=self._input_batch_size,
            write_interval=100,
        )
        return self._detection_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        df = self._ensure_dataframe(context)
        if df is None or not df.has_input_data():
            return {}

        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        frame_id0: int = int(context.get("frame_id", -1))
        video_len = len(track_data_sample)
        for i in range(video_len):
            fid: int = frame_id0 + i
            img_data_sample = track_data_sample[i]
            ds = getattr(df, "get_sample_by_frame", None)
            det_ds = ds(frame_id=fid) if callable(ds) else None
            if det_ds is None:
                # Attach empty detections to mirror detector behavior
                inst = InstanceData()
                inst.scores = torch.empty((0,), dtype=torch.float32)
                inst.labels = torch.empty((0,), dtype=torch.long)
                inst.bboxes = torch.empty((0, 4), dtype=torch.float32)
                img_data_sample.pred_instances = inst
            else:
                inst = getattr(det_ds, "pred_instances", None)
                if inst is None:
                    # fall back to dict-based path
                    rec = df.get_data_dict_by_frame(frame_id=fid)
                    if rec is None:
                        inst = InstanceData()
                        inst.scores = torch.empty((0,), dtype=torch.float32)
                        inst.labels = torch.empty((0,), dtype=torch.long)
                        inst.bboxes = torch.empty((0, 4), dtype=torch.float32)
                    else:
                        inst = InstanceData()
                        inst.scores = torch.as_tensor(
                            rec.get("scores", np.empty((0,), dtype=np.float32))
                        )
                        inst.labels = torch.as_tensor(
                            rec.get("labels", np.empty((0,), dtype=np.int64))
                        )
                        inst.bboxes = torch.as_tensor(
                            rec.get("bboxes", np.empty((0, 4), dtype=np.float32))
                        )
                img_data_sample.pred_instances = inst
            img_data_sample.set_metainfo({"frame_id": int(fid)})
        return {"detection_dataframe": df}

    def input_keys(self):
        return {"data_samples", "frame_id"}

    def output_keys(self):
        return {"detection_dataframe"}


class LoadTrackingPlugin(Plugin):
    """
    Loads tracks from `tracking_dataframe` and attaches `pred_track_instances`.

    Produces `data`, `nr_tracks`, and `max_tracking_id` analogous to TrackerPlugin.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - frame_id: first frame in batch
      - tracking_dataframe: TrackingDataFrame (input_file provided)
    """

    def __init__(
        self,
        enabled: bool = True,
        input_path_key: str = "tracking_data_path",
        game_dir_key: str = "game_dir",
        file_stem: str = "tracking",
        input_batch_size: int = 1,
    ):
        super().__init__(enabled=enabled)
        self._input_path_key = input_path_key
        self._game_dir_key = game_dir_key
        self._file_stem = file_stem
        self._input_batch_size = input_batch_size
        self._tracking_dataframe: Optional[TrackingDataFrame] = None
        self._warned_missing = False

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[TrackingDataFrame]:
        if self._tracking_dataframe is not None:
            return self._tracking_dataframe
        path = _ctx_value(context, self._input_path_key)
        if not path:
            path = find_latest_dataframe_file(
                _ctx_value(context, self._game_dir_key), self._file_stem
            )
        if not path or not os.path.exists(path):
            if not self._warned_missing:
                logger.info("LoadTrackingPlugin: no %s CSV found; skipping load", self._file_stem)
                self._warned_missing = True
            return None
        dataframe_class = TrackingDataFrame
        if str(path).endswith((".db", ".sqlite")):
            from hmlib.telemetry.tracking import DatabaseTrackingDataFrame

            dataframe_class = DatabaseTrackingDataFrame
        self._tracking_dataframe = dataframe_class(
            input_file=path,
            input_batch_size=self._input_batch_size,
            write_interval=100,
        )
        return self._tracking_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
        if df is None or not df.has_input_data():
            return {}

        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples

        frame_id0: int = int(context.get("frame_id", -1))
        video_len = len(track_data_sample)
        max_tracking_id = 0
        active_track_count = 0
        jersey_results_all: List[List[Any]] = []

        for i in range(video_len):
            img_data_sample = track_data_sample[i]
            ds = getattr(df, "get_sample_by_frame", None)
            source_ids = context.get("frame_ids", context.get("ids"))
            fid: int = int(source_ids[i]) if source_ids is not None else frame_id0 + i
            img_data_sample.set_metainfo({"frame_id": fid})
            rec = None
            rec = df.get_data_dict_by_frame(frame_id=fid)
            jersey_results: List[Any] = []
            if isinstance(rec, dict):
                for info in rec.get("jersey_info") or []:
                    if info is None:
                        continue
                    if getattr(info, "tracking_id", -1) < 0:
                        continue
                    jersey_results.append(info)
            jersey_results_all.append(jersey_results)
            track_ds = ds(frame_id=fid) if callable(ds) else None
            if track_ds is None:
                # attach empty tracks instance
                inst = InstanceData(
                    instances_id=torch.empty((0,), dtype=torch.long),
                    bboxes=torch.empty((0, 4), dtype=torch.float32),
                    scores=torch.empty((0,), dtype=torch.float32),
                    labels=torch.empty((0,), dtype=torch.long),
                )
                img_data_sample.pred_track_instances = inst
                try:
                    img_data_sample.set_metainfo({"nr_tracks": 0})
                except Exception:
                    pass
                continue
            inst = getattr(
                track_ds[0] if hasattr(track_ds, "__getitem__") else track_ds,
                "pred_track_instances",
                None,
            )
            if inst is None:
                # Fallback to dict-based reconstruction
                rec = df.get_data_dict_by_frame(frame_id=fid)
                if rec is None:
                    inst = InstanceData(
                        instances_id=torch.empty((0,), dtype=torch.long),
                        bboxes=torch.empty((0, 4), dtype=torch.float32),
                        scores=torch.empty((0,), dtype=torch.float32),
                        labels=torch.empty((0,), dtype=torch.long),
                    )
                else:
                    tlwh = rec.get("bboxes", np.empty((0, 4), dtype=np.float32))
                    if isinstance(tlwh, np.ndarray):
                        tlwh_t = torch.as_tensor(tlwh)
                        bboxes = tlwh_to_tlbr_multiple(tlwh_t)
                    else:
                        bboxes = tlwh_to_tlbr_multiple(tlwh)
                    ids = rec.get("tracking_ids", np.empty((0,), dtype=np.int64))
                    scores = rec.get("scores", np.empty((0,), dtype=np.float32))
                    labels = rec.get("labels", np.empty((0,), dtype=np.int64))
                    inst = InstanceData(
                        instances_id=torch.as_tensor(ids),
                        bboxes=bboxes,
                        scores=torch.as_tensor(scores),
                        labels=torch.as_tensor(labels),
                    )
            img_data_sample.pred_track_instances = inst
            try:
                track_mask = get_track_mask(inst)
                if isinstance(track_mask, torch.Tensor):
                    track_count = int(track_mask.sum().item())
                else:
                    track_count = len(inst.instances_id)
                img_data_sample.set_metainfo({"nr_tracks": track_count})
                img_data_sample.set_metainfo({"frame_id": int(fid)})
            except Exception:
                pass

            track_mask = get_track_mask(inst)
            if isinstance(track_mask, torch.Tensor):
                track_count = int(track_mask.sum().item())
                active_track_count = max(active_track_count, track_count)
                if track_count:
                    ids = inst.instances_id
                    if isinstance(ids, torch.Tensor):
                        ids = ids[track_mask]
                        max_id = int(torch.max(ids).item()) if ids.numel() else 0
                    else:
                        max_id = int(np.max(np.asarray(ids))) if len(ids) else 0
                    max_tracking_id = max(max_tracking_id, max_id)
            else:
                active_track_count = max(active_track_count, len(inst.instances_id))
                if len(inst.instances_id):
                    max_id = int(torch.max(inst.instances_id))
                    max_tracking_id = max(max_tracking_id, max_id)

        out: Dict[str, Any] = {
            "nr_tracks": active_track_count,
            "max_tracking_id": max_tracking_id,
            "tracking_dataframe": df,
        }
        if jersey_results_all:
            out["jersey_results"] = jersey_results_all
        return out

    def input_keys(self):
        return {"data_samples", "frame_id", "frame_ids", "ids", "shared"}

    def output_keys(self):
        return {"jersey_results", "nr_tracks", "max_tracking_id", "tracking_dataframe"}


class LoadPosePlugin(Plugin):
    """
    Loads per-frame pose JSON from `pose_dataframe` and exposes `pose_results`.

    The stored format is a simplified structure, but PoseToDetPlugin tolerates dict predictions
    if extended accordingly. Downstream postprocess uses pose results only if required.

    Expects in context:
      - pose_dataframe: PoseDataFrame (input_file provided)
      - data_samples: TrackDataSample (or list) to infer video_len
      - frame_id: int first frame
    """

    def __init__(
        self,
        enabled: bool = True,
        input_path_key: str = "pose_data_path",
        game_dir_key: str = "game_dir",
        file_stem: str = "pose",
    ):
        super().__init__(enabled=enabled)
        self._input_path_key = input_path_key
        self._game_dir_key = game_dir_key
        self._file_stem = file_stem
        self._pose_dataframe: Optional[PoseDataFrame] = None
        self._warned_missing = False

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[PoseDataFrame]:
        if self._pose_dataframe is not None:
            return self._pose_dataframe
        path = _ctx_value(context, self._input_path_key)
        if not path:
            path = find_latest_dataframe_file(
                _ctx_value(context, self._game_dir_key), self._file_stem
            )
        if not path or not os.path.exists(path):
            if not self._warned_missing:
                logger.info("LoadPosePlugin: no %s CSV found; skipping load", self._file_stem)
                self._warned_missing = True
            return None
        self._pose_dataframe = PoseDataFrame(input_file=path, write_interval=100)
        return self._pose_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        df = self._ensure_dataframe(context)
        if df is None or not df.has_input_data():
            return {}

        frame_id0: int = int(context.get("frame_id", -1))
        track_samples = context.get("data_samples")
        if track_samples is None:
            return {}
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        pose_results: List[Any] = []
        for i in range(video_len):
            get_s = getattr(df, "get_sample_by_frame", None)
            fid: int = frame_id0 + i
            pose_ds = get_s(frame_id=fid) if callable(get_s) else None
            if pose_ds is None:
                rec = df.get_data_dict_by_frame(frame_id=fid)
                if rec is None:
                    pose_results.append({"predictions": []})
                else:
                    pose_results.append(rec.get("pose", {"predictions": []}))
            else:
                pose_ds.set_metainfo({"frame_id": int(fid)})
                # Wrap PoseDataSample to match downstream expectations: {'predictions':[PoseDataSample]}
                pose_results.append({"predictions": [pose_ds]})

        return {"pose_results": pose_results, "pose_dataframe": df}

    def input_keys(self):
        return {"data_samples", "frame_id"}

    def output_keys(self):
        return {"pose_results", "pose_dataframe"}


class LoadCameraPlugin(Plugin):
    """
    Loads per-frame camera boxes from `camera_dataframe` and exposes them in context.

    Expects in context:
      - frame_id: int first frame in batch
      - camera_data_path (optional): explicit CSV path; otherwise inferred from game_dir
      - game_dir (optional): used with latest camera.csv when camera_data_path missing
    """

    def __init__(
        self,
        enabled: bool = True,
        input_path_key: str = "camera_data_path",
        game_dir_key: str = "game_dir",
        file_stem: str = "camera",
        input_batch_size: int = 1,
    ):
        super().__init__(enabled=enabled)
        self._input_path_key = input_path_key
        self._game_dir_key = game_dir_key
        self._file_stem = file_stem
        self._input_batch_size = input_batch_size
        self._camera_dataframe: Optional[CameraTrackingDataFrame] = None
        self._warned_missing = False

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[CameraTrackingDataFrame]:
        if self._camera_dataframe is not None:
            return self._camera_dataframe
        path = _ctx_value(context, self._input_path_key)
        if not path:
            path = find_latest_dataframe_file(
                _ctx_value(context, self._game_dir_key), self._file_stem
            )
        if not path or not os.path.exists(path):
            if not self._warned_missing:
                logger.info("LoadCameraPlugin: no %s CSV found; skipping load", self._file_stem)
                self._warned_missing = True
            return None
        self._camera_dataframe = CameraTrackingDataFrame(
            input_file=path,
            input_batch_size=self._input_batch_size,
        )
        return self._camera_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
        if df is None or not df.has_input_data():
            return {}

        frame_id0 = int(context.get("frame_id", -1))
        cam_rec = df.get_data_dict_by_frame(frame_id=frame_id0) if frame_id0 >= 0 else None
        if cam_rec is None:
            return {"camera_dataframe": df}

        return {
            "camera_dataframe": df,
            "camera_frame_id": cam_rec.get("frame_id"),
            "camera_bboxes": cam_rec.get("bboxes"),
        }

    def input_keys(self):
        return {"frame_id"}

    def output_keys(self):
        return {"camera_dataframe", "camera_frame_id", "camera_bboxes"}
