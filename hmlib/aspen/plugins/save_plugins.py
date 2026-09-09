from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hmlib.camera.camera_dataframe import CameraTrackingDataFrame
from hmlib.tracking_utils.action_dataframe import ActionDataFrame
from hmlib.tracking_utils.detection_dataframe import DetectionDataFrame
from hmlib.tracking_utils.pose_dataframe import PoseDataFrame
from hmlib.tracking_utils.tracking_dataframe import TrackingDataFrame
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.finalization import finalize_resources
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor
from hmlib.utils.path import add_prefix_to_filename

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


def _output_label(context: Dict[str, Any]) -> Optional[str]:
    label = _ctx_value(context, "output_label") or _ctx_value(context, "label")
    if label is None:
        return None
    label_str = str(label).strip()
    return label_str if label_str else None


def _apply_output_label(filename: str, context: Dict[str, Any]) -> str:
    label = _output_label(context)
    if not label:
        return filename
    try:
        return str(add_prefix_to_filename(filename, label))
    except Exception:
        return filename


def _apply_track_mask(inst, tids, tlbr, scores, labels):
    mask = get_track_mask(inst)
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
        if isinstance(tids, StreamTensorBase):
            tids = unwrap_tensor(tids)
        if isinstance(tids, torch.Tensor):
            tids = tids[mask.to(device=tids.device)]
        else:
            tids = np.asarray(tids)[mask_np]
        if isinstance(tlbr, StreamTensorBase):
            tlbr = unwrap_tensor(tlbr)
        if isinstance(tlbr, torch.Tensor):
            tlbr = tlbr[mask.to(device=tlbr.device)]
        else:
            tlbr = np.asarray(tlbr)[mask_np]
        if isinstance(scores, StreamTensorBase):
            scores = unwrap_tensor(scores)
        if isinstance(scores, torch.Tensor):
            scores = scores[mask.to(device=scores.device)]
        else:
            scores = np.asarray(scores)[mask_np]
        if isinstance(labels, StreamTensorBase):
            labels = unwrap_tensor(labels)
        if isinstance(labels, torch.Tensor):
            labels = labels[mask.to(device=labels.device)]
        else:
            labels = np.asarray(labels)[mask_np]
    return tids, tlbr, scores, labels


class SavePluginBase(Plugin):
    """Base class for save plugins with common utilities."""

    # Save plugins must run at their topological position because downstream
    # stages can mutate the shared data sample objects they persist.
    disable_in_cuda_graph_pipeline = False

    def is_output(self) -> bool:
        """If enabled, this node is an output."""
        return self.enabled


class SaveDetectionsPlugin(SavePluginBase):
    """
    Saves per-frame detections into `detection_dataframe`.

    Expects in context:
      - data: dict with 'data_samples' (TrackDataSample or [TrackDataSample])
      - frame_id: int for first frame in batch
      - detection_dataframe: DetectionDataFrame
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "detections.csv",
        write_interval: int = 1000,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._detection_dataframe: Optional[DetectionDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[DetectionDataFrame]:
        if self._detection_dataframe is not None:
            return self._detection_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, _apply_output_label(self._output_filename, context))
        self._detection_dataframe = DetectionDataFrame(
            output_file=output_path, write_interval=self._write_interval
        )
        return self._detection_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
        if df is None:
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
            img_data_sample = track_data_sample[i]
            inst = getattr(img_data_sample, "pred_instances", None)
            if inst is None:
                # No detections: still record an empty frame
                try:
                    df.add_frame_sample(frame_id=int(frame_id0 + i), data_sample=img_data_sample)
                except Exception:
                    df.add_frame_records(
                        frame_id=int(frame_id0 + i),
                        scores=np.empty((0,), dtype=np.float32),
                        labels=np.empty((0,), dtype=np.int64),
                        bboxes=np.empty((0, 4), dtype=np.float32),
                    )
                continue
            # Determine frame id
            fid = img_data_sample.metainfo.get("frame_id", None)
            try:
                if isinstance(fid, torch.Tensor):
                    fid = int(fid.reshape([1])[0].item())
            except Exception:
                fid = None
            if fid is None:
                fid = frame_id0 + i

            try:
                df.add_frame_sample(frame_id=int(fid), data_sample=img_data_sample)
            except Exception:
                df.add_frame_records(
                    frame_id=int(fid),
                    scores=getattr(inst, "scores", np.empty((0,), dtype=np.float32)),
                    labels=getattr(inst, "labels", np.empty((0,), dtype=np.int64)),
                    bboxes=getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32)),
                )

        return {"detection_dataframe": df}

    def input_keys(self):
        return {"data_samples", "frame_id"}

    def output_keys(self):
        return {"detection_dataframe"}

    def finalize(self):
        if self._detection_dataframe is not None:
            self._detection_dataframe.close()


class SaveTrackingPlugin(SavePluginBase):
    """
    Saves per-frame tracking results into `tracking_dataframe`.

    Expects in context:
      - data_samples: TrackDataSample (or list)
      - frame_id: int for first frame in batch
      - tracking_dataframe: TrackingDataFrame
      - jersey_results: Optional per-frame jersey info list
      - action_results: Optional per-frame action result list (from ActionFromPosePlugin)
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "tracking.csv",
        write_interval: int = 1000,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._tracking_dataframe: Optional[TrackingDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[TrackingDataFrame]:
        if self._tracking_dataframe is not None:
            return self._tracking_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, _apply_output_label(self._output_filename, context))
        self._tracking_dataframe = TrackingDataFrame(
            output_file=output_path,
            input_batch_size=1,
            write_interval=self._write_interval,
        )
        return self._tracking_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
        if df is None:
            return {}

        jersey_results_all = context.get("jersey_results")
        action_results_all = context.get("action_results")
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

        for i in range(video_len):
            img_data_sample = track_data_sample[i]
            inst = getattr(img_data_sample, "pred_track_instances", None)
            if inst is None:
                # No tracks: still record an empty frame
                try:
                    df.add_frame_sample(
                        frame_id=frame_id0 + i,
                        data_sample=img_data_sample,
                        jersey_info=None,
                        action_info=None,
                    )
                except Exception:
                    df.add_frame_records(
                        frame_id=frame_id0 + i,
                        tracking_ids=np.empty((0,), dtype=np.int64),
                        tlbr=np.empty((0, 4), dtype=np.float32),
                        scores=np.empty((0,), dtype=np.float32),
                        labels=np.empty((0,), dtype=np.int64),
                        jersey_info=None,
                    )
                continue
            jersey_results = (
                jersey_results_all[i]
                if isinstance(jersey_results_all, list) and i < len(jersey_results_all)
                else None
            )
            action_results = (
                action_results_all[i]
                if isinstance(action_results_all, list) and i < len(action_results_all)
                else None
            )

            try:
                df.add_frame_sample(
                    frame_id=frame_id0 + i,
                    data_sample=img_data_sample,
                    jersey_info=jersey_results,
                    action_info=action_results,
                )
            except Exception:
                tids = getattr(inst, "instances_id", np.empty((0,), dtype=np.int64))
                tlbr = getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32))
                scores = getattr(inst, "scores", np.empty((0,), dtype=np.float32))
                labels = getattr(inst, "labels", np.empty((0,), dtype=np.int64))
                tids, tlbr, scores, labels = _apply_track_mask(inst, tids, tlbr, scores, labels)
                df.add_frame_records(
                    frame_id=frame_id0 + i,
                    tracking_ids=tids,
                    tlbr=tlbr,
                    scores=scores,
                    labels=labels,
                    jersey_info=jersey_results,
                    action_info=action_results,
                )
        return {"tracking_dataframe": df}

    def input_keys(self):
        return {"data_samples", "frame_id", "jersey_results", "action_results"}

    def output_keys(self):
        return {"tracking_dataframe"}

    def finalize(self):
        if self._tracking_dataframe is not None:
            self._tracking_dataframe.close()


class SavePosePlugin(SavePluginBase):
    """
    Saves per-frame pose results from `data['pose_results']` into `pose_dataframe`.

    We serialize a simplified structure capturing keypoints/bboxes/scores to JSON.

    Expects in context:
      - pose_results: list of pose results
      - data_samples: used only to determine clip length when pose_results missing
      - frame_id: int for first frame in batch
      - pose_dataframe: PoseDataFrame
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "pose.csv",
        write_interval: int = 1000,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._pose_dataframe: Optional[PoseDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[PoseDataFrame]:
        if self._pose_dataframe is not None:
            return self._pose_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, _apply_output_label(self._output_filename, context))
        Path(output_path).touch(exist_ok=True)
        self._pose_dataframe = PoseDataFrame(
            output_file=output_path, write_interval=self._write_interval
        )
        return self._pose_dataframe

    @staticmethod
    def _to_list(x):
        try:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass
        return x

    @classmethod
    def _simplify_pose_item(cls, pose_result_item: Any) -> Dict[str, Any]:
        preds = None
        try:
            preds = pose_result_item.get("predictions")
        except Exception:
            preds = None
        out_preds: List[Dict[str, Any]] = []
        if isinstance(preds, list):
            for ds in preds:
                inst = getattr(ds, "pred_instances", None)
                item: Dict[str, Any] = {}
                if inst is not None:
                    for k in (
                        "bboxes",
                        "scores",
                        "bbox_scores",
                        "labels",
                        "keypoints",
                        "keypoint_scores",
                    ):
                        if hasattr(inst, k):
                            item[k] = cls._to_list(getattr(inst, k))
                out_preds.append(item)
        return {"predictions": out_preds}

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        df = self._ensure_dataframe(context)
        if df is None:
            return {}

        pose_results: Optional[List[Any]] = context.get("pose_results")

        frame_id0: int = int(context.get("frame_id", -1))
        # If pose_results is missing, write empty entries for each frame
        if not pose_results:
            track_samples = context.get("data_samples")
            if isinstance(track_samples, list):
                track_data_sample = track_samples[0]
            else:
                track_data_sample = track_samples
            video_len = len(track_data_sample) if track_data_sample is not None else 0
            for i in range(video_len):
                df.add_frame_records(
                    frame_id=frame_id0 + i, pose_json=json.dumps({"predictions": []})
                )
        else:
            for i, item in enumerate(pose_results):
                # Prefer direct PoseDataSample storage
                try:
                    df.add_frame_sample(frame_id=frame_id0 + i, pose_item=item)
                except Exception:
                    simp = self._simplify_pose_item(item)
                    df.add_frame_records(frame_id=frame_id0 + i, pose_json=json.dumps(simp))
        return {"pose_dataframe": df}

    def input_keys(self):
        return {"pose_results", "data_samples", "frame_id"}

    def output_keys(self):
        return {"pose_dataframe"}

    def finalize(self):
        if self._pose_dataframe is not None:
            self._pose_dataframe.close()


class SaveActionsPlugin(SavePluginBase):
    """
    Saves per-frame action results. By default, writes into the `tracking_dataframe`
    action columns, if a TrackingDataFrame is provided in context. This trunk is
    optional since SaveTrackingPlugin already persists action results when placed
    after the `actions` trunk; include this only if you need a dedicated action
    saving pass.

    Expects in context:
      - action_results: per frame
      - data_samples: TrackDataSample (or list)
      - frame_id: int for first frame in batch
      - tracking_dataframe: TrackingDataFrame (will update action columns)
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "actions.csv",
        write_interval: int = 1000,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._write_interval = write_interval
        self._action_dataframe: Optional[ActionDataFrame] = None

    def _ensure_action_dataframe(self, context: Dict[str, Any]) -> Optional[ActionDataFrame]:
        if self._action_dataframe is not None:
            return self._action_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, _apply_output_label(self._output_filename, context))
        self._action_dataframe = ActionDataFrame(
            output_file=output_path, write_interval=self._write_interval
        )
        return self._action_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = context.get("tracking_dataframe")
        action_df = context.get("action_dataframe") or self._ensure_action_dataframe(context)
        if df is None and action_df is None:
            return {}
        action_results_all = context.get("action_results")
        if not action_results_all:
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
        for i in range(video_len):
            img_data_sample = track_data_sample[i]
            inst = getattr(img_data_sample, "pred_track_instances", None)
            if inst is None:
                continue
            actions = action_results_all[i] if i < len(action_results_all) else None
            # Update tracking with action columns if tracking df present
            if df is not None:
                tids = getattr(inst, "instances_id", np.empty((0,), dtype=np.int64))
                tlbr = getattr(inst, "bboxes", np.empty((0, 4), dtype=np.float32))
                scores = getattr(inst, "scores", np.empty((0,), dtype=np.float32))
                labels = getattr(inst, "labels", np.empty((0,), dtype=np.int64))
                tids, tlbr, scores, labels = _apply_track_mask(inst, tids, tlbr, scores, labels)
                df.add_frame_records(
                    frame_id=frame_id0 + i,
                    tracking_ids=tids,
                    tlbr=tlbr,
                    scores=scores,
                    labels=labels,
                    jersey_info=None,
                    action_info=actions,
                )
            # Optionally write dedicated action dataframe
            if action_df is not None and actions is not None:
                try:
                    # Build list of ActionDataSample-like dicts (tracking_id, label_index/label, score)
                    action_df.add_frame_sample(frame_id=frame_id0 + i, data_samples=actions)
                except Exception:
                    action_df.add_frame_records(
                        frame_id=frame_id0 + i, action_json=json.dumps(actions)
                    )
        return {"action_dataframe": action_df} if action_df is not None else {}

    def input_keys(self):
        return {"data_samples", "frame_id", "tracking_dataframe", "action_results"}

    def output_keys(self):
        return {"action_dataframe"}

    def finalize(self):
        if self._action_dataframe is not None:
            self._action_dataframe.close()


class SaveCameraPlugin(SavePluginBase):
    """
    Saves per-frame camera boxes into `camera_dataframe`.

    Expects in context:
      - frame_id: int first frame in batch
      - current_box: TLBR camera box tensor or array
      - work_dir: output directory for camera.csv
    """

    def __init__(
        self,
        enabled: bool = True,
        work_dir_key: str = "work_dir",
        output_filename: str = "camera.csv",
        fast_output_filename: str = "camera_fast.csv",
        save_fast: bool = True,
        write_interval: int = 1000,
    ):
        super().__init__(enabled=enabled)
        self._work_dir_key = work_dir_key
        self._output_filename = output_filename
        self._fast_output_filename = fast_output_filename
        self._save_fast = bool(save_fast)
        self._write_interval = write_interval
        self._camera_dataframe: Optional[CameraTrackingDataFrame] = None
        self._camera_fast_dataframe: Optional[CameraTrackingDataFrame] = None

    def _ensure_dataframe(self, context: Dict[str, Any]) -> Optional[CameraTrackingDataFrame]:
        if self._camera_dataframe is not None:
            return self._camera_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(work_dir, _apply_output_label(self._output_filename, context))
        self._camera_dataframe = CameraTrackingDataFrame(
            output_file=output_path,
            input_batch_size=1,
        )
        self._camera_dataframe.write_interval = self._write_interval
        return self._camera_dataframe

    def _ensure_fast_dataframe(self, context: Dict[str, Any]) -> Optional[CameraTrackingDataFrame]:
        if not self._save_fast:
            return None
        if self._camera_fast_dataframe is not None:
            return self._camera_fast_dataframe
        work_dir = _ctx_value(context, self._work_dir_key)
        if not work_dir:
            return None
        os.makedirs(work_dir, exist_ok=True)
        output_path = os.path.join(
            work_dir, _apply_output_label(self._fast_output_filename, context)
        )
        self._camera_fast_dataframe = CameraTrackingDataFrame(
            output_file=output_path,
            input_batch_size=1,
        )
        self._camera_fast_dataframe.write_interval = self._write_interval
        return self._camera_fast_dataframe

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        df = self._ensure_dataframe(context)
        fast_df = self._ensure_fast_dataframe(context)
        if df is None and fast_df is None:
            return {}

        frame_id0 = int(context.get("frame_id", -1))
        current_box = context.get("current_box")
        current_fast_box = context.get("current_fast_box_list")

        def _to_tlbr_array(box_obj: Any) -> Optional[np.ndarray]:
            if box_obj is None:
                return None
            box_obj = unwrap_tensor(box_obj)
            if isinstance(box_obj, torch.Tensor):
                arr = box_obj.detach().cpu().numpy()
            else:
                arr = np.asarray(box_obj)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError(f"Camera CSV requires Nx4 boxes; received shape {arr.shape}")
            return arr.astype(np.float32, copy=False)

        for dataframe, box in ((df, current_box), (fast_df, current_fast_box)):
            if dataframe is None or box is None:
                continue
            tlbr = _to_tlbr_array(box)
            if frame_id0 < 0:
                raise ValueError("Camera CSV requires a nonnegative frame_id")
            for i in range(int(tlbr.shape[0])):
                dataframe.add_frame_records(frame_id=frame_id0 + i, tlbr=tlbr[i : i + 1])

        out: Dict[str, Any] = {}
        if df is not None:
            out["camera_dataframe"] = df
        if fast_df is not None:
            out["camera_fast_dataframe"] = fast_df
        return out

    def input_keys(self):
        return {"frame_id", "current_box", "current_fast_box_list", "shared"}

    def output_keys(self):
        return {"camera_dataframe", "camera_fast_dataframe"}

    def finalize(self):
        actions = []
        if self._camera_dataframe is not None:
            actions.append(("camera CSV", self._camera_dataframe.close))
        if self._camera_fast_dataframe is not None:
            actions.append(("fast camera CSV", self._camera_fast_dataframe.close))
        finalize_resources(actions)
