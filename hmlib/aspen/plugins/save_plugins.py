from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hmlib.camera.camera_dataframe import CameraPolicyDataFrame, CameraTrackingDataFrame
from hmlib.camera.camera_policy import POLICY_SCHEMA, camera_policy_path
from hmlib.telemetry.capture import CAPTURE_KEYS
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

    telemetry_kind = "detections"

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

        if "telemetry_batch" in context:
            context["telemetry_batch"].capture("detections", context)
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
        return CAPTURE_KEYS | {"data_samples", "frame_id"}

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

    telemetry_kind = "tracks"

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

        if "telemetry_batch" in context:
            context["telemetry_batch"].capture("tracks", context)
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
        return CAPTURE_KEYS | {"data_samples", "frame_id", "jersey_results", "action_results"}

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

    telemetry_kind = "cameras"

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
        self._camera_policy_dataframes: List[CameraPolicyDataFrame] = []
        self._last_policy_frame: Optional[int] = None
        self._last_camera_frame: Optional[int] = None

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

        if "telemetry_batch" in context:
            context["telemetry_batch"].capture("cameras", context)
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

        batches = [
            (dataframe, _to_tlbr_array(box))
            for dataframe, box in ((df, current_box), (fast_df, current_fast_box))
            if dataframe is not None and box is not None
        ]
        batch_sizes = {len(boxes) for _, boxes in batches}
        if len(batch_sizes) > 1:
            raise ValueError("Slow/fast camera CSV batches must contain the same frames")
        batch_size = next(iter(batch_sizes), 0)
        source_ids = context.get("frame_ids")
        if source_ids is None:
            frame_ids = list(range(frame_id0, frame_id0 + batch_size))
        else:
            source_ids = unwrap_tensor(source_ids)
            if isinstance(source_ids, torch.Tensor):
                source_ids = source_ids.detach().cpu().numpy()
            source_ids = np.asarray(source_ids)
            if source_ids.shape != (batch_size,) or source_ids.dtype.kind not in "iu":
                raise ValueError("Camera CSV frame_ids must be a one-dimensional integer batch")
            frame_ids = [int(frame) for frame in source_ids]
        if any(frame < 0 for frame in frame_ids):
            raise ValueError("Camera CSV requires nonnegative source frame IDs")
        if "camera_policy_events" in context and batch_size:
            self._save_policy_events(
                context["camera_policy_events"], frame_ids, [dataframe for dataframe, _ in batches]
            )
        elif self._camera_policy_dataframes and batch_size:
            raise ValueError("Camera policy provenance is missing after export started")

        for dataframe, tlbr in batches:
            for i in range(int(tlbr.shape[0])):
                dataframe.add_frame_records(frame_id=frame_ids[i], tlbr=tlbr[i : i + 1])

        out: Dict[str, Any] = {}
        if df is not None:
            out["camera_dataframe"] = df
        if fast_df is not None:
            out["camera_fast_dataframe"] = fast_df
        return out

    def _save_policy_events(
        self,
        events: List[Dict[str, Any]],
        frame_ids: List[int],
        cameras: List[CameraTrackingDataFrame],
    ) -> None:
        if not isinstance(events, list):
            raise ValueError("Camera policy events must be a list")
        if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])) or (
            self._last_camera_frame is not None and frame_ids[0] <= self._last_camera_frame
        ):
            raise ValueError("Camera policy export requires increasing source frame IDs")
        last = self._last_policy_frame
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Invalid camera policy event")
            frame = event.get("frame")
            if (
                isinstance(frame, bool)
                or not isinstance(frame, int)
                or frame not in frame_ids
                or event.get("schema") != POLICY_SCHEMA
                or event.get("kind") != ("startup" if last is None else "change")
                or not isinstance(event.get("policy"), dict)
                or (last is not None and frame <= last)
            ):
                raise ValueError("Camera policy event does not match its source frame batch")
            if last is None and frame != frame_ids[0]:
                raise ValueError("Camera policy startup must precede the first exported frame")
            last = frame
        if last is None:
            raise ValueError("Camera policy export is missing its startup event")
        if not self._camera_policy_dataframes:
            self._camera_policy_dataframes = [
                CameraPolicyDataFrame(output_file=str(camera_policy_path(camera.output_file)))
                for camera in cameras
            ]
        elif [dataframe.output_file for dataframe in self._camera_policy_dataframes] != [
            str(camera_policy_path(camera.output_file)) for camera in cameras
        ]:
            raise ValueError("Camera policy outputs changed during export")
        for dataframe in self._camera_policy_dataframes:
            dataframe.add_events(events)
        self._last_policy_frame = last
        self._last_camera_frame = frame_ids[-1]

    def input_keys(self):
        return CAPTURE_KEYS | {
            "frame_id",
            "frame_ids",
            "current_box",
            "current_fast_box_list",
            "camera_policy_events",
            "shared",
        }

    def output_keys(self):
        return {"camera_dataframe", "camera_fast_dataframe"}

    def finalize(self):
        actions = []
        if self._camera_dataframe is not None:
            actions.append(("camera CSV", self._camera_dataframe.close))
        if self._camera_fast_dataframe is not None:
            actions.append(("fast camera CSV", self._camera_fast_dataframe.close))
        for dataframe in self._camera_policy_dataframes:
            actions.append((f"camera policy CSV {dataframe.output_file}", dataframe.close))
        finalize_resources(actions)
