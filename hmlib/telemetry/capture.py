"""Snapshot observation metadata at Aspen's existing save positions."""

from __future__ import annotations

import dataclasses
import json
import threading

import numpy as np
import torch

from hmlib.utils.gpu import unwrap_tensor

CAPTURE_KEYS = {
    "telemetry_batch",
    "original_images",
    "rink_profile",
    "camera_input_geometry",
    "frame_ids",
    "ids",
    "frame_id",
    "fps",
    "pts_ns",
    "data_samples",
    "shared",
}


def array(value):
    value = unwrap_tensor(value)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _json_value(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported telemetry attribute: {type(value).__name__}")


def _json_copy(value):
    return json.loads(json.dumps(value, default=_json_value, allow_nan=False))


def _integer_ids(value, length):
    ids = array(value)
    if ids.shape != (length,) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("Telemetry requires a nonnegative integer source frame batch")
    return [int(i) for i in ids]


def _boxes(value):
    boxes = array(value)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Telemetry requires finite Nx4 TLBR boxes")
    boxes = boxes.astype(np.float64)
    boxes[:, 2:] -= boxes[:, :2]
    if np.any(boxes[:, 2:] <= 0):
        raise ValueError("Telemetry boxes must have positive width and height")
    return boxes.tolist()


def _observations(sample, kind, context, index):
    field = "pred_instances" if kind == "detections" else "pred_track_instances"
    instances = (
        sample.get(field, None) if isinstance(sample, dict) else getattr(sample, field, None)
    )
    if instances is None:
        return []
    bboxes = array(instances.bboxes)
    scores = array(instances.scores).reshape(-1)
    labels = array(instances.labels).reshape(-1)
    count = len(bboxes)
    if len(scores) != count or len(labels) != count or labels.dtype.kind not in "iu":
        raise ValueError("Telemetry observation arrays disagree")
    metadata = sample.get("metainfo", {}) if isinstance(sample, dict) else sample.metainfo
    keep = np.ones(count, dtype=bool)
    count_keys = (
        ("num_detections", "num_valid_after_nms", "num_valid")
        if kind == "detections"
        else ("num_tracks",)
    )
    for key in count_keys:
        valid = getattr(instances, key, metadata.get(key))
        if valid is not None:
            valid_array = array(valid)
            if valid_array.size != 1:
                raise ValueError("Invalid telemetry observation count")
            valid_count = int(valid_array.item())
            if not 0 <= valid_count <= count:
                raise ValueError("Telemetry observation count exceeds allocation")
            keep &= np.arange(count) < valid_count
            break
    if kind == "tracks":
        track_mask = getattr(instances, "track_mask", None)
        if track_mask is not None:
            track_mask = array(track_mask)
            if track_mask.shape != (count,) or track_mask.dtype != np.bool_:
                raise ValueError("Invalid telemetry track mask")
            keep &= track_mask
        ids = array(instances.instances_id).reshape(-1)
        if len(ids) != count or ids.dtype.kind not in "iu":
            raise ValueError("Telemetry track IDs must be integers")
        ids = ids[keep]
        if np.any(ids < 0):
            raise ValueError("Telemetry track IDs must be nonnegative")
    boxes = _boxes(bboxes[keep])
    scores, labels = scores[keep], labels[keep]
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite telemetry scores")
    attributes = {}
    if kind == "tracks":
        for key in ("jersey_results", "action_results"):
            batches = context.get(key)
            if batches is not None and index < len(batches) and batches[index] is not None:
                for entry in batches[index]:
                    entry = _json_copy(entry)
                    if isinstance(entry, dict) and "tracking_id" in entry:
                        attributes.setdefault(str(entry["tracking_id"]), {})[key] = entry
    rows = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        row = (*box, float(score), int(label))
        if kind == "tracks":
            track_id = str(int(ids[i]))
            row = (track_id, *row, json.dumps(attributes.get(track_id, {}), allow_nan=False))
        rows.append(row)
    return rows


class TelemetryBatch:
    def __init__(self, recorder, index):
        self.recorder = recorder
        self.index = index
        self.stages = {}
        self._lock = threading.Lock()

    def capture(self, kind, context):
        self.recorder.check()
        samples = context.get("data_samples")
        if isinstance(samples, list):
            if len(samples) != 1:
                raise ValueError("Telemetry expects one video sample batch")
            samples = samples[0]
        boxes = {}
        if kind == "cameras":
            for role, key in (("program", "current_box"), ("fast", "current_fast_box_list")):
                if context.get(key) is not None:
                    boxes[role] = _boxes(context[key])
            length = len(next(iter(boxes.values()))) if boxes else len(samples or [])
            if any(len(value) != length for value in boxes.values()):
                raise ValueError("Camera telemetry batches disagree")
        else:
            length = len(samples) if samples is not None else 0
        if length <= 0:
            raise ValueError("Telemetry capture stage has no source frames")
        source_ids = context.get("frame_ids", context.get("ids"))
        if source_ids is not None:
            ids = _integer_ids(source_ids, length)
        elif samples is not None:
            ids = []
            for i, sample in enumerate(samples):
                metadata = (
                    sample.get("metainfo", {}) if isinstance(sample, dict) else sample.metainfo
                )
                ids.append(
                    metadata.get("frame_id", metadata.get("img_id", context["frame_id"] + i))
                )
            ids = _integer_ids(ids, length)
        else:
            ids = _integer_ids(np.arange(context["frame_id"], context["frame_id"] + length), length)
        images = unwrap_tensor(context["original_images"])
        if images.ndim not in (3, 4) or images.shape[-3] not in (1, 3, 4):
            raise ValueError("Telemetry expects CHW/BCHW original images")
        width, height = int(images.shape[-1]), int(images.shape[-2])
        if not 0 < width <= 32768 or not 0 < height <= 32768:
            raise ValueError("Invalid telemetry canvas dimensions")
        profile = context.get("rink_profile") or {}
        mask = profile.get("combined_mask")
        if mask is not None:
            mask = unwrap_tensor(mask)
            if torch.is_tensor(mask):
                if mask.device.type != "cpu":
                    raise ValueError("Telemetry requires the existing CPU calibration mask")
                mask = mask.detach().numpy()
            if mask.shape != (height, width):
                raise ValueError("Rink mask dimensions differ from the telemetry canvas")
        revision = json.dumps(
            {
                "rink": profile.get("geometry_revision", "calibration"),
                "stitching": context.get("camera_input_geometry", {}),
            },
            sort_keys=True,
            allow_nan=False,
        )
        if context.get("pts_ns") is not None:
            pts = _integer_ids(context["pts_ns"], length)
        else:
            fps = float(context.get("fps") or context.get("shared", {}).get("fps") or 0)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("Telemetry needs source PTS or a positive source FPS")
            pts = [round(frame * 1_000_000_000 / fps) for frame in ids]
        stage = {"ids": ids, "pts": pts, "geometry": (width, height, revision, mask)}
        if kind == "cameras":
            stage["boxes"] = boxes
            stage["events"] = _json_copy(context.get("camera_policy_events", []))
        else:
            stage["rows"] = [
                _observations(sample, kind, context, i) for i, sample in enumerate(samples)
            ]
        with self._lock:
            if kind not in self.recorder.stages or kind in self.stages:
                raise ValueError(f"Unexpected or duplicate telemetry stage: {kind}")
            self.stages[kind] = stage
            complete = self.stages.keys() == self.recorder.stages
        if complete:
            self.recorder.submit(self)
