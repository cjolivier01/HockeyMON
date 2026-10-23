"""Native hockey PARSeq and causal STGCN++ backends for the Aspen graph."""

from __future__ import annotations

import math

import numpy as np
import torch

from hmlib.aspen.player_semantics import (
    ActionHistory,
    JerseyConsensus,
    JerseyLabel,
    SettledActions,
    SettledJerseys,
    decode_parseq,
    jersey_roi,
    parseq_input,
)
from hmlib.models.player_analytics import PlayerModel
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.gpu import unwrap_tensor

from .base import Plugin


def _array(value):
    value = unwrap_tensor(value)
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _tracks(frame):
    instances = _field(frame, "pred_track_instances")
    if instances is None:
        return {}
    ids, boxes = _field(instances, "instances_id"), _field(instances, "bboxes")
    if ids is None or boxes is None:
        return {}
    ids, boxes = _array(ids), _array(boxes)
    mask = get_track_mask(instances)
    if mask is not None:
        mask = _array(mask).astype(bool)
        ids, boxes = ids[mask], boxes[mask]
    return {
        int(identifier): box
        for identifier, box in zip(ids[:256], boxes[:256], strict=True)
        if identifier >= 0
    }


def fresh_poses(result, tracks) -> dict[int, np.ndarray]:
    """Associate only fresh inference with distinct current tracks, never display holds."""
    if not result or not tracks:
        return {}
    poses, boxes = [], []
    for prediction in result.get("predictions", []):
        instances = _field(prediction, "pred_instances", prediction)
        points = _field(instances, "keypoints")
        scores = _field(instances, "keypoint_scores")
        bounds = _field(instances, "bboxes")
        if points is None:
            continue
        if scores is None or bounds is None:
            raise ValueError(
                "Native analytics requires fresh keypoint confidence and source bounding boxes"
            )
        points, scores, bounds = _array(points), _array(scores), _array(bounds)
        if points.ndim != 3 or points.shape[1:] != (17, 2) or scores.shape != points.shape[:2]:
            raise ValueError("Native player analytics requires COCO17 pose topology")
        if len(bounds) != len(points):
            raise ValueError("Fresh pose boxes and keypoints must align")
        poses.extend(np.concatenate((points, scores[..., None]), axis=-1))
        boxes.extend(bounds[:, :4])
    if not poses:
        return {}
    ids, target = list(tracks), np.asarray(list(tracks.values()))
    boxes = np.asarray(boxes)
    intersection = np.maximum(
        0,
        np.minimum(boxes[:, None, 2:], target[None, :, 2:])
        - np.maximum(boxes[:, None, :2], target[None, :, :2]),
    ).prod(axis=-1)
    area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=-1)
    target_area = np.maximum(0, target[:, 2:] - target[:, :2]).prod(axis=-1)
    overlap = intersection / np.maximum(1e-9, area[:, None] + target_area[None, :] - intersection)
    used_poses, output = set(), {}
    for flat in np.argsort(-overlap, axis=None, kind="stable"):
        pose_index, track_index = np.unravel_index(flat, overlap.shape)
        if overlap[pose_index, track_index] < 0.3:
            break
        track_id = ids[track_index]
        if pose_index not in used_poses and track_id not in output:
            output[track_id] = poses[pose_index]
            used_poses.add(pose_index)
    return output


def _frames(context):
    images = unwrap_tensor(context["original_images"])
    samples = context["data_samples"]
    if isinstance(samples, list):
        if len(samples) != 1:
            raise ValueError("Player analytics expects one video batch")
        samples = samples[0]
    frames = _field(samples, "video_data_samples", samples)
    if len(frames) != len(images):
        raise ValueError("Player analytics tracking and image batches must align")
    shared = context.get("shared", {})
    frame_ids = context.get("frame_ids")
    if frame_ids is not None:
        frame_ids = _array(frame_ids)
    pose_results = context.get("pose_results") or [{} for _ in frames]
    if len(pose_results) != len(frames):
        raise ValueError("Fresh poses must align with source images")
    for index, (image, frame, poses) in enumerate(zip(images, frames, pose_results, strict=True)):
        metadata = _field(frame, "metainfo", {})
        if "pts_ns" in metadata:
            timestamp = int(metadata["pts_ns"])
        else:
            fps = context.get("fps", shared.get("fps", metadata.get("fps")))
            identifier = frame_ids[index] if frame_ids is not None else metadata.get("frame_id")
            if identifier is None and context.get("frame_id") is not None:
                identifier = int(context["frame_id"]) + index
            if (
                fps is None
                or not math.isfinite(float(fps))
                or float(fps) <= 0
                or identifier is None
            ):
                raise ValueError("Native player analytics requires source PTS or frame IDs and FPS")
            timestamp = round(float(identifier) / float(fps) * 1e9)
        if timestamp < 0:
            raise ValueError("Player analytics source timestamps must be nonnegative")
        if image.ndim != 3:
            raise ValueError("Player analytics requires three-dimensional source images")
        if image.shape[0] in (3, 4):
            height, width = image.shape[-2:]
        elif image.shape[-1] in (3, 4):
            height, width = image.shape[:2]
        else:
            raise ValueError("Player analytics requires three color channels")
        tracks = _tracks(frame)
        yield timestamp, image, int(width), int(height), tracks, poses


class _NativeAnalytics(Plugin):
    feature = ""

    def __init__(self, model_path, provider, enabled):
        super().__init__(enabled=enabled)
        self.model = PlayerModel(self.feature, model_path, provider)
        self._reset()

    def _reset(self):
        self.source = None
        self.timestamp = None
        self.states = {}
        self.last_seen = {}
        self.attempts = {}

    def _advance(self, context, timestamp, width, height, tracks):
        shared = context.get("shared", {})
        geometry = context.get("camera_input_geometry") or {}
        source = (
            shared.get("game_id"),
            shared.get("source_id"),
            tuple(shared.get("source_video_paths") or []),
            geometry.get("stitched_geometry_revision"),
            width,
            height,
        )
        if source != self.source or (self.timestamp is not None and timestamp < self.timestamp):
            self._reset()
        self.source, self.timestamp = source, timestamp
        self.last_seen.update({track_id: timestamp for track_id in tracks})
        keep = sorted(
            (identifier for identifier, seen in self.last_seen.items() if timestamp - seen <= 2e9),
            key=lambda identifier: (-self.last_seen[identifier], identifier),
        )[:256]
        self.states = {
            identifier: state for identifier, state in self.states.items() if identifier in keep
        }
        self.attempts = {
            identifier: attempt
            for identifier, attempt in self.attempts.items()
            if identifier in keep
        }
        self.last_seen = {identifier: self.last_seen[identifier] for identifier in keep}

    def _settings(self, context, rate, threshold):
        config = context.get("shared", {}).get("game_config", {})
        settings = config.get("pipeline", {}).get("player-analytics", {}).get(self.feature, {})
        rate = float(settings.get("rate-hz", rate))
        threshold = float(settings.get("confidence-threshold", threshold))
        if not math.isfinite(rate) or not 0 < rate <= 120 or not 0 <= threshold <= 1:
            raise ValueError(f"Invalid {self.feature} cadence or confidence threshold")
        return round(1e9 / rate), threshold

    def _due(self, timestamp, tracks, interval):
        return sorted(
            (
                identifier
                for identifier in tracks
                if timestamp - self.attempts.get(identifier, -(10**20)) >= interval
            ),
            key=lambda identifier: (self.attempts.get(identifier, -(10**20)), identifier),
        )

    def input_keys(self):
        return {
            "original_images",
            "data_samples",
            "pose_results",
            "frame_ids",
            "frame_id",
            "fps",
            "camera_input_geometry",
        }


class TemporalActionPlugin(_NativeAnalytics):
    feature = "action"

    def __init__(
        self,
        model_path: str | None = None,
        provider: str = "auto",
        enabled: bool = True,
        top_k: int = 3,
        score_threshold: float = 0.5,
    ):
        super().__init__(model_path, provider, enabled)
        if not 1 <= top_k <= 60:
            raise ValueError("Action top_k must be between 1 and 60")
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.labels = self.model.contract["labels"]

    def forward(self, context):
        if not self.enabled:
            self._reset()
            return {}
        interval, minimum = self._settings(context, 1, self.score_threshold)
        outputs = []
        for timestamp, _, width, height, tracks, result in _frames(context):
            self._advance(context, timestamp, width, height, tracks)
            for track_id, pose in fresh_poses(result, tracks).items():
                state = self.states.setdefault(
                    track_id,
                    {"history": ActionHistory(), "label": None, "contender": None, "count": 0},
                )
                history = state["history"]
                previous = history.previous_ns
                history.observe(timestamp, pose, width, height)
                if previous is not None and (
                    history.previous_ns is None or timestamp - previous > 150_000_000
                ):
                    state.update(label=None, contender=None, count=0)
            tensors, identifiers = [], []
            for track_id in self._due(timestamp, tracks, interval):
                state = self.states.get(track_id)
                tensor = state["history"].tensor(timestamp) if state else None
                if tensor is not None:
                    tensors.append(tensor)
                    identifiers.append(track_id)
                    self.attempts[track_id] = timestamp
                    if len(tensors) == 32:
                        break
            if tensors:
                logits = self.model(np.stack(tensors))
                probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probabilities /= probabilities.sum(axis=-1, keepdims=True)
                for track_id, scores in zip(identifiers, probabilities, strict=True):
                    state = self.states[track_id]
                    indices = np.argsort(-scores, kind="stable")[: self.top_k]
                    label, score = int(indices[0]), float(scores[indices[0]])
                    if score < minimum:
                        state.update(contender=None, count=0)
                        continue
                    state["count"] = (
                        min(2, state["count"] + 1) if state["contender"] == label else 1
                    )
                    state["contender"] = label
                    current = state["label"]
                    if (
                        current is None
                        or current["label_index"] == label
                        or score >= current["score"] + 0.1
                        or state["count"] >= 2
                    ):
                        state["label"] = {
                            "tracking_id": track_id,
                            "label": self.labels[label],
                            "label_index": label,
                            "score": score,
                            "topk_indices": indices.tolist(),
                            "topk_scores": scores[indices].tolist(),
                            "window_start_ns": state["history"].samples[0][0],
                            "observed_at_ns": timestamp,
                        }
            frame_results = SettledActions()
            for track_id in tracks:
                state = self.states.get(track_id)
                label = state["label"] if state else None
                if label is not None and timestamp - label["observed_at_ns"] <= 2e9:
                    frame_results.append(dict(label))
                elif state:
                    state.update(label=None, contender=None, count=0)
            outputs.append(frame_results)
        return {"action_results": outputs}

    def output_keys(self):
        return {"action_results"}


class ParseqJerseyPlugin(_NativeAnalytics):
    feature = "jersey"

    def __init__(
        self,
        model_path: str | None = None,
        provider: str = "auto",
        enabled: bool = True,
        roi_mode: str = "bbox",
        min_confidence: float = 0.8,
    ):
        super().__init__(model_path, provider, enabled)
        if roi_mode not in {"bbox", "pose"}:
            raise ValueError("Jersey ROI mode must be bbox or pose")
        self.roi_mode = roi_mode
        self.min_confidence = min_confidence

    def forward(self, context):
        if not self.enabled:
            self._reset()
            return {}
        interval, minimum = self._settings(context, 2, self.min_confidence)
        outputs = []
        for timestamp, image, width, height, tracks, result in _frames(context):
            self._advance(context, timestamp, width, height, tracks)
            poses = fresh_poses(result, tracks) if self.roi_mode == "pose" else {}
            crops, identifiers = [], []
            pixels = None
            for track_id in self._due(timestamp, tracks, interval):
                roi = jersey_roi(
                    tracks[track_id], poses.get(track_id), self.roi_mode, width, height
                )
                if roi is None:
                    continue
                if pixels is None:
                    pixels = _array(image)
                    if pixels.shape[0] in (3, 4):
                        pixels = pixels.transpose(1, 2, 0)
                    if pixels.dtype != np.uint8:
                        if not np.isfinite(pixels).all():
                            raise ValueError("PARSeq source pixels must be finite")
                        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
                    pixels = pixels[..., :3]
                crops.append(parseq_input(pixels, roi))
                identifiers.append(track_id)
                self.attempts[track_id] = timestamp
                if len(crops) == 32:
                    break
            if crops:
                for track_id, logits in zip(identifiers, self.model(np.stack(crops)), strict=True):
                    observation = decode_parseq(logits)
                    if observation is not None and observation[1] >= minimum:
                        state = self.states.setdefault(track_id, JerseyConsensus())
                        state.observe(timestamp, *observation)
            frame_results = SettledJerseys()
            for track_id in tracks:
                state = self.states.get(track_id)
                observed = state.result(timestamp) if state else None
                if observed is not None:
                    text, score = observed
                    frame_results.append(
                        JerseyLabel(tracking_id=track_id, number=text, score=score)
                    )
            outputs.append(frame_results)
        return {"jersey_results": outputs}

    def output_keys(self):
        return {"jersey_results"}
