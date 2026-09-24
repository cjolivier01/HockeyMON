"""Compare detectors on identical inputs while tracking the selected model."""

from __future__ import annotations

import time

import numpy as np
import torch

from hmlib.models.detector_comparison import ComparisonReport
from hmlib.utils.gpu import unwrap_tensor
from hmlib.utils.image import make_channels_first

from .detector_factory_plugin import DetectorFactoryPlugin
from .detector_plugin import DetectorInferencePlugin


def original_shape(meta: dict) -> tuple[int, int]:
    shape = [int(v) for v in meta["ori_shape"]]
    # HmCrop can preserve a full NHWC/NCHW tensor shape in ori_shape.
    if len(shape) == 4:
        shape = shape[1:]
    if len(shape) == 3:
        shape = shape[:2] if shape[-1] in (1, 3, 4) else shape[-2:]
    if len(shape) != 2:
        raise ValueError(f"Unexpected original image shape: {shape}")
    return tuple(shape)


class DetectorComparePlugin(DetectorInferencePlugin):
    def __init__(
        self,
        models: dict,
        selected: str,
        output_dir: str,
        annotations: str | None = None,
        sample_every: int = 30,
        preview_frames: int = 20,
        threshold: float = 0.25,
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        if sample_every < 1 or preview_frames < 0 or not 0 <= threshold <= 1:
            raise ValueError("Invalid comparison sampling/preview/threshold settings")
        self.report = ComparisonReport(
            output_dir,
            models,
            annotations,
            threshold,
            metadata={"tracking_model": selected, "precision": "float32"},
        )
        self.selected = selected
        self.sample_every = sample_every
        self.preview_frames = preview_frames
        self.threshold = threshold
        self._frames_seen = 0
        self._previews_written = 0
        self.factories = torch.nn.ModuleDict(
            {
                name: DetectorFactoryPlugin(
                    **spec,
                    nms_backend="head",
                    cuda_graph=False,
                    static_detections={"enable": False},
                )
                for name, spec in models.items()
                if name != selected
            }
        )
        self.inference = DetectorInferencePlugin()

    def forward(self, context: dict):
        if context.get("using_precalculated_detection", False):
            raise ValueError("Comparison cannot run with precomputed detections")
        inputs = make_channels_first(unwrap_tensor(context["inputs"]))
        # Use a common accuracy reference precision. The installed MMCV CUDA
        # NMS also requires boxes and scores to have matching float32 dtypes.
        inputs = inputs.float()
        context = dict(context, inputs=inputs, fp16=False)
        self.report.metadata["game_id"] = context.get("game_id")
        samples = context["data_samples"]
        track = samples[0] if isinstance(samples, list) else samples
        # The selected detector still runs on every frame and supplies tracking.
        super().forward(context)
        from mmdet.structures import DetDataSample, TrackDataSample

        for index in range(len(track)):
            sample = track[index]
            frame_id = int(sample.metainfo["frame_id"])
            compare = (
                frame_id in self.report.frames
                if self.report.ground_truth is not None
                else self._frames_seen % self.sample_every == 0
            )
            self._frames_seen += 1
            if not compare:
                continue
            predictions, elapsed = {}, {}
            for name in self.report.models:
                # Copy metadata only. GPU prediction wrappers own CUDA events
                # that cannot be deep-copied, and candidates need no old predictions.
                comparison_track = TrackDataSample(
                    video_data_samples=[DetDataSample(metainfo=sample.metainfo)]
                )
                candidate_context = dict(
                    context,
                    inputs=inputs[index : index + 1],
                    data_samples=comparison_track,
                    detect_timer=None,
                )
                if name != self.selected:
                    candidate_context.update(self.factories[name](candidate_context))
                if inputs.is_cuda:
                    torch.cuda.synchronize(inputs.device)
                start = time.perf_counter()
                self.inference(candidate_context)
                if inputs.is_cuda:
                    torch.cuda.synchronize(inputs.device)
                elapsed[name] = (time.perf_counter() - start) * 1000
                inst = comparison_track[0].pred_instances
                boxes = unwrap_tensor(inst.bboxes).detach().float().cpu().numpy()
                scores = unwrap_tensor(inst.scores).detach().float().cpu().numpy()
                labels = unwrap_tensor(inst.labels).detach().cpu().numpy()
                predictions[name] = [
                    dict(
                        bbox=[float(x), float(y), float(right - x), float(bottom - y)],
                        score=float(score),
                    )
                    for (x, y, right, bottom), score, label in zip(boxes, scores, labels)
                    if label == 0 and score > 0 and right > x and bottom > y
                ]
            shape = original_shape(sample.metainfo)
            self.report.record(frame_id, shape, predictions, elapsed, tuple(inputs.shape[-2:]))
            if self._previews_written < self.preview_frames:
                self._preview(frame_id, inputs[index], sample.metainfo, predictions)
                self._previews_written += 1
        return {}

    def _preview(self, frame_id: int, image: torch.Tensor, meta: dict, predictions: dict):
        import cv2

        rgb = image.detach().float().cpu().permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = torch.as_tensor(meta["scale_factor"]).cpu().numpy().reshape(-1)[:2]
        pad = torch.as_tensor(meta.get("pad_param", [0, 0, 0, 0])).cpu().numpy()
        panels = []
        for name, detections in predictions.items():
            panel = base.copy()
            for detection in detections:
                if detection["score"] < self.threshold:
                    continue
                x, y, w, h = detection["bbox"]
                first = np.array([x, y]) * scale + [pad[2], pad[0]]
                last = np.array([x + w, y + h]) * scale + [pad[2], pad[0]]
                cv2.rectangle(
                    panel, tuple(first.astype(int)), tuple(last.astype(int)), (0, 255, 0), 2
                )
                cv2.putText(
                    panel,
                    f"{detection['score']:.2f}",
                    tuple(first.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )
            factor = min(1.0, 1000 / panel.shape[1])
            panel = cv2.resize(
                panel, (round(panel.shape[1] * factor), round(panel.shape[0] * factor))
            )
            cv2.putText(
                panel,
                f"{name} | frame {frame_id} | score >= {self.threshold}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            panels.append(panel)
        output = self.report.output / f"frame_{frame_id:08d}.jpg"
        if not cv2.imwrite(str(output), np.concatenate(panels, axis=0)):
            raise OSError(f"Could not write detector preview: {output}")

    def input_keys(self):
        return super().input_keys() | {"device", "game_id", "work_dir"}

    def finalize(self):
        self.report.finalize()
