import contextlib
from contextlib import nullcontext
from typing import Any, Dict

import torch

from hmlib.log import get_logger
from hmlib.utils.gpu import unwrap_tensor, wrap_tensor
from hmlib.utils.image import make_channels_first

from .base import Plugin
from .detector_factory_plugin import _strip_static_padding

logger = get_logger(__name__)


def _snapshot_tensor(value):
    tensor = unwrap_tensor(value)
    if torch.is_tensor(tensor):
        return tensor.clone()
    return tensor


#: Detection count fields, in the order `hmlib/telemetry/capture.py` consults
#: them. Each is a *prefix* count: rows beyond it are static padding.
_VALID_COUNT_KEYS = ("num_detections", "num_valid_after_nms", "num_valid")


def _valid_prefix(instances, total: int) -> tuple[str | None, Any | None, int]:
    """The ``(key, value, limit)`` of whichever prefix-count field is present."""
    for key in _VALID_COUNT_KEYS:
        candidate = unwrap_tensor(getattr(instances, key, None))
        if candidate is not None:
            try:
                limit = max(0, min(total, int(torch.as_tensor(candidate).reshape(-1)[0])))
            except (ValueError, IndexError, RuntimeError):
                limit = total
            return key, candidate, limit
    return None, None, total


def _cap_referees(instances, referee_label: int, max_referees: int, demote_to: int = 0):
    """Hold the referee count to what can physically be on the ice.

    A game has a fixed, small number of officials, which the detector has no
    way to know -- it scores each box independently. Measured on a
    vegas-gknights-1 warmup frame it reported five referees where there were
    none, all distant gold-sweater skaters scoring 0.48-0.77, while the real
    officials in live frames scored 0.79-0.90. Ranking by score and keeping
    only the top few converts that into the right answer far more reliably
    than a score threshold does: the confidence gap between the worst true
    positive and the best false positive was 0.014 on the frames measured,
    which is too thin to threshold on.

    Surplus referees are *relabelled* as players, not dropped. The detector
    did find a person there; it was only wrong about the role, and deleting
    the box would lose a skater from tracking entirely.
    """
    labels = unwrap_tensor(getattr(instances, "labels", None))
    scores = unwrap_tensor(getattr(instances, "scores", None))
    if not torch.is_tensor(labels) or not torch.is_tensor(scores) or labels.numel() == 0:
        return instances

    _, _, limit = _valid_prefix(instances, labels.shape[0])
    is_referee = labels == int(referee_label)
    is_referee[limit:] = False
    indices = torch.nonzero(is_referee).reshape(-1)
    if indices.numel() <= max_referees:
        return instances

    # Keep the highest-scoring officials; demote the rest.
    ranked = indices[torch.argsort(scores[indices], descending=True)]
    demoted = ranked[max_referees:]
    new_labels = labels.clone()
    new_labels[demoted] = int(demote_to)
    instances.labels = new_labels
    logger.info(
        "Capped referees at %d: demoted %d lower-scoring detection(s) to class %d",
        max_referees,
        int(demoted.numel()),
        demote_to,
    )
    return instances


def _drop_degenerate_boxes(instances):
    """Compact away detections with non-positive width or height.

    A zero-area box is meaningless to every consumer, and the telemetry writer
    rejects the whole batch over one of them
    (`hmlib/telemetry/capture.py::_boxes`), taking down the run. The tracker and
    PlayTracker already guard against this shape of input; detections were the
    one path that did not, so a model emitting a collapsed box -- more likely
    from a freshly fine-tuned head -- killed the pipeline rather than losing a
    box.

    The array cannot simply be filtered. With static detection outputs the
    tensors are fixed-size and a separate count says how many leading rows are
    real; dropping rows outright leaves that count larger than the array and
    trips a different telemetry assertion. So the valid prefix is compacted in
    place -- survivors slide to the front, the tail stays padding -- and the
    count is lowered to match.

    Returns the instances unchanged when everything is already valid, so the
    common path allocates nothing.
    """
    boxes = unwrap_tensor(getattr(instances, "bboxes", None))
    if not torch.is_tensor(boxes) or boxes.numel() == 0 or boxes.ndim != 2:
        return instances

    total = boxes.shape[0]
    count_key, valid, limit = _valid_prefix(instances, total)

    ok = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & torch.isfinite(boxes).all(1)
    within = ok.clone()
    within[limit:] = False
    kept = int(within.sum())
    if kept == limit:
        return instances

    order = torch.cat(
        (
            torch.nonzero(within).reshape(-1),
            torch.nonzero(~within).reshape(-1),
        )
    )
    compacted = instances[order.to(device=boxes.device)]
    if count_key is not None:
        new_count = torch.as_tensor(valid).reshape(-1).clone()
        new_count[0] = kept
        # The count is a scalar, so it lives in metainfo: InstanceData requires
        # every *data* field to match the instance count and refuses a length-1
        # tensor on an N-row container.
        if count_key in compacted.metainfo_keys():
            compacted.set_metainfo({count_key: new_count})
        else:
            setattr(compacted, count_key, new_count)
    return compacted


class DetectorInferencePlugin(Plugin):
    """
    Runs a pure detection model per-frame and attaches `pred_instances`.

    Expects in context:
      - inputs: model input tensor (T, C, H, W) or (T, H, W, C)
      - data_samples: TrackDataSample or [TrackDataSample]
      - detector_model: mmdet detector with .predict
      - fp16: bool (optional)
      - detect_timer: optional timer with tic/toc

    Produces in context:
      - Side-effects: attaches per-frame `pred_instances` to each `img_data_sample` in data_samples
    """

    def __init__(
        self,
        enabled: bool = True,
        referee_label: int | None = None,
        max_referees: int = 3,
    ):
        super().__init__(enabled=enabled)
        # `referee_label` is written by hmlib/models/detector_selection.py from
        # the chosen profile; it stays None on every single-class detector, and
        # the cap is then a no-op.
        self._referee_label = None if referee_label is None else int(referee_label)
        self._max_referees = max(0, int(max_referees))

    def __call__(self, *args, **kwargs) -> Any:
        self._iter_num += 1
        do_trace = self._iter_num == 4
        if do_trace:
            pass
        # from cuda_stacktrace import CudaStackTracer

        with (
            # CudaStackTracer(functions=["cudaStreamSynchronize"], enabled=do_trace),
            contextlib.nullcontext(),
        ):
            results = super().__call__(*args, **kwargs)
        if do_trace:
            pass
        return results

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        if bool(context.get("using_precalculated_detection", False)):
            return {}
        detector = context.get("detector_model")
        if detector is None:
            return {}

        fp16: bool = bool(context.get("fp16", False))
        detect_timer = context.get("detect_timer")
        if detect_timer is not None:
            detect_timer.tic()

        inputs_any = context.get("inputs")
        if inputs_any is None:
            return {}
        detection_image = unwrap_tensor(inputs_any)

        track_samples = context.get("data_samples")
        # Accept either TrackDataSample or [TrackDataSample]
        if isinstance(track_samples, list):
            assert len(track_samples) == 1
            track_data_sample = track_samples[0]
        else:
            track_data_sample = track_samples
        video_len = len(track_data_sample)

        with torch.no_grad():
            det_device = (
                detection_image.device
                if isinstance(detection_image, torch.Tensor)
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            use_autocast = bool(fp16 and det_device.type == "cuda")
            amp_ctx = (
                torch.amp.autocast("cuda", dtype=torch.float16, enabled=True)
                if use_autocast
                else nullcontext()
            )

            if isinstance(detection_image, torch.Tensor):
                batch_inputs = detection_image
            else:
                batch_inputs = torch.as_tensor(detection_image)
                batch_inputs = batch_inputs
            if batch_inputs.ndim != 4:
                raise AssertionError(
                    "DetectorInferencePlugin expects per-frame tensors shaped (T, C, H, W)"
                )
            batch_inputs = make_channels_first(batch_inputs).contiguous()
            assert batch_inputs.size(0) == video_len, "Mismatch between frames and detection inputs"

            batch_data_samples = []
            for frame_id in range(video_len):
                img_data_sample = track_data_sample[frame_id]
                # Ensure frame_id metainfo is present for downstream postprocessing
                fid = img_data_sample.metainfo.get("img_id")
                img_data_sample.set_metainfo(
                    {
                        "frame_id": fid,
                    }
                )
                batch_data_samples.append(img_data_sample)

            with amp_ctx:
                det_results = detector.predict(batch_inputs / 255.0, batch_data_samples)
                # det_results = detector.predict(batch_inputs, batch_data_samples)

            assert (
                isinstance(det_results, (list, tuple)) and len(det_results) == video_len
            ), "Batch inference must return one result per frame"
            for img_data_sample, det_sample in zip(batch_data_samples, det_results):
                # Attach detections to the video data sample for downstream tracker
                instances = _drop_degenerate_boxes(
                    _strip_static_padding(det_sample.pred_instances, strip=False)
                )
                if self._referee_label is not None:
                    instances = _cap_referees(
                        instances,
                        referee_label=self._referee_label,
                        max_referees=self._max_referees,
                    )
                img_data_sample.pred_instances = instances
                img_data_sample.pred_instances.bboxes = wrap_tensor(
                    _snapshot_tensor(img_data_sample.pred_instances.bboxes)
                )
                img_data_sample.pred_instances.labels = wrap_tensor(
                    _snapshot_tensor(img_data_sample.pred_instances.labels)
                )
                img_data_sample.pred_instances.scores = wrap_tensor(
                    _snapshot_tensor(img_data_sample.pred_instances.scores)
                )
        if detect_timer is not None:
            detect_timer.toc()

        return {}

    def input_keys(self):
        return {
            "inputs",
            "data_samples",
            "detector_model",
            "fp16",
            "detect_timer",
            "using_precalculated_detection",
        }

    def output_keys(self):
        return set()
