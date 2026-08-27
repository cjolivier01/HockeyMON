import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from mmcv.transforms import Compose
from mmdet.models.mot.base import BaseMOTModel
from mmdet.registry import MODELS
from mmdet.structures import OptTrackSampleList
from mmengine.structures import InstanceData

from hmlib.aspen.plugins.base import Plugin
from hmlib.datasets.dataframe import HmDataFrameBase
from hmlib.utils.hockeymon_compat import HmByteTrackConfig, HmTracker, HmTrackerPredictionMode

logger = logging.getLogger(__name__)


def _use_cpp_tracker(dflt: bool = False) -> bool:
    s = os.environ.get("HM_USE_CPP_TRACKER")
    if not s:
        return False
        # return True
    return int(s) != 0


def _should_use_cpp_bytetrack(requested: bool) -> bool:
    return bool(requested)


@MODELS.register_module()
class HmEndToEnd(BaseMOTModel, Plugin):
    def __init__(
        self,
        *args,
        neck: Optional[Callable] = None,
        # post-detection pipeline removed; use dedicated plugins instead
        post_tracking_pipeline: List[Dict[str, Any]] = None,
        pose_model: str = None,
        pose_weights: str = None,
        enabled: bool = True,
        num_classes_override: Optional[int] = None,
        dataset_meta: Dict[str, Any] = None,
        # cpp_bytetrack: bool = _use_cpp_tracker(),
        cpp_bytetrack: bool = True,
        # cpp_bytetrack: bool = False,
        data_preprocessor: Optional[Dict[str, Any]] = None,
        detector: Optional[Dict[str, Any]] = None,
        tracker: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Build data_preprocessor if provided as a dict
        if data_preprocessor and isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)

        # We intentionally ignore `detector` and `tracker` here to decouple from ByteTrack.
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self._enabled = enabled
        self._cpp_bytetrack = _should_use_cpp_bytetrack(bool(cpp_bytetrack))
        # legacy post-detection pipeline removed
        self.post_tracking_pipeline = post_tracking_pipeline
        self.post_tracking_composed_pipeline = None
        self.neck = None
        self._num_classes_override = num_classes_override
        self.dataset_meta = dataset_meta
        self._pose_model = pose_model
        self._pose_weights = pose_weights

        if self._cpp_bytetrack:
            config = HmByteTrackConfig()
            config.init_track_thr = 0.7

            #
            # Low threshold Needs to be low in case through glass/poles
            #
            config.obj_score_thrs_low = 0.1
            config.obj_score_thrs_high = 0.3

            config.match_iou_thrs_high = 0.1
            config.match_iou_thrs_low = 0.5
            config.match_iou_thrs_tentative = 0.3
            config.track_buffer_size = 60
            config.return_user_ids = False
            config.return_track_age = False
            prediction_mode = getattr(HmTrackerPredictionMode, "BoundingBox", None)
            if prediction_mode is not None:
                config.prediction_mode = prediction_mode

            self._hm_byte_tracker = HmTracker(config)
        else:
            self._hm_byte_tracker = None

        if neck is not None:
            assert False
            # self.neck = build_neck(neck)

    def init_weights(self):
        # No-op: this container does not have submodules that require init.
        pass

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    # @auto_fp16(apply_to=("img",))
    def forward(self, img, return_loss=True, **kwargs):
        # Plugin-mode: if a dict is passed (AspenNet), treat it as context
        if isinstance(img, dict):
            context: Dict[str, Any] = img
            return self._forward_plugin(context)
        # When used as a model in non-trunk mode, we don't implement end-to-end forward anymore.
        raise NotImplementedError(
            "HmEndToEnd no longer implements ByteTrack forward; use Aspen plugins instead."
        )

    # AspenNet trunk interface: runs tracking + returns context updates
    def _forward_plugin(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not getattr(self, "_enabled", True):
            return {}

        # Lazy build of Compose pipelines
        if self.post_tracking_pipeline and self.post_tracking_composed_pipeline is None:
            self.post_tracking_composed_pipeline = Compose(self.post_tracking_pipeline)

        using_precalculated_tracking: bool = bool(
            context.get("using_precalculated_tracking", False)
        )
        using_precalculated_detection: bool = bool(
            context.get("using_precalculated_detection", False)
        )
        tracking_dataframe = context.get("tracking_dataframe")
        detection_dataframe = context.get("detection_dataframe")
        frame_id: int = int(context.get("frame_id", -1))
        track_samples = context.get("data_samples")
        if isinstance(track_samples, list):
            track_data_sample = track_samples[0] if track_samples else None
        else:
            track_data_sample = track_samples
        if track_data_sample is None:
            return {}

        detect_timer = context.get("detect_timer")
        if detect_timer is not None:
            detect_timer.tic()

        # With separated detector/tracker plugins, this trunk should not be invoked.
        # Keep a minimal no-op to avoid breaking existing graphs if enabled=false.
        if False:
            pass

        if detect_timer is not None:
            detect_timer.toc()

        nr_tracks = 0
        max_tracking_id = 0

        jersey_results = context.get("jersey_results")
        for frame_index, video_data_sample in enumerate(track_data_sample.video_data_samples):
            pred_track_instances = getattr(video_data_sample, "pred_track_instances", None)
            if pred_track_instances is None:
                continue
            if not using_precalculated_tracking and tracking_dataframe is not None:
                tracking_dataframe.add_frame_records(
                    frame_id=frame_id + frame_index,
                    tracking_ids=pred_track_instances.instances_id,
                    tlbr=pred_track_instances.bboxes,
                    scores=pred_track_instances.scores,
                    labels=pred_track_instances.labels,
                    jersey_info=(
                        jersey_results[frame_index] if jersey_results is not None else None
                    ),
                )
            if not using_precalculated_detection and detection_dataframe is not None:
                detection_dataframe.add_frame_records(
                    frame_id=frame_id,
                    scores=video_data_sample.pred_instances.scores,
                    labels=video_data_sample.pred_instances.labels,
                    bboxes=video_data_sample.pred_instances.bboxes,
                )

        # data = data.copy()
        # Avoid passing big tensors downstream unnecessarily
        # Re-attach original images from pre-model context if missing
        # if preserved_original_images is not None:
        #     data["original_images"] = preserved_original_images
        # elif "original_images" in data:
        #     data["original_images"] = data["original_images"]
        # if "img" in data:
        #     del data["img"]

        return {"nr_tracks": nr_tracks, "max_tracking_id": max_tracking_id}

    # Plugin introspection for AspenNet minimal_context mode
    def input_keys(self):
        return {
            "data_samples",
            "jersey_results",
            "fp16",
            "using_precalculated_tracking",
            "using_precalculated_detection",
            "tracking_dataframe",
            "detection_dataframe",
            "frame_id",
            "detect_timer",
        }

    def output_keys(self):
        return {"nr_tracks", "max_tracking_id"}

    def predict(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        data_samples: OptTrackSampleList = None,
        **kwargs: Dict[str, Any],
    ):
        # No longer supports end-to-end predict outside Aspen plugins.
        return self.simple_test(inputs=inputs, data_samples=data_samples, **kwargs)

    # Satisfy BaseMOTModel abstract method; not used in this container.
    def loss(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        data_samples: OptTrackSampleList = None,
        **kwargs: Dict[str, Any],
    ):
        raise NotImplementedError("HmEndToEnd is not intended for training.")

    @staticmethod
    def get_dataframe(
        dataset_results: Union[Dict[str, Any], None], name: str
    ) -> Optional[HmDataFrameBase]:
        if not dataset_results:
            return None
        return dataset_results.get(name)

    def simple_test(
        self,
        inputs,
        data_samples: OptTrackSampleList = None,
        track: bool = True,
        **kwargs,
    ):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        if isinstance(data_samples, list):
            assert len(data_samples) == 1
            track_data_sample = data_samples[0]
        else:
            track_data_sample = data_samples
        video_len = len(track_data_sample)

        assert inputs.ndim == 5
        # only one video, but can be multiple frames
        assert inputs.shape[0] == 1
        det_inputs = inputs.squeeze(0)
        del inputs

        tracking_dataframe = None
        # Maybe use pre-saved detection results
        dataset_results = kwargs.get("dataset_results")
        detection_dataframe = self.get_dataframe(dataset_results, "detection_dataframe")
        if detection_dataframe:
            # TODO: support multi-batch
            assert len(track_data_sample) == 1
            video_data_sample = track_data_sample.video_data_samples[0]
            assert not hasattr(video_data_sample, "pred_instances")
            video_data_sample.pred_instances = InstanceData(
                scores=torch.from_numpy(detection_dataframe["scores"]),
                labels=torch.from_numpy(detection_dataframe["labels"]),
                bboxes=torch.from_numpy(detection_dataframe["bboxes"]),
            )
            all_det_results = track_data_sample
        elif True:
            # makes a difference?
            det_inputs = det_inputs.contiguous()

        if any(not hasattr(vds, "pred_instances") for vds in track_data_sample.video_data_samples):

            def _predict(det_inputs, track_data_sample):
                return self.detector.predict(det_inputs, track_data_sample)

            all_det_results = _predict(det_inputs, track_data_sample)

        del det_inputs

        all_frame_jersey_info = []
        for frame_index in range(video_len):
            det_data_sample = all_det_results[frame_index]
            img_data_sample = track_data_sample[frame_index]
            det_bboxes = det_data_sample.pred_instances.bboxes
            det_labels = det_data_sample.pred_instances.labels
            det_scores = det_data_sample.pred_instances.scores

            if self.post_detection_composed_pipeline is not None or (
                track and self.post_tracking_composed_pipeline is not None
            ):
                # We may prune the detections to relevent items
                data: Dict[str, Any] = {
                    "det_bboxes": det_bboxes,
                    "labels": det_labels,
                    "scores": det_scores,
                    "prune_list": ["det_bboxes", "labels", "scores"],
                    "ori_shape": det_data_sample.metainfo["ori_shape"],
                    "data_samples": data_samples,
                }
                data.update(**kwargs)

            if self.post_detection_composed_pipeline is not None:
                data = self.post_detection_composed_pipeline(data)
                det_bboxes = data["det_bboxes"]
                det_labels = data["labels"]
                det_scores = data["scores"]

                instance_data = InstanceData()
                instance_data["scores"] = det_scores
                instance_data["labels"] = det_labels
                instance_data["bboxes"] = det_bboxes
                det_data_sample.pred_instances = instance_data
                assert len(det_bboxes) == len(det_labels)
                assert len(det_scores) == len(det_labels)

            # Tracker will want to know the frame id so that it can expire lost tracks
            frame_id = det_data_sample.metainfo["img_id"].reshape([1])
            det_data_sample.set_metainfo({"frame_id": frame_id.item()})

            # track = False
            active_track_count: int = 0
            if track and not hasattr(img_data_sample, "pred_track_instances"):
                tracking_dataframe = self.get_dataframe(dataset_results, "tracking_dataframe")
                if tracking_dataframe:
                    pred_track_instances = InstanceData(
                        instances_id=torch.from_numpy(tracking_dataframe["tracking_ids"]),
                        scores=torch.from_numpy(tracking_dataframe["scores"]),
                        labels=torch.from_numpy(tracking_dataframe["labels"]),
                        bboxes=torch.from_numpy(tracking_dataframe["bboxes"]),
                    )
                    frame_jersey_results = []
                    for tid, jersey_info in zip(
                        pred_track_instances.instances_id, tracking_dataframe["jersey_info"]
                    ):
                        if jersey_info is not None:
                            frame_jersey_results.append(jersey_info)
                    all_frame_jersey_info.append(frame_jersey_results)
                if self._cpp_bytetrack:
                    ll1: int = len(det_data_sample.pred_instances.bboxes)
                    assert len(det_data_sample.pred_instances.labels) == ll1
                    assert len(det_data_sample.pred_instances.scores) == ll1
                    results = self._hm_byte_tracker.track(
                        data=dict(
                            frame_id=frame_id,
                            bboxes=det_data_sample.pred_instances.bboxes,
                            labels=det_data_sample.pred_instances.labels,
                            scores=det_data_sample.pred_instances.scores,
                        )
                    )
                    if "user_ids" in results:
                        ids = results["user_ids"]
                    else:
                        ids = results["ids"]
                    ll2: int = len(ids)
                    assert len(results["bboxes"]) == ll2
                    assert len(results["scores"]) == ll2
                    assert len(results["labels"]) == ll2
                    pred_track_instances = InstanceData(
                        instances_id=ids.cpu(),
                        bboxes=results["bboxes"].cpu(),
                        scores=results["scores"].cpu(),
                        labels=results["labels"].cpu(),
                    )
                else:
                    pred_track_instances = self.tracker.track(data_sample=det_data_sample, **kwargs)
                active_track_count: int = max(
                    active_track_count, len(pred_track_instances.instances_id)
                )
                img_data_sample.pred_track_instances = pred_track_instances

            # For performance purposes, add in the number of tracks we're tracking (both active and inactive)
            det_data_sample.set_metainfo({"nr_tracks": active_track_count})

        if all_frame_jersey_info and any(j for j in all_frame_jersey_info):
            data["jersey_results"] = all_frame_jersey_info

        if track and self.post_tracking_composed_pipeline is not None:
            data = self.post_tracking_composed_pipeline(data)

        return data
