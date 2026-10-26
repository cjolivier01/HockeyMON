from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import torch

from hmlib.builder import HM
from hmlib.jersey.jersey_tracker import JerseyTracker
from hmlib.log import logger
from hmlib.team.team_tracker import TeamLabelRenderer
from hmlib.tracking_utils import visualization as vis
from hmlib.tracking_utils.utils import get_track_mask
from hmlib.utils.gpu import StreamTensorBase, unwrap_tensor, wrap_tensor
from hmlib.utils.image import is_channels_last, make_channels_last
from hmlib.vis.labels import draw_label_box

from .base import Plugin

#: BGR badge colour for officials. Amber, so it reads as "not a skater"
#: against both the team palette and the per-id tracking colours.
_REFEREE_COLOR = (0, 200, 240)


def _batch_tlbrs_to_tlwhs(tlbrs: torch.Tensor) -> torch.Tensor:
    tlwhs = tlbrs.clone()
    tlwhs[:, 2] = tlwhs[:, 2] - tlwhs[:, 0]
    tlwhs[:, 3] = tlwhs[:, 3] - tlwhs[:, 1]
    return tlwhs


def _summarize_value(v: Any) -> str:
    if isinstance(v, StreamTensorBase):
        return f"StreamTensor({tuple(v.shape)})"
    if isinstance(v, torch.Tensor):
        return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    if isinstance(v, dict):
        return f"dict(keys={len(v)})"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)})"
    return type(v).__name__


@HM.register_module()
class OverlayPlugin(Plugin):
    """Render late-stage overlays using data produced by other trunks.

    This is meant to run *after* compute-heavy trunks (pose, jersey, etc.) so
    those trunks can execute in parallel with play tracking, and overlays can
    be composed right before the camera crop/output stage.

    Expects in context:
      - img: batched image tensor [T,H,W,C] or [T,C,H,W] (from PlayTrackerPlugin)
      - data_samples: TrackDataSample (or list) with tracking results
      - pose_results: optional list of pose results
      - jersey_results: optional list of jersey results
      - pose_inferencer: optional (used only for its visualizer)

    Produces in context:
      - img: updated with overlays
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        plot_pose: bool = False,
        pose_kpt_thr: float | None = None,
        plot_jersey_numbers: bool = False,
        plot_teams: bool = False,
        plot_referees: bool = True,
        referee_label: int | None = None,
        goalie_label: int | None = None,
        plot_tracking: bool = False,
        plot_all_detections: float | None = None,
        plot_trajectories: bool = False,
        draw_tracking_circles: bool = True,
        trajectory_history: int = 75,
        print_available: bool = True,
        print_every: int = 0,
    ) -> None:
        super().__init__(enabled=enabled)
        self._plot_pose = bool(plot_pose)
        self._pose_kpt_thr = pose_kpt_thr
        self._plot_jersey_numbers = bool(plot_jersey_numbers)
        self._plot_teams = bool(plot_teams)
        self._plot_referees = bool(plot_referees)
        self._referee_label = None if referee_label is None else int(referee_label)
        self._goalie_label = None if goalie_label is None else int(goalie_label)
        self._plot_tracking = bool(plot_tracking)
        self._plot_all_detections = plot_all_detections
        self._plot_trajectories = bool(plot_trajectories)
        self._draw_tracking_circles = bool(draw_tracking_circles)
        self._trajectory_history = max(1, int(trajectory_history))
        self._print_available = bool(print_available)
        self._print_every = max(0, int(print_every))
        self._printed_once = False
        self._jersey_tracker = JerseyTracker(show=bool(plot_jersey_numbers))
        self._team_renderer = TeamLabelRenderer()
        self._traj: dict[int, list[tuple[int, int]]] = {}

    def _maybe_print_available(self, context: dict[str, Any]) -> None:
        if not self._print_available:
            return
        if self._print_every <= 0:
            if self._printed_once:
                return
        else:
            if (self._iter_num % self._print_every) != 0:
                return

        keys = sorted(context.keys())
        msg = (
            f"OverlayPlugin available keys: {keys} | "
            f"img={_summarize_value(context.get('img'))} | "
            f"frame_ids={_summarize_value(context.get('frame_ids'))} | "
            f"pose_results={_summarize_value(context.get('pose_results'))} | "
            f"jersey_results={_summarize_value(context.get('jersey_results'))}"
        )
        logger.info(msg)
        self._printed_once = True

    @staticmethod
    def _to_batched_image(img_any: Any) -> torch.Tensor | None:
        if img_any is None:
            return None
        if isinstance(img_any, StreamTensorBase):
            img_any = img_any.wait()
        if isinstance(img_any, torch.Tensor):
            img = img_any
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.ndim != 4:
                return None
            if not is_channels_last(img):
                img = make_channels_last(img)
            return img
        if isinstance(img_any, np.ndarray):
            img = torch.from_numpy(img_any)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.ndim != 4:
                return None
            if not is_channels_last(img):
                img = make_channels_last(img)
            return img
        return None

    @staticmethod
    def _get_track_data_sample(data: dict[str, Any]) -> Any:
        tds = data.get("data_samples")
        if isinstance(tds, list):
            if not tds:
                return None
            return tds[0]
        return tds

    @staticmethod
    def _assign_frame(img_batch: torch.Tensor, index: int, new_img: Any) -> None:
        if isinstance(new_img, StreamTensorBase):
            new_img = new_img.wait()
        if isinstance(new_img, np.ndarray):
            new_img = torch.from_numpy(new_img)
        if not isinstance(new_img, torch.Tensor):
            return
        if new_img.ndim == 4 and new_img.shape[0] == 1:
            new_img = new_img.squeeze(0)
        if new_img.ndim != 3:
            return
        if not is_channels_last(new_img):
            new_img = make_channels_last(new_img)
        if new_img.device != img_batch.device:
            new_img = new_img.to(device=img_batch.device, non_blocking=True)
        img_batch[index] = new_img

    def _update_and_draw_trajectories(
        self, img: torch.Tensor, tracking_ids: torch.Tensor, bboxes_tlwh: torch.Tensor
    ) -> torch.Tensor:
        if (not self._plot_trajectories) or tracking_ids.numel() == 0:
            return img

        for (x1, y1, w, h), tid in zip(bboxes_tlwh, tracking_ids):
            tid_i = int(tid)
            cx = int(float(x1 + 0.5 * w))
            cy = int(float(y1 + h))
            pts = self._traj.get(tid_i)
            if pts is None:
                pts = []
                self._traj[tid_i] = pts
            pts.append((cx, cy))
            if len(pts) > self._trajectory_history:
                del pts[: len(pts) - self._trajectory_history]

        for tid_i, pts in list(self._traj.items()):
            if len(pts) < 2:
                continue
            color = vis.get_color(abs(int(tid_i)))
            for (x0, y0), (x1p, y1p) in itertools.pairwise(pts):
                img = vis.plot_line(img, (x0, y0), (x1p, y1p), color=color, thickness=2)
        return img

    def _draw_tracking_and_dets(self, img: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
        if (
            (not self._plot_tracking)
            and (self._plot_all_detections is None)
            and (not self._plot_trajectories)
        ):
            return img

        track_data_sample = self._get_track_data_sample(data)
        vds = getattr(track_data_sample, "video_data_samples", None) if track_data_sample else None
        if vds is None:
            return img

        frame_count = min(img.shape[0], len(vds))
        for frame_index in range(frame_count):
            vs = vds[frame_index]

            if self._plot_all_detections is not None:
                det_inst = getattr(vs, "pred_instances", None)
                dets = getattr(det_inst, "bboxes", None) if det_inst is not None else None
                scores = getattr(det_inst, "scores", None) if det_inst is not None else None
                if isinstance(dets, torch.Tensor) and isinstance(scores, torch.Tensor):
                    img_f = img[frame_index]
                    for det, score in zip(dets, scores):
                        if float(score) >= float(self._plot_all_detections):
                            img_f = vis.plot_rectangle(
                                img=img_f, box=det, color=(64, 64, 64), thickness=1
                            )
                            if float(score) < 0.7:
                                img_f = vis.plot_text(
                                    img_f,
                                    format(float(score), ".2f"),
                                    (int(det[0]), int(det[1])),
                                    1,
                                    1.0,
                                    (255, 255, 255),
                                    thickness=1,
                                )
                    self._assign_frame(img, frame_index, img_f)

            track_inst = getattr(vs, "pred_track_instances", None)
            if track_inst is None:
                continue
            ids = unwrap_tensor(track_inst.instances_id)
            bboxes_tlbr = unwrap_tensor(track_inst.bboxes)
            if (not isinstance(ids, torch.Tensor)) or (not isinstance(bboxes_tlbr, torch.Tensor)):
                continue
            if bboxes_tlbr.numel() == 0:
                continue

            labels = self._track_labels(track_inst, ids)

            track_mask = get_track_mask(track_inst)
            if isinstance(track_mask, torch.Tensor):
                ids = ids[track_mask]
                bboxes_tlbr = bboxes_tlbr[track_mask]
                if labels is not None:
                    labels = labels[track_mask.to(device=labels.device)]
            if ids.numel() == 0:
                continue

            bboxes_tlwh = _batch_tlbrs_to_tlwhs(bboxes_tlbr)

            if self._plot_trajectories:
                img_f = img[frame_index]
                img_f = self._update_and_draw_trajectories(img_f, ids, bboxes_tlwh)
                self._assign_frame(img, frame_index, img_f)

            if self._plot_tracking:
                frame_id_val = int(getattr(vs, "frame_id", frame_index))
                img_f = img[frame_index]
                img_f = vis.plot_tracking(
                    img_f,
                    bboxes_tlwh,
                    ids,
                    frame_id=torch.tensor(frame_id_val),
                    speeds=[],
                    line_thickness=2,
                    ignore_frame_id=True,
                    ignore_tracking_ids=None,
                    ignored_color=(0, 0, 0),
                    draw_tracking_circles=self._draw_tracking_circles,
                )
                img_f = self._draw_referees(img_f, ids, bboxes_tlwh, labels)
                self._assign_frame(img, frame_index, img_f)

        return img

    def _track_labels(self, track_inst: Any, ids: torch.Tensor) -> torch.Tensor | None:
        """Per-track detector class index, or None when the head is single-class.

        Role indices belong to the selected detector profile; ordinary person
        heads leave these unset and draw no role labels.
        """
        if self._referee_label is None and self._goalie_label is None:
            return None
        labels = unwrap_tensor(getattr(track_inst, "labels", None))
        if not isinstance(labels, torch.Tensor) or labels.numel() != ids.numel():
            return None
        return labels.reshape(-1)

    def _draw_referees(
        self,
        img_f: torch.Tensor,
        ids: torch.Tensor,
        bboxes_tlwh: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        """Badge officials so they read as distinct from players at a glance.

        `vis.plot_tracking` has already drawn a per-id coloured box for every
        track; this overprints a REF label on the officials rather than
        recolouring, so the track identity stays readable.
        """
        if labels is None:
            return img_f
        is_referee = torch.zeros_like(labels, dtype=torch.bool)
        for role in (self._referee_label if self._plot_referees else None, self._goalie_label):
            if role is not None:
                is_referee |= labels == role
        if not bool(is_referee.any()):
            return img_f

        img_w = int(img_f.shape[-1] if not is_channels_last(img_f) else img_f.shape[1])
        font_scale = max(0.6, float(img_w) / 1800.0)
        for index in torch.nonzero(is_referee).reshape(-1).tolist():
            x1, y1, w, _ = bboxes_tlwh[index]
            img_f = draw_label_box(
                image=img_f,
                text="GOALIE" if int(labels[index]) == self._goalie_label else "REF",
                center_x=int(float(x1) + float(w) * 0.5),
                bottom_y=int(float(y1)) - int(font_scale * 4),
                font_scale=font_scale,
                fg=(255, 255, 255),
                bg=_REFEREE_COLOR,
                bg_alpha=215,
            )
        return img_f

    def _draw_pose(
        self, img: torch.Tensor, pose_results: Any, pose_inferencer: Any
    ) -> torch.Tensor:
        if not self._plot_pose:
            return img
        if not pose_results or pose_inferencer is None:
            return img

        pose_impl = getattr(pose_inferencer, "inferencer", None)
        vis = getattr(pose_impl, "visualizer", None)
        if vis is None:
            return img

        kpt_thr = self._pose_kpt_thr
        if kpt_thr is None:
            kpt_thr = getattr(pose_inferencer, "filter_args", {}).get("kpt_thr", None)
        if kpt_thr is None:
            kpt_thr = 0.3

        frame_count = min(img.shape[0], len(pose_results))
        for i in range(frame_count):
            frame_res = pose_results[i] if i < len(pose_results) else None
            if not isinstance(frame_res, dict):
                continue
            preds = frame_res.get("predictions")
            if not preds:
                continue
            out_img = vis.add_datasample(
                name="pose results",
                image=img[i],
                data_sample=preds[0],
                clone_image=False,
                draw_gt=False,
                draw_bbox=False,
                show_kpt_idx=False,
                kpt_thr=float(kpt_thr),
            )
            self._assign_frame(img, i, out_img)
        return img

    def _draw_jerseys(self, img: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
        if not self._plot_jersey_numbers:
            return img
        jersey_results = data.get("jersey_results")
        if not jersey_results:
            return img
        track_samples = data.get("data_samples")
        if track_samples is None:
            return img
        if isinstance(track_samples, list):
            if not track_samples:
                return img
            track_samples = track_samples[0]

        vds = getattr(track_samples, "video_data_samples", None)
        if vds is None:
            return img

        frame_count = min(img.shape[0], len(vds), len(jersey_results))
        frame_ids_any = data.get("frame_ids", None)
        if frame_ids_any is None:
            frame_ids_any = data.get("frame_id", None)

        for frame_index in range(frame_count):
            vs = vds[frame_index]
            track_inst = getattr(vs, "pred_track_instances", None)
            if track_inst is None:
                continue
            ids = unwrap_tensor(track_inst.instances_id)
            bboxes_tlbr = unwrap_tensor(track_inst.bboxes)

            if not isinstance(ids, torch.Tensor) or not isinstance(bboxes_tlbr, torch.Tensor):
                continue
            if bboxes_tlbr.numel() == 0:
                continue

            track_mask = get_track_mask(track_inst)
            if isinstance(track_mask, torch.Tensor):
                ids = ids[track_mask]
                bboxes_tlbr = bboxes_tlbr[track_mask]

            tlwh = _batch_tlbrs_to_tlwhs(bboxes_tlbr)

            frame_id_val = frame_index
            frame_id_val = int(getattr(vs, "frame_id", frame_index))

            per_frame = jersey_results[frame_index]
            from hmlib.aspen.player_semantics import SettledJerseys
            from hmlib.vis.player_labels import draw_current_jerseys

            if isinstance(per_frame, SettledJerseys):
                self._assign_frame(
                    img,
                    frame_index,
                    draw_current_jerseys(img[frame_index], ids, tlwh, per_frame),
                )
                continue
            if per_frame:
                for info in per_frame:
                    self._jersey_tracker.observe_tracking_id_number_info(
                        frame_id=frame_id_val, info=info
                    )
            img_frame = self._jersey_tracker.draw(
                image=img[frame_index], tracking_ids=ids, bboxes=tlwh
            )
            self._assign_frame(img, frame_index, img_frame)
        return img

    def _draw_teams(self, img: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
        """Render settled home/away labels published by the team trunk.

        The labels are already aggregated upstream, so this only smooths the
        anchor point and draws. Officials are *not* filtered out on the way
        here: the team trunk does not touch ``track_mask``. It skips them when
        clustering where the detector head can tell them apart, but a track
        the head called a player is clustered and badged like any other.
        """
        if not self._plot_teams:
            return img
        team_results = data.get("team_results")
        if not team_results:
            return img
        track_samples = data.get("data_samples")
        if track_samples is None:
            return img
        if isinstance(track_samples, list):
            if not track_samples:
                return img
            track_samples = track_samples[0]

        vds = getattr(track_samples, "video_data_samples", None)
        if vds is None:
            return img

        frame_count = min(img.shape[0], len(vds), len(team_results))
        for frame_index in range(frame_count):
            vs = vds[frame_index]
            track_inst = getattr(vs, "pred_track_instances", None)
            if track_inst is None:
                continue
            ids = unwrap_tensor(track_inst.instances_id)
            bboxes_tlbr = unwrap_tensor(track_inst.bboxes)
            if not isinstance(ids, torch.Tensor) or not isinstance(bboxes_tlbr, torch.Tensor):
                continue
            if bboxes_tlbr.numel() == 0:
                continue

            track_mask = get_track_mask(track_inst)
            if isinstance(track_mask, torch.Tensor):
                ids = ids[track_mask.to(device=ids.device)]
                bboxes_tlbr = bboxes_tlbr[track_mask.to(device=bboxes_tlbr.device)]
            if ids.numel() == 0:
                continue

            self._team_renderer.update(team_results[frame_index])
            img_frame = self._team_renderer.draw(
                image=img[frame_index],
                tracking_ids=ids,
                bboxes=_batch_tlbrs_to_tlwhs(bboxes_tlbr),
            )
            self._assign_frame(img, frame_index, img_frame)
        return img

    def forward(self, context: dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}

        controls = context.get("shared", {}).get("overlay_controls", {})
        for key, attribute in (
            ("plot_individual_player_tracking", "_plot_tracking"),
            ("plot_tracking_circles", "_draw_tracking_circles"),
            ("plot_trajectories", "_plot_trajectories"),
        ):
            if key in controls:
                setattr(self, attribute, controls[key])

        self._maybe_print_available(context)

        img_any = context.get("img")
        if img_any is None:
            img_any = context.get("original_images")
        img = self._to_batched_image(img_any)
        if img is None:
            return {}

        should_render = (
            self._plot_pose
            or self._plot_jersey_numbers
            or self._plot_teams
            or self._plot_tracking
            or (self._plot_all_detections is not None)
            or self._plot_trajectories
        )
        if not should_render and not self._print_available:
            return {}

        # Copy so in-place writes are safe even if upstream shares storage.
        if should_render:
            img = img.clone()

        pose_inferencer = context.get("pose_inferencer") or context.get("shared", {}).get(
            "pose_inferencer"
        )
        pose_results = context.get("pose_results")
        if pose_results is not None:
            img = self._draw_pose(
                img=img, pose_results=pose_results, pose_inferencer=pose_inferencer
            )

        payload: dict[str, Any] = {
            "data_samples": context.get("data_samples"),
            "jersey_results": context.get("jersey_results"),
            "team_results": context.get("team_results"),
            "frame_ids": context.get("frame_ids"),
            "frame_id": context.get("frame_id"),
        }
        img = self._draw_tracking_and_dets(img=img, data=payload)
        img = self._draw_jerseys(img=img, data=payload)
        img = self._draw_teams(img=img, data=payload)

        # Preserve channels-last tensor for downstream ApplyCamera/VideoOutput.
        img = make_channels_last(img)
        return {"img": wrap_tensor(img)}

    def input_keys(self):
        return {
            "img",
            "original_images",
            "data_samples",
            "pose_results",
            "jersey_results",
            "team_results",
            "frame_ids",
            "frame_id",
            "pose_inferencer",
            "shared",
        }

    def output_keys(self):
        return {"img"}
