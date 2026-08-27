"""Camera post-processing utilities for arena estimation and video output.

This module computes the play/arena box from rink profiles and input frames,
configures `HockeyMON` video geometry, and optionally drives `VideoOutput`
for camera debugging/training videos. Camera play tracking logic is handled
by `PlayTrackerPlugin` and the underlying `PlayTracker` module.
"""

from __future__ import absolute_import, division, print_function

import argparse
from typing import Any, Dict, Optional

import torch

from hmlib.aspen.plugins.postprocess_plugin import CamPostProcessPlugin
from hmlib.builder import HM


@HM.register_module()
class CamTrackPostProcessor(CamPostProcessPlugin):
    """
    Backwards-compatible wrapper around CamPostProcessPlugin.

    This class preserves the legacy CamTrackPostProcessor constructor used by
    older code paths while delegating all behaviour to CamPostProcessPlugin,
    which now owns the implementation.
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        args: argparse.Namespace,
        device: torch.device,
        fps: float,
        save_dir: str,
        output_video_path: Optional[str],
        camera_name: str,
        original_clip_box,
        video_out_pipeline: Dict[str, Any],
        save_frame_dir: Optional[str] = None,
        data_type: str = "mot",
        postprocess: bool = True,
        video_out_device: Optional[torch.device] = None,
        no_cuda_streams: bool = False,
    ):
        super().__init__(enabled=True)
        self._configure(
            opt=opt,
            args=args,
            device=device,
            fps=fps,
            save_dir=save_dir,
            output_video_path=output_video_path,
            camera_name=camera_name,
            original_clip_box=original_clip_box,
            video_out_pipeline=video_out_pipeline,
            save_frame_dir=save_frame_dir,
            data_type=data_type,
            postprocess=postprocess,
            video_out_device=video_out_device,
            no_cuda_streams=no_cuda_streams,
        )
