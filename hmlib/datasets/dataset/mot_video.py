"""Video dataset helpers for MOT-style tracking and stitching pipelines."""

import contextlib as _contextlib
import os
import threading
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from threading import Lock
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hmlib.log import get_root_logger
from hmlib.tracking_utils.timer import Timer
from hmlib.utils.containers import create_queue
from hmlib.utils.gpu import (
    StreamCheckpoint,
    StreamTensorBase,
    cuda_stream_scope,
    unwrap_tensor,
    wrap_tensor,
)
from hmlib.utils.image import image_height, image_width, make_channels_first, make_channels_last
from hmlib.video.ffmpeg import BasicVideoInfo
from hmlib.video.video_stream import VideoStreamReader

_CPU_DEVICE = torch.device("cpu")


@contextmanager
def optional_with(resource):
    """A context manager that works even if the resource is None."""
    if resource is None:
        # If the resource is None, yield nothing but still enter the with block
        yield None
    else:
        # If the resource is not None, use it as a normal context manager
        with resource as r:
            yield r


class MOTLoadVideoWithOrig(Dataset):  # for inference

    _instance_counter: int = 0

    def __init__(
        self,
        path: str | list[str],
        game_id: str | None = None,
        video_id: int = 1,
        preproc=None,
        max_frames: int = 0,
        batch_size: int = 1,
        start_frame_number: int = 0,
        multi_width_img_info: bool = True,
        embedded_data_loader=None,
        original_image_only: bool = False,
        device: torch.device = _CPU_DEVICE,
        decoder_device: torch.device = _CPU_DEVICE,
        decoder_type: str | None = None,
        device_for_original_image: torch.device = None,
        log_messages: bool = False,
        dtype: torch.dtype = None,
        data_pipeline: Callable | None = None,
        frame_step: int = 1,
        adjust_exposure: float | None = None,
        no_cuda_streams: bool = False,
        async_mode: bool = True,
        prefetch_batches: int = 2,
        checkerboard_input: bool = False,
        source_frame_offset: float = 0,
    ):
        self._instance_id = MOTLoadVideoWithOrig._instance_counter
        MOTLoadVideoWithOrig._instance_counter += 1
        if isinstance(path, list):
            self._path_list = path
        elif path:
            self._path_list = [path]
        else:
            self._path_list = None

        if original_image_only and device_for_original_image is None:
            device_for_original_image = device

        assert (
            not original_image_only or data_pipeline is None
        ), "original_image_only cannot be used with a data_pipeline at this time"

        self._current_path_index = 0
        self._game_id = game_id
        self._no_cuda_streams = no_cuda_streams
        # The delivery device of the letterbox image
        self._device = device
        self._decoder_device = decoder_device
        if decoder_type is not None:
            decoder_type = str(decoder_type).strip().lower()
            if decoder_type in ("", "auto"):
                decoder_type = None
        self._requested_decoder_type = decoder_type
        self._preproc = preproc
        self._logger = get_root_logger()
        self._dtype = dtype
        if frame_step < 1 or source_frame_offset < 0 or start_frame_number < 0:
            raise ValueError("Frame step must be positive and frame offsets nonnegative")
        self._frame_step = frame_step
        self._source_frame_offset = source_frame_offset
        self._next_source_frame = start_frame_number * frame_step + source_frame_offset
        # Requested positions drive the existing seek/processing policy. Only
        # complete decoding from the first chapter proves a source ordinal.
        self._next_exact_source_frame: int | None = None
        self._chapter_frames = None
        self._data_pipeline = data_pipeline
        self._log_messages = log_messages
        self._device_for_original_image = device_for_original_image
        self._start_frame_number = start_frame_number
        self._adjust_exposure = adjust_exposure
        self._multi_width_img_info = multi_width_img_info
        self._original_image_only = original_image_only
        self._async_mode = bool(async_mode)
        self._prefetch_batches = max(1, int(prefetch_batches))
        self._pending_worker_requests = 0
        self._checkerboard_input = bool(checkerboard_input)
        self.width_t = None
        self.height_t = None
        self._scale_color_tensor = None
        self._count = torch.tensor([0], dtype=torch.int64)
        self._next_frame_id = torch.tensor(
            [start_frame_number + source_frame_offset], dtype=torch.int32
        )
        if self._next_frame_id == 0:
            # frame number is 1-based
            self._next_frame_id = 1
        self.video_id = torch.tensor([video_id], dtype=torch.int32)
        self._last_size = None
        self._max_frames = max_frames
        self._batch_size = batch_size
        self._timer = Timer()
        self._timer_counter = 0
        self.vn = None
        self.vw = None
        self.vh = None
        self.cap = None
        self._vid_iter = None
        self._sample_decoded_timing = []
        self._mapping_offset = None
        self._thread = None
        self._scale_inscribed_to_original = None
        self._embedded_data_loader = embedded_data_loader
        self._embedded_data_loader_iter = None
        self._cuda_stream = None
        self._decoder_type = None
        self._next_counter = 0
        self._frame_read_count = 0
        self._video_info = None
        self._seek_lock = Lock()
        # Optional per-sample debug metadata propagated from embedded loaders (e.g., StitchDataset).
        self._embedded_debug: dict[str, Any] | None = None
        self._to_worker_queue = create_queue(
            mp=False,
            name="mot-video-to-worker" + ("-EDL" if embedded_data_loader is not None else ""),
            warn_after=10.0,
        )
        self._from_worker_queue = create_queue(
            mp=False,
            name="mot-video-from-worker" + ("-EDL" if embedded_data_loader is not None else ""),
            warn_after=10.0,
        )

        self.load_video_info()

    # ------------------------------------------------------------------
    # Debug helpers: RGB stats and checkerboard generation
    # ------------------------------------------------------------------
    @staticmethod
    def compute_rgb_stats(
        img: torch.Tensor, cpu: bool = False
    ) -> dict[str, tuple[float, float, float]]:
        """Compute per-channel min, max, mean for R/G/B over an image or batch.

        Accepts tensors with layout (B, C, H, W), (B, H, W, C), (C, H, W) or
        (H, W, C). Only the first three channels are considered.
        """
        if img is None:
            raise ValueError("compute_rgb_stats expected a tensor, got None")
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"compute_rgb_stats expects torch.Tensor, got {type(img)}")

        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.ndim != 4:
            raise ValueError(
                f"compute_rgb_stats expects 3D/4D tensor, got shape {tuple(img.shape)}"
            )

        # Normalize to channels-last for simpler slicing
        img_cl = make_channels_last(img)
        # Work in float for accurate mean; clamp for integer-like tensors.
        work = img_cl[..., :3].to(torch.float32)
        flat = work.reshape(-1, 3)
        min_vals, _ = flat.min(dim=0)
        max_vals, _ = flat.max(dim=0)
        mean_vals = flat.mean(dim=0)
        return {
            "min": min_vals.cpu() if cpu else min_vals,
            "max": max_vals.cpu() if cpu else max_vals,
            "mean": mean_vals.cpu() if cpu else mean_vals,
        }

    @staticmethod
    def check_rgb_stats(
        img: torch.Tensor,
        reference: dict[str, tuple[float, float, float]],
        *,
        atol: float = 1e-3,
        rtol: float = 1e-5,
    ) -> tuple[bool, bool]:
        """Compare current RGB stats against a reference.

        Returns (changed, unchanged) where:
          - changed: True if any per-channel min/max/mean differs beyond tolerance.
          - unchanged: the logical negation of 'changed'.
        """
        if reference is None:
            raise ValueError("check_rgb_stats requires a non-None reference stats dict.")
        current = MOTLoadVideoWithOrig.compute_rgb_stats(img)

        def _to_tensor(vals: tuple[float, float, float]) -> torch.Tensor:
            return torch.tensor(vals, dtype=torch.float32)

        changed = False
        for key in ("min", "max", "mean"):
            ref_vals = reference.get(key)
            cur_vals = current.get(key)
            if ref_vals is None or cur_vals is None:
                changed = True
                break
            ref_t = ref_vals
            cur_t = cur_vals
            if not torch.allclose(ref_t, cur_t, atol=atol, rtol=rtol):
                changed = True
                break
        return changed, not changed

    @staticmethod
    def make_checkerboard_like(
        img: torch.Tensor,
        tile_size: int = 32,
    ) -> torch.Tensor:
        """Generate a checkerboard pattern with the same shape/device/dtype as ``img``.

        The pattern alternates 0 and 255 across tiles of size ``tile_size``.
        Only the first three channels are written; any extra channels (e.g., alpha)
        are left unchanged when present.
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"make_checkerboard_like expects torch.Tensor, got {type(img)}")
        squeeze_batch = False
        if img.ndim == 3:
            img = img.unsqueeze(0)
            squeeze_batch = True
        if img.ndim != 4:
            raise ValueError(
                f"make_checkerboard_like expects 3D/4D tensor, got shape {tuple(img.shape)}"
            )

        # Track whether the original layout was channels-last so we can restore it.
        was_channels_last = img.shape[-1] in (3, 4)
        # Normalize to channels-last for easier pattern creation
        cl = make_channels_last(img)
        h, w = cl.shape[1], cl.shape[2]
        device = cl.device

        # Build a 0/1 checkerboard over HxW
        ys = torch.arange(h, device=device) // max(1, tile_size)
        xs = torch.arange(w, device=device) // max(1, tile_size)
        pattern = (ys.view(-1, 1) + xs.view(1, -1)) % 2  # HxW in {0,1}
        pattern = pattern.to(dtype=torch.float32)
        pattern = pattern * 255.0  # scale to [0,255]

        # Expand to (B, H, W, 3)
        pattern = pattern.unsqueeze(0).unsqueeze(-1)  # 1xHxWx1
        pattern = pattern.expand(cl.shape[0], h, w, min(3, cl.shape[-1]))

        out = cl.clone()
        out[..., : pattern.shape[-1]] = pattern

        # Restore original dtype
        if torch.is_floating_point(img):
            out = (out / 255.0).to(dtype=img.dtype)
        else:
            out = out.clamp(0.0, 255.0).to(dtype=img.dtype)

        # Restore original layout
        if not was_channels_last:
            out = make_channels_first(out)
        if squeeze_batch:
            out = out.squeeze(0)
        return out

    def load_video_info(self) -> None:
        if not self._path_list:
            return
        if self._video_info is not None:
            return
        assert self._embedded_data_loader is None
        self._video_info = BasicVideoInfo(",".join(self._path_list))
        self._chapter_frames = (
            [self._video_info.frame_count]
            if len(self._path_list) == 1
            else [BasicVideoInfo(str(path)).frame_count for path in self._path_list]
        )

    def _chapter_position(self, source_frame: float) -> tuple[int, float]:
        for index, count in enumerate(self._chapter_frames):
            if source_frame < count:
                return index, source_frame
            source_frame -= count
        raise ValueError("Requested source frame is beyond the end of the video playlist")

    def _open_video(self, *, continue_exact_ordinal: bool = False):
        self.load_video_info()
        if self._embedded_data_loader is None:
            if self._requested_decoder_type is not None:
                self._decoder_type = self._requested_decoder_type
            else:
                self._decoder_type = "cv2"
                if self._decoder_device is not None and self._decoder_device.type == "cuda":
                    # Preserve the old NVIDIA path, but prefer the AMD FFmpeg/VAAPI
                    # backend when running on ROCm.
                    from hmlib.utils.torch_backend import is_rocm_backend

                    self._decoder_type = "pyamdcodec" if is_rocm_backend() else "pynvcodec"
            self._current_path_index, local_frame = self._chapter_position(self._next_source_frame)
            file_path = str(self._path_list[self._current_path_index])
            if not os.path.isfile(file_path):
                raise RuntimeError(f"Video file {file_path} does not exist")
            self.cap = VideoStreamReader(
                filename=file_path,
                type=self._decoder_type,
                batch_size=1 if self._frame_step > 1 else self._batch_size,
                device=self._decoder_device,
            )
            self.vw = self.cap.width
            self.vh = self.cap.height
            assert self._video_info.width == self.vw and self._video_info.height == self.cap.height
            self.vn = self._video_info.frame_count
            if not self.vn:
                raise RuntimeError(
                    f"Video {self._path_list[self._current_path_index]!s} either does not exist or has no usable video content"
                )
            if local_frame:
                self.cap.seek(frame_number=local_frame)
            self._vid_iter = iter(self.cap)
            if not continue_exact_ordinal:
                self._next_exact_source_frame = (
                    0 if self._current_path_index == 0 and not local_frame else None
                )
            if (
                local_frame
                or getattr(self._vid_iter, "source_ordinals_complete", False) is not True
            ):
                self._next_exact_source_frame = None
        else:
            self.vn = len(self._embedded_data_loader)
        if self.vn is not None and self._log_messages:
            print(f"Lenth of the video: {self.vn:d} frames")

    def _close_video(self):
        if self.cap is not None:
            self.cap.close()
            self.cap = None
        if self._embedded_data_loader is not None:
            self._embedded_data_loader.close()

    def goto_next_video(self, *, decoded_to_eof: bool = False) -> bool:
        if not self._path_list:
            return False
        if self._current_path_index + 1 >= len(self._path_list):
            return False
        assert self._embedded_data_loader is None
        if (
            not decoded_to_eof
            or getattr(self._vid_iter, "source_ordinals_complete", False) is not True
        ):
            self._next_exact_source_frame = None
        self._next_source_frame = sum(self._chapter_frames[: self._current_path_index + 1])
        self._current_path_index += 1
        self._close_video()
        self._open_video(continue_exact_ordinal=True)
        return True

    @property
    def fps(self) -> float:
        if self._embedded_data_loader is not None:
            return self._embedded_data_loader.fps
        return self._video_info.fps / self._frame_step

    @property
    def bit_rate(self) -> int:
        if self._embedded_data_loader is not None:
            return self._embedded_data_loader.bit_rate
        return self._video_info.bit_rate

    @property
    def dataset(self):
        return self

    @property
    def batch_size(self):
        if self._embedded_data_loader is not None:
            return self._embedded_data_loader.batch_size * self._batch_size
        return self._batch_size

    @property
    def letterbox_dw_dh(self):
        return self.letterbox_dw, self.letterbox_dh

    def _next_frame_worker(self):
        try:
            self._open_video()

            # Create the CUDA stream outside of the main loop
            if (
                self._async_mode
                and not self._no_cuda_streams
                and self._cuda_stream is None
                and self._device.type == "cuda"
            ):
                self._cuda_stream = torch.cuda.Stream(self._device)

            while True:
                cmd = self._to_worker_queue.get()
                if cmd.startswith("seek:"):
                    self._seek_file_frame(int(cmd.split(":", 1)[1]))
                    self._from_worker_queue.put("seek_ok")
                    continue
                if cmd != "ok":
                    break
                next_batch = self._get_next_batch()
                self._from_worker_queue.put(next_batch)
        except Exception as ex:  # noqa: BLE001 -- marshal worker failures to the consumer
            if not isinstance(ex, StopIteration):
                print(ex)
                traceback.print_exc()
            self._from_worker_queue.put(ex)
            return
        finally:
            self._close_video()
        self._from_worker_queue.put(StopIteration())
        return

    def _start_worker(self):
        assert self._async_mode
        self._thread = threading.Thread(
            target=self._next_frame_worker, name="MOTVideoNextFrameWorker"
        )
        self._next_counter = 0
        self._pending_worker_requests = 0
        self._thread.start()

    def _request_next_batch(self) -> None:
        if not self._async_mode or self._thread is None:
            return
        self._to_worker_queue.put("ok")
        self._pending_worker_requests += 1

    def _refill_prefetch(self) -> None:
        if not self._async_mode or self._thread is None:
            return
        while self._pending_worker_requests < self._prefetch_batches:
            self._request_next_batch()

    def _reset_prefetch(self) -> None:
        self._pending_worker_requests = 0

    def close(self):
        if self._async_mode:
            if self._thread is not None:
                self._to_worker_queue.put("stop")
                self._thread.join()
                self._thread = None
            self._reset_prefetch()
        else:
            # In synchronous mode, just close any open video or embedded loader.
            self._close_video()

        # Ensure any embedded loader (e.g., StitchDataset) is also shut down so
        # its coordinator/worker threads do not keep the process alive after
        # hmtrack exits with an error.
        if self._embedded_data_loader is not None and hasattr(self._embedded_data_loader, "close"):
            self._embedded_data_loader.close()

        if self._path_list is not None:
            print(
                f"MOTLoadVideoWithOrig delivered {self._frame_read_count} frames for file {self._path_list[self._current_path_index]!s}"
            )

    def __iter__(self):
        self._timer = Timer()
        if self._embedded_data_loader is not None:
            self._embedded_data_loader_iter = iter(self._embedded_data_loader)
        if self._async_mode:
            if self._thread is None:
                self._start_worker()
            self._refill_prefetch()
        else:
            # Synchronous mode: open video lazily on first iteration if needed.
            if self.cap is None:
                self._open_video()
        return self

    def _read_source_batch(self):
        try:
            image = next(self._vid_iter)
        except StopIteration:
            if not self.goto_next_video(decoded_to_eof=True):
                raise
            image = next(self._vid_iter)
        count = len(image) if image.ndim == 4 else 1
        if getattr(self._vid_iter, "source_ordinals_complete", False) is not True:
            self._next_exact_source_frame = None
        exact_frames = (
            [self._next_exact_source_frame + index for index in range(count)]
            if self._next_exact_source_frame is not None
            else [None] * count
        )
        if self._next_exact_source_frame is not None:
            self._next_exact_source_frame += count
        decoded = getattr(self._vid_iter, "last_frame_timing", None)
        if decoded is not None and len(decoded) != count:
            raise ValueError("Decoded timestamps do not align with the source frame batch")
        timing = (
            [item.document(self._current_path_index) for item in decoded]
            if decoded is not None
            else [None] * count
        )
        frames = [self._next_source_frame + index for index in range(count)]
        self._next_source_frame += count
        return image, frames, timing, exact_frames

    def _read_next_image(self):
        img = None
        if self.cap is not None:
            if self._frame_step == 1:
                (
                    img,
                    self._sample_source_frames,
                    self._sample_decoded_timing,
                    self._sample_exact_source_frames,
                ) = self._read_source_batch()
            else:
                (
                    images,
                    self._sample_source_frames,
                    self._sample_decoded_timing,
                    self._sample_exact_source_frames,
                ) = ([], [], [], [])
                for _ in range(self._batch_size):
                    for _ in range(self._frame_step):
                        image, frames, timing, exact_frames = self._read_source_batch()
                    images.append(image[0] if image.ndim == 4 else image)
                    self._sample_source_frames.extend(frames)
                    self._sample_decoded_timing.extend(timing)
                    self._sample_exact_source_frames.extend(exact_frames)
                img = np.stack(images) if isinstance(images[0], np.ndarray) else torch.stack(images)
        else:
            if self._batch_size == 1:
                item = next(self._embedded_data_loader_iter)
                # When wrapping a stitching dataset in debug mode, the embedded loader
                # may return (img, debug_info). Preserve the debug metadata so it can
                # be attached to the final data dict later.
                if isinstance(item, tuple) and len(item) == 2:
                    img, debug = item
                    self._embedded_debug = debug
                else:
                    img = item
                    self._embedded_debug = None
            else:
                imgs = []
                for _ in range(self._batch_size):
                    imgs.append(next(self._embedded_data_loader_iter))
                assert imgs[0].ndim == 4
                if isinstance(imgs[0], np.ndarray):
                    img = np.concatenate(imgs, axis=0)
                else:
                    img = torch.cat(imgs, dim=0)
        assert img is not None
        if self.vw is None:
            self.vw = image_width(img)
            self.vh = image_height(img)
        return True, img

    def _is_cuda(self):
        return self._device.type == "cuda"

    def _get_next_batch(self):
        current_count = self._count.item()
        if current_count >= len(self) * self._batch_size or (
            self._max_frames and current_count >= self._max_frames
        ):
            raise StopIteration

        cuda_stream = self._cuda_stream

        with cuda_stream_scope(cuda_stream), torch.no_grad():

            # BEGIN READ NEXT IMAGE
            while True:
                try:
                    prof = getattr(self, "_profiler", None)
                    rctx = (
                        prof.rf("dataloader.read_next_image")
                        if getattr(prof, "enabled", False)
                        else _contextlib.nullcontext()
                    )
                    with rctx:
                        res, img0 = self._read_next_image()
                    embedded_debug = self._embedded_debug
                    self._embedded_debug = None
                    if not res or img0 is None:
                        print(f"Error loading frame: {self._count + self._start_frame_number}")
                        raise StopIteration()
                    else:
                        break
                except StopIteration:
                    if self.goto_next_video():
                        continue
                    raise
            # END READ NEXT IMAGE

            if isinstance(img0, np.ndarray):
                img0 = torch.from_numpy(img0)

            if img0.ndim == 3:
                assert self._batch_size == 1
                img0 = img0.unsqueeze(0)
            assert img0.ndim == 4

            if self._preproc is not None:
                prof = getattr(self, "_profiler", None)
                pctx = (
                    prof.rf("dataloader.preproc")
                    if getattr(prof, "enabled", False)
                    else _contextlib.nullcontext()
                )
                with pctx:
                    img0 = self._preproc(img0)

            if self._device.type != "cpu" and img0.device != self._device:
                img0 = img0.to(self._device, non_blocking=True)

            if (
                self._adjust_exposure is not None
                and self._adjust_exposure != 0
                and self._adjust_exposure != 1
            ):
                if isinstance(img0, StreamTensorBase):
                    img0 = img0.get()
                if not torch.is_floating_point(img0.dtype):
                    img0 = img0.to(torch.float, non_blocking=True)
                img0 = img0 * self._adjust_exposure

            data_item: dict[str, Any] = {}
            if self._embedded_data_loader is None:
                data_item["source_frame_ids"] = list(self._sample_exact_source_frames)
                data_item["decoded_timing"] = list(self._sample_decoded_timing)
                from hmlib.video.frame_timing import timing_ns

                data_item["decoded_pts_ns"] = [
                    timing_ns(item) for item in self._sample_decoded_timing
                ]
                data_item["pts_ns"] = [
                    round(frame * 1e9 / self._video_info.fps)
                    for frame in self._sample_source_frames
                ]

            # Optional checkerboard + RGB stats debug mode for MOTLoadVideoWithOrig.
            checkerboard_image: torch.Tensor | None = None
            if self._checkerboard_input:
                checkerboard_image = MOTLoadVideoWithOrig.make_checkerboard_like(img0)
                # Compute stats on the tensor that will flow downstream.
                stats = MOTLoadVideoWithOrig.compute_rgb_stats(checkerboard_image)
                data_item.setdefault("debug_rgb_stats", {})["mot_input"] = stats

            # Does this need to be in imgs_info this way as an array?
            ids = torch.tensor(
                [int(self._next_frame_id + i) for i in range(len(img0))],
                dtype=torch.int64,
            )

            fps_value = self.fps
            if fps_value and fps_value > 0:
                fps_scalar = float(fps_value)
                data_item["hm_real_time_fps"] = [fps_scalar for _ in range(len(ids))]

            self._next_frame_id += len(ids)

            if self._data_pipeline is not None:
                img0 = unwrap_tensor(img0)

                original_img0 = img0

                if torch.is_floating_point(img0) and img0.dtype != self._dtype:
                    img0 = img0.to(dtype=self._dtype, non_blocking=True)

                data_item.update(
                    {
                        "img": make_channels_last(img0),
                        "img_info": {"frame_id": ids[0]},
                        "img_prefix": None,
                        "img_id": ids,
                    }
                )
                # Propagate any debug metadata coming from an embedded loader (e.g., StitchDataset).
                if embedded_debug is not None:
                    data_item.setdefault("debug_rgb_stats", {})["stitch"] = embedded_debug
                dctx = (
                    prof.rf("dataloader.data_pipeline")
                    if getattr(prof, "enabled", False)
                    else _contextlib.nullcontext()
                )
                with dctx:
                    pipeline_result = self._data_pipeline(data_item.copy())
                assert "img" not in pipeline_result
                data_item["img"] = pipeline_result.pop("inputs")
                data_item.update(pipeline_result)
                img = data_item["img"]
            else:
                if isinstance(img0, StreamTensorBase):
                    img0 = img0.wait()
                # Synchronizing here makes it a lot faster for some reason
                # cuda_stream.synchronize()

                original_img0 = img0

                if not self._original_image_only:
                    if self._dtype is not None and img0.dtype != self._dtype:
                        img0 = img0.to(self._dtype, non_blocking=True)
                    assert False  # Don't use this path anymore
                    img = self.make_letterbox_images(make_channels_first(img0))
                else:
                    if self._dtype is not None and self._dtype != original_img0.dtype:
                        original_img0 = original_img0.to(self._dtype)
                    img = original_img0
                data_item.update(
                    {
                        "img": make_channels_last(img),
                        "img_id": ids,
                    }
                )

            if self.width_t is None:
                self.width_t = torch.tensor([image_width(img)], dtype=torch.int64)
                self.height_t = torch.tensor([image_height(img)], dtype=torch.int64)

            original_img0 = make_channels_first(original_img0)

            path = str(self._path_list[self._current_path_index]) if self._path_list else None
            imgs_info = [
                (self.height_t.repeat(len(ids)) if self._multi_width_img_info else self.height_t),
                (self.width_t.repeat(len(ids)) if self._multi_width_img_info else self.width_t),
                ids,
                self.video_id.repeat(len(ids)),
                [path if path is not None else self._game_id],
            ]

        self._count += self._batch_size

        # BEGIN _wrap_original_image
        # Do this last, after any cuda events are taken for the non-original image
        def _wrap_original_image(
            orig_img: torch.Tensor,
        ) -> StreamCheckpoint | torch.Tensor:
            if checkerboard_image is not None:
                orig_img = checkerboard_image.to(
                    device=orig_img.device, dtype=orig_img.dtype, non_blocking=True
                )
            if (
                not self._original_image_only
                and self._device_for_original_image is not None
                and self._device_for_original_image != orig_img.device
            ):
                orig_img = orig_img.to(device=self._device_for_original_image, non_blocking=True)
            return wrap_tensor(orig_img)

        # END _wrap_original_image

        if self._data_pipeline is not None:
            assert isinstance(data_item["img"], torch.Tensor)
            data_item["img"] = StreamCheckpoint(tensor=data_item["img"])
            prof = getattr(self, "_profiler", None)
        if self._original_image_only:
            if not isinstance(original_img0, StreamTensorBase):
                original_img0 = _wrap_original_image(original_img0)
            data_item.update({"img": original_img0, "imgs_info": imgs_info, "frame_ids": ids})
        else:
            data_item.update(
                {
                    "original_images": _wrap_original_image(original_img0),
                    "img": wrap_tensor(img),
                    "imgs_info": imgs_info,
                    "ids": ids,
                }
            )
            # assert False # Don't use this path anymore
            # return _wrap_original_image(original_img0), img, None, imgs_info, ids
        return data_item

    def __next__(self):
        with self._seek_lock:
            self._timer.tic()
            prof = getattr(self, "_profiler", None)
            qctx = (
                prof.rf("dataloader.dequeue")
                if getattr(prof, "enabled", False)
                else _contextlib.nullcontext()
            )
            with qctx:
                if self._async_mode:
                    rctx = (
                        prof.rf("dataloader.prefetch_refill")
                        if getattr(prof, "enabled", False)
                        else _contextlib.nullcontext()
                    )
                    with rctx:
                        self._refill_prefetch()
                    wctx = (
                        prof.rf("dataloader.dequeue_wait")
                        if getattr(prof, "enabled", False)
                        else _contextlib.nullcontext()
                    )
                    with wctx:
                        results = self._from_worker_queue.get()
                    if self._pending_worker_requests > 0:
                        self._pending_worker_requests -= 1
                    if not isinstance(results, Exception):
                        with rctx:
                            self._refill_prefetch()
                    else:
                        self._reset_prefetch()
                else:
                    try:
                        results = self._get_next_batch()
                    except Exception as ex:  # noqa: BLE001 -- common sync/async error propagation
                        results = ex
            if isinstance(results, Exception):
                if not isinstance(results, StopIteration):
                    print(results)
                    traceback.print_exc()
                self.close()
                raise results
            self._timer.toc()

            self._timer_counter += self._batch_size
            if self._log_messages and self._next_counter and self._next_counter % 20 == 0:
                self._logger.info(
                    f"Video Dataset frame delivery {self._timer_counter} ({self._batch_size * 1.0 / max(1e-5, self._timer.average_time):.2f} fps)"
                )
                if self._next_counter and self._next_counter % 100 == 0:
                    self._timer = Timer()
                    self._timer_counter = 0
            self._next_counter += 1
            self._frame_read_count += 1
            return results

    def tell(self) -> int:
        """
        Get which frame the dataset is on (which one it will load next)
        """
        return self._next_frame_id.clone()

    def _seek_file_frame(self, frame_number: int) -> None:
        logical_frame = frame_number - int(self._source_frame_offset)
        if logical_frame < 0:
            raise ValueError("Seek precedes the synchronized source origin")
        source_frame = logical_frame * self._frame_step + self._source_frame_offset
        self._chapter_position(source_frame)
        self._close_video()
        self._next_source_frame = source_frame
        self._open_video()
        self._next_frame_id = torch.tensor([frame_number], dtype=torch.int32)

    def seek(self, frame_number: int) -> None:
        """
        Seek to the frame number
        """
        with self._seek_lock:
            assert frame_number > 0  # 1-based
            if frame_number == self._next_frame_id:
                return
            if self._async_mode:
                self._to_worker_queue.put(f"seek:{frame_number}")
                self._reset_prefetch()
                while True:
                    response = self._from_worker_queue.get()
                    if isinstance(response, Exception):
                        raise response
                    if isinstance(response, str) and response == "seek_ok":
                        break
                        # otherwise it's probably a queued frame, so discard it
                self._refill_prefetch()
            else:
                # Best-effort synchronous seek for file-backed videos; embedded loaders are unsupported.
                if self._embedded_data_loader is not None:
                    raise NotImplementedError(
                        "seek is not supported for embedded data loaders in sync mode"
                    )
                self._seek_file_frame(int(frame_number))
        return

    def __len__(self):
        if self._embedded_data_loader is not None:
            return len(self._embedded_data_loader)
        if self.vn is None:
            self.vn = self._video_info.frame_count
        return self.vn // self._frame_step // self._batch_size  # delivered frames

    # Optional: external wiring for profiler instance
    def set_profiler(self, profiler):
        self._profiler = profiler
