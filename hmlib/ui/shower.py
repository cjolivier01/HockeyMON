"""Asynchronous image viewer for debugging and live previews.

`Shower` consumes tensors or numpy arrays from a queue and displays them
via OpenCV or (optionally) Tkinter windows.

@see @ref hmlib.ui.show "show" for simpler one-off display helpers.
"""

import contextlib
import queue
import threading
import time
import tkinter as tk
from collections import OrderedDict
from typing import Any

from hmlib.ui.display_env import sanitize_display_env_for_cv2

sanitize_display_env_for_cv2()

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

from hmlib.log import get_root_logger
from hmlib.ui.headless_preview import (
    BrowserPreviewServer,
    FFmpegLivePublisher,
    has_local_display,
    mask_stream_url,
    resolve_youtube_stream_url,
    validate_youtube_stream_url,
)
from hmlib.utils.containers import create_queue
from hmlib.utils.gpu import StreamTensorBase, cuda_stream_scope, unwrap_tensor, wrap_tensor
from hmlib.utils.image import make_channels_last, make_visible_image

try:
    from hockeymon.core import show_cuda_tensor
except ImportError:  # pragma: no cover - optional native extension
    show_cuda_tensor = None

from .show import cv2_has_opengl, show_gpu_tensor

# from .tk import get_tk_root


def _is_rocm_runtime() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _native_cuda_preview_available() -> bool:
    return show_cuda_tensor is not None and not _is_rocm_runtime()


def _opencv_cuda_preview_available() -> bool:
    return not _is_rocm_runtime()


class ImageDisplayer:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        # Initially set up with an empty image
        self.image_label = tk.Label(master, text="No Image")
        self.image_label.pack()

    def display(self, tensor):
        # Convert the PyTorch tensor to a PIL Image
        image = Image.fromarray(make_channels_last(tensor).cpu().numpy().astype("uint8"))

        # Convert the PIL Image to a format Tkinter can use
        tk_image = ImageTk.PhotoImage(image)

        # Update the label with the new image
        self.image_label.configure(image=tk_image)

        # Keep a reference, avoid garbage collection
        self.image_label.image = tk_image


class Shower:
    def __init__(
        self,
        label: str,
        show_scaled: float | None = None,
        max_size: int = 1,
        fps: float | None = None,
        cache_on_cpu: bool = False,
        logger=None,
        use_tk: bool = False,
        allow_gpu_gl: bool = True,
        profiler: Any = None,
        step: int = 1,
        hold_tensor_ref: bool = True,
        skip_frame_when_full: bool = False,
        drop_oldest_when_full: bool = False,
        enable_local_display: bool = True,
        show_youtube: bool = False,
        youtube_stream_url: str | None = None,
        youtube_stream_key: str | None = None,
        headless_preview_host: str = "0.0.0.0",
        headless_preview_port: int = 0,
        always_stream: bool = False,
    ):
        self._label = label
        self._allow_gpu_gl = allow_gpu_gl
        self._show_scaled = show_scaled
        self._max_size: int = max(max_size, 1)
        self._fps = fps
        self._cache_on_cpu = cache_on_cpu
        self._stream = None
        self._profiler = profiler
        if self._fps is not None:
            self._label += " (" + str(self._fps) + " fps)"
        self._cv2_has_opengl_support = cv2_has_opengl()
        self._opencv_cuda_preview = _opencv_cuda_preview_available()
        self._step: int = step
        self._iter: int = 0
        self._next_frame_time = None
        self._skip_frame_when_full = skip_frame_when_full
        self._drop_oldest_when_full = drop_oldest_when_full
        self._use_tk = use_tk
        self._tk_displayer = None
        # Holds a ref to a displayed tensor so the memory pointer is valid
        self._hold_tensor_ref = hold_tensor_ref
        self._displayed_tensor: torch.Tensor | None = None
        self._requested_local_display = bool(enable_local_display)
        self._has_local_display = has_local_display()
        self._enable_local_display = self._requested_local_display and self._has_local_display
        self._always_stream = bool(always_stream)
        self._native_cuda_preview = _native_cuda_preview_available()
        self._headless_preview: BrowserPreviewServer | None = None
        self._show_youtube = bool(show_youtube)
        self._youtube_publish_failed = False
        self._youtube_stream_url = (
            resolve_youtube_stream_url(youtube_stream_url or "", youtube_stream_key)
            if self._show_youtube
            else None
        )
        if self._show_youtube and self._youtube_stream_url is not None:
            validate_youtube_stream_url(self._youtube_stream_url)
        self._youtube_publisher: FFmpegLivePublisher | None = (
            FFmpegLivePublisher(
                output_url=self._youtube_stream_url,
                label=self._label,
                fps=self._fps if self._fps is not None else 30.0,
                logger=logger,
                profiler=self._profiler,
            )
            if self._show_youtube and self._youtube_stream_url
            else None
        )
        if self._requested_local_display and not self._has_local_display:
            self._headless_preview = BrowserPreviewServer(
                label=self._label,
                host=headless_preview_host,
                port=headless_preview_port,
                logger=logger,
                profiler=self._profiler,
            )
        self._logger = logger if logger is not None else get_root_logger()
        if self._enable_local_display and _is_rocm_runtime():
            self._logger.warning(
                "ROCm local preview is using the CPU OpenCV display path because native "
                "CUDA/OpenGL preview interop is unavailable on this runtime."
            )
        self._q = create_queue(mp=False)
        self._thread = threading.Thread(target=self._worker)
        self._thread.start()

    def get_progress_stream_entries(self) -> OrderedDict[str, str]:
        entries: OrderedDict[str, str] = OrderedDict()
        if self._headless_preview is not None:
            preview_url = self._headless_preview.preferred_url
            if preview_url:
                entries["Preview URL"] = preview_url
        if self._youtube_stream_url is not None:
            entries["Publish URL"] = mask_stream_url(self._youtube_stream_url)
        return entries

    def update_progress_table(self, table_map: Any) -> None:
        for key in ("Preview URL", "Publish URL"):
            table_map.pop(key, None)
        for key, value in self.get_progress_stream_entries().items():
            table_map[key] = value

    def _prof_ctx(self, name: str):
        prof = self._profiler
        if prof is not None and getattr(prof, "enabled", False):
            return prof.rf(name)
        return contextlib.nullcontext()

    def _do_show(self, img: torch.Tensor | np.ndarray):
        with self._prof_ctx("shower._do_show"), cuda_stream_scope(self._stream):
            if self._use_tk and self._tk_displayer is None:
                # root = get_tk_root()
                # self._tk_displayer = ImageDisplayer(root)
                assert False

            if img.ndim == 3:
                if isinstance(img, torch.Tensor | StreamTensorBase):
                    img = img.unsqueeze(0)
                elif isinstance(img, np.ndarray):
                    img = np.expand_dims(img, axis=0)

            img = unwrap_tensor(img)
            for s_img in img:
                if self._headless_preview is not None:
                    with self._prof_ctx("shower.browser_preview"):
                        self._headless_preview.publish(
                            s_img,
                            show_scaled=self._show_scaled,
                            require_clients=not self._always_stream,
                        )
                if self._youtube_publisher is not None:
                    try:
                        with self._prof_ctx("shower.live_publish"):
                            self._youtube_publisher.write_frame(
                                s_img, show_scaled=self._show_scaled
                            )
                    except (BrokenPipeError, OSError, RuntimeError, ValueError) as exc:
                        if not self._youtube_publish_failed:
                            self._logger.warning(
                                "Live preview publish stopped for %s: %s",
                                self._label,
                                exc,
                            )
                            self._youtube_publish_failed = True
                        try:
                            self._youtube_publisher.close()
                        except (BrokenPipeError, OSError, RuntimeError) as close_exc:
                            self._logger.debug(
                                "Ignoring live preview close failure for %s after publisher "
                                "error: %s",
                                self._label,
                                close_exc,
                            )
                        self._youtube_publisher = None
                if not self._enable_local_display:
                    continue
                with self._prof_ctx("shower.local_display"):
                    if self._use_tk:
                        self._tk_displayer.display(s_img)
                    else:
                        if (
                            isinstance(s_img, torch.Tensor)
                            and self._opencv_cuda_preview
                            and self._cv2_has_opengl_support
                            and s_img.device.type == "cuda"
                        ):
                            # This doesn't work last time I checked
                            show_gpu_tensor(label=self._label, tensor=s_img, wait=False)
                        else:
                            if (
                                isinstance(s_img, torch.Tensor)
                                and self._allow_gpu_gl
                                and self._native_cuda_preview
                                and s_img.device.type == "cuda"
                            ):
                                s_img = make_visible_image(
                                    s_img, enable_resizing=self._show_scaled, force_numpy=False
                                )
                                stream_handle = (
                                    self._stream.cuda_stream if self._stream is not None else None
                                )
                                show_cuda_tensor(self._label, s_img, True, stream_handle)
                                if self._hold_tensor_ref:
                                    # Holds a ref to this image to keep its GPU surface valid
                                    # (is this necessary? Do we create a separate texture out of this?)
                                    self._displayed_tensor = s_img
                            else:
                                cv2.imshow(
                                    self._label,
                                    make_visible_image(
                                        s_img,
                                        enable_resizing=self._show_scaled,
                                        force_numpy=True,
                                    ),
                                )
                                cv2.waitKey(1)

    def close(self):
        if self._thread is not None:
            self._q.put(None)
            self._thread.join()
            self._thread = None
        if self._headless_preview is not None:
            self._headless_preview.close()
            self._headless_preview = None
        if self._youtube_publisher is not None:
            self._youtube_publisher.close()
            self._youtube_publisher = None
        self._displayed_tensor = None

    def _worker(self):
        last_frame = None
        next_frame_time = time.time()
        frame_interval = 1.0 / self._fps if self._fps is not None else None

        with cuda_stream_scope(self._stream):
            while True:
                if self._fps is None:
                    img = self._q.get()
                    if img is None:
                        break
                    self._do_show(img=img)
                else:
                    current_time = time.time()
                    sleep_duration = next_frame_time - current_time

                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

                    # Update the next frame time
                    next_frame_time += frame_interval

                    # Determine which frame to show
                    frame_to_show = last_frame
                    shutdown_requested = False

                    while self._q.qsize() != 0:
                        potential_frame = self._q.get()
                        if potential_frame is None:
                            shutdown_requested = True
                            frame_to_show = None
                            last_frame = None
                            break
                        if time.time() >= next_frame_time - frame_interval:
                            frame_to_show = potential_frame

                    if shutdown_requested:
                        break

                    # If we have a frame to show, display it
                    if frame_to_show is not None:
                        self._do_show(frame_to_show)
                        last_frame = frame_to_show

    def _ensure_stream(self, device: torch.device):
        if self._stream is None and device.type == "cuda":
            # Keep preview copies off the compute streams once producer work is ready.
            self._stream = torch.cuda.Stream(device, priority=-1)

    def show(self, img: torch.Tensor | np.ndarray | StreamTensorBase, clone: bool = True):
        self._iter += 1
        if self._iter % self._step != 0:
            return
        with self._prof_ctx("shower.show"):
            img_unwrapped = unwrap_tensor(img)
            img_device = img_unwrapped.device if isinstance(img_unwrapped, torch.Tensor) else None
            if self._stream is None and img_device is not None and img_device.type == "cuda":
                self._ensure_stream(img_device)
            if self._thread is not None:
                counter: int = 0
                while self._q.qsize() >= self._max_size:
                    if self._skip_frame_when_full:
                        if self._drop_oldest_when_full:
                            try:
                                with self._prof_ctx("shower.drop_stale_frame"):
                                    self._q.get(block=False)
                                continue
                            except queue.Empty:
                                break
                        return
                    # print("Too many items in Shower queue...")
                    time.sleep(0.01)
                    counter += 1
                    if counter % 20 == 0:
                        self._logger.info("Too many items in Shower queue...")
                if self._cache_on_cpu and not isinstance(img, np.ndarray):
                    img = img.cpu()
                if self._fps is None or img.ndim == 3:
                    if not self._cache_on_cpu:
                        if isinstance(img_unwrapped, torch.Tensor):
                            prev_stream = (
                                torch.cuda.current_stream(img_unwrapped.device)
                                if img_unwrapped.device.type == "cuda"
                                else None
                            )
                            with cuda_stream_scope(self._stream):
                                if prev_stream is not None and self._stream is not None:
                                    self._stream.wait_stream(prev_stream)
                                img = unwrap_tensor(img)
                            if clone:
                                img = img.clone()
                            img = wrap_tensor(img)
                        elif clone:
                            img = np.array(img, copy=True)
                    self._q.put(img)
                else:
                    assert img.ndim == 4
                    for s_img in img:
                        assert False  # stream issues here sometimes if it cant be strided maybe?
                        self._q.put(s_img)
