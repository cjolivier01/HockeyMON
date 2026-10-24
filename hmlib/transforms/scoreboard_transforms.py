import copy
import logging
import os
from pathlib import Path
from typing import Any

import torch
from mmengine.registry import TRANSFORMS

from hmlib.config import get_clip_box, get_config, get_nested_value
from hmlib.scoreboard.scoreboard import Scoreboard
from hmlib.scoreboard.selector import configure_scoreboard
from hmlib.utils.image import image_height, image_width, make_channels_last

logger = logging.getLogger(__name__)


def _try_pop(d: dict[str, Any], k: str) -> Any | None:
    if k in d:
        return d.pop(k)
    return None


def _is_rocm_runtime() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _scoreboard_bbox(
    scoreboard_points: torch.Tensor | list[list[float]],
) -> tuple[int, int, int, int]:
    if isinstance(scoreboard_points, torch.Tensor):
        points = scoreboard_points.detach().to(dtype=torch.float32)
    else:
        points = torch.tensor(scoreboard_points, dtype=torch.float32)
    if points.shape != (4, 2):
        raise ValueError(
            "Scoreboard perspective polygon must contain exactly four [x, y] points; "
            f"got shape {tuple(points.shape)}"
        )

    mins = torch.floor(torch.min(points, dim=0).values)
    maxs = torch.ceil(torch.max(points, dim=0).values)
    return tuple(int(v) for v in torch.cat((mins, maxs), dim=0).tolist())


def _scoreboard_is_inside_image(
    scoreboard_points: torch.Tensor | list[list[float]],
    img: torch.Tensor,
) -> tuple[bool, tuple[int, int, int, int], tuple[int, int]]:
    x0, y0, x1, y1 = _scoreboard_bbox(scoreboard_points)
    img_w = image_width(img)
    img_h = image_height(img)
    is_inside = 0 <= x0 < x1 <= img_w and 0 <= y0 < y1 <= img_h
    return is_inside, (x0, y0, x1, y1), (img_w, img_h)


@TRANSFORMS.register_module()
class HmConfigureScoreboard:
    def __init__(
        self,
        game_id: str | None = None,
        allow_interactive_setup: bool | None = None,
    ):
        self._game_id = game_id
        if allow_interactive_setup is None:
            self._allow_interactive_setup = (
                not _is_rocm_runtime() and "HM_UI_SESSION_DIR" not in os.environ
            )
        else:
            self._allow_interactive_setup = bool(allow_interactive_setup)
        self._scoreboard_config = None
        self._configured = False
        self._polygon_revision = None
        self._shared_canvas_saved = False

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        if self._game_id is None:
            self._game_id = results.get("game_id", None)
        if (
            not self._shared_canvas_saved
            and self._game_id
            and os.environ.get("HM_SHARED_UI_BACKEND") == "1"
            and "HM_SHARED_NATIVE_CALIBRATION" not in os.environ
        ):
            import cv2

            from hmlib.config import get_game_dir
            from hmlib.utils.image import to_uint8_image

            image = make_channels_last(to_uint8_image(results["img"]))
            if image.ndim == 4:
                image = image[0]
            data = image.detach().cpu().numpy()
            valid, encoded = cv2.imencode(".png", data)
            if not valid:
                raise OSError("Cannot encode the shared scoreboard reference canvas")
            target = Path(get_game_dir(self._game_id)) / "s.png"
            temporary = target.with_name(f".s-{os.getpid()}.png")
            try:
                temporary.write_bytes(encoded.tobytes())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            self._shared_canvas_saved = True
        runtime_config = results.get("game_config")
        if runtime_config is not None:
            preferences = get_nested_value(runtime_config, "rink.scoreboard", {}) or {}
            results["scoreboard_preferences"] = preferences
            polygon = preferences.get("perspective_polygon")
            if polygon != self._polygon_revision:
                self._polygon_revision = copy.deepcopy(polygon)
                self._configured = False
                self._scoreboard_config = None
            if not preferences.get("enabled", True):
                results.pop("scoreboard_cfg", None)
                return results
        if self._game_id and not self._configured:
            self._configured = True
            game_config = (
                runtime_config if runtime_config is not None else get_config(game_id=self._game_id)
            )
            current_scoreboard = get_nested_value(
                game_config, "rink.scoreboard.perspective_polygon"
            )
            if current_scoreboard is None and not self._allow_interactive_setup:
                logger.warning(
                    "No scoreboard perspective polygon configured for %s; "
                    "skipping scoreboard capture for this run. Select its corners "
                    "in the desktop Calibration page or scoreboard selector.",
                    self._game_id,
                )
                return results
            scoreboard_points = current_scoreboard
            if scoreboard_points is None:
                try:
                    scoreboard_points = configure_scoreboard(game_id=self._game_id)
                except FileNotFoundError:
                    scoreboard_points = configure_scoreboard(
                        game_id=self._game_id, image=results["img"]
                    )
            if (
                scoreboard_points is not None
                and torch.sum(torch.tensor(scoreboard_points, dtype=torch.float)).item() != 0
                and game_config
            ):
                clip_box = get_clip_box(game_id=self._game_id)
                if clip_box:
                    scoreboard_points = [
                        [point[0] - clip_box[0], point[1] - clip_box[1]]
                        for point in scoreboard_points
                    ]
                self._scoreboard_config: dict[str, Any] = {
                    "scoreboard_points": scoreboard_points,
                    "dest_width": get_nested_value(game_config, "rink.scoreboard.projected_width"),
                    "dest_height": get_nested_value(
                        game_config, "rink.scoreboard.projected_height"
                    ),
                }
        if self._scoreboard_config:
            results["scoreboard_cfg"] = self._scoreboard_config
        else:
            results.pop("scoreboard_cfg", None)
        return results


@TRANSFORMS.register_module()
class HmCaptureScoreboard:
    def __init__(
        self,
        scoreboard_scale: float = 1.0,
    ):
        self._scoreboard = None
        self._scoreboard_scale = scoreboard_scale
        self._skip_capture = False
        self._geometry_signature = None

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        scoreboard = results.get("scoreboard_cfg")
        geometry = repr(scoreboard)
        if geometry != self._geometry_signature:
            self._geometry_signature = geometry
            self._scoreboard = None
            self._skip_capture = False
        if not scoreboard or self._skip_capture:
            results.pop("scoreboard_cfg", None)
            return results
        img = results["img"]
        preferences = results.get("scoreboard_preferences", {})
        signature = (
            preferences.get("scoreboard_scale", self._scoreboard_scale),
            preferences.get("resize_filter", "nearest"),
            preferences.get("sharpen_amount", 0),
        )
        if self._scoreboard is not None and signature != self._signature:
            self._scoreboard = None
        if self._scoreboard is None:
            scoreboard_points = scoreboard["scoreboard_points"]
            is_inside, bbox, image_size = _scoreboard_is_inside_image(scoreboard_points, img)
            if not is_inside:
                logger.warning(
                    "Scoreboard polygon bbox %s is outside current frame size %s; "
                    "skipping scoreboard capture for this run.",
                    bbox,
                    image_size,
                )
                self._skip_capture = True
                results.pop("scoreboard_cfg", None)
                return results

            dest_width = scoreboard["dest_width"]
            if isinstance(dest_width, str) and dest_width.startswith("%"):
                ratio = float(dest_width[1:]) / 100
                if "video_frame_cfg" in results:
                    dw = results["video_frame_cfg"]["output_frame_width"]
                else:
                    dw = results["ori_shape"][-1]
                dest_width = dw * ratio
            dest_height = scoreboard["dest_height"]
            if isinstance(dest_height, str) and dest_height.startswith("%"):
                ratio = float(dest_height[1:]) / 100
                if "video_frame_cfg" in results:
                    dh = results["video_frame_cfg"]["output_frame_height"]
                else:
                    dh = results["ori_shape"][-2]
                dest_height = dh * ratio
            self._scoreboard = Scoreboard(
                src_pts=scoreboard_points,
                dest_width=int(dest_width),
                dest_height=int(dest_height),
                scoreboard_scale=signature[0],
                resize_filter=signature[1],
                sharpen_amount=signature[2],
                dtype=torch.float,
                device=img.device,
            )
            self._signature = signature
        # w/h may have been adjusted based upon aspect ratio of the given points, etc.
        scoreboard_img = make_channels_last(self._scoreboard.forward(img))
        results["scoreboard_img"] = scoreboard_img

        return results


@TRANSFORMS.register_module()
class HmRenderScoreboard:
    def __init__(self, image_labels: list[str]):
        self._image_labels = image_labels
        self._scoreboard_width: int | None = None
        self._scoreboard_height: int | None = None

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        scoreboard_img = _try_pop(results, "scoreboard_img")
        if scoreboard_img is None:
            return results

        assert scoreboard_img.ndim == 4
        self._scoreboard_height = int(scoreboard_img.shape[1])
        self._scoreboard_width = int(scoreboard_img.shape[2])
        preferences = results.get("scoreboard_preferences", {})
        position = preferences.get("position", "upper-left")
        if position not in {"upper-left", "upper-center", "upper-right"}:
            raise ValueError(f"Unknown scoreboard position: {position}")

        results.pop("scoreboard_cfg", None)
        for img_label in self._image_labels:
            img = results.get(img_label)
            if img is not None:
                img = make_channels_last(img)
                scoreboard_img = scoreboard_img.to(
                    dtype=img.dtype, device=img.device, non_blocking=True
                )
                assert self._scoreboard_height is not None and self._scoreboard_width is not None
                sh = self._scoreboard_height
                sw = self._scoreboard_width
                # Keep large user-selected scale values within the output.
                if sh > img.shape[1] or sw > img.shape[2]:
                    ratio = min(img.shape[1] / sh, img.shape[2] / sw)
                    sh, sw = max(1, int(sh * ratio)), max(1, int(sw * ratio))
                    board = (
                        torch.nn.functional.interpolate(
                            scoreboard_img.permute(0, 3, 1, 2).float(),
                            size=(sh, sw),
                            mode="bilinear",
                            align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .to(img.dtype)
                    )
                else:
                    board = scoreboard_img
                x = (
                    0
                    if position == "upper-left"
                    else (img.shape[2] - sw) // (2 if position == "upper-center" else 1)
                )
                img[:, :sh, x : x + sw, :] = board
                results[img_label] = img
        return results
