"""Tracking-related overlay helpers for drawing boxes, labels and scores.

Bridges between OpenCV-based visualization and PyTorch tensor-based drawing
functions from :mod:`hmlib.vis`.
"""

from typing import List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

import hmlib.vis.pt_text as ptt
import hmlib.vis.pt_visualization as ptv
from hmlib.utils.gpu import StreamTensorBase
from hmlib.utils.image import is_channels_last, make_channels_first, make_channels_last


def tlwhs_to_tlbrs(tlwhs):
    tlbrs = np.copy(tlwhs)
    if len(tlbrs) == 0:
        return tlbrs
    tlbrs[:, 2] += tlwhs[:, 0]
    tlbrs[:, 3] += tlwhs[:, 1]
    return tlbrs


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def to_cv2(image: torch.Tensor | np.ndarray) -> np.ndarray:
    # OpenCV likes [Height, Width, Channels]
    assert image.ndim <= 3
    if isinstance(image, StreamTensorBase):
        image = image.get()
    if isinstance(image, torch.Tensor):
        if image.dtype == torch.float16:
            image = image.to(torch.float32)
        image = image.cpu().numpy()
    if image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)
    return np.ascontiguousarray(image)


def plot_circle(
    image: Union[torch.Tensor, np.ndarray],
    center: Tuple[int, int],
    radius: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(image, np.ndarray):
        image = to_cv2(image)
        cv2.circle(
            image,
            center=[int(i) for i in center],
            radius=int(radius),
            color=color,
            thickness=thickness,
        )
        return image
    if thickness < 0:
        # This is probably cv2.FILLED
        thickness = int(radius)
    return ptv.draw_circle(
        image=image,
        center_x=center[0],
        center_y=center[1],
        radius=int(radius),
        color=color,
        thickness=thickness,
    )


def plot_ellipse(
    image: Union[torch.Tensor, np.ndarray],
    center: Tuple[int, int],
    axes: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(image, StreamTensorBase):
        image = image.get()
    if isinstance(image, np.ndarray):
        image = to_cv2(image)
        cv2.ellipse(
            image,
            center=(int(center[0]), int(center[1])),
            axes=(int(axes[0]), int(axes[1])),
            angle=0.0,
            startAngle=0,
            endAngle=360,
            color=normalize_color(image, color),
            thickness=thickness,
            lineType=cv2.LINE_4,
        )
        return image
    fill = False
    if thickness < 0:
        thickness = int(max(axes))
        fill = True
    return ptv.draw_ellipse_axes(
        image=image,
        center_x=center[0],
        center_y=center[1],
        radius_x=int(axes[0]),
        radius_y=int(axes[1]),
        color=normalize_color(image, color),
        thickness=thickness,
        fill=fill,
    )


def plot_rectangle(
    img,
    box: List[int],
    color: Tuple[int, int, int],
    thickness: int,
    label: Union[str, None] = None,
    text_scale: int = 1,
):
    assert thickness > 0
    intbox = [int(i) for i in box]
    if isinstance(img, torch.Tensor):
        img = plot_torch_rectangle(
            image=img,
            tlbr=box,
            color=normalize_color(img, color),
            thickness=thickness,
        )
    else:
        img = to_cv2(img)
        cv2.rectangle(
            img,
            intbox[0:2],
            intbox[2:4],
            color=normalize_color(img, color),
            thickness=thickness,
        )

    if label:
        text_thickness = 2
        img = my_put_text(
            img,
            label,
            (intbox[0], intbox[1] + 30),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 128),
            thickness=text_thickness,
        )
    return img


def normalize_color(img, color):
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            color = [float(i) for i in color]
    elif isinstance(img, torch.Tensor):
        if torch.is_floating_point(img):
            color = [float(i) for i in color]
    return color


def plot_alpha_rectangle(
    img,
    box: List[int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    label: str = None,
    text_scale: int = 1,
    opacity_percent: int = 100,
):
    if opacity_percent == 100:
        return plot_rectangle(img, box, color, thickness, label, text_scale)
    intbox = [int(i) for i in box]

    # TODO: Do just a small portion, like how the watermark is done

    if isinstance(img, np.ndarray):
        if img.dtype == np.float:
            color = [float(i) for i in color]
        color = [float(i) for i in color]

        mask = np.copy(img)

        cv2.rectangle(
            mask,
            intbox[0:2],
            intbox[2:4],
            color=normalize_color(img, color),
            thickness=cv2.FILLED,
        )

        alpha = float(opacity_percent) / 100
        # Blend the mask with the original imagealpha =
        rectangled_image = cv2.addWeighted(mask, alpha, img, 1 - alpha, 0)
    else:
        # PyTorch
        if torch.is_floating_point(img):
            color = [float(i) for i in color]
        rectangled_image = plot_torch_rectangle(
            img,
            tlbr=intbox,
            color=color,
            thickness=thickness,
            alpha=(255 * opacity_percent / 100),
            filled=True,
        )

    if label:
        text_thickness = 2
        rectangled_image = my_put_text(
            rectangled_image,
            label,
            (intbox[0], intbox[1] + 30),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 128),
            thickness=text_thickness,
        )
    return rectangled_image


def plot_torch_rectangle(
    image: torch.Tensor,
    tlbr: torch.Tensor,
    color: Tuple[int, int, int],
    thickness: int = 1,
    alpha: int = 255,
    filled: bool = False,
):
    assert isinstance(image, torch.Tensor)
    sq = image.ndim == 3
    if sq:
        image = image.unsqueeze(0)
    assert image.ndim == 4
    image = ptv.draw_box(
        image=image, tlbr=tlbr, color=color, thickness=thickness, alpha=alpha, filled=filled
    )
    if sq:
        image = image.squeeze(0)
    return image


def draw_dashed_rectangle(img, box, color, thickness, dash_length: int = 10):
    """
    Draw a dashed-line rectangle on an image.

    Parameters:
    img (numpy.ndarray): The image.
    top_left (tuple): The top-left corner of the rectangle (x, y).
    bottom_right (tuple): The bottom-right corner of the rectangle (x, y).
    color (tuple): Color of the rectangle (B, G, R).
    thickness (int): Thickness of the rectangle lines.
    dash_length (int): Length of each dash.
    """
    x1, y1 = int(box[0]), int(box[1])
    x2, y2 = int(box[2]), int(box[3])

    # Draw top and bottom sides
    for x in range(x1, x2, dash_length * 2):
        my_draw_line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        my_draw_line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)

    # Draw left and right sides
    for y in range(y1, y2, dash_length * 2):
        my_draw_line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        my_draw_line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    return img


def _to_int(vals):
    return [int(i) for i in vals]


def plot_line(
    img: torch.Tensor | np.ndarray,
    src_point,
    dest_point,
    color: Tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    return my_draw_line(
        img,
        _to_int(src_point),
        _to_int(dest_point),
        color=normalize_color(img, color),
        thickness=thickness,
    )


def plot_point(
    img: torch.Tensor | np.ndarray,
    point,
    color: Tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    img = to_cv2(img)
    x = int(point[0] + 0.5 * thickness)
    y = int(point[1] + 0.5 * thickness)
    cv2.circle(
        img,
        [x, y],
        radius=int((thickness + 1) // 2),
        color=normalize_color(img, color),
        thickness=thickness,
    )
    return img


last_frame_id = -1


def plot_frame_id_and_speeds(im, frame_id, vel_x, vel_y, accel_x, accel_y):
    text_scale = max(2, im.shape[1] / 1600.0)

    y_delta = int(15 * text_scale)
    text_y_offset = y_delta

    im = to_cv2(im)

    cv2.putText(
        im,
        "frame: %d" % (frame_id),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "vel_x: %.2f" % (vel_x),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "vel_y: %.2f" % (vel_y),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "accel_x: %.2f" % (accel_x),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )

    text_y_offset += y_delta
    cv2.putText(
        im,
        "accel_y: %.2f" % (accel_y),
        (0, text_y_offset),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2,
    )
    return im


def my_put_text(
    img: Union[torch.Tensor, np.ndarray],
    text: str,
    org: Tuple[int, int],
    fontFace: int,
    fontScale: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> Union[torch.Tensor, np.ndarray]:
    assert isinstance(text, str)
    if isinstance(img, torch.Tensor):
        was_channels_last = is_channels_last(img)
        if was_channels_last:
            img = make_channels_first(img)
        img = ptt.draw_text(
            image=img,
            x=int(org[0]),
            y=int(org[1]),
            text=text,
            font_size=fontScale * 2,
            color=color,
            position_is_text_bottom=True,
        )
        if was_channels_last:
            img = make_channels_last(img)
        return img
    img = to_cv2(img)
    cv2.putText(
        img,
        text,
        org,
        fontFace,
        fontScale,
        color,
        thickness=thickness,
    )
    return img


def unsqueeze(t: Union[torch.Tensor, np.ndarray], dim: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(t, torch.Tensor):
        return t.unsqueeze(dim)
    return np.expand_dims(t, axis=dim)


# def plot_frame_number(image, frame_id):
#     text_scale = max(2, image_width(image) / 800.0)
#     text_thickness = 2
#     text_offset = int(8 * text_scale)
#     if image.ndim == 3:
#         image = unsqueeze(image, 0)
#     result_images = []
#     frame_id = int(frame_id)
#     for i in range(image.shape[0]):
#         img = my_put_text(
#             img=image[i],
#             text=f"F: {frame_id + i}",
#             org=(text_offset, int(15 * text_scale)),
#             fontFace=cv2.FONT_HERSHEY_PLAIN,
#             fontScale=text_scale,
#             color=(0, 0, 255),
#             thickness=text_thickness,
#         )
#         result_images.append(img)
#     if isinstance(image, torch.Tensor):
#         image = torch.stack(result_images)
#     else:
#         image = np.stack(result_images)
#     return image


def plot_tracking(
    image,
    tlwhs,
    obj_ids,
    frame_id,
    scores=None,
    fps=0.0,
    ids2=None,
    speeds: List[float] = None,
    box_color: Tuple[int, int, int] = None,
    show_frame_heading: bool = False,
    line_thickness: int = 1,
    ignore_frame_id: bool = False,
    print_track_id: bool = True,
    ignore_tracking_ids: Set[int] = None,
    ignored_color: Optional[Tuple[int, int, int]] = None,
    draw_tracking_circles: bool = True,
    circle_flatten_ratio: float = 0.35,
    circle_radius_scale: float = 0.5,
):
    if ignore_tracking_ids is None:
        ignore_tracking_ids = set()
    if not ignore_frame_id:
        global last_frame_id
        # don't call this more than once per frame
        assert frame_id > last_frame_id
        last_frame_id = frame_id.clone()
    assert len(tlwhs) == len(obj_ids)
    if speeds:
        assert len(speeds) == len(obj_ids)
    im = image

    text_scale = max(2, image.shape[1] / 1600.0)
    text_thickness = 2
    text_offset = int(8 * text_scale)

    if show_frame_heading:
        im = my_put_text(
            im,
            "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            (0, 0, 255),
            thickness=2,
        )

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        x1f = float(x1)
        y1f = float(y1)
        wf = float(w)
        hf = float(h)
        intbox = tuple(map(int, (x1f, y1f, x1f + wf, y1f + hf)))
        obj_id = int(obj_ids[i])
        color = box_color if box_color is not None else get_color(abs(obj_id))
        norm_color = normalize_color(im, color)

        if draw_tracking_circles:
            center_x = int(x1f + 0.5 * wf)
            radius_x = max(1, int(round(abs(wf) * circle_radius_scale)))
            radius_y = max(1, int(round(radius_x * circle_flatten_ratio)))
            ground_offset = max(1, int(round(hf * 0.1 + abs(wf) * 0.05)))
            center_y = int(y1f + hf - ground_offset)
            circle_thickness = 2
            circle_color = norm_color
            if obj_id in ignore_tracking_ids:
                circle_color = (
                    normalize_color(im, ignored_color) if ignored_color is not None else norm_color
                )
            if radius_x > 0 and radius_y > 0:
                im = plot_ellipse(
                    im,
                    center=(center_x, center_y),
                    axes=(radius_x, radius_y),
                    color=circle_color,
                    thickness=circle_thickness,
                )
            if obj_id in ignore_tracking_ids:
                im = my_put_text(
                    im,
                    "IGNORED",
                    (intbox[0], intbox[1] + text_offset),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 0, 128),
                    thickness=text_thickness,
                )
            elif print_track_id:
                id_text = "{}".format(int(obj_id))
                if ids2 is not None:
                    id_text = id_text + ", {}".format(int(ids2[i]))
                im = my_put_text(
                    im,
                    id_text,
                    (intbox[0], intbox[1] + text_offset),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 0, 255),
                    thickness=text_thickness,
                )
        else:
            if obj_id in ignore_tracking_ids:
                im = plot_rectangle(
                    im,
                    box=intbox,
                    color=ignored_color,
                    thickness=1,
                    label="IGNORED",
                )
            else:
                im = plot_rectangle(
                    im,
                    box=intbox,
                    color=norm_color,
                    thickness=line_thickness,
                )

                if print_track_id:
                    id_text = "{}".format(int(obj_id))
                    if ids2 is not None:
                        id_text = id_text + ", {}".format(int(ids2[i]))
                    im = my_put_text(
                        im,
                        id_text,
                        (intbox[0], intbox[1] + text_offset),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        (0, 0, 255),
                        thickness=text_thickness,
                    )
        if speeds:
            speed = speeds[i]
            if not np.isnan(speed):
                if speed < 10:
                    speed_text = "{:.1f}".format(speeds[i])
                else:
                    speed_text = f"{int(speed + 0.5)}"
            else:
                speed_text = "NaN"
            pos_x = intbox[2] - text_offset
            pos_y = intbox[3]
            im = my_put_text(
                im,
                speed_text,
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (150, 0, 150),
                thickness=text_thickness,
            )
    return im


def plot_trajectory(image, track_id, tlwhs):
    color = get_color(int(track_id))
    thickness = 2
    for tlwh in tlwhs:
        x1 = int(tlwh[0])
        y1 = int(tlwh[1])
        w = int(tlwh[2])
        h = int(tlwh[3])
        # Trailing off of one foot...
        cx = int(x1 + 0.5 * w)
        cy = int(y1 + h)
        image = plot_rectangle(
            img=image,
            box=[cx - thickness, cy - thickness, cx + thickness, cy + thickness],
            color=color,
            thickness=thickness,
        )

    return image


def plot_detections(image, tlbrs, scores=None, color=(255, 0, 0), ids=None):
    im = np.copy(image)
    text_scale = max(1, image.shape[1] / 800.0)
    thickness = 2 if text_scale > 1.3 else 1
    for i, det in enumerate(tlbrs):
        x1, y1, x2, y2 = np.asarray(det[:4], dtype=np.int)
        if len(det) >= 7:
            label = "det" if det[5] > 0 else "trk"
            if ids is not None:
                text = "{}# {:.2f}: {:d}".format(label, det[6], ids[i])
                im = my_put_text(
                    im,
                    text,
                    (x1, y1 + 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 255, 255),
                    thickness=thickness,
                )
            else:
                text = "{}# {:.2f}".format(label, det[6])

        if scores is not None:
            text = "{:.2f}".format(scores[i])
            im = my_put_text(
                im,
                text,
                (x1, y1 + 30),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 255, 255),
                thickness=thickness,
            )

        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)

    return im


def plot_text(*args, **kwargs):
    return my_put_text(*args, **kwargs)


def draw_arrows(img, bbox, horizontal=True, vertical=True):
    """
    Draw arrows on the bounding box edges.

    Parameters:
    - img: The image on which to draw.
    - bbox: The bounding box specified as (x1, y1, x2, y2).
    - horizontal: Boolean indicating direction of arrows on left/right edges.
                  True for outward, False for inward.
    - vertical: Boolean indicating direction of arrows on top/bottom edges.
                True for outward, False for inward.
    """
    intbox = [int(i) for i in bbox]
    x, y = intbox[:2]
    w = intbox[2] - intbox[0]
    h = intbox[3] - intbox[1]

    arrow_length = max(w // 20, 20)  # Length of the arrow
    # tip_length = arrow_length // 20
    tip_length = 0.1
    arrow_thickness = 2  # Thickness of the arrow
    color = (0, 255, 0)  # Arrow color (Green)

    # Calculate midpoints
    left_mid = (x, y + h // 2)
    right_mid = (x + w, y + h // 2)
    top_mid = (x + w // 2, y)
    bottom_mid = (x + w // 2, y + h)

    # Horizontal arrows
    if horizontal:
        # Left arrow pointing outwards
        cv2.arrowedLine(
            img,
            (left_mid[0] + arrow_length, left_mid[1]),
            left_mid,
            color,
            arrow_thickness,
            tipLength=tip_length,
        )
        # Right arrow pointing outwards
        cv2.arrowedLine(
            img,
            (right_mid[0] - arrow_length, right_mid[1]),
            right_mid,
            color,
            arrow_thickness,
            tipLength=tip_length,
        )
    else:
        # Left arrow pointing inwards
        cv2.arrowedLine(
            img,
            left_mid,
            (left_mid[0] + arrow_length, left_mid[1]),
            color,
            arrow_thickness,
            tipLength=tip_length,
        )
        # Right arrow pointing inwards
        cv2.arrowedLine(
            img,
            right_mid,
            (right_mid[0] - arrow_length, right_mid[1]),
            color,
            arrow_thickness,
            tipLength=tip_length,
        )

    # Vertical arrows
    if vertical:
        # Top arrow pointing outwards
        cv2.arrowedLine(
            img,
            (top_mid[0], top_mid[1] + arrow_length),
            top_mid,
            color,
            arrow_thickness,
            tipLength=tip_length,
        )
        # Bottom arrow pointing outwards
        cv2.arrowedLine(
            img,
            (bottom_mid[0], bottom_mid[1] - arrow_length),
            bottom_mid,
            color,
            arrow_thickness,
            tipLength=tip_length,
        )
    else:
        # Top arrow pointing inwards
        cv2.arrowedLine(
            img,
            top_mid,
            (top_mid[0], top_mid[1] + arrow_length),
            color,
            arrow_thickness,
            tipLength=tip_length,
        )
        # Bottom arrow pointing inwards
        cv2.arrowedLine(
            img,
            bottom_mid,
            (bottom_mid[0], bottom_mid[1] - arrow_length),
            color,
            arrow_thickness,
            tipLength=tip_length,
        )
    return img


def draw_centered_lines(image, bbox, thickness=2, color=(0, 255, 0)):
    intbox = [int(i) for i in bbox]
    x, y = intbox[:2]
    w = intbox[2] - intbox[0]
    h = intbox[3] - intbox[1]

    # Calculate 26% of the box's width and height for the line lengths
    line_length_w = int(0.26 * w)
    line_length_h = int(0.26 * h)

    # Calculate the start and end points for lines parallel to width
    start_x_w = x + (w - line_length_w) // 2
    end_x_w = start_x_w + line_length_w
    # And for lines parallel to height
    start_y_h = y + (h - line_length_h) // 2
    end_y_h = start_y_h + line_length_h

    # Draw lines centered on each side
    # Line on top side
    my_draw_line(image, (start_x_w, y), (end_x_w, y), color, thickness)
    # Line on bottom side
    my_draw_line(image, (start_x_w, y + h), (end_x_w, y + h), color, thickness)

    # Line on left side
    my_draw_line(image, (x, start_y_h), (x, end_y_h), color, thickness)
    # Line on right side
    my_draw_line(image, (x + w, start_y_h), (x + w, end_y_h), color, thickness)

    return image


def my_draw_line(
    image: Union[torch.Tensor, np.ndarray],
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(image, np.ndarray):
        image = to_cv2(image)
        cv2.line(image, pt1, pt2, color, thickness)
        return image

    return ptv.draw_line(
        image=image,
        x1=pt1[0],
        y1=pt1[1],
        x2=pt2[0],
        y2=pt2[1],
        color=color,
        thickness=thickness,
    )


def draw_corner_boxes(image, bbox, thickness=2, color=(0, 255, 0)):
    intbox = [int(i) for i in bbox]
    x, y = intbox[:2]
    w = intbox[2] - intbox[0]
    h = intbox[3] - intbox[1]

    # Calculate 10% of the box's width and height
    corner_length_w = int(0.1 * w)
    corner_length_h = int(0.1 * h)

    # Define points for the corners - each corner will be represented as two lines
    # Top-left corner
    my_draw_line(image, (x, y), (x + corner_length_w, y), color, thickness)
    my_draw_line(image, (x, y), (x, y + corner_length_h), color, thickness)

    # Top-right corner
    my_draw_line(image, (x + w, y), (x + w - corner_length_w, y), color, thickness)
    my_draw_line(image, (x + w, y), (x + w, y + corner_length_h), color, thickness)

    # Bottom-left corner
    my_draw_line(image, (x, y + h), (x + corner_length_w, y + h), color, thickness)
    my_draw_line(image, (x, y + h), (x, y + h - corner_length_h), color, thickness)

    # Bottom-right corner
    my_draw_line(image, (x + w, y + h), (x + w - corner_length_w, y + h), color, thickness)
    my_draw_line(image, (x + w, y + h), (x + w, y + h - corner_length_h), color, thickness)

    return image


# def plot_kmeans_intertias(hockey_mon: HockeyMON):
#     inertias = []
#     object_count = len(hockey_mon.online_image_center_points)
#     for i in range(1, object_count):
#         kmeans = KMeans(n_clusters=i, n_init="auto")
#         kmeans.fit(hockey_mon.online_image_center_points)
#         inertias.append(kmeans.inertia_)

#     plt.plot(range(1, object_count), inertias, marker="o")
#     plt.title("Elbow method")
#     plt.xlabel("Number of clusters")
#     plt.ylabel("Inertia")
#     plt.show()


# def plot_kmeans_scatter(hockey_mon: HockeyMON, n_clusters: int = 3):
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(hockey_mon.online_image_center_points)
#     for x, y in hockey_mon.online_image_center_points:
#         plt.scatter(x, y, c=kmeans.labels_)
#     plt.show()
