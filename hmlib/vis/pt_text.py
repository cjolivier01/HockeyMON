"""PyTorch-based text drawing utilities on tensors.

Provides cached glyph rendering to RGB tensors and helpers for annotating
images with text directly on GPU tensors.

@see @ref hmlib.vis.pt_visualization "pt_visualization" for geometric shapes.
"""

import string
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..utils.image import is_channels_first, make_channels_first

# Key: (ttf file path, point size, device) -> letter -> RGBA tensor
CHARACTERS: dict[tuple[str, int, str], dict[str, torch.Tensor]] = {}


def print_rgba_planes(image_tensor):
    """
    Prints the values of each plane (R, G, B, A) of an RGBA image tensor, pixel by pixel.
    The values will be printed in the shape of the image, lined up in columns and rows.

    Args:
        image_tensor (np.ndarray): A NumPy array of shape (height, width, 4) representing an RGBA image.
    """
    if len(image_tensor.shape) != 3 or image_tensor.shape[2] != 4:
        raise ValueError(
            "Input image tensor must have shape (height, width, 4) representing an RGBA image."
        )

    height, width, _ = image_tensor.shape

    planes = ["R", "G", "B", "A"]

    for plane_idx, plane_name in enumerate(planes):
        print(f"\n{plane_name} Plane:")
        for row in range(height):
            row_values = [f"{int(image_tensor[row, col, plane_idx]):3}" for col in range(width)]
            print(" ".join(row_values))


_TRANSPARENT_PIXEL: int = 255


def _create_text_images(
    font_path: str, font_size: int, font_color: tuple[int, int, int], device=torch.device
) -> dict[str, torch.Tensor]:
    assert font_path

    key = (font_path, font_size, str(device))
    found = CHARACTERS.get(key)
    if found is not None:
        return found

    # Define the set of characters we want to render
    # characters = string.ascii_letters + string.digits  # All upper and lowercase letters and digits
    # Line breaks are handled by draw_text; RAQM cannot shape control glyphs.
    characters = string.ascii_letters + string.digits + string.punctuation + " "

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Dictionary to hold the character images as tensors
    char_tensors: dict[str, torch.Tensor] = {}

    max_height = 0
    border_size: int = 1
    dbl_border_size: int = int(2 * border_size)

    # Create an image for each character
    for char in characters:
        char_width = font.getlength(text=char)

        # Create an image with transparent background
        image = Image.new(
            "RGBA",
            (int(char_width + dbl_border_size), font_size + dbl_border_size),
            (_TRANSPARENT_PIXEL, _TRANSPARENT_PIXEL, _TRANSPARENT_PIXEL, 0),
        )
        draw = ImageDraw.Draw(image)

        # Draw the text
        draw.text((border_size, border_size), char, font=font, fill=font_color)

        numpy_image = np.array(image)

        # get rid of aliasing and make a solid color
        # alpha_channel = numpy_image[:, :, 3]
        # numpy_image[alpha_channel != 0] = [font_color[0], font_color[1], font_color[2], 255]

        max_height = max(max_height, numpy_image.shape[0])

        # print_rgba_planes(numpy_image)

        # show_image("char", numpy_image, wait=True)

        # Convert the PIL image to a PyTorch tensor
        tensor = (
            torch.from_numpy(numpy_image).permute(2, 0, 1).to(device).to(torch.float)
        )  # Normalize the tensor
        char_tensors[char] = tensor

    char_tensors["max_height"] = max_height
    CHARACTERS[key] = char_tensors

    return char_tensors


def find_font_path(font_name_list: list[str] | None = None):
    if not font_name_list:
        font_name_list = ["Ubuntu Sans:style=Thin", "DejaVuSerif.ttf", "FreeSerif.ttf"]
    for font_name in font_name_list:
        # Command to find the font file
        command = ["fc-list", ":", "family", "file"]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.split("\n")
        for line in lines:
            if font_name in line:
                font_path = line.split(":")[0]
                return font_path
    return None


def alpha_blend(base_img: torch.Tensor, letter_img: torch.Tensor, start_x: int, start_y: int):
    # Calculate the region of interest in the base image
    end_y = start_y + letter_img.shape[1]
    end_x = start_x + letter_img.shape[2]

    base_img = make_channels_first(base_img)
    # Extract this region from the base image
    region = base_img[:, start_y:end_y, start_x:end_x]

    # Alpha blending
    alpha = letter_img[3:4]  # Alpha channel of the letter image
    inv_alpha = 1.0 - alpha

    # Foreground and background blending
    blended_region = letter_img[0:3] * alpha + region[0:3] * inv_alpha
    base_img[:, start_y:end_y, start_x:end_x] = blended_region.clamp(0, 255)

    return base_img


# We only support default font for draw_text right now in order to conserve resources
draw_text_SIZE_TO_FONT_PATHS: dict[int, str] = {}


def draw_text(
    image: torch.Tensor,
    x: int,
    y: int,
    text: str,
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 0, 0),
    position_is_text_bottom: bool = False,
) -> torch.Tensor:
    # font_size = int(font_size * 25)
    assert is_channels_first(image)
    font_size = int(font_size * 10)
    font_path = draw_text_SIZE_TO_FONT_PATHS.get(font_size)
    if font_path is None:
        font_path = find_font_path()
        draw_text_SIZE_TO_FONT_PATHS[font_size] = font_path
    char_images = _create_text_images(
        font_path=font_path, font_size=font_size, font_color=color, device=image.device
    )
    if position_is_text_bottom:
        y = int(y - char_images["max_height"])
        y = max(0, y)
    ndims = image.ndim
    if ndims == 4:
        assert image.shape[0] == 1
        image = image.squeeze(0)
    start_x: int = x
    for char in text:
        if char == "\n":
            x = start_x
            y += int(char_images["max_height"])
        elif char == "\r":
            x = start_x
        elif char in char_images:
            letter_img = char_images[char]
            base_w = image.shape[-1]
            base_h = image.shape[-2]
            letter_w = letter_img.shape[-1]
            letter_h = letter_img.shape[-2]
            if x + letter_w > base_w or y + letter_h > base_h:
                # Going off the edge of the image
                break
            if x >= 0:
                image = alpha_blend(image, letter_img, x, y)
            x += letter_img.shape[2]  # Move x to the right for the next character
    if ndims == 4:
        image = image.unsqueeze(0)
    return image


def measure_text(text: str, font_size: int = 20) -> tuple[int, int]:
    """Return (width_px, height_px) for `draw_text` with the same `font_size`.

    This is used by overlays to draw a background box before rendering text.
    """
    if not text:
        return 0, 0
    font_size = int(font_size * 10)
    font_path = draw_text_SIZE_TO_FONT_PATHS.get(font_size)
    if font_path is None:
        font_path = find_font_path()
        draw_text_SIZE_TO_FONT_PATHS[font_size] = font_path
    char_images = _create_text_images(
        font_path=font_path,
        font_size=font_size,
        font_color=(255, 255, 255),
        device="cpu",
    )
    line_width = 0
    max_width = 0
    height = int(char_images["max_height"])
    total_height = height
    for ch in text:
        if ch == "\n":
            max_width = max(max_width, line_width)
            line_width = 0
            total_height += height
            continue
        if ch == "\r":
            max_width = max(max_width, line_width)
            line_width = 0
            continue
        glyph = char_images.get(ch)
        if glyph is not None:
            line_width += int(glyph.shape[-1])
    max_width = max(max_width, line_width)
    return int(max_width), int(total_height)
