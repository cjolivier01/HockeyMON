"""Choose output bitrate from source pixel density and final encoder geometry."""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Iterable, Optional

from hmlib.log import logger

DEFAULT_OUTPUT_BITRATE = 55_000_000


def _positive_integer(value, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer, received {value!r}") from error
    if isinstance(value, bool) or number <= 0 or value != number:
        raise ValueError(f"{name} must be a positive integer, received {value!r}")
    return number


def select_source_bitrate_density(
    paths: Iterable[str], *, probe: Optional[Callable] = None
) -> Optional[Fraction]:
    """Return the greatest video bitrate/pixel ratio across cameras and chapters."""
    if probe is None:
        from hmlib.video.ffmpeg import BasicVideoInfo

        probe = BasicVideoInfo
    selected = None
    for path in dict.fromkeys(str(path) for path in paths):
        info = probe(path)
        if not info.bit_rate or not info.width or not info.height:
            logger.warning(
                "Source bitrate or dimensions unavailable for %s; excluding it from bitrate selection",
                path,
            )
            continue
        bitrate = _positive_integer(info.bit_rate, f"source bitrate for {path}")
        width = _positive_integer(info.width, f"source width for {path}")
        height = _positive_integer(info.height, f"source height for {path}")
        density = Fraction(bitrate, width * height)
        if selected is None or density > selected:
            selected = density
    return selected


def resolve_output_bitrate(
    bit_rate: Optional[int],
    source_density: Optional[Fraction],
    width: int,
    height: int,
) -> int:
    """Honor an explicit bitrate, otherwise scale to actual output pixels."""
    if bit_rate is not None:
        return _positive_integer(bit_rate, "output bitrate")
    width = _positive_integer(width, "output width")
    height = _positive_integer(height, "output height")
    if source_density is None:
        logger.warning(
            "No source bitrate density is available; using %d bps for output",
            DEFAULT_OUTPUT_BITRATE,
        )
        return DEFAULT_OUTPUT_BITRATE
    if source_density <= 0:
        raise ValueError("Source bitrate density must be positive")
    scaled = source_density * width * height
    # Round half up, matching hstream's rational integer arithmetic.
    result = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)
    if result <= 0:
        raise ValueError("Source bitrate density rounds to a zero output bitrate")
    logger.info("Output bitrate for %dx%d from source density: %d bps", width, height, result)
    return result
