"""Follower zoom tuning shared by native and Python camera boxes."""

from numbers import Integral


def zoom_in_shrink_thresholds(aggressiveness: int) -> tuple[float, float]:
    """Return width/height deadbands matching HStream's 0..100 zoom control.

    The default 25 retains the original 8%/10% shrink thresholds. Larger
    values zoom in sooner; growth thresholds and camera velocity are unchanged.
    """
    if (
        isinstance(aggressiveness, bool)
        or not isinstance(aggressiveness, Integral)
        or not 0 <= aggressiveness <= 100
    ):
        raise ValueError("rink.camera.zoom_in_aggressiveness must be an integer from 0 to 100")
    if aggressiveness <= 25:
        multiplier = 2.0 - aggressiveness / 25.0
    else:
        multiplier = 1.0 - (aggressiveness - 25) / 75.0 * 0.9
    return 0.08 * multiplier, 0.10 * multiplier
