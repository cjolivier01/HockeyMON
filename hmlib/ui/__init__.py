"""Lightweight UI helpers for image and video visualization.

Expose :func:`show_image` and :class:`Shower` for use in CLIs and notebooks.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .show import show_image
    from .shower import Shower

__all__ = [
    "Shower",
    "show_image",
]


def __getattr__(name: str):
    if name == "show_image":
        from .show import show_image

        return show_image
    if name == "Shower":
        from .shower import Shower

        return Shower
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
