"""Model-owned class indices used only to select camera subjects."""

import torch

from hmlib.config import get_nested_value


def excluded_role_labels(config: dict) -> tuple[int, ...]:
    roles = get_nested_value(config, "detector.role_labels", {})
    excluded = []
    for role in ("referee", "goalie"):
        enabled = get_nested_value(config, f"rink.tracking.cam_ignore_{role}s", False)
        if not isinstance(enabled, bool):
            raise TypeError(f"cam_ignore_{role}s must be a boolean")
        if enabled and roles.get(role) is not None:
            excluded.append(roles[role])
    return tuple(excluded)


def camera_role_mask(labels: torch.Tensor, config: dict) -> torch.Tensor:
    keep = torch.ones_like(labels, dtype=torch.bool)
    for label in excluded_role_labels(config):
        keep &= labels != label
    return keep
