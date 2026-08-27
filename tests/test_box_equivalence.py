import sys
from typing import Dict, Tuple


def _torch():
    try:
        import torch  # type: ignore
    except Exception:
        print("SKIP: torch not available", file=sys.stderr)
        sys.exit(0)
    return torch


def _core():
    try:
        import hockeymon.core as core  # type: ignore
    except Exception:
        print("SKIP: hockeymon.core not available", file=sys.stderr)
        sys.exit(0)
    return core


def make_cpp_config(core, arena_box: Tuple[float, float, float, float], cfg: Dict):
    AllLivingBoxConfig = core.AllLivingBoxConfig
    BBox = core.BBox
    c = AllLivingBoxConfig()
    # Translation constraints (defaults already enabled in config)
    max_speed_x = cfg.get("max_speed_x", 20.0)
    max_speed_y = cfg.get("max_speed_y", 20.0)
    max_accel_x = cfg.get("max_accel_x", 5.0)
    max_accel_y = cfg.get("max_accel_y", 5.0)
    c.max_speed_x = max_speed_x
    c.max_speed_y = max_speed_y
    c.max_accel_x = max_accel_x
    c.max_accel_y = max_accel_y
    c.stop_translation_on_dir_change = cfg.get("stop_on_change", False)
    c.stop_translation_on_dir_change_delay = cfg.get("stop_delay", 0)
    c.cancel_stop_on_opposite_dir = cfg.get("cancel", False)
    c.time_to_dest_speed_limit_frames = cfg.get("time_to_dest_speed_limit_frames", 10)
    c.time_to_dest_stop_speed_threshold = cfg.get("time_to_dest_stop_speed_threshold", 0.0)
    c.dynamic_acceleration_scaling = 0.0
    c.arena_angle_from_vertical = 30.0
    c.arena_box = BBox(*arena_box)
    # Disable sticky/damping features for determinism unless asked for
    c.sticky_translation = cfg.get("sticky_translation", False)
    c.sticky_size_ratio_to_frame_width = cfg.get("sticky_size_ratio_to_frame_width", 10.0)
    c.sticky_translation_gaussian_mult = cfg.get("sticky_translation_gaussian_mult", 5.0)
    c.unsticky_translation_size_ratio = cfg.get("unsticky_translation_size_ratio", 0.75)
    # Resizing constraints (defaults already enabled in config)
    c.max_speed_w = cfg.get("max_speed_w", max_speed_x / 1.8)
    c.max_speed_h = cfg.get("max_speed_h", max_speed_y / 1.8)
    c.max_accel_w = cfg.get("max_accel_w", max_accel_x)
    c.max_accel_h = cfg.get("max_accel_h", max_accel_y)
    c.min_width = 10.0
    c.min_height = 10.0
    c.max_width = arena_box[2] - arena_box[0]
    c.max_height = arena_box[3] - arena_box[1]
    c.stop_resizing_on_dir_change = cfg.get(
        "stop_resizing_on_dir_change", cfg.get("resize_stop_on_change", True)
    )
    c.resizing_stop_on_dir_change_delay = cfg.get("resize_stop_delay", 0)
    c.resizing_cancel_stop_on_opposite_dir = cfg.get("resize_cancel", False)
    c.resizing_stop_cancel_hysteresis_frames = cfg.get("resize_hyst", 0)
    c.resizing_stop_delay_cooldown_frames = cfg.get("resize_cooldown", 0)
    c.resizing_time_to_dest_speed_limit_frames = cfg.get("resize_ttg", 10)
    c.resizing_time_to_dest_stop_speed_threshold = cfg.get("resize_ttg_stop_threshold", 0.0)
    c.sticky_sizing = cfg.get("sticky_sizing", False)
    # Living config
    c.scale_dest_width = 1.0
    c.scale_dest_height = 1.0
    c.clamp_scaled_input_box = True
    return c


def make_py_box(torch, arena_box_t, init_box_t, cfg: Dict):
    from hmlib.camera.moving_box import MovingBox

    # Map cpp config into MovingBox params
    return MovingBox(
        label="py",
        bbox=init_box_t.clone(),
        arena_box=arena_box_t.clone(),
        max_speed_x=torch.tensor(cfg.get("max_speed_x", 20.0), dtype=torch.float),
        max_speed_y=torch.tensor(cfg.get("max_speed_y", 20.0), dtype=torch.float),
        max_accel_x=torch.tensor(cfg.get("max_accel_x", 5.0), dtype=torch.float),
        max_accel_y=torch.tensor(cfg.get("max_accel_y", 5.0), dtype=torch.float),
        max_speed_w=(
            torch.tensor(cfg["max_speed_w"], dtype=torch.float) if "max_speed_w" in cfg else None
        ),
        max_speed_h=(
            torch.tensor(cfg["max_speed_h"], dtype=torch.float) if "max_speed_h" in cfg else None
        ),
        max_width=arena_box_t[2],
        max_height=arena_box_t[3],
        stop_on_dir_change=cfg.get("stop_on_change", False),
        stop_on_dir_change_delay=int(cfg.get("stop_delay", 0)),
        cancel_stop_on_opposite_dir=bool(cfg.get("cancel", False)),
        time_to_dest_speed_limit_frames=int(cfg.get("time_to_dest_speed_limit_frames", 10)),
        time_to_dest_stop_speed_threshold=float(cfg.get("time_to_dest_stop_speed_threshold", 0.0)),
        pan_smoothing_alpha=float(cfg.get("pan_smoothing_alpha", 0.0)),
        sticky_translation=bool(cfg.get("sticky_translation", False)),
        sticky_sizing=bool(cfg.get("sticky_sizing", False)),
        resize_stop_on_dir_change=cfg.get(
            "stop_resizing_on_dir_change", cfg.get("resize_stop_on_change", True)
        ),
        resize_stop_on_dir_change_delay=int(cfg.get("resize_stop_delay", 0)),
        resize_cancel_stop_on_opposite_dir=bool(cfg.get("resize_cancel", False)),
        resize_stop_cancel_hysteresis_frames=int(cfg.get("resize_hyst", 0)),
        resize_stop_delay_cooldown_frames=int(cfg.get("resize_cooldown", 0)),
        resize_time_to_dest_speed_limit_frames=int(cfg.get("resize_ttg", 10)),
        resize_time_to_dest_stop_speed_threshold=float(cfg.get("resize_ttg_stop_threshold", 0.0)),
        device="cpu",
    )


def make_dest_sequence_w(torch, init_box_t, step: float, n1: int, n2: int):
    """Increase width (n1 steps), then decrease width (n2 steps), keep center fixed."""
    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    dests = []
    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    # Grow width
    for i in range(n1):
        ww = w + step * (i + 1)
        dests.append(make_box_at_center(c, w=ww, h=h))
    # Shrink width
    last_w = w + step * n1
    for j in range(n2):
        ww = last_w - step * (j + 1)
        dests.append(make_box_at_center(c, w=ww, h=h))
    return dests


def make_dest_sequence_h(torch, init_box_t, step: float, n1: int, n2: int):
    """Increase height (n1 steps), then decrease height (n2 steps), keep center fixed."""
    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    dests = []
    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    # Grow height
    for i in range(n1):
        hh = h + step * (i + 1)
        dests.append(make_box_at_center(c, w=w, h=hh))
    # Shrink height
    last_h = h + step * n1
    for j in range(n2):
        hh = last_h - step * (j + 1)
        dests.append(make_box_at_center(c, w=w, h=hh))
    return dests


def make_dest_sequence_x(torch, init_box_t, step: float, n1: int, n2: int):
    """Move right (n1 steps), then left (n2 steps), return list of TLBR tensors with same size."""
    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    dests = []
    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    # Right
    for i in range(n1):
        cc = torch.tensor([c[0] + step * (i + 1), c[1]], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    # Left
    last = c[0] + step * n1
    for j in range(n2):
        cc = torch.tensor([last - step * (j + 1), c[1]], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    return dests


def make_dest_sequence_y(torch, init_box_t, step: float, n1: int, n2: int):
    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    dests = []
    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    # Down (increasing y)
    for i in range(n1):
        cc = torch.tensor([c[0], c[1] + step * (i + 1)], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    # Up
    last = c[1] + step * n1
    for j in range(n2):
        cc = torch.tensor([c[0], last - step * (j + 1)], dtype=torch.float)
        dests.append(make_box_at_center(cc, w=w, h=h))
    return dests


def _to_cpp_bbox(core, t):
    return core.BBox(float(t[0]), float(t[1]), float(t[2]), float(t[3]))


def _close(a: float, b: float, eps=1e-3):
    return abs(a - b) <= eps


def _assert_boxes_close(py_t, cpp_b, eps=1e-2):
    assert _close(float(py_t[0]), cpp_b.left, eps)
    assert _close(float(py_t[1]), cpp_b.top, eps)
    assert _close(float(py_t[2]), cpp_b.right, eps)
    assert _close(float(py_t[3]), cpp_b.bottom, eps)


def _run_case(cfg: Dict, axis: str):
    torch = _torch()
    core = _core()

    # Arena and initial box
    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    # Reasonable constraints to see behavior
    cfg_full = dict(
        max_speed_x=30.0,
        max_speed_y=30.0,
        max_accel_x=10.0,
        max_accel_y=10.0,
        stop_on_change=cfg.get("stop_on_change", False),
        stop_delay=cfg.get("stop_delay", 0),
        cancel=cfg.get("cancel", False),
        pan_smoothing_alpha=0.0,
        sticky_translation=False,
        sticky_sizing=False,
    )

    cpp_cfg = make_cpp_config(core, arena, cfg_full)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)

    py_box = make_py_box(torch, arena_t, init_box_t, cfg_full)

    # Build destination sequence with a direction change and potential cancel
    step = 20.0
    pre = 4
    post = 6
    if axis == "x":
        dests = make_dest_sequence_x(torch, init_box_t, step, pre, post)
    else:
        dests = make_dest_sequence_y(torch, init_box_t, step, pre, post)

    # If cancel requested, force a hard reversal then flip back to trigger cancel
    if cfg_full["cancel"] and cfg_full["stop_delay"] > 0:
        idx = pre
        far = step * (pre + 2)
        if axis == "x":
            base = dests[idx - 1]
            dests[idx] = base + torch.tensor([-far, 0.0, -far, 0.0], dtype=torch.float)
            dests[idx + 1] = base + torch.tensor([far, 0.0, far, 0.0], dtype=torch.float)
        else:
            base = dests[idx - 1]
            dests[idx] = base + torch.tensor([0.0, -far, 0.0, -far], dtype=torch.float)
            dests[idx + 1] = base + torch.tensor([0.0, far, 0.0, far], dtype=torch.float)

    canceled_seen = False
    # Step both in lock-step and compare
    for i, d in enumerate(dests):
        cpp_new = cpp_box.forward(_to_cpp_bbox(core, d))
        py_new = py_box.forward(d)
        _assert_boxes_close(py_new, cpp_new)

        # Compare core translation state values (current speeds)
        tstate = cpp_box.translation_state()
        # Python internals
        vx = float(py_box._current_speed_x)
        vy = float(py_box._current_speed_y)
        assert _close(vx, float(tstate.current_speed_x), 5e-2)
        assert _close(vy, float(tstate.current_speed_y), 5e-2)

        if cfg_full["cancel"] and cfg_full["stop_delay"] > 0:
            if bool(
                getattr(tstate, "canceled_stop_x", False)
                or getattr(tstate, "canceled_stop_y", False)
            ):
                assert bool(py_box._cancel_stop_x_flash or py_box._cancel_stop_y_flash)
                canceled_seen = True
    if cfg_full["cancel"] and cfg_full["stop_delay"] > 0:
        assert canceled_seen


def _run_resize_case(cfg: Dict, axis: str):
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    # Keep translation idle; focus on resizing behavior.
    cfg_full = dict(
        max_speed_x=0.0,
        max_speed_y=0.0,
        max_accel_x=10.0,
        max_accel_y=10.0,
        max_speed_w=30.0,
        max_speed_h=30.0,
        max_accel_w=10.0,
        max_accel_h=10.0,
        stop_on_change=False,
        stop_delay=0,
        cancel=False,
        pan_smoothing_alpha=0.0,
        sticky_translation=False,
        stop_resizing_on_dir_change=cfg.get("resize_stop_on_change", True),
        resize_stop_delay=cfg.get("resize_stop_delay", 0),
        resize_cancel=cfg.get("resize_cancel", False),
        resize_hyst=cfg.get("resize_hyst", 0),
        resize_cooldown=cfg.get("resize_cooldown", 0),
        resize_ttg=cfg.get("resize_ttg", 10),
        sticky_sizing=False,
    )

    cpp_cfg = make_cpp_config(core, arena, cfg_full)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg_full)

    step = 20.0
    pre = 4
    post = 6
    if axis == "w":
        dests = make_dest_sequence_w(torch, init_box_t, step, pre, post)
    else:
        dests = make_dest_sequence_h(torch, init_box_t, step, pre, post)

    # If cancel requested, force a hard resize reversal then flip back to trigger cancel
    if cfg_full["resize_cancel"] and cfg_full["resize_stop_delay"] > 0:
        idx = pre
        far = step * (pre + 2)
        base = dests[idx - 1]
        from hmlib.bbox.box_functions import center, make_box_at_center, width, height

        c = center(base)
        w = width(base)
        h = height(base)
        if axis == "w":
            dests[idx] = make_box_at_center(c, w=w - far, h=h)
            dests[idx + 1] = make_box_at_center(c, w=w + far, h=h)
        else:
            dests[idx] = make_box_at_center(c, w=w, h=h - far)
            dests[idx + 1] = make_box_at_center(c, w=w, h=h + far)

    canceled_seen = False
    for i, d in enumerate(dests):
        cpp_new = cpp_box.forward(_to_cpp_bbox(core, d))
        py_new = py_box.forward(d)
        _assert_boxes_close(py_new, cpp_new)

        rstate = cpp_box.resizing_state()
        vw = float(py_box._current_speed_w)
        vh = float(py_box._current_speed_h)
        assert _close(vw, float(rstate.current_speed_w), 5e-2)
        assert _close(vh, float(rstate.current_speed_h), 5e-2)

        if cfg_full["resize_cancel"] and cfg_full["resize_stop_delay"] > 0:
            if bool(
                getattr(rstate, "canceled_stop_w", False)
                or getattr(rstate, "canceled_stop_h", False)
            ):
                assert bool(
                    py_box._resize_cancel_stop_w_flash or py_box._resize_cancel_stop_h_flash
                )
                canceled_seen = True
    if cfg_full["resize_cancel"] and cfg_full["resize_stop_delay"] > 0:
        assert canceled_seen


def main():
    cases = [
        (dict(stop_delay=0, stop_on_change=False, cancel=False), "x"),
        (dict(stop_delay=0, stop_on_change=True, cancel=False), "x"),
        (dict(stop_delay=6, stop_on_change=False, cancel=False), "x"),
        (dict(stop_delay=6, stop_on_change=True, cancel=False), "x"),
        (dict(stop_delay=6, stop_on_change=True, cancel=True), "x"),
        (dict(stop_delay=6, stop_on_change=True, cancel=True), "y"),
    ]
    for cfg, axis in cases:
        _run_case(cfg, axis)
    resize_cases = [
        (dict(resize_stop_delay=0, resize_stop_on_change=False, resize_cancel=False), "w"),
        (dict(resize_stop_delay=0, resize_stop_on_change=True, resize_cancel=False), "w"),
        (dict(resize_stop_delay=6, resize_stop_on_change=False, resize_cancel=False), "w"),
        (dict(resize_stop_delay=6, resize_stop_on_change=True, resize_cancel=False), "w"),
        (dict(resize_stop_delay=6, resize_stop_on_change=True, resize_cancel=True), "w"),
        (dict(resize_stop_delay=6, resize_stop_on_change=True, resize_cancel=True), "h"),
    ]
    for cfg, axis in resize_cases:
        _run_resize_case(cfg, axis)
    # Extra: ensure braking doesn't start when not moving enough
    _should_not_brake_when_not_moving()
    _should_not_resize_brake_when_not_moving()
    _should_respect_resize_cooldown()
    _should_limit_resize_speed_ttg()
    _should_snap_translation_speed_ttg()
    _should_snap_resize_speed_ttg()
    print("box equivalence tests: OK")


def _should_not_brake_when_not_moving():
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    cfg = dict(
        max_speed_x=60.0,
        max_speed_y=60.0,
        max_accel_x=1.0,
        max_accel_y=1.0,
        stop_on_change=True,
        stop_delay=6,
        cancel=False,
        pan_smoothing_alpha=0.0,
        sticky_translation=False,
    )
    cpp_cfg = make_cpp_config(core, arena, cfg)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg)

    # Step 1: small accel to the right (x increasing)
    step = 10.0
    dest1 = init_box_t + torch.tensor([step, 0.0, step, 0.0], dtype=torch.float)
    cpp_box.forward(_to_cpp_bbox(core, dest1))
    py_box.forward(dest1)

    # Step 2: reverse direction (x decreasing), but speed < max_speed/6 so no braking should start
    dest2 = dest1 + torch.tensor([-step, 0.0, -step, 0.0], dtype=torch.float)
    cpp_box.forward(_to_cpp_bbox(core, dest2))
    py_box.forward(dest2)

    tstate = cpp_box.translation_state()
    assert int(getattr(tstate, "stop_delay_x", 0) or 0) == 0
    assert int(py_box._stop_delay_x) == 0


def _should_not_resize_brake_when_not_moving():
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    cfg = dict(
        max_speed_x=0.0,
        max_speed_y=0.0,
        max_accel_x=1.0,
        max_accel_y=1.0,
        max_speed_w=60.0,
        max_speed_h=60.0,
        max_accel_w=1.0,
        max_accel_h=1.0,
        resize_stop_on_change=True,
        resize_stop_delay=6,
        resize_cancel=False,
        resize_hyst=0,
        resize_cooldown=0,
        resize_ttg=10,
        sticky_translation=False,
    )
    cpp_cfg = make_cpp_config(core, arena, cfg)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg)

    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    # Step 1: small resize (width increasing)
    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    dest1 = make_box_at_center(c, w=w + 2.0, h=h)
    cpp_box.forward(_to_cpp_bbox(core, dest1))
    py_box.forward(dest1)

    # Step 2: reverse direction, but speed < max_speed/6 so no braking should start
    dest2 = make_box_at_center(c, w=w, h=h)
    cpp_box.forward(_to_cpp_bbox(core, dest2))
    py_box.forward(dest2)

    rstate = cpp_box.resizing_state()
    assert int(getattr(rstate, "stop_delay_w", 0) or 0) == 0
    assert int(py_box._resize_stop_delay_w) == 0


def _should_respect_resize_cooldown():
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    cfg = dict(
        max_speed_x=0.0,
        max_speed_y=0.0,
        max_accel_x=20.0,
        max_accel_y=20.0,
        max_speed_w=60.0,
        max_speed_h=60.0,
        max_accel_w=20.0,
        max_accel_h=20.0,
        resize_stop_on_change=True,
        resize_stop_delay=4,
        resize_cancel=True,
        resize_hyst=1,
        resize_cooldown=3,
        resize_ttg=10,
        sticky_translation=False,
    )
    cpp_cfg = make_cpp_config(core, arena, cfg)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg)

    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    # Grow width to build speed
    dest_grow = make_box_at_center(c, w=w + 40.0, h=h)
    cpp_box.forward(_to_cpp_bbox(core, dest_grow))
    py_box.forward(dest_grow)
    # Reverse to trigger braking
    dest_shrink = make_box_at_center(c, w=w - 40.0, h=h)
    cpp_box.forward(_to_cpp_bbox(core, dest_shrink))
    py_box.forward(dest_shrink)
    # Flip back to cancel (hysteresis=1)
    cpp_box.forward(_to_cpp_bbox(core, dest_grow))
    py_box.forward(dest_grow)

    rstate = cpp_box.resizing_state()
    assert int(getattr(rstate, "stop_delay_w", 0) or 0) == 0
    assert int(py_box._resize_stop_delay_w) == 0

    # During cooldown, another direction change should not start a stop-delay
    dest_shrink2 = make_box_at_center(c, w=w - 20.0, h=h)
    cpp_box.forward(_to_cpp_bbox(core, dest_shrink2))
    py_box.forward(dest_shrink2)
    rstate = cpp_box.resizing_state()
    assert int(getattr(rstate, "stop_delay_w", 0) or 0) == 0
    assert int(py_box._resize_stop_delay_w) == 0


def _should_limit_resize_speed_ttg():
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    cfg = dict(
        max_speed_x=0.0,
        max_speed_y=0.0,
        max_accel_x=200.0,
        max_accel_y=200.0,
        max_speed_w=200.0,
        max_speed_h=200.0,
        max_accel_w=200.0,
        max_accel_h=200.0,
        resize_stop_on_change=True,
        resize_stop_delay=0,
        resize_cancel=False,
        resize_hyst=0,
        resize_cooldown=0,
        resize_ttg=5,
        sticky_translation=False,
    )
    cpp_cfg = make_cpp_config(core, arena, cfg)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg)

    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    dist = 50.0
    dest = make_box_at_center(c, w=w + dist, h=h)

    cpp_box.forward(_to_cpp_bbox(core, dest))
    py_box.forward(dest)

    rstate = cpp_box.resizing_state()
    limit = dist / cfg["resize_ttg"]
    assert abs(float(rstate.current_speed_w)) <= limit + 1e-3
    assert abs(float(py_box._current_speed_w)) <= limit + 1e-3


def _should_snap_translation_speed_ttg():
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    cfg = dict(
        max_speed_x=200.0,
        max_speed_y=0.0,
        max_accel_x=200.0,
        max_accel_y=0.0,
        stop_on_change=False,
        stop_delay=0,
        cancel=False,
        time_to_dest_speed_limit_frames=10,
        time_to_dest_stop_speed_threshold=0.25,
        pan_smoothing_alpha=0.0,
        sticky_translation=False,
    )
    cpp_cfg = make_cpp_config(core, arena, cfg)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg)

    # Small distance should clamp speed below threshold and snap to zero.
    step = 2.0
    dest = init_box_t + torch.tensor([step, 0.0, step, 0.0], dtype=torch.float)
    cpp_box.forward(_to_cpp_bbox(core, dest))
    py_box.forward(dest)

    tstate = cpp_box.translation_state()
    assert abs(float(tstate.current_speed_x)) <= 1e-3
    assert abs(float(py_box._current_speed_x)) <= 1e-3


def _should_snap_resize_speed_ttg():
    torch = _torch()
    core = _core()

    arena = (0.0, 0.0, 1920.0, 1080.0)
    init_box = (400.0, 400.0, 800.0, 600.0)
    init_box_t = torch.tensor(init_box, dtype=torch.float)
    arena_t = torch.tensor(arena, dtype=torch.float)

    cfg = dict(
        max_speed_x=0.0,
        max_speed_y=0.0,
        max_accel_x=0.0,
        max_accel_y=0.0,
        max_speed_w=200.0,
        max_speed_h=200.0,
        max_accel_w=200.0,
        max_accel_h=200.0,
        resize_stop_on_change=True,
        resize_stop_delay=0,
        resize_cancel=False,
        resize_hyst=0,
        resize_cooldown=0,
        resize_ttg=10,
        resize_ttg_stop_threshold=0.25,
        sticky_translation=False,
    )
    cpp_cfg = make_cpp_config(core, arena, cfg)
    cpp_box = core.LivingBox("cpp", core.BBox(*init_box), cpp_cfg)
    py_box = make_py_box(torch, arena_t, init_box_t, cfg)

    from hmlib.bbox.box_functions import center, make_box_at_center, width, height

    c = center(init_box_t)
    w = width(init_box_t)
    h = height(init_box_t)
    dist = 2.0
    dest = make_box_at_center(c, w=w + dist, h=h)
    cpp_box.forward(_to_cpp_bbox(core, dest))
    py_box.forward(dest)

    rstate = cpp_box.resizing_state()
    assert abs(float(rstate.current_speed_w)) <= 1e-3
    assert abs(float(py_box._current_speed_w)) <= 1e-3


if __name__ == "__main__":
    main()
