# Player size filtering

Program Controls → Tracking provides three controls:

- **Ignore largest players (0 = off)** excludes the configured number of largest tracked bounding-box areas. The default is 1; enter 0 to disable it.
- **Ignore oversized players** enables the additional percentage filter. It defaults to unchecked.
- **Larger than other players' average (%)** sets the percentage above the average area of the other players. The default, 100%, excludes areas strictly greater than twice that average.

Changes apply live and Save Preset stores them for the game. These filters select players before both play framing and breakaway detection; they apply independently of the fast/follower motion tuning checkboxes.

The largest-count filter runs first. Each remaining player is then compared with the mean area of the other remaining players, excluding itself. Percentage decisions all use that same snapshot, without repeatedly recalculating the average after removals. Size filtering always leaves at least three players; when fewer than four are tracked, neither filter removes players. Ties in the largest-count filter follow input order.

The matching defaults in hstream's `configs/baseline.yaml` and HockeyMON's `hmlib/config/baseline.yaml` are:

```yaml
rink:
  tracking:
    cam_ignore_largest: true
    cam_ignore_largest_count: 1
    cam_ignore_oversized: false
    cam_oversized_percent: 100
```

`cam_ignore_largest` remains the compatibility enable switch for existing rink and game configurations. A legacy `false` still disables the count filter. Editing the count in the Program UI updates both the count and that switch. The percentage filter is independent.

Native play-tracker YAML uses `ignore-largest-bbox`, `ignore-largest-bbox-count`, `ignore-oversized-bboxes`, and `oversized-bbox-percent`. HockeyMON's native and Python play trackers use the same selection rules. Size means bounding-box **area**, not height or real-world player size; the filter helps with nearby referees and bench traffic but does not classify those roles.
