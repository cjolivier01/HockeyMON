# Camera-policy boundaries in training exports

The standard Aspen camera export writes `camera_policy.csv` alongside
`camera.csv`, and `camera_fast_policy.csv` alongside `camera_fast.csv`. Each
headerless row contains `Frame,PolicyJSON`. The JSON uses schema
`hm-camera-policy-v1` and records the startup policy or a subsequent change.
Output labels, custom camera filenames, and publication generation suffixes
are shared with each companion: `trial_camera-3.csv` pairs with
`trial_camera_policy-3.csv`.

The tracker freezes each event after live controls are applied and before that
frame advances the camera policy. A batch may contain several events. The
snapshot contains camera settings, controller mode, applied fast/follower
selectors, applied stitch rotation, and canvas/play-area dimensions. Resetting or
reloading camera settings creates a boundary when those effective values
change. Repeating unchanged controls does not create another boundary. Color
settings are excluded from camera-policy comparisons. Aspen stitching attaches
the rotation used to render each batch, so a UI request during a batch creates
a geometry boundary when the affected pixels arrive. Legacy stitching uses
its pinned startup rotation; the CLI requires Aspen stitching for live
multi-camera controls and the legacy dataloader return format is unchanged.

Events are flushed and synchronized before their associated camera rows can
be flushed. A persistence failure propagates through pipeline finalization and
prevents successful output publication. The existing output publisher copies
all companion CSVs into the same generation before publishing `tracking.csv`.
Keep policy companions with their camera CSVs when copying datasets manually.

GPT/DriveGPT and transformer training loaders split windows at each policy
boundary and reset previous-camera features at the start of each run. For
example, a policy change at source frame 105 splits contiguous frames
100–109 into 100–104 and 105–109. Original source frame IDs and every camera,
tracking, and detection row are preserved. Fast-camera-only training reads
the fast camera's companion; combined slow/fast training validates both files
and splits at either set of boundaries. Older exports without policy files keep their
existing numeric-gap behavior; an existing malformed or mismatched companion
raises an error instead of silently joining policies.

This adapts hstream's live telemetry policy boundaries to HM's CSV training
loaders. DeepStream timestamps, seek epochs, and native configuration-event
sidecars are not required by HM's sequential source-frame export.
