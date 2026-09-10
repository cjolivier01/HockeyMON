# Stitching calibration

`hmstitch` and two-camera `hmtrack` resolve the effective `stitching` configuration
before extracting frames or changing a cached calibration. The dataset calibration
path uses the same settings. Invalid combinations raise an error before cleanup.

The bundled default uses `opencv-magsac`, rectilinear output, and no Hugin
optimizer. `opencv-affine-ransac` also supports rectilinear output. Other
projections require NONA and explicit optimizer opt-in:

```yaml
stitching:
  mapping_backend: nona
  run_autooptimizer: true
  projection: general-panini
  projection_parameters:
    general-panini: [100, 0, 0]
  camera_config: gopro-mission-1
  rink_config: vallco
  projection_framing:
    auto_fov: false
    horizontal_fov: 180
    auto_canvas: true
    auto_crop: false
    crop: [0, 1, 0, 1]
  max_output_width: 7680
```

All 22 Hugin projections are supported. Parameterized projections are
`albers-equal-area-conic`, `biplane`, `triplane`, and `general-panini`; values
must respect Hugin's ranges and use increments of 0.01. Unsupported FOVs are
rejected, including rectilinear FOVs above 179 degrees. The generated PTO is
checked to detect Hugin clamping a requested value.

Camera presets select source-image FOVs. Override either axis with
`camera_fov.horizontal_fov` or `camera_fov.vertical_fov`. A custom camera identifier
requires a definition under `camera_configs` or both explicit FOVs.

A selected rink supplies shared camera-space yaw, pitch, and roll before NONA
projection. Omitted or null `projection_framing.rotation_degrees` inherits those
angles; `[0, 0, 0]` explicitly disables the inherited rotation. Reading settings
does not modify this inheritance. This is separate from rotating the stitched
bitmap after calibration.

NONA framing supports Auto FOV/canvas/crop and a manual crop expressed as
`[left, right, top, bottom]` fractions of the full projected canvas. A custom crop
and Auto crop cannot be active together. OpenCV mapping is determined by the fitted rectilinear transform. Non-default
projection framing, including a selected rink with nonzero rotation, requires
NONA; unsupported framing is rejected before calibration.

`max_output_dimension` caps both canvas dimensions. `max_output_width` caps width.
NONA caps the full projected canvas before remapping (including any crop), so
cropping may produce a smaller width. Native OpenCV applies caps during map
construction. Width-cap support requires rebuilding the HockeyMON native
extension. Both calibration and cache reuse record the complete resolved settings;
changing FOV, projection, framing, optimizer choice or caps invalidates old maps.

## Multiple calibration frames

`stitching.calibration_frame_count` selects 1–64 synchronized pairs (default 4).
Sampling starts at the configured calibration time and takes consecutive pairs;
short clips use the remaining available pairs with a warning. Source dimensions
must remain stable. Pooled correspondences are tried first, followed by the
individual pairs with the most matches. Duplicate static correspondences are
removed before pooling is sampled to the requested control-point limit.

Only rejected geometric alignment can trigger another candidate. Model, disk,
seam, resource-limit and publication errors remain terminal with their original
cause. NONA alignment requires a reported optimization RMS at most 50 pixels;
OpenCV distinguishes geometric rejection in the rebuilt native extension.
Older native extensions retain terminal behavior for untyped mapping errors.
Frame count and the control-point limit are recorded in calibration provenance.

## AKAZE and GoPro KB4 profiles

Select `stitching.control_point_matcher: akaze-hamming` (`akaze` is an alias) for
OpenCV M-LDB detection and mutual Hamming matching without model downloads.
Rebuild the native extension to expose its CPU AKAZE detector; this does not
depend on the Python OpenCV package exporting AKAZE.
Detection is limited to 1920 pixels and 2000 keypoints per camera, using the
facing camera halves and epipolar filtering. At least six matches are required.

When present, game-local `left_calibration.json` must contain both
`left_uniforms` and `right_uniforms` objects with `width`, `height`, `fx`, `fy`,
`cx`, `cy`, and four KB4 coefficients in `d`. Missing profiles are reported and
use original image coordinates. Existing malformed or incomplete profiles fail.

Calibrated AKAZE detects on rectified images and composes the inverse projective
or affine transform with KB4 distortion when generating original-camera remap
coordinates. It requires an OpenCV backend and a rebuilt native extension; NONA
rejects calibrated points. MAGSAC validates spatial consensus and can search up
to twelve projective hypotheses without lowering consensus requirements when
removing a rejected hypothesis. Profile contents are pinned and fingerprinted
for the generation so profile edits invalidate cached maps.

## Standalone control-point command

`python -m hmlib.cli.create_control_points --game-id GAME` reads the same game
calibration settings as the tracker, including when `--left` and `--right`
override the input files. Video inputs use synchronized multi-frame calibration;
two PNG inputs use a single pair through the same project builder.

Unspecified matcher, mapping backend and control-point limits defer to the
effective settings. For example, `--mapping-backend nona --run-autooptimizer`
explicitly enables the Hugin optimizer. `--no-run-autooptimizer` can disable it
when using an OpenCV backend. `--device`, `--calibration-frame-count`,
`--stitch-frame-time`, and paired integer `--lfo`/`--rfo` overrides are supported.
`--scale` is a positive Hugin-only relative scale; native backends use
`--max-output-dimension`.

Frame-array callers of `configure_stitching` use temporary input PNGs and the
public shared builder. A failed matcher or image write preserves existing
reference images and removes the temporary inputs.
