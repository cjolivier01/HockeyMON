# Rink leveling and panorama crop

After calibrating a game with `stitching.mapping_backend: nona` and
`stitching.run_autooptimizer: true`, open the editor:

```sh
hmlevel --game-id chicago-3
# From a checkout:
python -m hmlib.cli.level_stitching --game-id chicago-3
```

The command prints a loopback browser URL. `--no-browser` leaves browser
launching to you; `--port` selects a fixed port for an SSH tunnel. Keep the
full URL, including its fragment token. The editor requires Hugin's
`pano_trafo`, `pano_modify`, and a NONA version supporting PNG output with
internal blending. Missing tools and unsupported projects produce errors.

Select at least three tall, spatially separated vertical posts across the
source camera tabs. Click each post's two endpoints, then choose **Estimate
pitch and roll**. The calculation preserves the displayed yaw and reports
the inlier count, excluded marks, and angular residual. It uses the original
camera pixel coordinates and the published calibration's lenses and poses.
Short, inconsistent, or closely clustered marks cannot determine tilt and
must be adjusted. Projects with camera translation are unsupported.

You can also enter yaw, pitch and roll directly. **Render preview** applies
the requested rotation to a private project and renders the full panorama at
no more than 1920 pixels per dimension. Choose **Automatic crop**, drag a
manual rectangle, or enter normalized crop percentages. The crop overlay
shows the retained region of this full canvas; render again after editing.
Previews use NONA's internal seam blender, so runtime seam/color processing
may look different. The projection, rotation and crop geometry are shared.

**Save for next calibration** writes only the game's private
`stitching.projection_framing` config. Recalibrate and restart tracking to
apply it. The editor does not change an active panorama, mapping TIFFs, or
source images. Crop-only saves retain inherited rink rotations; choosing a
new rotation creates an explicit game override, including an explicit zero.
The saved Program edge-rotation angles remain available, but that extra
rotation is suppressed while the loaded NONA calibration includes pitch or
roll leveling. A yaw-only calibration keeps Program edge rotation enabled.

A save requires a successful preview of the current settings and marks. If
calibration, source images, effective calibration settings or private config
change, reopen the editor. Calibration lock contention produces a busy error.
The private config is replaced atomically under the calibration lock;
applications that also edit this config should use that lock.

Source previews are bounded to 2048 pixels per dimension after Pillow decodes
the original images. Pillow's image-size safeguards remain enabled; this is
not a streaming decoder for arbitrarily large source files.
