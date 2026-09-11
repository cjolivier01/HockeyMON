# Rink leveling during NONA calibration

Game-based Hugin/NONA calibration opens an optional browser selector after panorama alignment and initial
projection/framing preparation. Mark the top and bottom of at least three upright wall or glass posts, including at
least one post in each camera. Pitch and roll update automatically 150 ms after each completed selection or
dragged-point adjustment; yaw is preserved.

**Preview angles** runs `pano_modify` and a bounded, 1600-pixel NONA still render. Inspect the rink walls and both
ends before choosing **Use angles**. **Skip leveling** immediately continues with the previously configured framing.
**Cancel calibration** stops the complete calibration run rather than treating cancellation as a skip. Only **Use
angles** saves an explicit `stitching.projection_framing.rotation_degrees` value in the game's private configuration.

The selector does not rerun feature matching or `autooptimiser`. If accepted angles differ from the configured
angles, calibration restores the aligned PTO and reapplies only the inexpensive projection/framing step. The preview
also starts from that preserved pre-framing PTO, applies the same projection, parameters, absolute rotation, FOV,
canvas, and crop rules as the final output, and only then downscales the framed canvas. Final full-resolution NONA
maps and the Enblend seam are generated once the selector is complete. Output-size limiting also resizes the framed
PTO with `pano_modify`, without another optimizer pass. If the calibration backend exits while the selector is open,
its server closes as cancellation rather than silently continuing as though leveling were skipped.

The fit uses the calibrated Hugin lens model. `pano_trafo` maps each selected source-image endpoint to a viewing ray;
each post then defines a plane through the camera center. A robust fit finds the shared vertical direction, rejects
inconsistent marks, and reports the angular residual. Widely separated, long posts work better than clustered or
short marks. Avoid rink corners, sloping beams, and objects that are not truly vertical.
