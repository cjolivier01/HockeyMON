# Scoreboard selection

When a game has no `rink.scoreboard.perspective_polygon`, the selector runs the
trained Mask2Former R50 once and starts with editable suggested corners. Save
is still required. Existing selections and the four-zero "No Scoreboard"
sentinel skip detection, including when explicitly reopening the selector.
A missed detection leaves an empty editor. Model failures are reported in the
log and editor while manual selection remains available.

Inference uses PyTorch's GPU backend (CUDA or ROCm) when available and retries
once on CPU after GPU OOM, releasing failed GPU objects before retrying. With
no GPU device it uses CPU directly. Other errors do not trigger a CPU retry.
No model is retained for frame processing, and saved-polygon runs never load
or download it.

To try the automatic draft, run `./hm_run.sh --game-id=<game-id> -t=5` for a
game without a saved scoreboard polygon. The normal setup flow opens the
selector; review the suggested corners and save. The first use downloads
the model. To open just the editor on an existing `s.png`, run
`PYTHONPATH=. python -m hmlib.scoreboard.selector --game-id=<game-id>`.
This preserves a saved polygon as the initial draft instead of redetecting it.

The portable TorchScript model downloads on demand into
`~/.cache/hstream/models/scoreboard-mask2former-r50-2cd82b9e3f1d87d1.pt`, with
SHA-256 verification and atomic publication. An explicit local export can be
selected with `HM_SCOREBOARD_TORCH_MODEL`. The asset comes from the same
checkpoint as HStream's native ONNX model; runtime does not need MMDetection.
HStream's `scripts/export_scoreboard_model.py` and
`docs/scoreboard-proposals.md` document the export and shared tensor contract.
Mask processing is bounded to 2048x768 and corners retain original panorama
coordinates. Multiple boards remain ambiguous, so review the suggested board.

The browser selector displays a reduced panorama when the source exceeds 8192
pixels on either axis or a 96 MiB RGBA display budget. The Python
`max_display_height` option can reduce it further. Zoom, selection, initial
points, and saved polygons always use the original image's pixel coordinates.

Only the reduced RGB image is retained and PNG-encoded for the browser. Tensor
inputs are resized on their current device before host transfer. File inputs
still use Pillow's full PNG decoder, so source decoding needs memory for the
original image and remains subject to Pillow's decompression-bomb limits. The
selector additionally rejects sources larger than 65535 pixels per axis or
256 MiPixels before requesting pixel data; this is a display proxy, not a
streaming PNG decoder.
