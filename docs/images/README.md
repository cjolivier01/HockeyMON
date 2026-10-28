# Image provenance

## `blackhawks-p3_play_tracking.webp`

The clip at the top of the repository [README](../../README.md). It replaced
`tv-12-1-r2_fast_play.gif`, which was committed in `ed3b9e13` on 2026-02-04
with no record of how it was made and no game left on disk to redo it from.

Rendered on 2026-10-27 on an NVIDIA RTX 5090, from `8810b4cf`. The input is
`blackhawks-p3-stitched_output-with-audio-4.mp4`, an 8192 × 2732 HEVC stitched
panorama that HockeyMONStream produced for that game; `hmtrack` detected,
tracked and framed it from there, and wrote a 4856 × 2732 cropped output.

```sh
hmtrack --input-video blackhawks-p3-stitched_output-with-audio-4.mp4 \
  -s 00:36:25 -t 65 --no-audio --no-progress-bar \
  --plot-tracking --plot-teams --home-is-dark --plot-moving-boxes
```

What is on screen, since `--plot-moving-boxes` draws more than it sounds like
(`hmlib/camera/play_tracker.py:1301-1336`, `hmlib/camera/living_box.py`):

| Overlay | What it is |
| --- | --- |
| The frame | Live box 1, the camera. `hmtrack` crops to it, so its outline sits on the frame edge — magenta while it is moving, grey while translation is frozen, which is most of this clip (`living_box.py:116-121`) |
| Blue rectangle | Live box 0, the fast region of interest the camera chases. It is wider than the shot for most of the clip, so one of its two vertical edges is in frame about half the time and both never are; over the last two seconds the right edge and both of its corners sweep across |
| Two circles at frame centre | The camera's sticky and unsticky translation thresholds: how far the fast ROI has to move before the camera follows it (`living_box.py:202-220`) |
| Cyan line | Camera centre to fast-ROI centre, with two thin legs splitting that offset into X and Y. A line, not an arrow — `draw_arrows` is commented out (`living_box.py:201`) |
| Red numbers | Track ids from the tracker, not jersey numbers |
| Thin ellipses | One per track, in that track's colour |
| Red tick marks | The camera's shrink threshold. Its green grow counterpart is off-screen (`living_box.py:86-101`) |
| White dashed rectangle | The size the camera is converging toward: the fast ROI scaled by `follower_box_scale_width/height` (1.45) and centred on the camera. Turns red when the size is constrained, and is mostly off-screen, so it shows up as two horizontal dashed lines in about a third of the frames (`living_box.py:72-84`) |

Along the far boards the ids overlap into an unreadable row. That is what the
overlay does at this camera distance, not an encoding artefact.

Two notes for anyone repeating it. The shipped graph enables the TensorRT
detector (`hmlib/config/aspen/tracking.yaml:116`,
`detector_factory.params.trt.enable: true`), and the pinned
`torch-tensorrt==2.11.0+cu130` has no Python 3.14 wheel —
`requirements.txt:16` excludes it there for that reason. On a 3.14
environment the run dies on the first batch with `Requested TensorRT detector
execution failed`, and there is no fallback to PyTorch. Either use a 3.13
environment, or turn it off:

```sh
hmtrack --config-override aspen.plugins.detector_factory.params.trt.enable=false \
  --config-override aspen.plugins.detector_factory.params.nms_backend=torchvision \
  ...
```

That runs the detector on PyTorch, at roughly 0.07x real time on this machine;
65 s of video took about 15 minutes. Both overrides are needed: without the
second one `params.nms_backend` stays `trt`
(`hmlib/config/aspen/tracking.yaml:126`).

Do not combine the override pair with `--config`. `hmlib/hm_opts.py:325-326`
rejects an override whose key path does not already exist, and
`aspen.plugins` exists only after `hmlib/cli/hmtrack.py:2827-2833` merges the
shipped graph — which `:2817-2821` skips the moment any `--config` appears.
The render above predates the override form and used an equivalent sidecar
YAML, passed as `--config hmlib/config/aspen/tracking.yaml --config
no-trt.yaml` for that same reason: the shipped graph has to be named again.
It also passed `--detector-nms-backend torchvision`, which did nothing —
`hmlib/cli/hmtrack.py:1824` and `:1852` read that flag only on the ONNX and
TensorRT paths. The override pair above was run against the same input and
reaches frame delivery; it is the form to use.

`--plot-overhead-rink` is accepted but has no effect, and was not used here.
The renderer is written out — `hmlib/transforms/overlays.py:224-467` draws a
whole minimap, and `hmlib/camera/play_tracker.py:1832-1848` already supplies
its inputs — but the flag has no entry in `hmlib/hm_opts.py`'s
argument-to-config table, so nothing ever sets `overhead_rink` and the code is
unreachable rather than proven.

The excerpt is 6.5 s starting 42.5 s into that run. No HOME or AWAY badge
appears in it, and it is not for want of crops: `hmlib/config/baseline.yaml`
puts the first label about six seconds of play in at the defaults. Either the
clusterer could not rank its two groups — `min_lightness_gap` and, through the
balance check, `min_cluster_share` are the two conditions at
`hmlib/team/clusterer.py:258` — or it ranked them and no track accumulated
enough evidence past `team.min_margin` to earn a badge. Which it was is not
recorded; an empty screen is what the overlay does in both cases.

```sh
ffmpeg -ss 42.5 -i tracking_output.mkv -t 6.5 \
  -vf "fps=12,scale=1040:-2:flags=lanczos" -c:v libx264 -crf 10 -an src.mp4
ffmpeg -i src.mp4 -c:v libwebp_anim -q:v 54 -compression_level 6 -loop 0 \
  blackhawks-p3_play_tracking.webp
```

1040 × 586, 78 frames at 12 fps, infinite loop, 1,808,940 bytes. Trimming,
dropping the 59.94 fps source to 12, scaling and compression are the only
processing — through a near-lossless x264 intermediate, so three generations
in total. The 6.5 s run continuously; no cut, no overlay added or removed
afterwards, nothing retouched.

Animated WebP rather than GIF: the file it replaces was 9,322,167 bytes for
480 × 270, because a full-frame pan over ice texture defeats GIF interframe
compression. This one is 4.7 times the pixels at a fifth of the size. GitHub
serves a committed `.webp` as `image/webp`. The file has no still-image chunk,
so a decoder that cannot read animated WebP gets nothing rather than a first
frame; every browser that decodes WebP at all has handled the animated form
since 2020.

## `aspennet-*.svg`

Hand-drawn AspenNet execution diagrams for the threading section of the
repository README. No provenance beyond their commit.
