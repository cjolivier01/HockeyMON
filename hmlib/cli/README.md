## HockeyMOM command-line tools (hm*)

This folder contains user-facing CLI commands that are installed with hm-prefixed names when you build/install the wheel. Each command targets a common workflow like stitching two camera angles, tracking players, clipping highlights, etc.

### Installed commands
- **hmtrack** — Run tracking/inference on a game video directory
- **hmstitch** — Stitch left/right camera videos into one panorama
- **hmcreate_control_points** — Compute control points with SuperPoint/LightGlue, DeDoDe/LightGlue, or LoFTR and update a .pto
- **hmplayers** — Analyze tracked players and generate per-player timestamp files
- **hmfind_ice_rink** — Detect the ice rink mask and save configuration for a game
- **hmorientation** — Inspect a game directory and label left/right camera sets
- **hmconcatenate_videos** — Normalize and concatenate multiple videos with ffmpeg

### Conventions and prerequisites
- **Game directory layout**: Most tools expect a game-id directory under `$HOME/Videos/<game-id>` with `config.yaml` and a stitched frame `s.png` (some tools will generate these). Example: `$HOME/Videos/ev-stockton-1`.
- **GPU acceleration**: Many commands can use CUDA for decoding/encoding or ML inference. Set `--gpus` or device flags as needed. CPU-only can work but is slower.
- **Common options**: A number of flags are shared across commands via `hm_opts` (e.g., `--game-id`, `--gpus`, `--output`, `--lfo/--rfo`, `--start-frame-time`). When in doubt, run `<cmd> --help`.

---

## hmtrack — Tracking and post-processing

Run end-to-end tracking on a game: optional stitching, detections, tracking, jersey numbers, pose, boundaries, and render the output video. Can operate from raw left/right videos or a pre-stitched file.

### Quick start
```bash
# From a configured game directory
hmtrack --game-id ev-stockton-1
# -> $HOME/Videos/ev-stockton-1/ev-stockton-1-tracking_output-with-audio.mp4

# From an explicit input file
hmtrack --input-video path/to/stitched_output-with-audio.mp4 --output tracking.mp4
```

### Notable options
- `--game-id <id>` — Use `$HOME/Videos/<id>` and its config
- `--input-video <file|dir|left,right>` — Override inputs (two files or a dir implies stitching)
- `--stitch` / `--force-stitching` — Force stitching phase
- `--stitch-rotate-degrees <float>` — After stitching, rotate the panorama about its center by the given degrees; keeps same dimensions (use small values to level the horizon)
- `--detect-jersey-numbers`, `--plot-jersey-numbers` — Jersey number detection and overlay
- `--plot-pose` — Pose skeletons overlay
- `--plot-tracking`, `--plot-trajectories` — Visualize tracking overlays (circles by default) and paths; use `--no-plot-tracking-circles` for boxes
- `--cvat-output` — Export data for CVAT
- `--save-tracking-data`, `--save-detection-data`, `--save-camera-data` — Write CSV artifacts
- `--camera-controller rule|transformer|gpt|drivegpt`, `--camera-model <pt>` — Use a learned camera controller; `drivegpt` checkpoints are trained with `camgpt_train.py --model-kind=drivegpt` and initialize compatible blocks from OpenDriveLab UniAD planning weights.
- `--output-video <file>` and `--no-save-video` — Control rendered output
- `--checkpoint` / `--detector` / `--reid` — Override model weights

### Variant sweeps (experiment YAML)
Run the same clip multiple times with different config overrides and combine the results into a single video (concat or tiled grid). Each variant is labeled on-screen with its parameter overrides.

Example experiment file:
```yaml
experiment:
  name: zoom_sweep
  clip:
    start_time: "00:10:00"
    duration: "00:00:08"
  overlay:
    prefix: "variant: "
    color: [255, 255, 255]
    position: [40, 60]
    max_lines: 6
  output:
    mode: tile
    path: output_workdirs/ev-stockton-1/experiments/zoom_sweep.mkv
    tile:
      rows: 2
      cols: 2
  variants:
    - name: base
      config: {}
    - name: hyst20
      config:
        rink:
          camera:
            resizing_stop_cancel_hysteresis_frames: 20
    - name: stop_thresh
      config:
        rink:
          camera:
            resizing_time_to_dest_stop_speed_threshold: 0.2
```

Run it:
```bash
hmtrack --game-id ev-stockton-1 --experiment-config zoom_sweep.yaml
```

Notes:
- `clip` settings in the experiment file override `--start-time` / `--max-time`.
- Per-variant `args:` overrides can be added if needed (e.g., `plot_pose: true`).

### Data artifacts and cross‑references
- `tracking.csv` stores per-frame track boxes (TLWH), IDs, scores/labels, and optional action columns.
- `detections.csv` stores per-frame detection boxes (TLBR), scores, and labels.
- `pose.csv` stores `PoseJSON` with a `predictions` list per frame.

---

## hmstitch — Stitch left/right camera videos

Stitch the two perspectives into a single panorama, optionally writing only the configuration (pto, mapping files) or generating the stitched video.

### Quick start
```bash
# Auto-discover from game-id and generate a stitched video
hmstitch --game-id ev-stockton-1 -o stitched_output-with-audio.mp4

# Configure only (no render)
hmstitch --game-id ev-stockton-1 --configure-only

# Try DeDoDe + LightGlue and generate remap TIFFs with OpenCV MAGSAC++
hmstitch --game-id ev-stockton-1 \
  --control-point-matcher dedode-lightglue \
  --mapping-backend opencv-magsac

# Use detector-free LoFTR matching (nona remains the mapping backend)
hmstitch --game-id ev-stockton-1 --control-point-matcher loftr

# Constrain wide-angle geometry to an affine transform with robust RANSAC
hmstitch --game-id ev-stockton-1 \
  --control-point-matcher dedode-lightglue \
  --mapping-backend opencv-affine-ransac
```

### Notable options
- `--game-id <id>` — Scan `$HOME/Videos/<id>` for left/right files and chapters
- `--lfo` / `--rfo <frames>` — Frame offsets (if sync known)
- `--stitch-frame-time HH:MM:SS` — Choose reference frame for alignment; `--start-frame-time` to start processing at a time
- `--stitch-rotate-degrees <float>` — Rotate the stitched output about its center by the given degrees; keeps same dimensions (use small values to level the horizon)
- `--max-control-points N` — Control points for homography
- `--control-point-matcher superpoint-lightglue|dedode-lightglue|loftr` — Feature matching backend (default: `superpoint-lightglue`); DeDoDe caps its longest input dimension at 1920 pixels and restores matches to source-image coordinates
- `--mapping-backend nona|opencv-magsac|opencv-affine-ransac` — Mapping TIFF generator (default: `nona`); the native OpenCV options use either `findHomography` with MAGSAC++ or `estimateAffine2D` with RANSAC and write the same RGB/alpha and X/Y map artifacts
- `--blend-mode laplacian|multiblend|gpu-hard-seam` — Blending mode
- `--batch-size`, `--stitch-cache-size`, `--multi-gpu` — Performance tuning
- `--show` / `--show-scaled` — Preview frames

---

## hmcreate_control_points — Compute and write Hugin control points

Synchronize by audio (optional), extract frames, compute feature matches with a selectable learned matcher, and update the Hugin `.pto` project file with control points. Helpful before stitching. DeDoDe and LoFTR weights are downloaded by Kornia on first use.

### Quick start
```bash
# From game-id
hmcreate_control_points --game-id ev-stockton-1 --max-control-points 300

# From explicit videos
hmcreate_control_points --left left.mp4 --right right.mp4 --synchronize-only
```

### Notable options
- `--left` / `--right` — Input videos (or supply `--game-id`)
- `--synchronize-only` — Print left/right frame offsets and exit
- `--max-control-points N` — Limit control point matches
- `--control-point-matcher superpoint-lightglue|dedode-lightglue|loftr` — Feature matching backend
- `--mapping-backend nona|opencv-magsac|opencv-affine-ransac` — Mapping TIFF generator
- `--scale <float>` — Downscale when optimizing/visualizing

---

## hmplayers — Analyze tracked players and emit highlight ranges

Uses `tracking.csv` and `camera.csv` produced by `hmtrack` to infer per-shift intervals for jersey numbers and generate per-player timestamp files, plus a helper shell script to assemble highlights.

### Quick start
```bash
hmplayers --game-id ev-stockton-1
```

### Outputs
- Writes files like `player_29.txt` with HH:MM:SS ranges and a `make_player_highlights.sh` script in the game directory.

---

## hmfind_ice_rink — Detect ice rink mask and save

Run the rink segmentation model on the stitched frame `s.png` in a game directory, save the combined mask and related metadata into the private config, and print centroid/edges info.

### Quick start
```bash
hmfind_ice_rink --game-id ev-stockton-1 --show
```

### Notable options
- `--scale <float>` — Optional image scaling
- `--device cpu|cuda[:N]` — Choose inference device
- `--force` — Recompute and overwrite existing mask configuration

---

## hmorientation — Label left/right inputs and save to private config

Discover available video files in a game directory, infer which perspective is left vs right using rink masks, and write an ordered chapter list per side to the private config.

### Quick start
```bash
hmorientation --game-id ev-stockton-1
```

---

## hmconcatenate_videos — Normalize and concatenate multiple videos

Normalize resolution, fps, and audio across inputs, optionally trim each segment, and concatenate to one or more outputs using ffmpeg. Supports GPU NVENC and CPU x265 encoders.

### Quick start (manual inputs)
```bash
# NVENC VBR
hmconcatenate_videos --use-gpu --inputs a.mp4 b.mp4 c.mp4 -o out.mkv \
	--video-quality vbr --cq 19 --b-v 30M --maxrate 60M

# Lossless (NVENC)
hmconcatenate_videos --use-gpu --video-quality lossless --inputs a.mp4 b.mp4 -o out.mkv

# x265 CRF
hmconcatenate_videos --inputs a.mp4 b.mp4 -o out.mp4 --video-quality crf --crf 20 --preset slow
```

### Quick start (game-aware stitching lists)
```bash
# Concatenate left and right stitching chapter lists into two outputs under the game dir
hmconcatenate_videos --game-id ev-stockton-1
# -> $HOME/Videos/ev-stockton-1/left.mp4
# -> $HOME/Videos/ev-stockton-1/right.mp4

# Left-only (explicit output)
hmconcatenate_videos --game-id ev-stockton-1 --left-only -o /tmp/left_concat.mkv

# Right-only (default name in game dir)
hmconcatenate_videos --game-id ev-stockton-1 --right-only
# -> $HOME/Videos/ev-stockton-1/right.mp4
```

### Notable options
- Inputs:
  - `--inputs a.mp4 b.mp4 ...` — Explicit list (2+ files required).
  - `--game-id <id>` — Discover left/right chapter lists via `hmorientation`/private config.
  - `--left-only` / `--right-only` — Limit to one side when using `--game-id`.
  - When `--game-id` is used with both sides active (default), `--output` is optional and defaults to `<game-dir>/left.mp4` and `<game-dir>/right.mp4`.
  - When `--game-id` is used with a single side, `--output` (if provided) targets that side; otherwise a side-based default is used.
- Normalization:
  - `--clip START-END` — Per-input trims; repeat per input or leave empty for full.
  - `--aspect-mode pad|crop|stretch` — Aspect handling when normalizing.
  - `--force-res WxH`, `--force-fps NUM/DEN` — Force target profile.
  - `--audio-rate`, `--audio-channels`, `--audio-bitrate` — Audio normalization.
  - `--use-gpu` — Use NVENC (hevc_nvenc) and GPU scaling where applicable; otherwise use libx265.

---

## Shared/common flags (hm_opts)

Some commands accept these shared flags:
- `--game-id <id>` — Target game
- `--gpus "0,1"` — GPU selection for ML/inference
- `--output <file>` — Output artifact path
- `--lfo` / `--rfo` — Frame offsets for left/right
- `--start-frame-time HH:MM:SS` and `--stitch-frame-time HH:MM:SS` — Time selection
- `--stitch-rotate-degrees <float>` — Optional post-stitch rotation (stitching-enabled flows)
- `--fp16` — Mixed precision in some stages
- `--show` / `--show-scaled` — Preview frames
- `--cache-size`, `--stitch-cache-size` — Pipeline buffering

---

## Tips
- Use `hmorientation` first on new game folders to auto-label left/right and save chapter lists.
- If alignment drifts, run `hmcreate_control_points` to refresh control points; then `hmstitch`.
- After stitching, run `hmfind_ice_rink` to cache the rink mask; `hmtrack` will use it to improve boundaries.
- Use `hmplayers` to generate per-player time ranges, then `hmconcatenate_videos` to produce highlight reels.

---

## Help
Each command supports `--help` to print the full set of options and defaults.
