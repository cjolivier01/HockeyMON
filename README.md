HockeyMON

![Looping clip of hockey, cropped to the camera the play tracker chose. A red track id and a thin ellipse follow each skater, a blue rectangle marks the fast region of interest the camera is chasing, two circles at the centre of the frame show how far that region has to move before the camera follows it, and a cyan line runs from the centre to the middle of the blue rectangle](docs/images/blackhawks-p3_play_tracking.webp)

*6.5 continuous seconds of `hmtrack` output. The frame itself is the camera —
the outline on its edge is that box. The blue rectangle is the fast region of
interest it is chasing, wider than the shot for most of the clip so one of its
sides runs off the edge, and the cyan line is how far the camera still has to
go. Nothing was drawn after the run; the
[image notes](docs/images/README.md) name every overlay.*

## Docs
- Repository guidelines: see [AGENTS.md](AGENTS.md).
- How to contribute: see [CONTRIBUTING.md](CONTRIBUTING.md).
- Shared HStream desktop: [launch and install](hm-ui/README.md), [backend and workflow ownership](docs/shared-ui-backend.md).

## System dependencies and native tools

For a fresh Ubuntu/Debian machine you should run, from the repo root:

```bash
./env/install_deps.sh
./build_deps.sh
```

- `env/install_deps.sh` installs the system libraries needed by the native C++ tools used by HockeyMON:
  - Hugin CLI tools (`pto_gen`, `autooptimiser`, `pano_modify`, `pano_trafo`, `nona`) for stitching and optional rink leveling.
  - Enblend/Enfuse and image/codec libraries (tiff, jpeg, png, OpenEXR, FFTW, wxWidgets, etc.).
  - These are required so the Bazel/CMake builds in `external/hugin` and `external/enblend-enfuse` succeed and the CLI binaries can run.
- `build_deps.sh` runs the Hugin Bazel build:
  - Changes into `external/hugin` and runs `bazelisk run //:install_tree -- --prefix="$CONDA_PREFIX"`.
  - This drives CMake to build and install the Hugin command-line tools into the active conda environment.
  - Those tools are then available to `hmcreate_control_points` and `hmstitch` for pano generation.

If you are on a different distro, use `env/install_deps.sh` as a reference list of required packages and install the equivalents manually, then run `./build_deps.sh`.

If you do **not** build these native tools, you will not be able to run Hugin-based stitching from raw left/right videos, but you can still run tracking on an already-stitched panorama (see below).

## Building and running hmtrack and stitching

You can run the CLIs either from an installed wheel (preferred for end users) or directly from this repo for development.

### From an installed wheel (recommended)

After building wheels with `./bdist_wheel` and installing them into your environment (e.g. `pip install dist/hmlib-*.whl dist/hockeymon-*.whl`), the CLI entry points are available on your `PATH`:

```bash
# Full pipeline from a configured game directory (stitch + track)
hmtrack --game-id ev-stockton-1
# -> $HOME/Videos/ev-stockton-1/ev-stockton-1-tracking_output-with-audio-1.mp4

# Explicit stitching step
hmstitch --game-id ev-stockton-1 -o stitched_output-with-audio.mp4
```

### From the repo checkout (no wheel install)

From the repo root you can invoke the same commands via Python modules:

```bash
# Tracking (equivalent to hmtrack)
python -m hmlib.cli.hmtrack --game-id ev-stockton-1
# -> $HOME/Videos/ev-stockton-1/ev-stockton-1-tracking_output-with-audio-1.mp4

# Stitching (equivalent to hmstitch)
python -m hmlib.cli.stitch --game-id ev-stockton-1 -o stitched_output-with-audio.mp4
```

Ensure your `PYTHONPATH` includes the repo root when running directly (for example, see `hm_run.sh` for how we set `PYTHONPATH` in development).

`hm-ui` opens the shared HStream Qt desktop with HockeyMON processing. Adding
`--camera-ui=1` to `hmtrack` or `hmstitch` opens that desktop and starts the
requested workflow. Install the matching HStream desktop/native helpers or set
the executable paths described in [hm-ui/README.md](hm-ui/README.md).

When a game-based NONA calibration creates new stitching maps, it also opens the optional
[rink-post leveling selector](docs/rink-leveling-selection.md) after panorama alignment and before the final maps.

## Running hmtrack with a pre-stitched or panoramic video

If you already have a stitched panorama (for example, produced by `hmstitch` on another machine or by an external tool) and you do not have Hugin/enblend built locally, you can run tracking directly on that video and skip the stitching phase:

```bash
# Using an explicit stitched video (no game-id layout required)
hmtrack --input-video path/to/stitched_panorama.mp4 --output tracking.mp4

# From the repo without installed wheel
python -m hmlib.cli.hmtrack --input-video path/to/stitched_panorama.mp4 --output tracking.mp4
```

Optionally, if you still use a game directory layout under `$HOME/Videos/<game-id>`, you can combine `--game-id` with `--input-video` so the tracker can pick up game-specific config while reading frames from your pre-stitched file.

## Unified Configuration
- HM configs (`hmlib/config/` camera/rink/game/model) and Aspen graph configs (`hmlib/config/aspen/`) can be combined.
- Aspen YAMLs are nested under a top-level `aspen:` key (e.g., `aspen.plugins`, `aspen.inference_pipeline`, `aspen.video_out_pipeline`).
- The CLI merges multiple YAML files in order using `--config` (repeatable). Later files override earlier values and add new fields.
- Backward-compat `--aspen-config` has been removed. Use `--config` exclusively.

Examples
- Single file (everything in one YAML):
  - `./hm_run.sh --config my_unified.yaml`
- Compose baseline + Aspen graph:
  - `./hm_run.sh --config hmlib/config/baseline.yaml --config hmlib/config/aspen/tracking.yaml`
- Add rink- or game-specific overrides after base files:
  - `./hm_run.sh --config base.yaml --config rink/vallco.yaml --config hmlib/config/aspen/tracking_pose.yaml`

YAML structure (snippet)
```yaml
camera:
  name: GoPro
game:
  game_id: 2024-10-01
aspen:
  inference_pipeline: [...]
  video_out_pipeline: [...]
  plugins:
    detector_factory: {...}
    tracker: {...}
```

**AspenNet Execution Modes**
- Normal (sequential): single thread runs plugins in topological order.
  
  ![AspenNet Normal](docs/images/aspennet-normal.svg)

- Threaded + CUDA streams: each plugin executes in its own thread and CUDA stream; queues connect stages.
  
  ![AspenNet Threaded (CUDA streams)](docs/images/aspennet-threaded-streams.svg)

- Threaded without streams: threaded execution on default stream/CPU; same queueing.
  
  ![AspenNet Threaded (no streams)](docs/images/aspennet-threaded-no-streams.svg)

- Configure via YAML `aspen.pipeline`: `threaded: bool`, `queue_size: int`, `cuda_streams: bool`.
- CLI toggles: `--aspen-threaded`, `--aspen-thread-queue-size`, `--aspen-thread-cuda-streams` or `--no-aspen-thread-cuda-streams`.
