# HockeyMON operator desktop

`hmui` opens the Rust desktop without starting a pipeline. `hmtrack --camera-ui=1`
and `hmstitch --camera-ui=1` open the same workspace and start the requested run.
The desktop stays open when a run finishes. Python owns processing, configuration,
model preparation and publication; the desktop does not require HStream binaries.

```bash
bazelisk build --config=cuda //hm-ui:hm-ui
hmui --root "$HOME/Videos"
# Or retain an existing command and all its explicit options:
hmtrack --game-id <game> --camera-ui=1
hmstitch --game-id <game> --camera-ui=1
```

Use the backend appropriate for the machine (`--config=rocm` on ROCm). A local
Rust build is also supported:

```bash
cargo build --locked --manifest-path hm-ui/Cargo.toml
```

## Workspace

| Page | Operations |
| --- | --- |
| Games & Sources | Switch game libraries, create/select games, import or remove source references, join chapters and inspect recordings. Imports can link or copy original files. |
| Runs & Logs | Start/restart tracking or stitching, stop, seek, inspect outputs, export an executable job script and read job logs. |
| Pipeline | Inspect the resolved Aspen dependencies and stage settings without loading weights. |
| Setup | Select detection/pose engines, prepare INT8 and appearance models, configure analytics, synchronization, optics, projection, output and recording preferences. |
| Calibration | Edit scoreboard corners, matched camera points, rink landmarks/goals, crop and leveling; import measured lens calibration and rebuild derived stitching artifacts. |
| Experiments | Create isolated camera/stitching candidates, run them, compare telemetry and promote successful settings. Promotion rejects changed source configuration. |
| Highlights | Preview/seek archives, arrange clips and title cards, add text/boxes/arrows with keyframes or tracked motion, preview and publish numbered videos with audio. |
| Runtime | Monitor Camera 1/2, Stitched and Program; pause/resume/step, adjust speed and live controls, save/reset, zoom/pan/maximize the preview. |

Setup fields marked **next-run** are applied on the next launch. Explicit CLI
options override saved launch preferences. Save writes only managed settings;
stale editor revisions fail visibly instead of replacing newer changes.
Geometric edits invalidate dependent calibration products and rink maps.

Camera experiments can replay a selected telemetry recording against its original
source or matching full Stitched archive without rerunning detection. Select one
continuous source/geometry interval; its tracks, source identity and time range
are frozen with the candidate. Camera tuning values come from the shared baseline.
Highlight track capture uses the chosen archive's publication and recorded camera
geometry, and stops at source, seek, reset or geometry boundaries.

Program is the cropped follower-camera output; Stitched is the full panorama.
Preview labels report source dimensions before display downscaling. The optional
4K Program archive renders from the source image at 3840×2160. The rink mini-map
is a desktop overlay and is omitted from recordings and network outputs.

Pause and step operate at processing batch boundaries; choose batch size one to
step individual frames. Seek finalizes the current recording and restarts at the
requested source time. Stop allows queued processing and audio/video finalization
to finish before escalating to process termination. Forced stops are reported.

Live publishing controls start, redirect and stop the primary/spare RTMP routes
and local Program/Stitched RTSP mounts. Recording switches independently control
Program, Stitched and 4K Program. Switching an archive off finalizes its segment;
switching it back on creates the next segment. All segments and telemetry share
one publication version. A companion segment manifest preserves their offsets
within the run for archive playback and highlight tracking.

Single-source games can edit or request an initial scoreboard polygon directly
from the recording. Suggestions stay in the editor until explicitly saved;
saved selections and disabled scoreboards bypass proposal inference.

Model preparation uses source frames from the selected game. A ready INT8 model
has passed ONNX validation, CPU Runtime loading and digest verification; an
incompatible or missing prepared model prevents launch. Appearance extraction
uses the pinned NVIDIA ReIdentificationNet model or a compatible custom ONNX.
CUDA Runtime is preferred; CPU extraction is reported when CUDA is unavailable.
Player analytics include the pinned RTMPose COCO17 model, causal STGCN++ action
recognition and hockey PARSeq jersey recognition. Action history requires 100
fresh samples at 10 Hz; gaps reset the window. Jersey crops can use player boxes
or fresh pose landmarks. Pose cadence and its bounded drawing hold are separate
from enabling inference. Custom models must satisfy the selected backend's tensor
contract, and preparation or inference failures remain visible.

## Configuration and session ownership

Game files remain under `HM_GAME_DIR` (or `hmui --root`). Experiment and model
caches live in hidden directories within that game. Session commands, durable
results, logs and temporary renders remain in the printed/inspectable job paths
under a `hm-workspace-*` temporary directory. Each session supervises its process
groups, including cleanup if the desktop disappears.

The UI uses schema-versioned local JSON commands and acknowledgements. The
runtime bridge owns live controls; the workspace owns game/workflow mutations.
The internal `Final` preview identifier remains compatible with existing Python
producers while the desktop displays it as **Program**.

The bridge searches for the executable in this order: `HM_UI_BIN`, `PATH`, Bazel
runfiles, the installed `hmlib/bin/hm-ui`, Bazel output paths, then local Rust
release/debug paths. The wheel bundles the executable; `--config=portable-wheels`
uses the Ubuntu 22.04 glibc 2.35 sysroot for its baseline.

## Verification

```bash
cargo test --locked --manifest-path hm-ui/Cargo.toml
bazelisk test --config=cuda //hm-ui:gui_smoke_test \
  //tests:test_desktop_workspace //tests:test_operator_controls \
  //tests:test_operator_rendering //tests:test_calibration_workspace \
  //tests:test_stitching_optics //tests:test_rink_mask_sampling \
  //tests:test_pose_runtime //tests:test_reid_plugin \
  //tests:test_native_player_analytics \
  //tests:test_model_preparation //tests:test_model_backend_contract \
  //tests:test_ui_workspace //tests:test_ui_workspace_highlights \
  //tests:test_ui_workspace_tracks //tests:test_ui_workspace_replay \
  //tests:test_ui_output_routes
```

The GUI smoke test initializes a real X11/EGL window under Xvfb. Python tests cover
stale saves, geometry provenance, process cleanup, pause/step/seek, output
publication, real FFmpeg highlights and RTMP/RTSP media. Runtime tests require the
repository's configured Python environment; network tests use local receivers.
