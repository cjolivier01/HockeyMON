# Shared HStream desktop

`hm-ui`, `hmtrack --camera-ui=1` and `hmstitch --camera-ui=1` launch the
HStream Qt desktop with `--backend=hockeymon`. HockeyMON no longer builds a
separate Rust desktop. The default `hstream-ui` invocation still selects the
native HStream Program pipeline.

Install matching HStream `hstream-ui`, `hstream-cli` and
`hstream-calibration-export` binaries alongside HockeyMON's `hm-cli`. See
[the launcher README](../hm-ui/README.md) for source-checkout commands. A desktop
started from a Bazel build selects its matching native runner; it does not
silently use an unrelated installed executable. `HSTREAM_UI_BIN`, `HM_CLI_BIN`,
`HSTREAM_NATIVE_RUNNER` and `HM_HSTREAM_CALIBRATION_EXPORT` provide explicit
overrides. `HM_CONFIG_ROOT` selects the matching native configuration directory.

## Ownership

| Operation | Owner |
| --- | --- |
| Game selection, source lists, controls, output choices and all dialogs | Shared HStream Qt UI |
| Program inference, tracking, camera, color, recording and network output | HockeyMON Python/Aspen pipeline supervised by `hm-cli` |
| Camera leveling, crop, matching, calibration, rink map and stitching experiments | Shared HStream native engines |
| Camera Experiments and highlight authoring, playback and publication | Shared HStream native engines |
| Native calibration provenance, scoreboard transactions and output epochs | HStream validators exposed through `hstream-calibration-export` |
| Python recording publication and telemetry allocation | HockeyMON publication/telemetry modules |

The shared desktop's leveling workflow lives in HockeyMONStream.
HockeyMON invokes it and consumes its validated output. Python does not reinterpret
the native calibration provenance or rerun its solver. A run receives an isolated
snapshot of native maps, seam, mask, source identities, synchronized offsets and
resolved settings. Python-generated caches stay in that snapshot. Live output
rotation requires the native generation/authorization transaction; a seek exports
the currently accepted generation again before restarting Python.

The shared rink map is fitted and validated natively. Python consumes the fitted
plane-to-ray transformation and bounded Hugin projection grid for rink-aware camera
framing and the minimap. This also keeps live rotation and recorded geometry tied
to the same output epoch.

## Launch and runtime contract

`hm-cli --ui-backend-info` provides a versioned, dependency-light JSON handshake.
Normal launches accept the shared desktop's game, source/sink, time, model and
settings arguments. Transport settings that describe the native mux are checked
against their supported values; unknown settings fail visibly. Original `hmtrack`
arguments and working directory travel in a private initial-launch envelope and
apply only to that first run. Explicit initial output requests override the UI’s
starting route choices. Before processing, `HM_BACKEND_LAUNCH` reports the resolved
routes so the desktop displays them, disables incompatible seeks and requires
the correct publication artifacts. An initial `hmstitch` envelope is translated into
the native calibration workflow. Explicit initial settings also override generated
UI options during native preparation. Python-only processing arguments that have no
native calibration meaning are rejected with an explanation.

Native `--settings-overlay` carries structured one-run settings without rewriting
the game or misinterpreting `-c` as a settings overlay: native `-c` names a
separate pipeline application. Explicit native `--options` retain their normal
precedence. The same effective settings accompany the calibration export.

The desktop owns a private session and process group. `hm-cli` translates stdin
commands into authenticated, bounded requests. The frame owner applies a complete
request and returns a receipt; enqueueing does not acknowledge a successful
change. Pause/resume generations, progress, preview selection, inspector edits,
scoreboard selection and output rotation follow the shared native protocol.
Seek replaces the worker, clears stale preview/control files and restores only
acknowledged controls, overlays and selected preview. Seeking is refused while
archives, streams or telemetry are recording, matching the desktop's admission
rules. The desktop sends Python Stop through the supervisor’s `q` command so
archive and telemetry finalization can finish before forced process-group cleanup.
A Stop during native preparation produces a cancelled-before-processing receipt;
requested outputs are shown as having no file.

On Linux, a small process guard owns each worker and audio player, reaps their
decoder descendants after crashes or parent death, and bounds forced shutdown.
It stays in the desktop's process group so pause and resume cover the complete
processing tree. Seeking retains the original start-frame/time origin and stops
local audio until the replacement worker produces its first frame.
The desktop's thirty-second Stop deadline still escalates
to termination of the process group if shutdown does not finish.

Python preview frames use bounded JPEG files read by the existing Qt preview
surfaces. Preview-only players/play/rink drawings use immutable metadata and the
actual image transforms; they do not change archive overlays. Full source images
are not copied to the UI every frame. Only the selected channel is published,
at most fifteen times per second. Camera 1 covers single-source games and the
left camera; Camera 2 covers the right camera. Their raw decoded frames are
copied at preview resolution before downstream color or drawing changes.
The single-source scoreboard selector saves
one reference canvas, including when the scoreboard is initially disabled;
dual-source selection uses the native-owned canvas and its provenance.

Local monitor audio follows source time, pause, seek and Render video. It opens
the current source chapter lazily and does not concatenate an entire game before
playback. It is independent of archive/network audio. Real-time playback paces
the Python source frames and services runtime requests during waits.

High-bit-depth grading uses FFmpeg BGR48 software decode followed by floating
0..255 processing. This preserves fractional source samples through grading;
the current Python hardware decoder exposes only 8-bit RGB. Automatic mode
requires every selected source/chapter to have a known depth above eight bits.
The first real frame reports the resolved mode and source depths to the UI.
Preview JPEGs and the existing archive encoders remain their usual output formats.

## Models, recordings and jobs

HockeyMON detector selections live under `hstream_ui.hockeymon.detector` so Python
ONNX files cannot be mistaken for native TensorRT engines. Shared INT8 preparation
uses the native recording sampler, Python model export/quantization and an
immutable manifest containing the model identity and artifact digest. Selecting
INT8 validates exactly the accepted engine and manifest; it does not guess a
cache or start another export.

Python writes a publication receipt only after its publication transaction
commits. The supervisor reports the exact successful artifacts; it never scans
an output directory to infer success. Qt checks the launch's requested outputs,
game identity and actual files. Explicit destinations outside the game directory
are supported. Video and telemetry companions share the positive publication
version, as described in [output publication](output-publication.md).

Recorded telemetry includes native replay inputs, camera checkpoints and the
actual source start, plus the immutable geometry/mask used by each sample.
The shared Camera Experiments engine can validate and replay Python-produced
recordings, including empty input frames and live camera-policy changes.

Job export consumes the UI's saved recipe and produces a quoted executable
`hm-cli` shell script, retaining its start time and settings. It selects file
outputs, removes local previews/network routes and records scheduler directives.
The exported script resolves `${HM_CLI_BIN:-hm-cli}` on the destination host;
that host needs the matching native workflows and the same source/game paths.

## Validation

The focused Bazel targets cover launch/envelope precedence, detector manifests,
runtime requests, scoreboard HTTP transactions, preview geometry, native snapshot
validation, precision-preserving media and local audio lifecycle. Native HStream
tests cover both backend modes and actual child-process/JPEG/receipt handling.
The end-to-end workflow also exercises real GPU processing, all three archive
resolutions and telemetry, native replay/Camera Experiments and native leveling.
UI PR descriptions include externally hosted before/after screenshots; screenshot
files do not belong in either repository's Git tree.
