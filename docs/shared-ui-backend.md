# Shared HStream desktop

`hm-ui`, `hmtrack --camera-ui=1` and `hmstitch --camera-ui=1` launch the
HStream Qt desktop with `--backend=hockeymon`. HockeyMON no longer builds a
separate Rust desktop. The default `hstream-ui` invocation still selects the
native HStream Program pipeline.

Install matching HStream `hstream-ui`, `hstream-cli` and
`hstream-calibration-export`, `hstream-model-prepare` and `hstream-model-infer`
binaries alongside HockeyMON's `hm-cli`. See
[the launcher README](../hm-ui/README.md) for source-checkout commands. A desktop
started from a Bazel build selects its matching native runner; it does not
silently use an unrelated installed executable. `HSTREAM_UI_BIN`, `HM_CLI_BIN`,
`HSTREAM_NATIVE_RUNNER`, `HM_HSTREAM_CALIBRATION_EXPORT`, `HM_HSTREAM_MODEL_PREPARE`
and `HM_HSTREAM_MODEL_INFER` provide explicit
overrides. `HM_CONFIG_ROOT` selects the matching native configuration directory.

Before opening Qt for an initial `hmtrack` invocation, the launcher asks
`hm-cli --resolve-initial-sources` to resolve its recording paths with the same
CLI/config precedence as processing. The private envelope binds those paths to
the original game, arguments and working directory. Qt validates that binding
before calibration admission, so a single prestitched recording never cleans or
marks the saved game's Left/Right calibration pending. Legacy unresolved initial
envelopes must be recreated with the matching launcher; they fail before mutation.

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

The synchronized baseline also owns `preview_reference` dimensions used by Qt
and native browser overlay coordinates. Its `web_preview` section bounds the
native selectable transport; these values do not enable that transport in the
Python backend or replace its existing preview protocol.

The synchronized baseline carries the required native calibration prompt
preferences `hstream_ui.show_crop_dialog` and `hstream_ui.show_leveling_dialog`,
both enabled. Qt and browser Defaults save overrides for these preferences and
the `stitching.stitch_frame_time` reference time. Native calibration owns the
dialogs; Python retains the shared values without adding another prompt policy.

The same baseline also owns `hstream_ui.show_scoreboard_dialog`, enabled by
default. The shared Qt launcher resolves private and game overrides and retains
an explicit `HM_NO_SCOREBOARD` environment override. The browser exports the
same startup default while preserving saved false values. Live scoreboard
selection remains independent of automatic missing-selection prompts.

The desktop owns a private session and process group. `hm-cli` translates stdin
commands into authenticated, bounded requests. The frame owner applies a complete
request and returns a receipt; enqueueing does not acknowledge a successful
change. Pause/resume generations, progress, preview selection, inspector edits,
scoreboard selection and output rotation follow the shared native protocol.
Seek replaces the worker, clears stale preview/control files and restores only
acknowledged controls, overlays, selected preview and playback pause. A paused
replacement permits its first complete preview batch, then holds subsequent
frames; seek completion waits for that batch and local audio remains paused.
Seeking is refused while
archives, streams or telemetry are recording, matching the desktop's admission
rules. The desktop sends Python Stop through the supervisor’s `q` command so
archive and telemetry finalization can finish before forced process-group cleanup.
A Stop during native preparation produces a cancelled-before-processing receipt;
requested outputs are shown as having no file.

On Linux, a small process guard owns each worker and audio player, reaps their
decoder descendants after crashes or parent death, and bounds forced shutdown.
It stays in the desktop's process group for coordinated forced cleanup. Once
the Python frame-owner mailbox exists, pause/resume acknowledgements carry
`mode=cooperative` only after that owner applies the change. Qt keeps the
supervisor running so scoreboard edits, other runtime controls and graceful Stop
remain available while frames are held. During startup and native preparation,
receipts omit that mode and Qt retains its process-group signal fallback.
Seeking retains the original start-frame/time origin and stops
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
The single-source scoreboard selector publishes one private reference canvas in
the backend session, including when the scoreboard is initially disabled. It
never replaces the saved game's native `s.png`. The dialog displays the exact
pinned canvas named by its authenticated session descriptor. Native transactions
persist a selection to the game only when its saved single-source playlist still
matches the active source; a temporary source override keeps its selection within
the run, including seeks, and leaves the saved pair's polygon unchanged. The private run
configuration supplies selector drafts and seek reconstruction for all single-source
runs. Explicit initial corners (including the disabled sentinel) remain active until
the operator saves a new selection; a successful native save updates that private
selection as well as the saved game when its sources match. Temporary
source overrides also infer their rink mask from the active frame without reading
or replacing the saved game's masks or contour metadata, even at equal dimensions.
Normal saved single-source playback retains its mask cache. Source coordinates do not inherit a saved pair's crop offset. Selection remains available
while paused. Dual-source selection uses the native-owned canvas and provenance.
Selector canvas export and proposal inference run under the same subprocess
guard as playback. Closing a selector cancels lock waits and reaps those helpers
before reporting closure; supervisor death also releases their native locks.
An incomplete cleanup remains an error even if the selector thread has exited.

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

The shared pose, jersey, action, team and appearance selections use their canonical
`pipeline.player-analytics` and `pipeline.tracker` settings. Before Python starts,
the supervised `hstream-model-prepare` helper uses HStream's model cache and
publishes the resolved asset paths. Python validates model identities, bundle
hashes and inference contracts. ONNX analytics execute the selected native model;
the default DeepStream appearance option executes the prepared ETLT-derived
TensorRT engine through `hstream-model-infer`, using the same native TensorRT
runtime that built it. Its binary pipe carries batches of at most 32 preprocessed
player crops and returns embeddings; full video frames never enter that pipe.
This bounded host transfer isolates native engine compatibility from the Python
environment's TensorRT version.

Native preparation publishes resolved `pipeline.player-analytics.gpu-id` and
`pipeline.tracker.gpu-id` using its existing application inheritance. Python
carries each accepted device through the pose, jersey, action, team and ReID
stages into their ONNX CUDA providers or native helper, including after a seek;
it does not choose a separate shared default. Pose input/output bindings use
that same device. If preprocessing uses another GPU, only the bounded pose
crops move to the inference device, and results return to the preprocessing
model's device for coordinate decoding.
Custom bundles retain their selected paths. Explicit initial
Python model overrides remain authoritative for that launch. A seek retains the
accepted assets; selecting a different model requires a new run. Stop cancels
preparation before any Python worker starts.

The supervisor freezes accepted shared model bindings in its private
`accepted-models.json`. Each worker validates the version, session, game and
selection identity, with a 64 KiB read limit, after reloading the original YAML
and CLI overrides. Captured original custom-path spellings identify the same
accepted absolute bundle even when the worker has a different working directory;
other model/custom-path changes fail. The worker restores only those asset paths,
resolved devices and shared graph parameters. Explicit Python model overrides
and unrelated engine/control choices retain their precedence. This snapshot
survives seek restarts and is never saved into the user's game configuration.

ONNX analytics use the installed ONNX Runtime providers. Automatic provider
selection reports when CUDA is unavailable and uses CPU; explicitly requesting
CUDA fails if that provider is unavailable. The native appearance helper still
executes its prepared TensorRT engine on the GPU. Shared pose bundles produce
COCO17 SimCC predictions directly; MMPose supplies crop preprocessing and
coordinate decoding, while the selected ONNX replaces network execution.

Python writes a publication receipt only after its publication transaction
commits. The supervisor reports the exact successful artifacts; it never scans
an output directory to infer success. Qt checks the launch's requested outputs,
game identity and actual files. Explicit destinations outside the game directory
are supported. Video and telemetry companions share the positive publication
version, as described in [output publication](output-publication.md). The supervisor
validates every published companion outside its unique working directory, closes
worker/audio owners, and removes that directory before reporting success. Failure
or incomplete publication retains the working recordings for recovery. Telemetry
geometry, masks and camera replay inputs are stored within the published database.

Playback progress and audio share the synchronized recording clock: the left
source synchronization offset is removed from delivered frame IDs, and delivered
FPS accounts for any source frame stepping. Playlist starts and seeks resolve a
global source frame to its chapter-local position without resetting frame IDs.
Mixed camera rates require an integer ratio (for example, 60/30 or 59.94/29.97);
unsupported ratios such as 60/24 fail explicitly. The loader samples each output
frame at the chosen cadence even for multi-frame batches, and passes original
source PTS into telemetry separately from its legacy frame identifiers. Those
identifiers count delivered samples and retain the integer left-source offset;
`fps` describes delivered cadence, while `pts_ns` identifies the sampled original
source frame. Initial frame-based start/limits also use the delivered cadence.

CUDA stitched archives follow the native HEVC encoder's queried raster limits:
only the archive pixels are scaled on the GPU to proportional, even dimensions
within its width, height and macroblock budget. The stitching canvas, source
previews and telemetry keep their original geometry. For example, the RTX 5090
fits a 20415×5256 canvas into an 8192×2108 archive. A failed capability query or
an aspect ratio that cannot fit without upscaling is an explicit error.

Recorded telemetry includes native replay inputs, camera checkpoints and the
actual source start, plus the immutable geometry/mask used by each sample.
The shared Camera Experiments engine can validate and replay Python-produced
recordings, including empty input frames and live camera-policy changes.

Job export consumes the UI's saved recipe and produces a quoted executable
`hm-cli` shell script, retaining its start time and settings. It selects file
outputs, removes local previews/network routes and records scheduler directives.
Shared output selections disable legacy spare RTMP publishers; an explicit
initial CLI configuration can enable one, and the resolved network route then
prevents seeking. Exported scripts request one node, one task and one GPU by
default, with operator SBATCH directives following those defaults. They retain
the selected `HM_OUTPUT_WORK_DIR` and clear private desktop session variables.
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

The synchronized `hstream_ui.int8_preparation` block supplies the shared desktop
dialog's required sample count and Play-when-ready defaults. The desktop passes
the resolved count to `hm-cli --prepare-detector`; Python preparation and its
artifact protocol are unchanged. Browser recording preparation remains native-only.
