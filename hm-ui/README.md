# HockeyMON with the shared HStream desktop

`hm-ui` (also installed as `hmui`) opens HStream's Qt desktop with HockeyMON's
Python Program backend. `hmtrack --camera-ui=1` opens the same desktop and starts
the requested Program run. `hmstitch --camera-ui=1` opens native stitching
calibration. The desktop stays open when processing finishes.

The widgets, game browser, calibration editors, experiments and highlights are
maintained in HockeyMONStream's `src/apps/hstream-ui`. This directory contains
Python launchers and the `hm-cli` backend entry point; it does not contain a
second desktop implementation.

## Build and launch

Install matching HStream tools, including `hstream-ui`, `hstream-cli` and
`hstream-calibration-export`. A source checkout can build the desktop with its
normal platform configuration:

```bash
# In the HStream checkout; select its configuration for your machine.
bazelisk build --config=opt --config=x86_64 //src/apps/hstream-ui:hstream-ui
```

Build HockeyMON's launchers with the backend appropriate for the machine:

```bash
bazelisk build --config=cuda //hm-ui:hm-ui //hm-ui:hm-cli
# Use --config=rocm on ROCm.
```

The Bazel launchers select `PYTHON_BIN_PATH`, then the active `CONDA_PREFIX`
interpreter, then `python3` on `PATH`. Use the processing environment that matches
the compiled `hockeymon` extension; Bazel's build-tool Python does not contain
the processing dependencies. The built desktop launcher uses its matching built
`hm-cli` unless `HM_CLI_BIN` explicitly selects another backend. For example:

```bash
PYTHON_BIN_PATH=/absolute/environment/bin/python bazel-bin/hm-ui/hm-ui --game-id <game>
```

Wheel console scripts retain the interpreter of the environment where the wheel
was installed. They do not reselect Python from these environment variables.

For an installed HStream desktop:

```bash
hmui --root "$HOME/Videos"
hmtrack --game-id <game> --camera-ui=1
hmstitch --game-id <game> --camera-ui=1
```

For two source checkouts, point to the built desktop and this backend launcher:

```bash
export HSTREAM_UI_BIN=/absolute/hstream/bazel-bin/src/apps/hstream-ui/hstream-ui
export HM_CLI_BIN=/absolute/HockeyMON/hm-ui/hm-cli
./hm-ui/hm-ui --game-id <game> --root "$HOME/Videos"
```

`HSTREAM_UI_BIN` and `HM_CLI_BIN` are explicit executable selections; an invalid
selection fails with an error. Otherwise the launcher searches `PATH`, then
`/opt/hstream/bin/hstream-ui` for the desktop and the source `hm-ui/hm-cli` for
the backend. The desktop verifies the backend protocol before opening a game.
Its matching native runner and calibration exporter are passed to the Python
backend. Installing HockeyMON's wheel alone does not install HStream or Qt.

Original `hmtrack` and `hmstitch` arguments travel in a private launch document,
including the invocation directory used for relative paths. The document applies
to the first launch and is removed when the desktop closes. A launch with explicit
input files and no game ID uses a private temporary game for that desktop session.

## Processing and workflow ownership

| Operation | Owner |
| --- | --- |
| Program inference, tracking and camera processing | HockeyMON Python |
| Calibration, camera leveling, projection/crop and rink mapping | Shared native HStream tools |
| Stitching experiments, camera experiments and highlights | Shared HStream desktop and workflow engines |
| Program recordings and telemetry publication | HockeyMON Python |
| Stitched calibration preview and native workflow outputs | HStream |
| Program job export | `hm-cli` |

Both applications use the same controls and game configuration. Runtime changes
retain the shared command acknowledgments and run generations. Controls marked
for the next run, including output routing and detector precision, take effect on
the next Play. A single-source HockeyMON recording does not require stitching
calibration. Dual-camera Program uses checked native calibration artifacts before
Python processing starts.

HockeyMON previews use a bounded private JPEG transport for the selected view;
native HStream previews retain their GPU rendering path. Turning Render video off
or opening the Pipeline inspector suspends image-channel delivery. The final frame
remains visible after Stop, and Stop restores the ordinary preview layout.

Python detector choices and prepared models are stored under
`hstream_ui.hockeymon.detector`, separately from native TensorRT assets. Prepare
INT8 samples the calibrated recording through the shared native sampler, prepares
a private sample reel, and invokes HockeyMON's existing ONNX quantizer. The returned
model must pass ONNX validation, CPU Runtime loading and digest verification.
Preparations are immutable, so a failed refresh leaves the prior selection intact.
Cancelling preparation stops its process group. The saved model and manifest must
still match at the next launch; playback never silently quantizes another model.

The shared desktop accepts a completed publication receipt only after the requested
recordings and telemetry exist. Explicit output destinations may live outside the
game directory. Python owns their publication and cleanup; the desktop does not
run the native finalizer on Python outputs. Exported jobs resolve `hm-cli` on the
execution host and do not retain the private desktop session.

See [the backend contract](../docs/shared-ui-backend.md) and HStream's
`docs/shared-ui-backends.md` for launch translation, runtime control, preview and
publication details. Native workflow documentation, including camera leveling and
rink-map provenance, remains in the HStream repository.

## Verification

```bash
python -m pytest -q tests/test_hstream_detector.py \
  tests/test_desktop_launch_sources.py tests/test_desktop_workspace.py
```

The HStream checkout owns `backend_options_test`, `shared_backend_window_test`,
`int8_preparation_dialog_test` and its native desktop/X11 regression suites. The
shared window tests exercise actual subprocess output, rendered preview pixels,
view switching, completion receipts and final-frame retention. Python preparation
tests use real PNGs, FFmpeg video, quantized ONNX validation and process cancellation.
UI PR screenshots must be hosted outside the Git tree.
