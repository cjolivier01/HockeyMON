# hm-ui

Rust operator UI for HockeyMON runtime camera controls.

`hm-ui` is a sidecar process. The Python tracker owns the video pipeline and writes a JSON control spec/state file; `hm-ui` renders the controls and writes value changes back.

## Build

```bash
bazelisk build //hm-ui:hm-ui
```

or:

```bash
cargo build --locked --manifest-path hm-ui/Cargo.toml
```

## Use With hmtrack

```bash
hmtrack --game-id <game> --camera-ui=1
```

The UI provides separate Stitched and Final preview tabs while tracking, and groups controls
under the view they affect. Each control can be reset to its open-time value; the top bar can
reset all controls to open-time or system defaults. Save writes only values that differ from
system configuration to the game's private config.

The same UI is available for the stitching-only workflow. It shows only the stitched preview
and the alignment/input/output color controls that affect that image:

```bash
hmstitch --game-id <game> --camera-ui=1
```

For local source-tree runs, this also builds the sidecar first:

```bash
make hmtrack-rust-ui ARGS="--game-id <game>"
```

The Python bridge searches for `hm-ui` in this order:

1. `HM_UI_BIN`
2. `PATH`
3. Bazel runfiles
4. installed wheel path, `hmlib/bin/hm-ui`
5. `bazel-bin/hmlib/bin/hm-ui` or `bazel-bin/hm-ui/hm-ui-bin`
6. `hm-ui/target/release/hm-ui` or `hm-ui/target/debug/hm-ui`

The hmlib wheel bundles `hm-ui` at `hmlib/bin/hm-ui`.
The Bazel release target links that bundled executable dynamically against a pinned
Ubuntu 22.04 (glibc 2.35) sysroot, independent of the build host's glibc version.

To exercise actual X11/EGL initialization (not only `--help`), install `xvfb-run` and run:

```bash
bazelisk test //hm-ui:gui_smoke_test
```
