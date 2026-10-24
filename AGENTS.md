# Repository Guidelines

## Project Structure & Module Organization
- `hmlib/`: Primary Python library and CLI entry points (e.g., `hmlib.cli.hmtrack`). Packaged via Bazel wheel rules.
- `src/`: Ancillary modules (`core/`, `users/`) used by higher-level code.
- `tests/`: Bazel `py_test` targets and simple runtime checks.
- `tools/`, `scripts/`: Bazel helpers, formatting utilities, and dev scripts.
- `assets/`, `external/`, `openmm/`, `cmake/`: Assets and native/C++ build integration.

## Build, Test, and Development Commands
- Build all: `bazelisk build //...` (or `bazel build //...`).
- Run tests: `bazelisk test //...`.
- Coverage report: `./coverage.sh` (generates HTML in `reports/coverage/`).
- Apply formatters: `./format.sh` (Black/isort via Bazel aspect).
- Run a Bazel target: `./run.sh //path:target --args`.
- Package wheel: `./run.sh //hmlib:bdist_wheel`.
- Run tracker locally (example): `EXP_NAME=dev VIDEO=video.mp4 ./hm_run.sh`.
 - Plain `bazelisk` defaults to `--config=rocm` (`.bazelrc`); on a CUDA machine pass `--config=cuda` or use `./bazel/bazel.sh`, which appends the detected backend.

### Handy local debugging command

- Run TensorRT-enabled tracking on a short clip (5s) for `chicago-3`:
  - `PYTHONPATH=$(pwd) python hmlib/cli/hmtrack.py --game-id=chicago-3 --async-post-processing=0 --async-video-out=0 --show-scaled=0.5 --camera-ui=1 --detector-trt-enable --detector-static-detections --detector-static-max-detections=800 --plot-tracking -t=5`

## Coding Style & Naming Conventions
- Python: Black (see `pyproject.toml`) and isort; run `./format.sh` before committing.
- Before finalizing changes, run `python -m black` and `ruff check` on any modified/new Python files and fix issues until clean.
- Typing: mypy is configured; prefer typed public APIs in `hmlib/`.
- Naming: modules and functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- C/C++: follow `.clang-format`; standard set to C++17 in CMake.
- Indentation: use spaces (not tabs) in all source files (Python/C/C++/JS/HTML/CSS/etc). Tabs are only allowed where they have special meaning (e.g., Makefiles).

## Error Handling & CLI Args
- Never silently fail. Avoid `except Exception: pass`, bare `except:`, silent fallback-to-default behavior, or catching-and-returning-success. If a best-effort path is genuinely required, surface it explicitly with context so the caller/user can tell the degraded path was taken.
- For CLI argument access: never use `getattr(args, "flag", default)` (or `hasattr(args, ...)`) to paper over missing argparse attributes. Define all expected args in the parser (with defaults) and access via `args.flag`; if an attribute is missing, that's a bug (use subparsers or separate namespaces if modes differ).

## Testing Guidelines
- Framework: pytest conventions are supported; prefer test functions named `should_*` (see `pyproject.toml`).
- Location: add tests under `tests/` or as Bazel `py_test` targets alongside code.
- Run: `bazelisk test //...` for CI-equivalent runs; use `./coverage.sh` to inspect branch coverage locally.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., `fix(hmlib): handle empty frames`). Prefix with scope when helpful (`hmlib/`, `tools/`, `build/`).
- PRs: include a clear description, linked issues, what/why, and test evidence (logs, sample outputs). Update docs when behavior changes.
- UI changes: include clearly labeled before-and-after screenshots in the PR description. Upload screenshots externally (for example, to a Gist) if needed, and embed or link them in the PR. Never commit screenshot files or include them in the repository's Git tree.
- Create regular PRs, not draft PRs, unless the user explicitly asks for a draft.
- Use task-focused branch names without AI/tooling prefixes such as `codex/`.
- Do not mention AI agents or coding tools in PR titles or descriptions; PR comments may mention them when useful.
- Assets: do not commit large datasets or model weights; use `datasets/` and `pretrained/` symlinks.

### Worktrees

- Create every worktree as a sibling of the repository, never inside it: `../hm-<task>`, not `.claude/worktrees/<task>` or any other subdirectory.
- A worktree under the repository root carries its own `BUILD.bazel` files, so `//...` expands into it and the build fails on targets that belong to another checkout. `.bazelignore` excludes `.claude` for this reason; a worktree anywhere else under the root would still break `bazelisk build //...`.
- Remove a worktree once its branch has merged, and keep the branch name task-focused on the same terms as any other branch.

### Nested repositories and submodules

- Treat every nested Git repository as an independent repository with its own commit and remote.
- Commit and push changes from the innermost repository outward. Never record a submodule pointer to a child commit that has not been pushed.
- After pushing a child repository, stage its updated gitlink in the immediate parent, commit and push that parent, then repeat for every containing superproject until the outermost repository is updated.
- Before finishing, verify that each repository in the chain points at the intended child commit and that no task-related changes remain uncommitted.

## Security & Configuration Tips

- `--camera-ui`, `hm-ui` and `hm-cli` use the shared HStream Qt desktop and its versioned backend protocol. Native calibration, camera leveling, rink-map fitting, experiments and highlights remain owned by HockeyMONStream; Python Program processing consumes validated private native snapshots through `hstream-calibration-export`. Do not add a second desktop or duplicate native provenance parsing. Runtime mutations acknowledge only frame-owner receipts, and publication reports only files committed by this run. See `docs/shared-ui-backend.md` and `hm-ui/README.md`.
- Shared analytics choices are translated by `hmlib/ui/shared_analytics.py`; `hstream-model-prepare` owns native cache preparation before Python starts. Python executes the selected ONNX bundles or the exact default DeepStream ETLT-derived TensorRT appearance engine, validates their contracts and hashes, and retains accepted assets across seeks. Preserve explicit initial Python model overrides and never substitute a different model for a shared menu selection.
- OpenMMLab's registry scope is process-global. Use `hmlib/utils/model_registry.py` when constructing detector, pose or segmentation models and lazy segmentation transforms; concurrent Aspen workers must not change each other's registry scope during checkpoint loading.

- `hmlib/scoreboard/proposal.py` supplies an ephemeral initial polygon only when the selector has no saved selection. Its pinned TorchScript export shares the native HStream scoreboard model contract; CUDA is preferred, GPU OOM releases resources and retries once on CPU, and other failures remain visible with manual selection available. Saved and disabled polygons bypass model work. Keep inference out of per-frame transforms; see `hmlib/scoreboard/README.md`.

- Keep `hmlib/config/baseline.yaml` byte-for-byte synchronized with HockeyMONStream's `configs/baseline.yaml`. `stitching.control_point_execution_provider` defaults to `cuda` (`cpu` is an explicit alternative without a UI switch). `stitching.control_point_resolution` is the native HStream matcher setting (`2k` is the default on every platform, including missing or `auto` settings; explicit `native`/`1k` choices remain available); its runtime policy is documented in that repository's `docs/native-feature-matchers.md`.
- The shared `stitching.rink_mask_frame_time` and rink-profile field configure native HStream final-mask sampling: `null` inherits, `auto` uses the first stitched frame, and negative `HH:MM:SS[.mmm]` is relative to the synchronized recording end. TV Dublin defaults to `-00:00:01`; Python mask sampling is unchanged. Runtime ownership and UI overrides are documented in HockeyMONStream's `docs/rink-mask-frame-time-design.md`.
- The shared `pipeline.player-analytics.pose` block supplies native HStream cadence (`rate-hz: 10`, or `fps` for every incoming frame within its model budget) and `continuous-drawing: true` for a tracked-box display hold of at most 250 ms. These preferences do not enable inference and Python does not interpret them; keep both baselines synchronized. See HStream's `docs/player-analytics-runtime.md`.
- The shared baseline also carries HStream's detector menu and the seam settings both applications read: `stitching.blend_mode` defaults to `laplacian`, and `blend_feather_fraction` defaults to `0.05` for the alpha blend. Keep the copies synchronized when either application's defaults change.
- The shared camera catalog includes Auto, Manual and the verified Mission 1 factory reference. `stitching.camera_config: auto` is resolved from original video metadata by native HStream; Python does not import that metadata. HStream's `docs/auto-input-lens.md` describes the supported recording modes, fallback and frozen experiment optics.
- The baseline's measured camera modes are optional refinement seeds derived offline from pinned CC0 community calibrations. HStream owns their source files, converter and `docs/camera-lens-catalog.md`; keep its generated baseline block byte-identical here. They are not manufacturer factory values. Native HStream can use profiles with complete `metadata_match` constraints as automatic mode fallbacks; uncertain profiles remain manual choices. `stitching.camera_automation.fov/lens` independently controls native camera FOV and correction, separate from panorama framing; Python retains these keys but does not interpret them.
- The shared `stitching.control_point_matcher` default is `akaze-hamming`; explicit user/game choices still win. AKAZE runs on CPU without matcher model assets. Native HStream matcher behavior and lens-profile migration constraints are documented in its `docs/native-feature-matchers.md`. The shared `stitching.control_point_filtering` defaults enable epipolar RANSAC and conditional homography consistency for native automatic matching; Python retains these settings but does not interpret them.
- The shared `stitching.projection_framing` block is split by owner, and `hmlib/stitching/settings.py` enforces the split so a sync cannot quietly change meaning. The baseline must name `auto_fov`, `horizontal_fov`, `auto_canvas` and `auto_crop`, because Python resolves them from it rather than from literals of its own; dropping one or setting it `null` raises. The baseline must not give `rotation_degrees` or `crop` a value: those come from `stitching.rink_config` or the operator, and because `get_config` deep-merges the baseline into the game config a value there would be indistinguishable from the operator's and would beat the rink, while a config that never saw the baseline would still take the rink — one game, two answers. `null` is accepted for both, meaning "inherit" as it does in HStream. A framing key HockeyMON does not know raises rather than being ignored, including one HStream has added, which is the tripwire: HStream enforces the same closed key set, so neither side can extend it unilaterally without the other noticing. Only NONA applies framing, so the OpenCV backends accept framing that leaves every decision to the backend and refuse a fixed FOV or canvas, a crop or a rotation rather than silently dropping it; `horizontal_fov` is not a request while `auto_fov` holds. See `docs/stitching-calibration.md`.
- `hmlib/stitching/blend.py` owns the seam blend vocabulary. It accepts the same spellings as HStream's `ParseBlendMode` (case-insensitive, `_` and `-` interchangeable, plus `hard`/`hard-seam` for `gpu-hard-seam`), with `multiblend` as the one HockeyMON-only addition. Absent, null and blank all mean "inherit"; anything else unrecognized raises. `GPU_BLEND_MODES` and `PYTHON_BLEND_MODES` say which modes each renderer can run, and `require_gpu_mode`/`require_python_mode` refuse the rest rather than silently rendering a different seam. `multiblend` is in neither: it names the calibration-time enblend/multiblend binaries, and `create_blender_config` returns a seamless config for it that the Python blender then dereferences. Blending is a render-time choice, so it must stay out of `StitchingSettings`, whose manifest is calibration provenance.
- In the CUDA stitcher bindings, omitting `blend_mode` keeps hm-cupano's encoding where a positive `num_levels` is a pyramid and 0 is a hard seam, so callers that predate the argument are unchanged. A `blend_mode` that is given decides the operator instead, because a caller naming one and getting another is the mismatch the argument exists to prevent; `hmlib` always passes it. `WORKSPACE.bazel`'s `hm-cupano` pin must carry the alpha blend; keep it in step with HockeyMONStream's pin.
- The shared HStream Stitched view owns seam blend and feather controls. `hmstitch --camera-ui=1` opens that native calibration workflow; `hmtrack --camera-ui=1` consumes the accepted settings and validated maps through the shared backend. Python still validates each renderer's supported blend modes in `hmlib/stitching/blend.py`. The internal `StitchUiPlugin` and `PlayTracker` bridges serve frame/runtime control; they no longer define a separate desktop UI. See `docs/shared-ui-backend.md` for launch and calibration ownership.
- Secrets: never commit credentials; prefer environment variables.
- Large files: keep outside the repo (symlinks `datasets/`, `pretrained/`).
- Reproducibility: run via Bazel for consistent tooling; avoid ad‑hoc local installs unless developing isolated modules.
- Two config loaders exist and they are not interchangeable. `get_config` seeds from `baseline_config` and merges rink/camera/game/private; `get_game_config` merges only the public and private game YAML and never sees the baseline. `hmlib/cli/create_control_points.py` uses the latter, so that CLI's `stitching` section arrives without baseline defaults — 25 keys and 12 whole top-level sections differ from what `hmtrack`/`hmstitch` see for the same game. Resolved `StitchingSettings` happen to agree today, because `read_stitching_settings` baselines the keys it owns internally, but anything reading `game_config` directly does not get that protection. Prefer `get_config` for new callers, and do not assume a config reaching `read_stitching_settings` was baseline-merged.
- Versioned recordings/publications start at `-1`; video and telemetry companions share one positive version. Telemetry uses `<game-id>_telemetry-N.db`, with legacy unnumbered names retained for reading. `hmlib/utils/output_publication.py` owns publication and `hmlib/telemetry/recorder.py` owns working database allocation. See `docs/output-publication.md` and `docs/telemetry-databases.md`.

## AspenNet Architecture
- Graph runner built from the YAML `aspen.plugins` mapping (`class`, `depends`, `params`, optional `enabled`); `AspenNet.__init__` raises without it, so a config spelling the node mapping `aspen.trunks` cannot run. "Trunk" remains the name for a node throughout the code and this document. Missing deps or cycles raise; disabled trunks become no-op stubs to preserve graph shape; graph is exported to `aspennet.dot` on init.
- `AspenNet._instantiate` calls `cls(**params)`, so a parameter a graph omits takes the plugin constructor's own literal, silently. When that literal disagrees with `baseline.yaml` the graph runs something other than the shipped default: reference the baseline with `GLOBAL.<path>` in every graph declaring the plugin rather than relying on the constructor. `tests/test_aspen_graph_baseline_params.py` guards the parameters listed in its `_BASELINE_BACKED` table; add an entry when a parameter starts taking its shipped value from the baseline.
- Execution modes set under `aspen.pipeline`/`threaded_trunks`: sequential topological order by default, or threaded pipeline with one worker per trunk connected by bounded `Queue(queue_size)`; optional per-trunk CUDA streams (`cuda_streams`) wrap each trunk and synchronize before handoff; grad/no-grad follows the `training` flag.
- Context flow: `forward` threads a shared mutable `context` (injects `shared` and `trunks` namespaces); trunks can declare `input_keys`/`output_keys` and, when `minimal_context` is true, only requested keys plus `shared` are passed; outputs update context, `DeleteKey` removes entries, and each trunk's outputs are stored under `context["trunks"][name]`.
- Device selection for stream usage is inferred from `context`/`shared` (`device`, `cuda_stream`, tensor devices) with CUDA current-device fallback; profiling is plumbed through `shared["profiler"]` using trunk `profile_scope`.
- Shutdown: `finalize()` is invoked on trunks if present; DAG is available via `to_networkx`/`to_dot` helpers and `display_graphviz`.
