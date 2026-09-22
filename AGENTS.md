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
 - Exclude wheels: `bazelisk build --config=no-python-wheels //...` (or `./bld --no-python-wheels`).

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
- Create regular PRs, not draft PRs, unless the user explicitly asks for a draft.
- Use task-focused branch names without AI/tooling prefixes such as `codex/`.
- Do not mention AI agents or coding tools in PR titles or descriptions; PR comments may mention them when useful.
- Assets: do not commit large datasets or model weights; use `datasets/` and `pretrained/` symlinks.

## Security & Configuration Tips

- Keep `hmlib/config/baseline.yaml` byte-for-byte synchronized with HockeyMONStream's `configs/baseline.yaml`. `stitching.control_point_execution_provider` defaults to `cuda` (`cpu` is an explicit alternative without a UI switch). `stitching.control_point_resolution` is the native HStream matcher setting (`auto` defaults to `2k` on Jetson and `native` on desktop/SBSA; explicit `native`/`2k` overrides it); its runtime policy is documented in that repository's `docs/native-feature-matchers.md`.
- Secrets: never commit credentials; prefer environment variables.
- Shared `ice_boundaries` defaults include HStream's four signed pixel mask insets (zero by default) and independent left/right half-box-width sampling offsets. These are player-filter settings, not stitching artifacts; see HockeyMONStream's `docs/rink-extents.md` for native filtering and preview semantics.
- Large files: keep outside the repo (symlinks `datasets/`, `pretrained/`).
- Reproducibility: run via Bazel for consistent tooling; avoid ad‑hoc local installs unless developing isolated modules.

## AspenNet Architecture
- Graph runner built from YAML `aspen.trunks` mapping (`class`, `depends`, `params`, optional `enabled`); missing deps or cycles raise; disabled trunks become no-op stubs to preserve graph shape; graph is exported to `aspennet.dot` on init.
- Execution modes set under `aspen.pipeline`/`threaded_trunks`: sequential topological order by default, or threaded pipeline with one worker per trunk connected by bounded `Queue(queue_size)`; optional per-trunk CUDA streams (`cuda_streams`) wrap each trunk and synchronize before handoff; grad/no-grad follows the `training` flag.
- Context flow: `forward` threads a shared mutable `context` (injects `shared` and `trunks` namespaces); trunks can declare `input_keys`/`output_keys` and, when `minimal_context` is true, only requested keys plus `shared` are passed; outputs update context, `DeleteKey` removes entries, and each trunk's outputs are stored under `context["trunks"][name]`.
- Device selection for stream usage is inferred from `context`/`shared` (`device`, `cuda_stream`, tensor devices) with CUDA current-device fallback; profiling is plumbed through `shared["profiler"]` using trunk `profile_scope`.
- Shutdown: `finalize()` is invoked on trunks if present; DAG is available via `to_networkx`/`to_dot` helpers and `display_graphviz`.
