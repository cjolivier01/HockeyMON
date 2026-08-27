# Native stitching dependencies TODO

## Current wheel behavior

`make wheel` builds the `hockeymon` and `hmlib` wheels, but the wheels are not
self-contained for creating Hugin projects and remap files from raw camera
inputs.

The `hmlib` wheel deliberately packages these native ELF executables:

- `hmlib/bin/enblend`
- `hmlib/bin/multiblend` on x86-64; it is excluded on ARM
- `hmlib/bin/hm-ui`

`hmlib/stitching/configure_stitching.py` resolves `enblend` and `multiblend`
relative to the installed `hmlib` package before falling back to `PATH`.

The Hugin command-line tools below are not included in either wheel:

- `pto_gen`
- `autooptimiser` (British spelling)
- `nona`

They are passed to `subprocess` as bare executable names, so they must already
be available on `PATH`. This can make a wheel installation appear
self-contained on a development machine where `make deps` has installed them
into `$CONDA_PREFIX/bin`.

`make deps`/`build_deps.sh` builds the vendored `external/hugin` tree and
installs the Hugin tools separately. Their shared `libhuginbase` dependency is
installed under `$CONDA_PREFIX/lib/hugin`; the installed executables use an
`$ORIGIN/../lib/hugin` runpath. The container build follows the same model: it
installs the three executables and `libhuginbase` into `/usr/local` before it
builds and installs the Python wheels.

The generated `hockeymon` wheel also currently contains root-level
`multiblend` and `src/enblend` files because those Bazel targets are listed as
wheel dependencies. The package-local Python resolver does not use those
paths. The intentional runtime copies are the ones under `hmlib/bin`.

## Runtime roles

- `pto_gen` creates the initial PTO project when it does not already exist.
- `autooptimiser` optimizes the project and determines/scales the output
  canvas.
- `nona` renders the mapping TIFFs consumed by the stitching pipeline.
- `enblend` generates the panorama and seam mask during configuration.
- `multiblend` is the fallback when the enblend seam is unusable.

These Hugin tools are needed when creating or regenerating stitching artifacts
from raw left/right inputs. Tracking an already-stitched panorama does not need
them.

## Packaging follow-up

- Decide whether Hugin remains an explicitly documented system/Conda
  dependency or becomes part of the `hmlib` wheel.
- If the wheel should be self-contained, add Bazel embedding rules for
  `pto_gen`, `autooptimiser`, and `nona`, analogous to `embed_enblend`.
- Package `libhuginbase` and audit all other dynamic dependencies needed by the
  Hugin tools. Preserve or rewrite their runpaths so libraries resolve relative
  to the installed package.
- Add package-local resolver functions for all three Hugin tools and use them
  at every subprocess call site.
- Determine the intended ARM behavior for `multiblend` and Hugin before
  advertising ARM wheels as self-contained.
- Remove the duplicate/unusable blend-tool payload from the `hockeymon` wheel,
  if it is not required by the native extension.
- Add a clean-environment wheel test which installs only the produced wheels,
  removes repository and Conda tool paths from `PATH`, and exercises PTO
  generation, optimization, nona remapping, and seam generation.
- Run `auditwheel show` (and repair if appropriate) to validate the final wheel
  platform tag and external shared-library requirements.

## Relevant files

- `Makefile`: `wheel` and separate `deps` targets
- `build_deps.sh`: Hugin installation into `$CONDA_PREFIX`
- `hmlib/BUILD.bazel`: embedded blend tools and wheel data
- `hockeymon/BUILD.bazel`: native wheel dependencies
- `external/hugin/BUILD.bazel`: Hugin CLI build targets and install tree
- `hmlib/stitching/configure_stitching.py`: binary resolution and subprocesses
- `hmlib/cli/create_control_points.py`: Hugin subprocesses
- `env/Dockerfile`: system-level Hugin installation before wheel installation
