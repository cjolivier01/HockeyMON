# September 2026 hstream port

This series adapts applicable changes from `../hstream` dated August 9 through
September 9, 2026, at source revision
`f3bb170a87d7172dba762190343ad2763bf4a9b7` (identical tree to the subsequently
merged `e7943cd6e24905dd977585b7b4671146f27ec22d`), to HockeyMON master
`6bcc66684809362b0ebf259b59417fc4b7c0dfb9`.

## Pull requests and merge order

Each row is a regular pull request. Follow the dependencies within each stack;
the three stacks can be reviewed independently.

| PR | Change | Depends on |
| --- | --- | --- |
| [#145](https://github.com/cjolivier01/HockeyMON/pull/145) | Propagate finalization failures and flush camera CSVs durably | master |
| [#146](https://github.com/cjolivier01/HockeyMON/pull/146) | Publish complete output generations without replacing existing files | #145 |
| [#147](https://github.com/cjolivier01/HockeyMON/pull/147) | Scale automatic bitrate to source density and final encoder dimensions | #146 |
| [#157](https://github.com/cjolivier01/HockeyMON/pull/157) | Durable camera-policy companions and training boundaries aligned with actual frames | #147 |
| [#148](https://github.com/cjolivier01/HockeyMON/pull/148) | Zoom aggression, source preview dimensions, bounded scoreboard selection | master |
| [#149](https://github.com/cjolivier01/HockeyMON/pull/149) | Reference shadow lift and alpha/high-bit-safe image grading | #148 |
| [#158](https://github.com/cjolivier01/HockeyMON/pull/158) | Carry the reviewed #149 changes onto master | #147, #148 |
| [#150](https://github.com/cjolivier01/HockeyMON/pull/150) | Validated projection, camera, rink, crop, canvas and scale settings | master |
| [#151](https://github.com/cjolivier01/HockeyMON/pull/151) | Multi-frame calibration candidates and retained edited control points | #150 |
| [#152](https://github.com/cjolivier01/HockeyMON/pull/152) | AKAZE matching with optional pinned KB4 lens profiles | #151 |
| [#153](https://github.com/cjolivier01/HockeyMON/pull/153) | Shared standalone image/video calibration workflow | #152 |
| [#154](https://github.com/cjolivier01/HockeyMON/pull/154) | Recoverable stitching artifact publication, bounded stable readers and cache provenance | #153 |
| [#155](https://github.com/cjolivier01/HockeyMON/pull/155) | Reviewed Hugin previews, selected-post rink leveling and crop controls | #154 |
| [#156](https://github.com/cjolivier01/HockeyMON/pull/156) | Resumable projection-comparison captures with isolated calibration and logs | #154 |

PR #149 was merged into the camera branch after #148 had already landed on
master. PR #158 carries that exact reviewed implementation onto master.

## Changes already present in master

The audit found existing support for SuperPoint, DeDoDe and LoFTR matching,
MAGSAC and affine mapping, uneven camera chapter lengths, tracker resize
deadband and overshoot fixes, and camera-dataset integrity checks. The shared
baseline and current `hm-cupano`/`jetson-utils` fork fixes were already pinned.

## Architecture boundaries

HockeyMON uses AspenNet/Python orchestration and its existing native codecs and
Rust/web controls. The following source changes require hstream-specific
components and are outside this series:

- Qt/DeepStream pipeline inspection, seek reconstruction, native GPU preview
  window lifecycle and Windows/WSL installers or signing.
- GStreamer live-generation ownership and epoch handling. The applicable file
  publication, rollback, resource bounds and stable-reader rules are adapted
  to HockeyMON's stitching artifacts instead.
- Main10/P010 archive encoding. HockeyMON's current video path decodes and
  stitches uint8 frames and encodes YUV420. High-bit image grading does not
  provide an end-to-end 10-bit video path.
- Packaging learned models for hstream's native inference backends; existing
  HockeyMON matcher integrations remain the relevant implementations.
- DeepStream `nvvideoconvert`/`dsxvideoconvert` factory selection. The corresponding
  shared YAML key does not control HockeyMON's AspenNet video path.
- Qt archive-job sidecar logs and interrupted-job recovery. HockeyMON has no
  equivalent archive-job manager; existing CLI and Slurm logging remain in use.

The large rink-mask overlay fix addresses a DeepStream OpenGL texture path that
HockeyMON does not use. HockeyMON composites masks before its bounded preview
downsampling; GPU previews resize before transfer to host memory.

## Validation and use

The series received independent code reviews with fixes for the findings that
required changes. Focused Python and Bazel tests cover runtime failures,
publication rollback, bitrate geometry, camera controls, grading, calibration,
artifact loading/recovery and leveling. Real Hugin checks cover all 22
projections, relative scale, NONA/enblend/multiblend output and leveling with
Unicode source paths. Native AKAZE/homography and Rust UI checks also passed.
Projection capture also passed real Hugin capture, resume and forced-rerun checks
while preserving the source game, plus subprocess timeout/crash cleanup tests.

The combined checkout passed 467 Python tests, including native/Python tracker
parity, 23 Bazel targets and three Rust UI tests. Black and Ruff passed across all 76 modified
Python files. GitHub reported no configured status checks for these PRs; the
validation evidence is from local runs.

Rebuild the HockeyMON native extension before using the new AKAZE detector or
native output-cap support. In the Python 3.14 validation environment, native
tests used a local Bazel override pointing at newer pybind headers already
installed with Torch; repository dependency pins were not changed.
The subsequent build fix pins pybind11 3.0.1, so `make perf develop` supports
Python 3.14 without that override, including the standalone AKAZE binding.
