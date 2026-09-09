# Completing and publishing tracking output

A tracking or stitching run now fails if its encoder, muxer, CSV writer, or
pipeline finalizer fails. Shutdown still attempts the remaining resource
finalizers. When processing already failed, that original exception remains the
reported failure and cleanup errors are logged with their tracebacks.

CSV flushes append after the first successful write and synchronize their data
to storage. Empty streams produce empty headerless CSV files. An uncertain write
or synchronization failure makes that writer terminal: a later cleanup cannot
retry an append that may already have reached storage.

`hmtrack` publishes completed output into the game directory or `--deploy-dir`
using one numeric suffix for the video and all CSVs. It selects a generation
higher than existing matching artifacts, including orphaned companion files.
Files are copied to hidden staging files on the destination filesystem and
synchronized before any final name appears. Final names are created without
replacing existing entries, and `tracking.csv` (or its suffixed/labeled form) is
published after its companions and video. A persistent
`.hm-output-publication.lock` serializes HM publishers. The protocol uses
ordinary same-filesystem hard links and does not require `O_TMPFILE` support on
NFS. Published files are independent copies of the working files.

Publication failures fail the run and retain the working artifacts. Cleanup
removes only staging/final names that still reference files owned by the failed
attempt. A crash can leave hidden `.hm-publish-*.partial` files or incomplete
companion generations; filename-based dataset discovery cannot select these
without the tracking marker, and a subsequent publication skips their suffix.

An explicit `--output-video` filename is preserved. Its numeric suffix fixes the
CSV suffix, and an existing destination causes an error instead of being
overwritten. When that explicit video is outside the CSV destination directory,
its publication is a separate transaction; a later CSV failure retains the
already completed video as well as the working artifacts.

This ports the output lifecycle and publication behavior from hstream commits
`780308e0`, `89b7df83`, `58099b70`, and `51ffac90`. It does not add hstream's native
Main10/P010 archive backend. HM's current decoder outputs RGBP/uint8, its native
stitch input is uint8, and its encoder accepts uint8/YUV420. Floating-point image
processing alone does not preserve a 10-bit source through those conversions.

## Automatic output bitrate

The default `video_out.bit_rate: null` selects the greatest video bitrate per
pixel across the actual input cameras and chapters. That ratio is multiplied
by the final encoder width and height after crop/resize/letterboxing. For
example, a 30 Mbps 1920×1080 camera produces a 120 Mbps 3840×2160 archive, or a
7.5 Mbps 960×540 output. Ratios use integer arithmetic and round half up.

Set `--output-video-bit-rate` or `video_out.bit_rate` to a positive integer to
keep an explicit bitrate. Metadata that lacks a bitrate or dimensions produces
a warning and is excluded from selection. If no usable source metadata is
available, encoding uses the previous 55 Mbps default and reports that fallback.
Probe errors propagate. This behavior ports hstream commit `3078abb8`.
