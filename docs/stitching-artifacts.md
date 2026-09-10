# Stitching artifact generations

Calibration builds its PTOs, source images, coordinate TIFFs, seam, panorama and
reference preview in a private `.stitching-stage-*` directory under the game.
It validates the complete replacement before changing published files. A failed
matcher, optimizer, map generator or seam command leaves the previous calibration
available. Candidate retries remain restricted to rejected geometry.

Publication fsyncs the new files, hardlinked backups and a recovery journal before
replacing any final filename. The `.stitching.lock` excludes other builders,
clean operations and map loaders throughout publication. On the next access, an
interrupted publication restores the old generation, or finishes cleanup after a
committed generation. Recovery refuses to replace files whose identities changed
outside the transaction. Do not remove `.stitching-publication.json` or its stage
directory manually while recovery is pending.

Runtime initialization holds the same lock until native or Python map loading
finishes. Single-image remappers and legacy seam regeneration cooperate as well;
there is no per-frame filesystem locking. Seam regeneration uses checked commands
and staged replacement. Different games no longer share process-wide `chdir`
state during Hugin commands. A nonblocking lock is available to interactive
calibration editors so an active rebuild can be reported immediately.

The manifest records effective settings plus source video identity, frame offsets
the selected timestamp, effective output scale and pinned lens-profile fingerprint.
Changing a source, offset, scale, lens profile or framing invalidates the cache. Source references in published PTOs point to stable game-local `left.png`
and `right.png`. Image inputs are identified by contents, so repeated frame-array
calls reuse the generation despite fresh temporary filenames and skip matching
on a cache hit. Content checks also detect edits to published reference images.
A manually edited PTO retains its control points on a nonforced rebuild only
when its effective settings, source identity and reference images still match.
All entry points propagate the game identity when invalidating geometry caches.
Derived rink masks and panorama-space config caches are invalidated only after
the replacement has passed validation; timestamps/audio offsets remain intact.

Before decoding or allocation, TIFF and PNG headers are checked against 65,536
pixels per output edge and 256 MiPixels per canvas. Placement TIFFs are limited
to 2 GiB, coordinate TIFFs to four bytes per pixel plus 32 MiB, and seams to
512 MiB. TIFF metadata has separate per-tag and total budgets; duplicate tags,
invalid raster spans, mismatched coordinate dimensions, oversized PNG chunks and
out-of-canvas seams are rejected. The output limit does not change the uint16
source-coordinate sentinel used by existing map generators.

These changes adapt hstream's coherent calibration publication and map safety
work, including `ae97044e`, `63a95f63`, `b95b2c54`, `3f5baa15`, `48b19de2` and
`c9c3d835`, to HM's file-based pipeline.
