# DriveGPT telemetry databases

Hstream records one `hstream_telemetry.db` (or numbered `hstream_telemetry-N.db`) in its working directory. Publication copies a completed, closed database to the game directory using the finalized video's suffix, such as `hstream_telemetry-2.db`. Capture without a saved video publishes the next available numbered generation.


HM's `hmtrack` also records SQLite telemetry. Enabled `SaveDetectionsPlugin`,
`SaveTrackingPlugin`, and `SaveCameraPlugin` stages write one `hm_telemetry.db`
(or a fresh `hm_telemetry-N.db` in a reused working directory) instead of their
observation CSVs. The normal publication policy copies the current closed
recording to the game/deploy directory as `hm_telemetry-N.db`, using the video's
suffix when video is saved. Time-limited runs stay in the working directory
unless `--deploy-dir` is supplied, as before. Enabled pose/action outputs remain
supplementary CSV files; older CSV inputs remain readable.

Both producers still save calibration `rink_mask_*.png` files in the game
directory. HM archives the run's actual CPU calibration mask inside each database
geometry as lossless PNG, without writing an additional run mask companion.
Encoding and hashing happen once per geometry on the metadata writer thread.
Recording observes only image dimensions; it never downloads video pixels.
Small detection/tracking/camera metadata is copied at the existing save stages,
before downstream mutation, into a bounded queue. Separate stages and concurrent
batches are matched and ordered before writing. Missing stages, invalid geometry,
or writer errors fail the run rather than dropping telemetry. A stalled batch
that exceeds the bounded reorder window also fails explicitly.

HM stores source frame IDs separately from positive, sequential sample IDs.
When explicit source `pts_ns` are unavailable, timestamps are derived from source
frame IDs and source FPS; the configuration archive records this provenance.
Discontinuities and camera policy changes create training boundaries. HM has no
native replay checkpoint producer, so its native replay/checkpoint tables remain
empty. The complete flag is set only after Aspen, video output, and the remaining
pipeline resources have successfully shut down; failed/interrupted recordings
remain incomplete and cannot be trained on or published as completed runs.

Experiment `reuse_tracking` passes the exact first variant's completed database
to later variants and records the reused tracks with their new camera outputs.
An explicit `--input-tracking-data=/path/to/hm_telemetry-N.db` also works with
`LoadTrackingPlugin`. Reuse requires one completed run with one unambiguous
geometry/source/epoch and unique source frame IDs. Merged or discontinuous inputs
must be selected separately; missing frames are errors, while recorded empty
frames remain valid. Other legacy CSV-only loaders and analysis commands retain
their existing input formats. Database training and publication use the same
commands below for either producer.

The database includes the original stitched canvas dimensions, lossless rink mask, mask transform/hash/revision, ordered detections and tracks, fast/Program camera outputs, timestamps and source/reset identities, configuration history, exact native replay inputs, and periodic native checkpoints. Panorama pixels and encoded video are not stored. A 16,000 × 6,500 recording retains native coordinates regardless of the saved video's dimensions.

New hstream recordings also archive the complete resolved launch configuration, the loaded baseline/user/game layers, app and subconfiguration documents, and the contents of referenced text configuration files. This `hstream-run-configuration-v1` YAML archive is stored in `config_events` with `kind='run-configuration'`, `key='startup'`, and `artifact_name='run-config.yaml'`. Layer snapshots preserve parsed values; referenced files preserve their text. Directory references and unavailable optional files are recorded explicitly. Masks are stored separately in `geometries`; model/video binaries remain external. Copying, merging, fingerprinting, and dataset publication preserve the archive with its run GUID.

To retrieve the full configuration for a run:

```sql
SELECT artifact_contents FROM config_events
WHERE run_id = '<run-uuid>' AND kind = 'run-configuration';
```

Train directly from any combination of individual and merged databases:

```sh
./drivegpt_train.sh --database='/data/games/*/hstream_telemetry-*.db' \
  --database=/data/combined.db --no-pose --include-rink --rink-input=grid
```

`--database` accepts a file, directory (recursive database discovery), or glob and is repeatable. `--game-id` and `--file-list` also prefer databases in the selected game directories when no explicit CSV filename overrides are supplied. Legacy CSV datasets remain readable for existing recordings. Pose features are not present in hstream recordings; use `--no-pose`.

Every processing run has a UUID independent of the filename. Multiple passes over the same game remain separate runs. Copies and merges retain their UUIDs. Identical repeated UUIDs are included once; differing content under the same UUID raises an error. Training caches and worker sharding use run/geometry identities. All runs and geometry revisions with the same source game ID stay on the same side of the training/validation split. Each geometry revision has its own rink input. Sequences cannot cross source changes, resets, seeks, timestamp reversals, geometry changes, or configuration boundaries. Frames with zero tracked players remain available.

Use stable, consistent game IDs for the same physical game; renamed copies with different game IDs cannot be recognized as the same source game automatically.

For explicit dataset selection:

```yaml
schema: hockey-drivegpt-dataset-v2
root: /data
databases:
  - games/*/hstream_telemetry-*.db
  - combined.db
include: ['*']
exclude: []
split:
  seed: 0
  validation_fraction: 0.1
  # Alternatively, assign all runs of selected games to validation:
  # validation_games: [game-b]
```

Pass `--dataset-config=/path/to/dataset.yaml`. A split must leave at least one source game for training; for a one-game experiment, use `validation_fraction: 0` (or `--val-split=0`). Training derives normalization from training canvases and rejects larger validation canvases, avoiding silently clipped geometry.

Publish a portable dataset by copying database snapshots, preserving source database basenames and suffixes:

```sh
python -m hmlib.cli.drivegpt_dataset --database=/data/games \
  --database=/data/combined.db --out=/data/training-dataset
```

The output contains the database copies, a catalog, and `dataset.yaml`. The generated configuration verifies run fingerprints against the published catalog and explicitly selects usable run/geometry passages; short revisions remain in the databases and are listed as rejected in the catalog. Masks and configurations stay inside the databases; no CSV companions are created. The existing `--source` publishing option automatically uses this format when databases are present.

Inspect and merge:

```sh
python -m hmlib.cli.telemetry_db inspect /data/game/hstream_telemetry-2.db
python -m hmlib.cli.telemetry_db merge /data/combined.db /data/games
```

A merge transaction adds entire runs, preserves row ordering/identities, skips identical existing runs, and rolls back if any conflict or broken foreign key is found. The destination must not also appear among the inputs. Never use a live/incomplete recording as a training or merge input. A temporary SQLite journal may exist while writing; completed publications are standalone files.

The versioned schema is `hmlib/telemetry/schema.sql`, byte-for-byte identical to hstream's `src/libs/recording/schema.sql`. `PRAGMA application_id` and `user_version` identify compatible databases. Unknown schema versions fail explicitly. Track IDs are decimal strings to retain the entire unsigned 64-bit range; ordered child rows use `(run_id, sample_id, ordinal)` keys.

Validation includes C++-produced recordings consumed by the Python trainer, exact mask-grid equivalence, configuration/geometry boundaries, empty frames, mixed database selection, idempotent merging, conflict rollback, and numbered publication. Mask compression happens once per geometry on the C++ metadata writer thread. Snapshot frequency remains periodic; SQLite does not introduce full state serialization on every frame.
