# Native play-tracker checkpoints

Camera experiments can resume the native policy at a recorded frame boundary
without resetting its velocities, braking, frozen boxes or player histories.
Include `hockeymon/csrc/play_tracker/PlayTrackerSnapshot.h`:

```cpp
auto checkpoint = tracker.snapshot();
auto encoded = hm::play_tracker::serialize_snapshot(checkpoint);
auto restored = hm::play_tracker::PlayTracker::from_snapshot(
    hm::play_tracker::deserialize_snapshot(encoded));
// Apply the experiment's runtime controls, then process the first trial frame.
auto result = restored->forward(tracking_ids, tracking_boxes);
```

Capture before the first frame to replay, or after its predecessor. Camera CSV
rows contain the boxes after processing a frame; those rectangles alone do not
describe the hidden motion/history state needed to resume it.

`from_snapshot` constructs an independent tracker and rebinds the detector's
braking callback to it. Living boxes, player histories and configuration do not
alias the source. Plain copying and moving of `PlayTracker` are disabled because
they would preserve the detector's pointer to its original owner. An individual
tracker still requires external synchronization while processing, tuning or
capturing state; independent trackers can run concurrently.

The snapshot records the tracker configuration and counters; each living box's
effective translation, resizing and geometry configuration; pan/zoom speeds,
braking/cooldown/frozen state; detector overrides; and the complete retained
player trajectory buffers and their running velocity sums. Size clamps and
Gaussian arena bounds derive from the stored configuration when restoring.
Process-wide diagnostic log counters are not policy state and are not restored.

`snapshot.config` contains the construction configuration, with the current
player-size filter. Runtime box settings are in
`snapshot.living_boxes[i].config`; runtime detector braking overrides are in
`snapshot.detector`. A host that uses a separate baseline for zero-as-default
or zoom-aggressiveness controls must persist that baseline alongside the native
snapshot. Host state such as source identity, arena provenance, skipped policy
ticks, `has_received_tracks`, video timing and previous wrapper outputs likewise
belongs in its enclosing checkpoint.

Persistence is an explicit, versioned field format using the classic locale and
lossless finite floating-point representations. The deserializer checks the
schema and policy implementation identifier, field ordering, value types,
topology, geometry, history shapes and counters. Malformed or incompatible
state throws `std::invalid_argument` before a tracker is constructed. The
format bounds payloads to 16 MiB, 64 boxes, 4,096 players and 4,096 history
entries per player, with a cumulative value budget. It contains no object
memory dumps, addresses, ABI layout or external file references.

Persisted arenas use nonnegative pixel coordinates up to 1,000,000; fixed
aspect ratios are between 1/1,024 and 1,024. These deliberately broad video
geometry limits also bound native diagnostic work and exclude unusable float
arithmetic. Validation checks representable centers/scaled/aspect-constrained
boxes and the dynamic projection's positive horizontal span. Initial camera
boxes may still extend outside their arena, as native initialization permits.

Maintain `kPlayTrackerSnapshotSchema` when the state format changes and
`kPlayTrackerSnapshotImplementation` when camera-policy interpretation changes.
Record the application/native revision and execution configuration in capture
provenance as well. Tests establish exact continuation and repeatability on the
same build/platform; cross-platform bit identity is not promised, and OpenMP
team/nesting settings can affect floating-point accumulation during clustering.

Validation:

```sh
bazelisk test --config=x86_64 --config=debug \
  //hockeymon/csrc/play_tracker:play_tracker_snapshot_test \
  //hockeymon/csrc/play_tracker:resizing_box_test \
  //hockeymon/csrc/play_tracker:player_size_filter_test
```
