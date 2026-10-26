# HockeyMON web Program producer contract

The producer foundation is opt-in. It does not qualify HockeyMON for browser
Program playback: authenticated presentation, browser audio, runtime parity and
export qualification remain required. `--web-backend-info` reports `qualified: false`. The outer service
must keep deployment disabled until those workflows are implemented and compared
against the same Qt/native/HockeyMON installation.

## Admission

`hm-cli --web-backend-info` imports no Torch, CUDA, OpenCV or YAML. It returns
`protocol: hstream-web-program`, `version: 1`, the feature names and
`runtime`. Runtime provenance hashes the actual installed `hmlib` and `hockeymon`
package files, including native extensions, owned shared libraries and embedded
executables. Bytecode caches are excluded. Symlinked package files bind their
actual target bytes; linked package directories and missing files fail. Source
checkouts without built embedded helpers or the native extension are not an
installed runtime. The desktop `--ui-backend-info` handshake remains unchanged.

Launch with exactly:

```
hm-cli --web-program-request /RUN/hockeymon-program-request.json
```

No additional desktop arguments are accepted. The request is a nonsymlinked,
nonempty regular UTF-8 JSON file of at most 1 MiB. Its parent is the private run
root. `HM_UI_INITIAL_LAUNCH` is forbidden. The complete top-level field set is:

| Field | Meaning |
| --- | --- |
| `version` | Integer `1` |
| `run_id` | Canonical lower-case UUID |
| `deployment_id` | Operator-frozen 64-character lower-case SHA-256 deployment ID |
| `runtime` | Exact portable `runtime` object from the installed handshake |
| `request_digest` | SHA-256 of the canonical request without this field |
| `native_root` | Canonical absolute path to the frozen matching native installation |
| `native` | Exactly the six portable file bindings below |
| `game_config` | Portable binding of `game/config.yaml` |
| `sources` | Exactly `source`, or exactly `left` and `right`, each with an ordered chapter list |
| `launch` | Typed output, time, pacing and preview selection below |
| `calibration` | `null` for a single source; a pinned completed snapshot for a pair |

Portable file bindings are exactly `{"size": INTEGER, "sha256": HEX}`. They do
not contain device/inode numbers, which may differ across Slurm mount clients.
The compute producer opens and hashes each admitted input with bounded reads,
then retains its descriptor and local device/inode/size/mtime/ctime identity.
Source media are also hashed once during admission; this costs a sequential read
of every admitted chapter. Later child launches check local pins rather than
rehashing the recordings. No video frames are decoded or read back for this check.

The native binding keys and fixed paths below `native_root` are:

| Key | Path |
| --- | --- |
| `runner` | `bin/hstream-cli` |
| `calibration_export` | `bin/hstream-calibration-export` |
| `model_prepare` | `bin/hstream-model-prepare` |
| `model_infer` | `bin/hstream-model-infer` |
| `application` | `configs/ds_hockey_app_config.yaml` |
| `baseline` | `configs/baseline.yaml` |

Extra manifest entries cannot authorize other host paths. The native baseline
must match the installed HockeyMON baseline bytes.

Each source entry is exactly `{"alias": "game/NAME", "path": "/immutable/media",
"content": PORTABLE_BINDING}`. The alias must be a direct child of the private
game directory, and resolve to the admitted canonical absolute media path. Source
order and the resolved Program plan must agree. The outer service remains
responsible for owner-scoped source selection and immutable input retention.

`launch` contains exactly `sinks` (a nonempty unique list of supported desktop
sink names), `telemetry` (boolean), `start_seconds` (finite nonnegative number),
`duration_seconds` (positive finite number or null), `realtime` (boolean), and
`preview` (`none`, `program`, `stitched`, `source0` or `source1`). Preview naming
does not grant an authenticated browser transport. Local speaker audio is disabled
in this web mode; source audio for authored outputs retains its existing producer.

For a pair, `calibration` contains exactly `directory`, `manifest`, `generation`,
`output_generation` and `invalidation_id`. The directory is `calibration/NAME`
below the run root; `manifest` binds its `native-calibration.yaml`. The shared
native snapshot loader validates every map/mask/config checksum, source identity
and generation. Pending or missing calibration fails with a Prepare stitching
requirement. Web Program consumes the snapshot without running the native
calibration graph or opening desktop editors. Native calibration and later
workflow adoption remain owned by the outer supervisor.

## Canonical digest and private layout

Canonical JSON uses Python's `json.dumps(value, sort_keys=True,
separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`.
There is no Unicode normalization. `request_digest` excludes only its own
field. The outer Django adapter can compute this with the standard library;
it must not import `hmlib` to compute a digest. Numeric values must preserve
their JSON types when the request is written.

The fixed directories are `game`, `home`, `work/hockeymon-session` and
`work/hockeymon-output`. The producer rejects linked private directories and
rechecks their identities. An exclusive producer lock and a durable admission
record prevent concurrent or repeated execution of one admitted request. Child
processes receive private HOME/config/session/work paths and only pinned native
helpers. The existing process guard and outer supervisor retain process-tree
ownership during model preparation, Python processing, cancellation and cleanup.

The pinned `game/config.yaml` must contain the frozen initial `preview_overlays`
mapping with exactly Boolean `players`, `play` and `rink`. Admission derives
`HSTREAM_UI_PREVIEW_OVERLAYS` from those bytes for both the frame owner and runtime
replay state. Missing or malformed selections fail before child launch; inherited
environment values cannot replace this selection. The outer service resolves the
shared baseline/defaults before freezing the game config. No producer-local
overlay defaults or request schema extension are introduced.

## Selected preview snapshots

`selected-jpeg-snapshot-v1` adds a private producer transport. It does not grant
browser access or qualify playback. The selected route uses the existing bounded
preview worker: one pending frame per stream, selected preview cadence, GPU
resizing before its existing bounded device-to-host copy, and CPU JPEG encoding.
There is no additional full-frame readback. Qt still receives its fixed JPEG and
width/height files. The original source dimensions survive source-tab resizing.

The frame owner first publishes `work/hockeymon-session/web-preview-owner.json`
atomically. It contains exactly `protocol: hstream-web-preview-owner`, integer
`version: 1`, `run_id`, `request_digest`, `session_id`, `channel`, `generation`
and `status`. Status is `selected` or `retired`; `channel` also permits `none`.
This is the applied route identity even before its first JPEG exists. Route
changes retire the old manifest before replacing the owner record under the
same lock. Seek and shutdown retire ownership independently in the supervisor;
the retired record retains the last session, channel and generation. Missing or
retired ownership has no displayable frame. The outer supervisor must also fence
requested but unacknowledged route/seek changes and prove process liveness; the
owner file establishes neither liveness nor presentation.

The frame owner publishes `work/hockeymon-session/web-preview.json` atomically.
Its exact fields are:

| Field | Meaning |
| --- | --- |
| `protocol`, `version` | `hstream-web-preview`, integer `1` |
| `run_id`, `request_digest` | Exact admitted Program request binding |
| `session_id` | 32 lower-case hex characters identifying the current frame owner |
| `channel` | `program`, `stitched`, `source0` or `source1` |
| `generation` | Positive acknowledged route generation |
| `sequence` | Positive sequence within that frame owner; dropped frames may leave gaps |
| `frame` | Exactly `source_frame_id` and `pts_ns`, each an integer or `null` |
| `source` | Exactly integer `width`, `height` before preview scaling |
| `image` | Exactly `path`, encoded `width`, `height`, byte `size`, lower-case `sha256` |
| `overlays` | Exactly Boolean `players`, `play`, `rink` selections captured with the frame |

The image filename is
`preview-web-SESSION-GENERATION-SEQUENCE.jpg`. JPEG files are written once;
publishing their manifest is the commit point. The route lock covers that commit
and rejects an encode whose route, generation or overlay selection changed while
it was in flight. Selecting a route again requires a new generation. Selecting
`none` retires the manifest and claims no image. Seek retires it before replacing
the worker, generates a new session ID, restarts the sequence, and retains the
last acknowledged route generation. Shutdown also retires the manifest.

At most two JPEG snapshots remain. The current manifest's image remains while
its successor is written; a previously superseded image is pruned before another
is added. Route retirement removes the manifest before its files. Readers
must open the named file without following links, pin its descriptor, check its
size and SHA-256, and verify run/request/session/channel/generation before decoding.
`read_snapshot` provides the private supervisor's read path and verifies matching
selected ownership before and after reading bytes. A missing, replaced
or concurrently retired file is a retry without a presentation receipt. Readers
must compare decoded dimensions with the manifest before painting and must not
treat file existence or the desktop preview-ready log as proof of browser paint.

Bounds are 16 KiB for the manifest, 4 MiB for JPEG bytes, 4096 for each decoded
side, 4,194,304 decoded pixels (the Qt pixel bound), and 131072 for each source
side. Frame IDs, PTS, generation and sequence cannot exceed JavaScript's exact
integer range.

`source_frame_id` carries a proven decoded source ordinal. Complete decoding from
chapter 0 counts actual returned frames, including frames skipped by deliberate
sampling, and carries that count across EOF independently of estimated chapter
lengths. Any nonzero seek, initially skipped chapter, decoder-discarded surface
or partial batch, or unqualified decoder API makes the ordinal unknown; the field
then stays null. In particular, a requested frame-number seek is not proof of its
resulting ordinal: the AMD/high-bit time seek and NVIDIA SimpleDecoder's
time-to-index lookup can return another VFR frame. The existing processing counters,
seek policy and legacy FPS-derived `pts_ns` remain unchanged. Stitched and Program
retain the left source's evidence, while source tabs retain their own source's
evidence. Preview geometry transforms preserve null IDs and actual decoded timing.
The older loader field `pts_ns`, estimated from frame number/FPS, remains excluded
from this contract. Only `decoded_pts_ns` can populate the snapshot's `pts_ns`.
This is the actual source chapter's PTS converted with rational arithmetic and
rounded to nanoseconds. It is not a normalized playlist, synchronized pair or
browser audio clock. Chapter changes can reset it. Negative or non-JavaScript-safe
values remain null on v1, while the internal signed rational record is preserved.

NVIDIA captures `DecodedFrame.getPTS()` before conversion to GPU tensors and uses
the selected stream's rational `time_base` from the existing FFProbe metadata.
The installed SDK demuxer forwards original packet ticks to the display callback;
its constructor's `timescale=1000` comment does not make those ticks milliseconds.
No pixel readback is added. AMD and the shared BGR48 software reader attach
`showinfo=checksum=0` to the same FFmpeg decoder that supplies raw stdout frames.
The parser pairs sequential integer ordinals and their declared input timebase
with complete raw frames. `-copyts`, accurate input-side `-ss` and
`-fps_mode passthrough` preserve VFR gaps, reorder B-frames for presentation and
drop seek preroll before this metadata tap. Seeking retains the existing requested
time policy; timestamps describe the actual returned frame, not the requested
index divided by FPS. Late seeks use input seeking, not full-prefix trimming.

The metadata channel hashes no frame bytes. It retains at most 256 timestamp
records, rejects diagnostic lines above 64 KiB and fails on missing, reordered,
extra or malformed records. One metadata wait is bounded to five seconds after
its full raw frame arrives. AMD reads directly into the existing batch storage;
the timestamp path adds no full-frame copy, conversion or readback. The existing
AMD VAAPI path still downloads frames for its raw pipe; that prior limitation is
unchanged. Unsupported decoder APIs retain null timestamps rather than deriving
them from FPS.

Before a web preview encode is queued, its internal `decoded_source` freezes
`role` (`source`, `left` or `right`), `chapter_index`, the admitted chapter's
portable `content` binding, signed decimal-string `pts`, and integer
`time_base_num` / `time_base_den`. The source role follows the actual source batch;
the publisher checks it against the selected route and admitted source catalog.
This record survives sampling, asynchronous loader queues and preview geometry
transforms. It is deliberately not added to the strict v1 manifest. A coordinated
v2 manifest and authenticated receipt update must carry this complete evidence
before browser audio can synchronize across sources or chapters. Paired/chapter
origin normalization, GPU qualification, browser audio and authenticated
exact-paint/runtime comparison tests remain required.

A CPU metadata overhead check decoded 180 H.264 frames at 640×360 with two FFmpeg
threads, reading directly into raw batch storage. Three alternating trials used
10.7–11.3 ms of Python CPU without metadata and 14.3–15.1 ms with this parser,
approximately 22 microseconds of additional Python CPU per frame. Total elapsed
times overlapped (75–188 ms without, 89–107 ms with); this short CPU check cannot
qualify GPU throughput or end-to-end latency. No frame checksum was enabled.

## Typed runtime controls

`typed-runtime-controls-v1` uses the existing desktop frame-owner mailbox. Only
the confined web launch accepts this protocol; desktop commands remain unchanged.
The lightweight handshake adds `control`, with integer `version: 1`,
`maximum_request_bytes: 16384` and the explicit action list below. It does not
advertise the native CLI's runtime capability versions.

The supervisor emits `HM_WEB_PROGRAM_READY ` followed by canonical JSON with
exactly `version: 1`, `run_id`, `request_digest`, `session_id`, `actions` and
`owner`. The complete owner object is the `web-preview-owner.json` record above.
Readiness requires the current frame-owner spec, session-bound runtime record,
playback capabilities and selected owner to agree. It does not prove that a JPEG
exists, audio is ready or any browser has painted a frame. The outer supervisor
also checks child liveness and matching committed ownership before serving images.

Input is one newline-terminated `@web-control BASE64URL` line. BASE64URL is
unpadded and decodes to at most 16 KiB of canonical JSON. Its exact fields are
`version: 1`, `run_id`, `request_digest`, `session_id`, `request_id`, `action` and
`payload`. Run and digest match admission; session matches the last READY event.
Request IDs are positive JavaScript-safe integers, strictly increasing for the
whole launch, including seek replacements. Only one request may be pending.
Malformed envelopes produce a bounded diagnostic; valid envelopes with rejected
payloads or ownership produce a correlated failure. Legacy text commands,
arbitrary property names, filesystem paths, editor launches and captures are
unavailable through the web ingress.

| Action | Exact payload |
| --- | --- |
| `pause`, `resume`, `step`, `stop` | `{}` |
| `rate` | `{"rate": NUMBER}`; exactly 0, 0.25, 0.5, 1, 2 or 4; 0 means maximum |
| `seek` | `{"position_ns": INTEGER}`; nonnegative, JavaScript-safe, within this run |
| `preview` | `{"channel": NAME}`; none, program, stitched, source0 or source1 |
| `preview-overlays` | Exactly Boolean `players`, `play`, `rink` |
| `properties` | `{"assignments": [{"element": NAME, "property": NAME, "value": VALUE}, ...]}` |

READY advertises only available playback actions. `rate` requires admitted
real-time playback; `seek` requires recording and telemetry disabled. An absent
preview route or unsupported live property fails through its existing owner.
Property batches contain 1–64 distinct assignments. The closed property catalog
is `dsplaytracker0` fixed-edge-rotation-angle / -left / -right; `playcropper0`
those same rotations plus scoreboard-position, exposure, shadow-lift and
shadow-lift-black-point; `hmstitcher0` exposure, shadow-lift and
shadow-lift-black-point. Scoreboard position is upper-left, upper-center or
upper-right. Other values are finite JSON numbers, never numeric strings or
booleans; the actual live control enforces its range and exact scaled precision.
Tuning-file imports and stitched-output epochs remain outside this typed catalog.

Each terminal event is `HM_WEB_PROGRAM_CONTROL ` followed by canonical JSON
containing exactly the envelope's version/run/digest/session/request/action,
`status` (`applied` or `failed`), `result` and `error`. Success has a result object
and null error. Failure has null result and a message bounded to 1024 characters.
Input acceptance never produces applied: the existing frame owner must first
complete the mutation and publish its same-session receipt. Control timeouts,
negative or stale receipts, replacement and shutdown produce failures. A failure
does not establish unchanged pixels or authorize reopening a fenced preview.

Successful results echo the payload. `preview` and `preview-overlays` also include
the exact resulting `owner`; route generations are assigned by the producer.
`stop` returns only `owner`, with retired status, after the frame owner closes
preview publication under the encoder's commit lock. Queued and in-flight JPEGs
cannot republish afterward. Stop acceptance is distinct from descendant retirement
and output adoption, which remain the outer supervisor's responsibility.

Seek returns `position_ns` and `owner` only after the supervised replacement has
a different session and reports its first completed batch with `seek_ready: true`.
The acknowledgment echoes the old request session; `result.owner` carries the
replacement session. A new READY then announces that session. The existing
supervisor and outer job deadlines own seek/replacement cleanup. Acknowledged
properties, preview route/generation, overlays, pause and explicit rate survive
replacement. The outer bridge fences route, seek, overlay and Stop requests before
dispatch and reopens only against the correlated applied owner and matching file.

## Browser audio transport proposal and parity gap

The existing desktop `AudioMonitor` starts a supervised local `ffplay`. It chooses
an audio-bearing source with the smallest saved synchronization frame offset and
adds that offset divided by the source FPS to the legacy playback position.
It moves through the selected source's probed chapter durations. Pause and mute
stop/reap the player; resume starts another player after progress advances. A
progress stall longer than 0.5 seconds stops playback, and drift above 0.75 seconds
restarts it. Seeking holds audio until the replacement worker completes its first
batch. The monitor still runs for unrestricted playback; ffplay runs at normal
speed, so rapid progress or changed video rate can trigger repeated rebasing.
It performs no time stretching and publishes no browser packets or exact A/V
presentation receipts. These are existing behaviors, not a browser qualification.

A proposed browser transport needs a separate supervised audio decoder using
only pinned admitted chapters and the same explicit source-selection policy.
It should produce bounded PCM or Opus packets with run/request/session identity,
an audio epoch, source role, chapter content binding, signed source PTS/rational
timebase and exact sample count. The coordinated v2 video record must carry the
same source/chapter evidence. A frozen origin mapping should define how actual
selected video frames, synchronization offsets, chapter starts and audio samples
share a presentation timeline; summing frame counts divided by FPS cannot supply
that mapping for VFR sources or independently offset streams.

An authenticated browser AudioWorklet, or a unified WebRTC sender, can own the
playout clock. Video paint scheduling must use that clock and matching epochs.
Applied pause flushes bounded queued audio, resume establishes a fresh clock
anchor, and seek retires the old decoder/queue before admitting replacement audio.
Seek must remain silent until replacement readiness and the first matching video
presentation; Stop must reap audio descendants before cleanup/adoption. Source
changes, stale epochs and lost browser clients must not retain audible packets.

Unrestricted playback, non-unit rate, mute, stalls and first-frame behavior need
paired Qt/browser comparisons and an explicit shared policy. Silencing those
modes or enabling time stretching without the coordinated desktop behavior would
change the product. This proposal therefore adds no audio implementation or policy
change, and public HockeyMON browser admission remains closed.

## Publication

The existing publisher commits outputs below `game` and records their exact local
identities after temporary publication links retire. Metadata companions retain
the same producer-assigned positive version; media kinds come from the producer,
not filenames. No native finalizer is invoked.

After resources close and the owned working directory is removed, the supervisor
writes `hockeymon-program-receipt.json` (at most 4 MiB). It contains `protocol`,
`version`, `run_id`, `deployment_id`, `runtime`, `request_digest`, `status` and
`artifacts`. Status is `complete` or `cancelled`; cancellation has no artifacts.
Each artifact contains relative `path` (`game/NAME`), original logical `name`,
`kind`, positive `version`, and decimal-string `device`, `inode`, `size`,
`mtime_ns` and `ctime_ns`. Complete output kinds must exactly match the admitted
archive and telemetry selection. Replaced, escaped or missing files fail.

The bounded `HM_WEB_PROGRAM_PUBLICATION` event identifies the receipt filename
and its portable size/SHA-256 binding. It is emitted only after the receipt is
durable. The outer supervisor must still prove descendant retirement, compare the
run/deployment/request bindings and verify video, sidecar and telemetry contents
before adopting artifacts. A child exit code or the desktop publication log alone
is never an adoption receipt. Producer failures do not publish a success receipt.
