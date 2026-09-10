# DriveGPT training

DriveGPT learns the final smooth 16:9 camera box from tracked object boxes and its
own previous camera prediction. The official configuration uses `slow_tlwh` targets
and `players_prev_y` inputs, with pose disabled and a compact static rink grid. The catalog
retains fast-camera CSVs for provenance; `slow_fast_tlwh` remains available as an
optional dual-output training target. Runtime consumes live tracking tensors;
CSV files are only used to record training examples and supervision.

## Dataset

Publish a new dataset into an empty destination:

```bash
PYTHONPATH=. python -m hmlib.cli.drivegpt_dataset \
  --source "$HOME/Videos" \
  --out /mnt/ripper-data/datasets/HockeyDriveGPT
```

The curator recursively inspects game directories and requires a matched
`tracking[-N].csv`, `camera[-N].csv`, `camera_fast[-N].csv` triple. It validates
numeric frame/box columns, rejects incomplete telemetry publications and duplicate
camera frame IDs, and requires at least 32 aligned contiguous frames. Selection
maximizes aligned frames, breaking ties by the latest modification time. This means
a newer short test run does not replace a fuller game export.

Copied files keep their original names under `games/<source-relative-game-path>/`.
Matching policy/telemetry sidecars and available rink/config assets are preserved;
pose and video files are omitted. Copies are SHA-256 verified and the completed
dataset is published by a directory rename. Nonempty destinations are never
overwritten. `catalog.json` records every selection/rejection, source paths,
hashes, frame coverage, and duplicate tracking identities. Each game also has a
`provenance.json` file. Unsuffixed configuration companions are collection-time
snapshots; they are not guaranteed to describe the selected generation's teacher.

The generated `dataset.yaml` selects game IDs using `include` and `exclude` glob
lists. It excludes exact duplicate tracking exports by default. A seeded shuffle
of sorted game IDs defines the whole-game validation split; alternatively set
`split.validation_games` to an explicit list. Selected duplicate identities are
rejected to avoid split leakage. Use `split.groups` to map source-game names to
lists of related export/segment IDs; an entire group goes to validation if any
member is explicitly selected for validation. Random splits operate on groups.
Exact hashes cannot identify footage re-exported with different calibration;
review source identities and exclude aliases such as `demo` and `short` in the
official catalog. All selected artifact checksums and minimum sequence lengths
are verified before training. A changed dataset/split cannot silently resume an
old run. `drivegpt_dataset.example.yaml` illustrates the schema.

## Training configuration

```bash
./drivegpt_train.sh --config hmlib/config/training/drivegpt.yaml
```

The YAML controls model dimensions and initialization, live-available features,
context and rollout lengths/start stride, run weighting, loader/cache settings, losses,
scheduled sampling, validation, checkpoints, and distributed settings. YAML values
are defaults; explicit CLI options override them. Unknown options fail. Dataset
configuration paths in a training YAML are relative to that YAML; dataset roots
are relative to their dataset YAML. `--dataset-root` overrides the root on a host.
Output/checkpoint paths follow normal CLI working-directory semantics.

`seq_len` is the runtime attention context stored in checkpoints. `rollout_len`
controls how many consecutive training frames the model practices with that
sliding context; it defaults to `seq_len` and cannot be shorter. The official
recipe uses context 32 and training horizon 128 to expose accumulated feedback
drift. Longer rollouts retain more autograd activations; reduce training
`batch_size` if GPU memory requires it. `val_batch_size` defaults to `batch_size`;
set it explicitly to preserve validation sampling while changing training batches.
Training frame budgets count `rollout_len * batch_size * world_size` per step.
Changing training horizon/batch size preserves compatible evaluation history when
validation batch size and the other validation settings remain unchanged.

Games are sampled uniformly within each worker's shard. Unequal shard sizes
slightly change overall game probabilities. `run_sampling: windows` weights contiguous runs by
their number of eligible windows; `uniform` gives each run equal probability.
`sample_stride` controls eligible start positions and does not skip frames within
a sequence. Numeric frame gaps and camera-policy changes split sequences.

Set `drivegpt_source_checkpoint` to the downloaded UniAD checkpoint, or omit it to
download `ckpts/uniad_base_e2e.pth` from
`OpenDriveLab/UniAD2.0_R101_nuScenes` via Hugging Face. The official configuration
requires initialization to succeed. `drivegpt_init: none` trains from scratch.

## Static rink context

The rink defines the players' spatial domain. The official recipe uses
`include_rink: true` and `rink_input: grid`. The default 32-by-64 occupancy grid
preserves shape, placement, holes, and multiple mask components. Its axes use the
same `x / norm.scale_x`, `y / norm.scale_y` coordinate plane as tracking boxes;
smaller game canvases are padded, not independently stretched to fill the grid.
The new run's normalization contains the declared training-frame dimensions as
well as the observed boxes. Validation/live canvases outside that extent fail.

A float32 grid costs 8 KiB per sequence, independent of rollout length. The loader
caches this grid per game and passes it separately as `[B,1,H,W]`. Mixed-game
batches need each sequence's own rink. The learned encoder preserves spatial
position through flattening and projects the grid to a bounded embedding, computed
once inside each DDP rollout forward and broadcast across time. Raw grids can be
cached during training; learned embeddings are recomputed after optimizer updates.
Live inference caches the embedding by game, immutable geometry revision, model
parameter version, device, dtype, normalization, and grid schema. Geometry changes
reset camera history. Full masks never become per-frame model inputs.

Detector boxes are rescaled to the original stitched-image coordinates before
tracking; tracking CSVs store those coordinates. Live rink profiles must identify
that same coordinate space and match the original tracked frame dimensions. The
stitching producer identifies the actual stitcher instance, input/output shape,
and applied rotation. With `ice_config.params.require_geometry_provenance: true`, each new geometry
causes the rink producer to segment the
original frame in memory; it does not certify old saved masks from dimensions
alone or overwrite them. A changed geometry invalidates the profile and embedding.
Legacy pipelines lacking this identity cannot supply a grid checkpoint. The camera
model uses the same rasterizer as training. RLE/PNG are useful storage compression; the compact
occupancy grid is the neural input. Rasterization samples 4-by-4 points per grid
cell with nearest binary sampling, union before pooling, fractional boundary
occupancy, and zero outside the tracking canvas. Binary sampling makes component
union agree with sampling the combined live mask at component seams.
Grid height/width are YAML settings stored in the checkpoint.

Each selected tracking export needs a generation-matched descriptor: for example,
`tracking-3.csv` requires `rink_context-3.json`. This explicitly binds the mask to
the CSV and documents the source of its coordinate transform:

```json
{
  "schema": "hockey-drivegpt-rink-v1",
  "coordinate_space": "original_stitched_pixels",
  "frame_size": [1920, 1080],
  "tracking": {"file": "tracking-3.csv", "sha256": "<tracking SHA-256>"},
  "masks": [{
    "file": "rink_mask_0.png",
    "sha256": "<mask SHA-256>",
    "mask_to_tracking": [[1, 0, 0], [0, 1, 0]]
  }],
  "evidence": "Describe the matching source-frame/calibration artifact and its identity."
}
```

`frame_size` is original tracking width/height. `mask_to_tracking` maps mask pixel
edge coordinates into that canvas; identity applies to a matching full-resolution
mask. Multiple masks are unioned in that plane. Affines must be finite/invertible
and justified by known geometry. They cannot repair different lens projections or
stitching calibrations. Never infer scale from observed box extrema or silently
substitute a newer mask. The curator preserves descriptors and mask files; training
validates hashes, frame bounds, descriptor identity, PNG decoding, and transformed
occupancy in a coordinated preflight before workers start. The compact grids are
then reused by workers.
Missing/unverified rink context is an error, not an all-zero substitute.

Legacy checkpoints without the new fields retain `rink_input: stats`, their
previous seven per-frame features, and their original input dimensions. The CLI
also retains that legacy default; the official YAML explicitly selects `grid`.
Grid checkpoints store the encoder version and grid dimensions and require a new
output directory. The official grid run writes to `runs/slow-camera-rink`.

## Two-host DDP

Use the same repository revision and PyTorch/CUDA/NCCL release on both hosts.
All ranks need read access to the dataset and initialization weights; rank zero
also needs write access to the output directory. For one GPU each on `mini` and
`monster`, execute these in separate host sessions:

```bash
# mini (rank zero; writes checkpoints)
MASTER_ADDR=192.168.68.74 NODE_RANK=0 \
  PYTHON_BIN_PATH="$HOME/.venvs/hockey-drivegpt/bin/python" ./drivegpt_ddp.sh

# monster
MASTER_ADDR=192.168.68.74 NODE_RANK=1 \
  PYTHON_BIN_PATH="$HOME/.venvs/hockey-drivegpt/bin/python" ./drivegpt_ddp.sh
```

`NNODES`, `NPROC_PER_NODE`, and `MASTER_PORT` are configurable. Set
`NCCL_SOCKET_IFNAME`/`GLOO_SOCKET_IFNAME` to each host's LAN interface when needed.
`batch_size` is per GPU; frame budgets include world size. Each rank/worker has an
independent random sampling stream. A complete scheduled-sampling rollout runs
inside one DDP forward, gradients synchronize, validation sums are reduced across
ranks, and only rank zero writes metrics and atomic checkpoints. Loader workers
use spawn to avoid forking a live CUDA/NCCL process.

## Evaluation and resumption

The official target is **mean IoU >= 0.97 for the slow camera box** on held-out
games. It measures overlap of the final smooth 16:9 view, not classification
accuracy. Optional `slow_fast_tlwh` runs require both box metrics to meet the target. The default evaluates
fixed-seed sampled 256-frame autoregressive rollouts, with a sliding 32-frame
context matching runtime. Each rollout starts from the recorded previous camera
state (or a full-frame state at a run boundary); subsequent feedback comes from
the model with aspect fitting and normalized box clamping. Runtime additionally
fits boxes to the actual frame/arena bounds; the CSV metric does not reproduce
those per-game bounds. This evaluates about 8.5 seconds at
30 FPS; it is not a claim about uninterrupted full-game rollout accuracy.

`*.metrics.jsonl` records the dataset/split identity, resolved arguments,
validation protocol, distributed metrics, and whether the target was met.
Training stops only when all selected target metrics meet the threshold. Changing
target mode changes the input/output dimensions; start a separate output directory
so the new run cannot reuse incompatible checkpoints or validation history. Exhausting `steps`
saves the final state and reports `target_met: false` if the threshold remains
unmet. Increase `--steps` to continue the same run. The newest checkpoint by stored
step is selected across best and numbered checkpoints; optimizer and validation
history are restored. Resumption continues stochastic training but does not
replay the exact pre-interruption data-loader/RNG state.

```bash
bazelisk test //tests:test_camera_gpt //tests:test_drivegpt_training
```

The training tests include a two-rank CPU/Gloo rollout, validation, coordinated
early stopping, checkpoint writing, and resumption selection.

Future exports should save a generation-specific rink descriptor and mask with
original canvas dimensions, calibration/source identity, and any explicit affine.
These are static sidecars, not repeated CSV columns. If stitching geometry changes
during an export, each geometry interval needs its matching context before that
export can be used with this static-per-game recipe. A descriptor cannot certify
multiple different calibrations just because their output dimensions match.

When running a grid checkpoint, enable strict rink context in the runtime Aspen
YAML together with the camera model:

```yaml
aspen:
  plugins:
    ice_config:
      params:
        require_geometry_provenance: true
    camera_controller:
      params:
        controller: drivegpt
        model_path: /path/to/drivegpt_best.pt
```

Other controllers and legacy checkpoints retain their existing saved-mask behavior
by default. The grid controller reports a missing-provenance error when strict
context has not been enabled; it never silently substitutes geometry.
