# DriveGPT training

DriveGPT learns slow and fast camera boxes from tracked object boxes and its own
previous camera prediction. The official configuration uses `players_prev_y` and
disables pose and rink-mask features. Runtime consumes live tracking tensors;
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
window length/start stride, run weighting, loader/cache settings, losses,
scheduled sampling, validation, checkpoints, and distributed settings. YAML values
are defaults; explicit CLI options override them. Unknown options fail. Dataset
configuration paths in a training YAML are relative to that YAML; dataset roots
are relative to their dataset YAML. `--dataset-root` overrides the root on a host.
Output/checkpoint paths follow normal CLI working-directory semantics.

Games are sampled uniformly within each worker's shard. Unequal shard sizes
slightly change overall game probabilities. `run_sampling: windows` weights contiguous runs by
their number of eligible windows; `uniform` gives each run equal probability.
`sample_stride` controls eligible start positions and does not skip frames within
a sequence. Numeric frame gaps and camera-policy changes split sequences.

Set `drivegpt_source_checkpoint` to the downloaded UniAD checkpoint, or omit it to
download `ckpts/uniad_base_e2e.pth` from
`OpenDriveLab/UniAD2.0_R101_nuScenes` via Hugging Face. The official configuration
requires initialization to succeed. `drivegpt_init: none` trains from scratch.

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

The target is **mean IoU >= 0.97 for both slow and fast camera boxes** on held-out
games. It is box overlap, not classification accuracy. The default evaluates
fixed-seed sampled 256-frame autoregressive rollouts, with a sliding 32-frame
context matching runtime. Each rollout starts from the recorded previous camera
state (or a full-frame state at a run boundary); subsequent feedback comes from
the model with aspect fitting and normalized box clamping. Runtime additionally
fits boxes to the actual frame/arena bounds; the CSV metric does not reproduce
those per-game bounds. This evaluates about 8.5 seconds at
30 FPS; it is not a claim about uninterrupted full-game rollout accuracy.

`*.metrics.jsonl` records the dataset/split identity, resolved arguments,
validation protocol, distributed metrics, and whether the target was met.
Training stops only when both metrics meet the threshold. Exhausting `steps`
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
