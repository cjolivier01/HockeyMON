# Hockey pose and action training

The offline scripts in `scripts/*hockey*.py` fine-tune RTMPose-M and STGCN++ on
[VIP-HARPET](https://vip.uwaterloo.ca/vip-harpet-dataset/). Native playback lives in
[HockeyMONStream](https://github.com/cjolivier01/HockeyMONStream): select the
`rtmpose-m-harpet18-256x192` pose and `stgcnpp-harpet18-actions4` action models together.
The generic COCO/NTU60 selections remain available. This training adds no Python
dependency to native playback.

## Data and joint contract

The official [ZIP](https://drive.google.com/file/d/1gfA0iPC8ESXZdCnsSO0aFys_j-3UaaTF/view)
has SHA256 `69949b1d65400fe0b794316f8b1c46ce14ac07307cc0c0e963f0b1d0c46a3e42`.
Keep it outside Git, alongside extracted `annot_{train,valid,test}.h5` and
`images_{train,valid,test}/` directories. `hockey_data.py` checks dimensions,
sequence identity, frame order and image hashes before conversion.

HARPET is **MPII16 plus two stick endpoints**, not COCO17 plus one point:

| Index | Landmark | Index | Landmark |
|---|---|---|---|
| 0 | right ankle | 9 | head top |
| 1 | right knee | 10 | right wrist |
| 2 | right hip | 11 | right elbow |
| 3 | left hip | 12 | right shoulder |
| 4 | left knee | 13 | left shoulder |
| 5 | left ankle | 14 | left elbow |
| 6 | pelvis | 15 | left wrist |
| 7 | thorax | 16 | stick top |
| 8 | neck | 17 | stick tip |

The old HARPE configuration's endpoint naming/skeleton and the old HARPET
OpenPose graph do not describe these annotations correctly. These scripts use
one explicit topology, including the correct left/right permutation. Existing
18-joint checkpoints with undocumented split history were not used to initialize
the new training.

The official split contains 297/64/63 train/validation/test triplets. Identical
images cross its split boundaries. We keep test, then validation, and discard
whole lower-priority triplets in duplicate-connected components. This removes
training `Backward_79`, `Backward_81`, `Forward_51`, `Forward_80`, `Shooting_102`
and validation `Backward_56`. Training `Passing_47` has irregular frame indices
21,25,26 and is also excluded. The resulting split is **291/63/63 triplets**
(873/189/189 images). The audit JSON records every retained image hash.

This removes exact-image leakage; it does not establish player/game/video-disjoint
splits. Same-split overlapping images remain. The small official test set and
broadcast-to-stitched-camera domain change limit conclusions about live games.

Pose boxes use only the first 16 body landmarks. The stick cannot enlarge its own
evaluation crop. Padding 3.5 includes 99.3% of training stick tips, versus about
17% with padding 1.25. HStream applies the same padding to tracker boxes, with
192×256 RGB input, ImageNet mean/std and raw SimCC outputs. Reported pose results
use annotated body boxes; they are not detector-to-action end-to-end scores.

## Recipes and selection

Pose starts from the official COCO/AIC RTMPose-M checkpoint
`5e55be2a03f6e5dcd14d088afc4ae5afe94a4f9de93c22e5deb725ad0eee899d`.
Shared body filters are remapped; pelvis/thorax/neck/head are initialized from
related COCO filters and both stick filters are new. AdamW, AMP, batch 32,
learning rate 0.0003, 200 epochs, warmup/cosine decay and early stopping are used.
Selection maximizes validation PCKh@0.5, normalized by neck-to-head-top distance;
body and both stick endpoints are reported separately. No inference flip test.

Action starts from the official 2D COCO NTU60 STGCN++ checkpoint. Compatible
backbone weights transfer; adjacency, shape-dependent normalization and the four
class head are fresh. The HARPET graph connects the stick top to both wrists.
Inputs are three chronological skeletons at 1/6-second intervals, one player,
18 joints and `(x,y,confidence)`. The model itself masks confidence below 0.3,
centers on pelvis and divides by visible body height per frame. ONNX contains
this normalization. Training uses ground-truth skeletons with mirrored anatomy,
coordinate noise, confidence perturbation and joint dropout. Validation uses
estimated poses for deployment-oriented selection. Classes are, in order:
backward skating, forward skating, passing, shooting.

Action uses AdamW at 0.001, batch 32, 250-epoch cosine decay, weighted smoothed
cross entropy and 60-epoch patience. Selection maximizes validation balanced
accuracy, then minimizes validation cross entropy. Class predictions are closed
set: the model has no learned idle, goalie, collision or other-action class.
Native confidence/gap gates can abstain but do not make this an event detector.

## Reproduce

The selected checkpoints and portable graphs are available as release assets:

| Model | PyTorch checkpoint | ONNX |
|---|---|---|
| RTMPose-M HARPET18 | [checkpoint](https://github.com/cjolivier01/HockeyMON/releases/download/v0.0.1/rtmpose-m-harpet18-256x192-9edcf24e57fc88ee.pth) | [graph](https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/rtmpose-m-harpet18-256x192-de819b8bdf6f0c01.onnx) |
| STGCN++ four actions | [checkpoint](https://github.com/cjolivier01/HockeyMON/releases/download/v0.0.1/stgcnpp-harpet18-actions4-218bdaab24a27111.pth) | [graph](https://github.com/cjolivier01/HockeyMONStream/releases/download/pretrained-assets-v1/stgcnpp-harpet18-actions4-833d3ec938ca4018.onnx) |

The [training record archive](https://github.com/cjolivier01/HockeyMON/releases/download/v0.0.1/hockey-harpet18-training-record-33a11f97542bceee.tar.gz) includes scripts, configurations, package provenance, audit hashes, metrics and native parity reports.

For MMPose `init_model`/`inference_topdown`, use the
[portable inference config](https://github.com/cjolivier01/HockeyMON/releases/download/v0.0.1/rtmpose-m-harpet18-inference-d39ac0bd2b77f3b9.py)
(SHA256 `d39ac0bd2b77f3b9de32ccd29a5a3b6e1e9906e0d222c25023b7f48aef400210`).
It includes the image/box pipeline and supersedes the initial checkpoint-named
`.py`, which omitted that pipeline. Checkpoint and ONNX bytes are unchanged.
All four real-image fixtures reproduce the saved PyTorch joint coordinates
through this public MMPose API. The immutable training archive retains the
original export code; the current exporter adds pipeline validation and this
portable config fix.

Final results use the frozen weights and native-matched OpenCV preprocessing.
[The JSON record](hockey-training-results.json) contains confusion matrices,
checkpoint hashes and confidence-gate counts.

| Measure | Validation | Held-out test |
|---|---:|---:|
| Pose PCKh@0.5, all 18 points | 89.77% | 89.30% |
| Body PCKh | 91.30% | 90.97% |
| Stick top PCKh | 75.13% | 73.54% |
| Stick tip PCKh | 79.89% | 78.31% |
| Action accuracy on estimated poses | 84.13% | 76.19% (48/63) |
| Action balanced accuracy on estimated poses | 84.07% | 77.37% |
| Action accuracy on annotated poses | 85.71% | 73.02% |

On test, the native input/confidence gates accept 59/63 clips, of which 47 are
correct (79.66% accepted accuracy). These clip-level gates do not model tracking
losses, scheduling capacity, interpolation or runtime label hysteresis.

Three pose recipes were compared: COCO initialization with seed 21 (89.51%
validation PCKh), low-rate refinement (89.4%, no improvement), and seed 7
(best training validation checkpoint at epoch 146, selected). Five action recipes
compared a GT-selected baseline, estimated-pose selection with seeds 42/7 and
noise 0.04/0.025, and the latter two repeated using the selected pose. The earlier
pose/action pairing reached 87.30% validation action accuracy; with the pose
checkpoint selected for PCKh, the best action recipe reached 84.13%. We selected
the pose first, then action balanced accuracy on that fixed pose, and did not
choose between pipelines using test performance. The final action is seed 42,
noise 0.04, epoch 67. Training stopped after validation plateaus.

Training/selection originally ran with OpenCV 5.0.0. Its IPP HAL changes border
interpolation compared with HStream. Native parity caught this; export and final
evaluation use `opencv-python-headless==4.12.0.88` with IPP disabled. Both ONNX
graphs and both checkpoints remained byte-identical. Evaluation was repeated
with those already frozen weights to measure the deployed interpolation path;
no model was retuned after seeing test results. New training runs enforce the
same pinned crop implementation.

Initialize `openmm` and its pinned MMPose/MMACTION submodules. Use an isolated
environment with the repository's CUDA PyTorch/OpenMMLab build; the recorded run
used Python 3.14, PyTorch 2.11.0a0, an RTX 4090, `timm==0.9.16`,
`transformers==4.49.0`, `tokenizers==0.21.4` and `huggingface-hub==0.36.2`.
For native-matched crops install `opencv-python-headless==4.12.0.88` in this
isolated environment (the scripts check the version and disable IPP).
MMACTION's legacy imports are incompatible with Transformers 5.x. The release
provenance records actual package origins, versions and available Git revisions.
Seeds constrain variation; cuDNN/GPU training is not promised to be bit-identical.

With `hockey_data` pointing at the extracted ZIP and `hockey_runs` at an external
output directory, run from the HockeyMON root:

```bash
python scripts/train_hockey_pose.py --data-root "$hockey_data" \
  --work-dir "$hockey_runs/pose" --checkpoint "$coco_checkpoint" --seed 7
python scripts/evaluate_hockey_pose.py --data-root "$hockey_data" \
  --config "$hockey_runs/pose/config.py" --checkpoint "$selected_pose" \
  --split valid --output "$hockey_runs/pose-valid"
python scripts/train_hockey_action.py --data-root "$hockey_data" \
  --work-dir "$hockey_runs/action" --checkpoint "$ntu_checkpoint" --seed 42 --noise 0.04 \
  --pose-predictions "$hockey_runs/pose-valid/predictions.npz"
python scripts/evaluate_hockey_action.py --data-root "$hockey_data" \
  --checkpoint "$selected_action" --split valid \
  --predictions "$hockey_runs/pose-valid/predictions.npz" \
  --output "$hockey_runs/action-valid.json"
```

Choose checkpoints from validation only. Then repeat both evaluations with
`--split test` and fresh output paths, without retuning. Output directories are
immutable to prevent an experiment silently overwriting another one.

`export_hockey_models.py --feature pose --config ... --checkpoint ... --output ...`
creates the portable inference config, optimizer-free checkpoint and ONNX.
It rejects preprocessing settings outside the fixed native profile, including
non-3.5 crop padding, RGB normalization changes, flip testing and DARK decoding.
For action, use `--feature action --predictions ... --checkpoint ... --output ...`.
Every validation row is checked against PyTorch, including dynamic batches 1,2,8.
The exporter emits HStream-compatible `export.json` and binary parity fixtures.
Use HStream's `scripts/prepare_player_models.py build --export-dir ... --builder
bazel-bin/src/apps/player-model-builder/player-model-builder --output ...` to
build and verify FP16 TensorRT locally. Engines are GPU/SDK-specific and are not
release assets. `prepare_hockey_pose_fixture.py` creates real-image reference
fixtures for native crop, model and decoder checks, one per validation class.
`verify_hockey_runtime.py --builder ... --bundle ... --export-dir ... --report ...`
replays all validation references through an existing native engine without
rebuilding it. Four real-image fixtures match the GPU crop exactly; FP16 decoded
positions differ from FP32 by at most about half an input pixel on these fixtures.

Large data, weights, crops and exports stay outside Git. Release assets use
content-addressed names; the model catalog pins full SHA256 values. Keep the
training configuration, dataset audit, selection/evaluation results and software
provenance beside the checkpoints. The MMPose inference config expects the
18-joint head; consumers must also apply the HARPET crop and topology.

## Additional data search (October 2026)

| Source | Relevance and decision |
|---|---|
| [VIP-HARPE](https://vip.uwaterloo.ca/vip-harpe-dataset/) | Its current download resolves to the same HARPET ZIP. Do not count it as extra data. |
| [FineHockeySkate](https://mkoshkina.github.io/projects/) ([paper](https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Koshkina_Transition-Constrained_Sequence_Modeling_for_Fine-Grained_Skating_Action_Recognition_CVPRW_2026_paper.pdf)) | Promising: 11 skating classes, 27,637 action clips, game-disjoint evaluation. Project page offers paper/demo, with no verified annotation download in the checked author repositories. Not included. |
| [CP-Hockey](https://ecp.ep.liu.se/index.php/linhac/article/view/1206) | Six-class NHL action recognition with stick-aware skeletons. Closely related, but no verified public data download found. Not included. |
| [Tempuckey](https://github.com/R2D2oid/Tempuckey-Dataset) | 690 broadcast clips with faceoff/tripping temporal annotations. Useful for a future scene-event model, not these four per-player labels or pose supervision. |
| [HockeyAI](https://github.com/simulamet-host/HockeyAI-MMSys2025) | Object detection labels; possible detector improvement, no direct body-pose/action supervision. |
| [HockeyOrient](https://datasets.simula.no/hockeyorient/) | Player orientation labels, a possible auxiliary task. Not a replacement for action or stick annotations. |
| [HockeyRink](https://github.com/simulamet-host/HockeyRink-MMSys2025) | Rink landmarks, not player joints. Excluded from pose training. |

No additional corpus was mixed into this experiment. None of the verified
downloads supplied compatible extra per-player pose/four-action supervision.

## Attribution

Dataset: H. Cai et al., [Temporal Hockey Action Recognition via Pose and Optical
Flows](https://openaccess.thecvf.com/content_CVPRW_2019/papers/CVSports/Cai_Temporal_Hockey_Action_Recognition_via_Pose_and_Optical_Flows_CVPRW_2019_paper.pdf),
CVPR Workshops 2019, University of Waterloo VIP Lab. OpenMMLab's MMPose and
MMACTION2 code is Apache-2.0; retain the upstream COCO/AIC and NTU60 checkpoint
provenance too. The dataset landing page does not state a separate permissive
data license; code licensing does not relicense the broadcast imagery. Images
and annotations are not redistributed in the model releases.
