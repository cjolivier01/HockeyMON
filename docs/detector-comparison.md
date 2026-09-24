# Compare game detectors and select the tracking model

From the HM repository (the parent of `openmm`):

```bash
python scripts/compare_game_detectors.py --game-id GAME_ID
python scripts/compare_game_detectors.py --game-id GAME_ID --include-distilled
```

The tool processes the first 300 frames through the game's normal hmtrack
decoding, stitching, crop, resize, and color pipeline. It compares every 30th
frame and saves up to 20 stacked JPEG previews. Change `--max-frames`,
`--sample-every`, `--preview-frames`, and `--start-time` to choose the clip.
Additional hmtrack arguments go after `--`. Use `--dry-run` to print the command.
The output defaults to `output_workdirs/GAME_ID/detector_comparisons/TIMESTAMP/`.
Use a new `--output-dir` for each run; existing prediction files are not overwritten.

`summary.txt` and `summary.json` report person counts, inference time and box
agreement with the deployed detector. `predictions.jsonl` records low-threshold
person boxes in COCO xywh format for each sampled frame. The review threshold
defaults to 0.25 and affects counts, agreement and previews, not mAP input.
Only class 0 (person) is compared. Raw detections are compared before rink masks
and tracking. All candidates use identical HM preprocessing, FP32 inference and PyTorch head NMS;
this compares model weights at HM's configured resolution, not their differing
training resolutions or production TensorRT latency. Timings exclude model loading
and one untimed warmup per model/input shape; warmup timings are recorded separately
in the JSON report. These are diagnostic single-frame timings, not a steady-state
throughput benchmark. With batch size 1, the selected model's actual tracking
prediction is reused. Larger tracking batches require additional single-frame
comparison calls to keep timing workloads equivalent. Comparison adds compute overhead.

## Accuracy with labeled game frames

Game video and deployed predictions alone cannot establish accuracy. Supply
independent COCO-format person labels with `--annotations labels.json`, or place
`detector_ground_truth.json` in the game directory. Each COCO `images` record must
include an explicit `frame_id` matching hmtrack's `img_id` (visible in the saved
previews/JSONL). Do not assume a COCO image ID is a video frame number.

Boxes and image dimensions must refer to the stitched/cropped image **before**
HM resize/padding, not the camera-follow output video. For example:

```json
{
  "images": [{"id": 1, "frame_id": 31, "width": 3840, "height": 1440}],
  "categories": [{"id": 1, "name": "person"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "bbox": [100, 200, 40, 100], "area": 4000, "iscrowd": 0}]
}
```

With labels, every labeled frame encountered in the selected clip is compared
regardless of `--sample-every`. The report computes COCO mAP@0.50:0.95, AP50,
AP75 and AR100 on that same subset for every model, using COCO maxDets=100.
It lists labeled frames outside the processed clip. Zero-detection frames count
toward accuracy. Label all people, including referees, or use the appropriate
COCO crowd annotations. Counts and model agreement without labels are diagnostics,
not precision/recall or accuracy.

## Select the detector that feeds normal tracking

```bash
hmtrack --game-id GAME_ID --detector-model deployed
hmtrack --game-id GAME_ID --detector-model trained
hmtrack --game-id GAME_ID --detector-model distilled
hmtrack --game-id GAME_ID --detector-model trained --trained-checkpoint openmm/work_dirs/RUN/best.pth
hmtrack --game-id GAME_ID --detector-model distilled --distilled-checkpoint openmm/work_dirs/RUN/student.pth
```

With no selection flag, existing deployment configuration is unchanged.
`deployed` reads the detector from the effective Aspen configuration (including
custom `--config` files). `trained` uses the one-class YOLOv8-M architecture and
the saved `best_coco_bbox_mAP_epoch_*.pth` in the completed M training directory.
`distilled` uses the one-class S architecture and the highest-scoring `topk_*`
checkpoint in `openmm/work_dirs/hm_yolov8_s_distilled_crowdhuman_300e`.
It accepts both full distillation checkpoints and exported students. Explicit
paths override discovery; missing or ambiguous checkpoints fail clearly.
`--detector-checkpoint` (alias `--detector`) overrides the selected profile's weights.
Checkpoint shapes must match the profile and are checked strictly.
Deployed configs with a checkpoint prefix (for example, `detector.` in a complete
tracker checkpoint) retain that prefix when weights are overridden; raw exported
detector checkpoints are also accepted.

During a comparison, `--tracking-model trained` or `--tracking-model distilled`
chooses the predictions passed to normal rink filtering/tracking. The default is
`deployed`. Comparison results never replace the selected model's detections.
Replaying detection/tracking CSVs is incompatible with model selection.
Compiled detector caches are separated by architecture/checkpoint identity.

The equivalent integrated command is:

```bash
hmtrack --game-id GAME_ID --detector-model trained \
  --compare-detectors output_workdirs/GAME_ID/my_comparison --compare-distilled \
  --max-frames 300 --detector-sample-every 30
```

Comparison executes normal hmtrack tracking and output stages, so normal tracking
artifacts may also be produced. The wrapper disables audio, final video saving
and camera UI for a bounded comparison run. Comparison uses the sequential Aspen
pipeline rather than threaded/CUDA-graph execution.
