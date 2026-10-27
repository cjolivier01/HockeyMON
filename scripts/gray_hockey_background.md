# Gray-background hockey dataset export

Run from the repository root in the training Python environment:

```bash
python scripts/gray_hockey_background.py \
  --annotations /data/datasets/hockeyTrackingDataset-coco/annotations/train.json \
                /data/datasets/hockeyTrackingDataset-coco/annotations/val.json \
  --image-root /data/datasets/hockeyTrackingDataset \
  --output /data/datasets/hockeyTrackingDataset-gray \
  --device cuda:0 --batch-size 8 --workers 4
```

Use `--limit 20` and a different output directory for a preview. CPU inference
is available with `--device cpu`. GPU batch sizing depends on available VRAM;
run this separately from training, and lower the batch size if memory is tight.
The script does not launch training.

The exporter loads the current rink model once, uses actual batched `test_step`
inference, and parallelizes image decoding and PNG writes within each batch.
The default checkpoint is the locally cached `ice_rink_iter_60000.pth`; override
`--checkpoint` and `--config` together for a different model.

Pixels inside the predicted ice mask OR any player/goalie/official annotation
box retain their original values. All other pixels become RGB (128,128,128).
This binary mask avoids leaving faint spectators behind. PNG output is lossless
but will use more disk space than JPEG. Spectator categories and annotations
are removed; other annotation attributes, IDs, image dimensions and splits stay
intact. Point the training image root at the output directory and annotations
at its `annotations/{train,val}.json`; this is a three-class dataset and requires
a corresponding training class configuration, not the old four-class recipe.

People on the predicted ice, or inside a preserved box, remain visible. Mask
errors can preserve benches or erase unannotated ice; inspect a preview before
training. A missing mask stops with the offending frame named. Nothing is
written to the source. Outputs must be new directories; there is no resume or
overwrite mode. A failed export can contain partial images; `COMPLETE.json` is
written only after all requested splits finish successfully.

## Measured GPU batch sizes

RTX 4090 (24 GB), current FP32 rink model, 1280×720 frames, four I/O workers:

| Batch | Export frames/s | Peak PyTorch reserved VRAM |
| --- | --- | --- |
| 2 | 15.1 | 2.9 GiB |
| 4 | 16.4 | 4.9 GiB |
| 8 | 16.6 | 9.5 GiB |
| 16 | 16.3 | 18.9 GiB |
| 24 | OOM on second batch | — |

The initial sweep used 32 frames spread across the three validation sequences.
Batch 8 then sustained 16.2 frames/s on 96 frames spread across the training
split. Timings include JPEG reads, inference, masking, and PNG writes, but
exclude model startup. Short tests benefit from filesystem caching; the full
82,079-frame export is approximately 85 minutes at this rate, not a guaranteed
completion time. Batch 8 is recommended for this GPU; the portable script
default remains 2. No mixed precision or allocator overrides were used.

## Koshkina team-ID: person / ref

```bash
./grayit.sh --team-id
# Small preview, two frames from each split:
./grayit.sh --team-id --limit 2 --output /tmp/team-id-gray-preview
```

The wrapper reads `/data/datasets/koshkina/team-id/York` and writes a new
`/data/datasets/koshkina/team-id-gray` directory. It uses batch size 1 for the
large panoramic frames. The original `./grayit.sh` invocation still exports
hockeyTrackingDataset with its existing settings.

Direct invocation, including an alternative source/output location:

```bash
python scripts/gray_hockey_background.py \
  --team-id-root /data/datasets/koshkina/team-id/York \
  --output /data/datasets/koshkina/team-id-gray --batch-size 1
```

`scripts/team_id_annotations.py` joins the crop names in `gt.txt` to the original
XYXY coordinates in `bbox_data.txt`. Source classes 1 and 3 become COCO category
**1: person**; source class 2 becomes **2: ref**. Goalies retain their team label
and therefore become persons. There are no team-ID targets or goalie category
in this output. These are two detector classes (normally contiguous model
indices 0 and 1), so use a two-class training configuration, not the current
three-class hockey recipe unchanged.

Only frames where every supplied box has a role label, and every role label has
a box, are included. Many frames have only partial role labeling. `video*`
sequences provide `players_bbox_data.txt`, which omits referees; those sequences
are excluded rather than inventing geometry. The export records all exclusions
in `team-id-source-report.json`. Completeness refers to the supplied boxes;
it does **not** certify that their original detector found every on-ice person.
Gray masking also cannot remove unannotated people inside the predicted ice.
Inspect the preview before training.

For the supplied data this yields **144 frames / 1,754 boxes**:

| Split | Frames | Person boxes | Ref boxes |
| --- | ---: | ---: | ---: |
| Train | 76 | 720 | 240 |
| Validation | 68 | 646 | 148 |

Games 7, 8 and 9 are held out as validation, consistent with the York tracking
split; all other usable games form training. `--team-id-val-games` overrides
this list. No frame from a named game appears in both splits. This is a small
additional dataset, not a replacement for CrowdHuman or the larger hockey set.

The output contains lossless PNGs under `images/<game>/all_frames/`, COCO files
at `annotations/train.json` and `annotations/val.json`, export settings, the
source report, and `COMPLETE.json` after a successful run. The source files are
never modified. Ice and labeled boxes are preserved with the same graying rule
as the original exporter.
