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
