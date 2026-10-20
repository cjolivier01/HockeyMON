#!/usr/bin/env python3
"""Create one MMPose/PyTorch native-inference fixture per validation action class."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from export_hockey_models import PoseNetwork, arrays
from hockey_data import configure_native_crops, digest
from mmengine.config import Config
from mmpose.registry import DATASETS, MODELS
from mmpose.utils import register_all_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_native_crops()
    torch.set_num_threads(4)
    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).eval()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    )
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    seen = set()
    with torch.inference_mode():
        for index in range(len(dataset)):
            raw = dataset.get_data_info(index)
            label = Path(raw["img_path"]).name.split("_")[0].removeprefix("ImageSequences")
            if label in seen:
                continue
            seen.add(label)
            destination = args.output / label
            destination.mkdir()
            sample = dataset[index]
            batch = model.data_preprocessor(
                {key: [value] for key, value in sample.items()}, training=False
            )
            inputs = batch["inputs"]
            outputs = arrays(PoseNetwork(model)(inputs))
            prediction = model.predict(inputs, batch["data_samples"])[0].pred_instances
            pose = np.concatenate(
                (prediction.keypoints, np.clip(prediction.keypoint_scores, 0, 1)[..., None]),
                axis=-1,
            )
            inputs.numpy().tofile(destination / "input.f32")
            pose.astype(np.float32).tofile(destination / "reference_pose.f32")
            for number, output in enumerate(outputs):
                output.tofile(destination / f"reference_{number}.f32")
            image = cv2.imread(raw["img_path"])
            if image is None or not cv2.imwrite(str(destination / "scene.png"), image):
                raise ValueError(f"Cannot retain fixture image: {raw['img_path']}")
            x1, y1, x2, y2 = raw["bbox"][0].tolist()
            fixture = {
                "image": "scene.png",
                "roi_xywh": [x1, y1, x2 - x1, y2 - y1],
                "metadata_size": [image.shape[1], image.shape[0]],
                "checkpoint_sha256": digest(args.checkpoint),
                "source_image_sha256": digest(Path(raw["img_path"])),
            }
            (destination / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    if len(seen) != 4:
        raise ValueError(f"Expected four validation classes, found {seen}")


if __name__ == "__main__":
    main()
