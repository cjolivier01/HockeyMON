#!/usr/bin/env python3
"""Export a frozen hockey checkpoint and validate every validation input in ONNX.

The resulting export.json is accepted by HStream's prepare_player_models.py build.
Only run this offline with trusted, locally trained checkpoints.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from hockey_data import LABELS, METAINFO, configure_native_crops, digest, software_provenance
from hockey_pose_contract import PREPROCESSING, portable_pose_config
from torch import nn


class PoseNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, image):
        return self.head(self.backbone(image))


def load_model(args, state):
    if args.feature == "action":
        from hockey_action import HockeyAction

        model = HockeyAction().eval()
        model.load_state_dict(state, strict=True)
        with np.load(args.predictions, allow_pickle=False) as data:
            inputs = np.stack([data[key] for key in sorted(data.files)])[:, None]
        config = {"model": "hockey_action.HockeyAction", "labels": LABELS, "topology": METAINFO}
        return model, inputs, config

    from mmengine.config import Config
    from mmpose.registry import DATASETS, MODELS
    from mmpose.utils import register_all_modules

    configure_native_crops()
    register_all_modules()
    cfg = Config.fromfile(args.config)
    inference = Config(portable_pose_config(cfg.to_dict(), METAINFO))
    model = MODELS.build(cfg.model).eval()
    model.load_state_dict(state, strict=True)
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    images = []
    with torch.inference_mode():
        for start in range(0, len(dataset), 8):
            rows = [dataset[i] for i in range(start, min(start + 8, len(dataset)))]
            batch = {key: [row[key] for row in rows] for key in ("inputs", "data_samples")}
            images.append(model.data_preprocessor(batch, training=False)["inputs"].numpy())
    inference.dump(str(args.output / "inference.py"))
    return PoseNetwork(model).eval(), np.concatenate(images), cfg.model.to_dict()


def arrays(value):
    return [tensor.detach().numpy() for tensor in (value if isinstance(value, tuple) else (value,))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", choices=("pose", "action"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, help="Pose training config with validation data paths"
    )
    parser.add_argument(
        "--predictions", type=Path, help="Action validation poses from frozen pose model"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.feature == "pose" and args.config is None) or (
        args.feature == "action" and args.predictions is None
    ):
        parser.error("Pose requires --config; action requires --predictions")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    model, inputs, config = load_model(args, state)
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    shape = [3, 256, 192] if args.feature == "pose" else [1, 3, 18, 3]
    if (
        list(inputs.shape[1:]) != shape
        or not 8 <= len(inputs) <= 256
        or not np.isfinite(inputs).all()
    ):
        raise ValueError(f"Invalid validation tensors: {inputs.shape}")
    source_path = args.output / "checkpoint.pth"
    torch.save(
        {"state_dict": state, "meta": {"dataset_meta": METAINFO, "labels": LABELS}}, source_path
    )
    config_path = args.output / "architecture.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    source_sha = digest(source_path)
    model_id = (
        "rtmpose-m-harpet18-256x192" if args.feature == "pose" else "stgcnpp-harpet18-actions4"
    )
    input_name = "image" if args.feature == "pose" else "skeleton"
    outputs = (
        [("simcc_x", [18, 384]), ("simcc_y", [18, 512])]
        if args.feature == "pose"
        else [("logits", [4])]
    )
    output_names = [name for name, _ in outputs]
    graph_path = args.output / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.from_numpy(inputs[:2]),
            str(graph_path),
            input_names=[input_name],
            output_names=output_names,
            dynamic_axes={name: {0: "batch"} for name in (input_name, *output_names)},
            opset_version=17,
            dynamo=False,
            external_data=False,
        )
    onnx.checker.check_model(onnx.load(graph_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(graph_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [value.name for value in session.get_inputs()] != [input_name] or [
        value.name for value in session.get_outputs()
    ] != output_names:
        raise ValueError("Exported tensor names differ from the deployment contract")
    cases = []
    # Exercise dynamic profile boundaries and cover all validation rows.
    indices = [
        [0],
        [0, 1],
        *[list(range(start, min(start + 8, len(inputs)))) for start in range(0, len(inputs), 8)],
    ]
    with torch.inference_mode():
        for number, rows in enumerate(indices):
            value = inputs[rows]
            expected = arrays(model(torch.from_numpy(value)))
            actual = session.run(None, {input_name: value})
            filename = f"input_{number}.f32"
            value.tofile(args.output / filename)
            case = {
                "batch": len(rows),
                "input": filename,
                "input_indices": rows,
                "input_sha256": digest(args.output / filename),
                "outputs": [],
            }
            for (name, dimensions), reference, result in zip(
                outputs, expected, actual, strict=True
            ):
                if (
                    list(result.shape) != [len(rows), *dimensions]
                    or not np.isfinite(result).all()
                    or not np.allclose(result, reference, atol=1e-4, rtol=5e-4)
                ):
                    raise ValueError(
                        f"ONNX parity failed: {name}, batch {len(rows)}; max error {np.max(np.abs(result-reference))}"
                    )
                filename = f"reference_{number}_{name}.f32"
                reference.tofile(args.output / filename)
                case["outputs"].append(
                    {
                        "name": name,
                        "shape": list(reference.shape),
                        "file": filename,
                        "sha256": digest(args.output / filename),
                        "onnx_max_abs_error": float(np.max(np.abs(result - reference))),
                    }
                )
            cases.append(case)
    preprocessing = (
        PREPROCESSING
        if args.feature == "pose"
        else {
            "id": "stgcnpp-harpet18-causal-v1",
            "coordinate_normalization": "pixels_model_root_body_height",
            "samples": 3,
            "sample_period_ns": 166666667,
            "max_gap_ns": 250000000,
            "second_person": "absent",
        }
    )
    manifest = {
        "schema_version": 1,
        "feature": args.feature,
        "model_id": model_id,
        "model_family": "rtmpose" if args.feature == "pose" else "stgcnpp",
        "topology_id": "harpet18",
        "max_batch": 8,
        "inputs": [{"name": input_name, "dtype": "float32", "shape": [-1, *shape]}],
        "outputs": [
            {"name": name, "dtype": "float32", "shape": [-1, *dimensions]}
            for name, dimensions in outputs
        ],
        "onnx": {"file": "model.onnx", "opset": 17, "sha256": digest(graph_path)},
        "source": {
            "url": f"https://github.com/cjolivier01/HockeyMON/releases/download/v0.0.1/{model_id}-{source_sha[:16]}.pth",
            "sha256": source_sha,
            "config_sha256": digest(config_path),
            "license": "Apache-2.0 (OpenMMLab code); VIP-HARPET training data, retain dataset and pretrained-weight attribution; see hockey-training.md",
        },
        "preprocessing": preprocessing,
        "validation": {
            "input_count": len(inputs),
            "cases": cases,
            "description": "All deduplicated HARPET validation inputs, annotated body-only boxes for pose and frozen pose estimates for action",
            "onnx_atol": 1e-4,
            "onnx_rtol": 5e-4,
        },
    }
    if args.feature == "action":
        manifest["labels"] = LABELS
    provenance = {
        "training_checkpoint_sha256": digest(args.checkpoint),
        "software": software_provenance(),
        "validation_predictions_sha256": digest(args.predictions) if args.predictions else None,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"model": model_id, "onnx_sha256": digest(graph_path), "validated_cases": len(cases)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
