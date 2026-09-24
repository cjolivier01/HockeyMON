from __future__ import annotations

import copy
import io
import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from mmdet.structures import DetDataSample, TrackDataSample
from mmengine.structures import InstanceData

from hmlib.cli.hmtrack import make_parser
from hmlib.models.detector_comparison import ComparisonReport, box_agreement
from hmlib.models.detector_selection import ROOT, configure_detector_selection, resolve_detector


class DetectorComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with open(ROOT / "hmlib/config/aspen/tracking.yaml") as stream:
            self.config = yaml.safe_load(stream)

    def test_wrapper_command_is_accepted_by_hmtrack_parser(self):
        wrapper = runpy.run_path(str(ROOT / "scripts/compare_game_detectors.py"))
        output = self.root / "report"
        output.mkdir()
        (output / "summary.txt").write_text("Test summary\n")

        def check_command(command, **kwargs):
            args = make_parser().parse_args(command[3:])
            self.assertEqual(args.game_id, "comparison-parser-test")
            self.assertTrue(args.compare_distilled)
            self.assertEqual(args.detector_model, "trained")
            self.assertEqual(args.max_frames, 2)
            self.assertTrue(kwargs["check"])

        with patch("subprocess.run", side_effect=check_command) as run:
            with patch("sys.stdout", io.StringIO()):
                self.assertEqual(
                    wrapper["main"](
                        [
                            "--game-id",
                            "comparison-parser-test",
                            "--include-distilled",
                            "--tracking-model",
                            "trained",
                            "--max-frames",
                            "2",
                            "--output-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
            run.assert_called_once()

    def test_selected_model_reaches_factory_and_gets_separate_cache(self):
        checkpoint = self.root / "new.pth"
        checkpoint.touch()
        args = make_parser().parse_args(
            [
                "--game-id",
                "sample",
                "--detector-model",
                "distilled",
                "--distilled-checkpoint",
                str(checkpoint),
            ]
        )
        previous = copy.deepcopy(self.config)
        configure_detector_selection(args, self.config)
        params = self.config["aspen"]["plugins"]["detector_factory"]["params"]
        self.assertEqual(params["checkpoint"], str(checkpoint))
        self.assertEqual(params["detector"]["backbone"]["widen_factor"], 0.5)
        self.assertEqual(params["detector"]["bbox_head"]["head_module"]["num_classes"], 1)
        self.assertIsNone(params["detector"]["init_cfg"])
        self.assertIn("detector-", params["trt"]["engine"])
        self.assertEqual(
            previous["aspen"]["inference_pipeline"], self.config["aspen"]["inference_pipeline"]
        )

    def test_default_profile_leaves_deployment_untouched(self):
        original = copy.deepcopy(self.config)
        configure_detector_selection(make_parser().parse_args([]), self.config)
        self.assertEqual(original, self.config)

    def test_comparison_configures_all_models_without_changing_tracker_dependencies(self):
        trained = self.root / "trained.pth"
        distilled = self.root / "distilled.pth"
        trained.touch()
        distilled.touch()
        args = make_parser().parse_args(
            [
                "--game-id",
                "sample",
                "--detector-model",
                "trained",
                "--trained-checkpoint",
                str(trained),
                "--distilled-checkpoint",
                str(distilled),
                "--compare-detectors",
                str(self.root / "report"),
                "--compare-distilled",
            ]
        )
        before = copy.deepcopy(self.config["aspen"]["plugins"])
        configure_detector_selection(args, self.config)
        plugins = self.config["aspen"]["plugins"]
        factory = plugins["detector_factory"]["params"]
        comparison = plugins["detector"]["params"]
        self.assertEqual(factory["checkpoint"], str(trained))
        self.assertFalse(factory["trt"]["enable"])
        self.assertFalse(factory["onnx"]["enable"])
        self.assertEqual(factory["nms_backend"], "head")
        self.assertEqual(comparison["selected"], "trained")
        self.assertEqual(set(comparison["models"]), {"deployed", "trained", "distilled"})
        self.assertEqual(comparison["models"]["distilled"]["checkpoint"], str(distilled))
        for name, plugin in plugins.items():
            self.assertEqual(plugin.get("depends"), before[name].get("depends"))

    def test_csv_replay_cannot_silently_bypass_selected_detector(self):
        args = make_parser().parse_args(
            ["--detector-model", "trained", "--input-detection-data", "old.csv"]
        )
        with self.assertRaisesRegex(ValueError, "live detections"):
            configure_detector_selection(args, self.config)

    def test_selected_checkpoint_override_needs_no_default_checkpoint(self):
        selected_checkpoint = self.root / "selected.pth"
        trained_checkpoint = self.root / "trained.pth"
        selected_checkpoint.touch()
        trained_checkpoint.touch()
        for selected in ("trained", "distilled"):
            with self.subTest(selected=selected):
                config = copy.deepcopy(self.config)
                args = make_parser().parse_args(
                    [
                        "--detector-model",
                        selected,
                        "--detector-checkpoint",
                        str(selected_checkpoint),
                        "--compare-detectors",
                        str(self.root / "report"),
                    ]
                    + (
                        ["--trained-checkpoint", str(trained_checkpoint)]
                        if selected == "distilled"
                        else []
                    )
                )
                with patch(
                    "hmlib.models.detector_selection.default_checkpoint",
                    side_effect=AssertionError("Explicit checkpoints must bypass discovery"),
                ):
                    configure_detector_selection(args, config)
                plugins = config["aspen"]["plugins"]
                self.assertEqual(
                    plugins["detector"]["params"]["models"][selected]["checkpoint"],
                    str(selected_checkpoint),
                )
                self.assertEqual(
                    plugins["detector_factory"]["params"]["checkpoint"], str(selected_checkpoint)
                )

    def test_accuracy_uses_processed_frames_and_handles_empty_predictions(self):
        labels = self.root / "labels.json"
        labels.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": i, "frame_id": i * 10, "width": 100, "height": 100} for i in (1, 2)
                    ],
                    "categories": [{"id": 1, "name": "person"}],
                    "annotations": [
                        {
                            "id": i,
                            "image_id": i,
                            "category_id": 1,
                            "bbox": [10, 10, 20, 30],
                            "area": 600,
                            "iscrowd": 0,
                        }
                        for i in (1, 2)
                    ],
                }
            )
        )
        report = ComparisonReport(
            str(self.root / "report"), {"deployed": {}, "trained": {}}, str(labels), 0.25
        )
        report.record(
            10,
            (100, 100),
            {
                "deployed": [{"bbox": [10, 10, 20, 30], "score": 0.9}],
                "trained": [],
            },
            {"deployed": 1, "trained": 2},
            (128, 128),
        )
        summary = report.finalize()
        self.assertAlmostEqual(summary["models"]["deployed"]["accuracy"]["mAP"], 1.0)
        self.assertEqual(summary["models"]["trained"]["accuracy"]["mAP"], 0.0)
        self.assertEqual(summary["labeled_frames_outside_compared_clip"], [20])
        self.assertIs(report.finalize(), summary)

    def test_agreement_does_not_double_match_boxes(self):
        box = {"bbox": [1, 2, 10, 20], "score": 0.9}
        self.assertEqual(
            box_agreement([box, box], [box], 0.25),
            {"matched": 1, "deployed_only": 1, "candidate_only": 0},
        )

    def test_comparison_preserves_selected_predictions_and_filters_nonpeople(self):
        from hmlib.aspen.plugins.detector_compare_plugin import DetectorComparePlugin
        from hmlib.utils.gpu import unwrap_tensor

        class FakeModel:
            def __init__(self, x):
                self.x = x
                self.calls = []

            def predict(self, inputs, samples):
                self.calls.append(tuple(inputs.shape))
                if inputs.dtype != torch.float32:
                    raise AssertionError("Comparison inference must use FP32")
                for sample in samples:
                    if tuple(sample.metainfo["ori_shape"]) != (100, 100):
                        raise AssertionError("Normalize cropped image shape before prediction")
                    sample.pred_instances = InstanceData(
                        bboxes=torch.tensor(
                            [[self.x, 2.0, self.x + 10, 22.0], [1.0, 1.0, 2.0, 2.0]]
                        ),
                        scores=torch.tensor([0.9, 0.95]),
                        labels=torch.tensor([0, 1]),
                    )
                return samples

        selected = FakeModel(10.0)
        alternative = FakeModel(30.0)
        with patch(
            "hmlib.aspen.plugins.detector_compare_plugin.DetectorFactoryPlugin.forward",
            return_value={"detector_model": alternative},
        ):
            plugin = DetectorComparePlugin(
                models={"deployed": {}, "trained": {}},
                selected="deployed",
                output_dir=str(self.root / "report"),
                preview_frames=1,
                sample_every=1,
            )
            for frame_id, input_height in ((7, 100), (8, 100), (9, 128), (10, 100)):
                if frame_id == 10:
                    plugin.sample_every = 10  # This frame is not compared.
                sample = DetDataSample(
                    metainfo={
                        "img_id": frame_id,
                        "ori_shape": (1, 100, 100, 3),
                        "scale_factor": (1.0, 1.0),
                    }
                )
                track = TrackDataSample(video_data_samples=[sample])
                plugin(
                    {
                        "inputs": torch.zeros(1, 3, input_height, 100, dtype=torch.float16),
                        "fp16": True,
                        "data_samples": track,
                        "detector_model": selected,
                        "device": torch.device("cpu"),
                    }
                )
                self.assertEqual(float(unwrap_tensor(track[0].pred_instances.bboxes)[0, 0]), 10.0)
                self.assertEqual(len(selected.calls), {7: 2, 8: 3, 9: 5, 10: 6}[frame_id])
                self.assertEqual(len(alternative.calls), {7: 2, 8: 3, 9: 5, 10: 5}[frame_id])
            self.assertEqual(len(plugin.report.metadata["warmups"]), 4)
            plugin.finalize()
        records = [
            json.loads(line)
            for line in (self.root / "report/predictions.jsonl").read_text().splitlines()
        ]
        self.assertEqual([r["frame_id"] for r in records], [7, 8, 9])
        record = records[0]
        self.assertEqual(record["predictions"]["trained"][0]["bbox"][0], 30.0)
        self.assertEqual(len(record["predictions"]["deployed"]), 1)
        self.assertTrue((self.root / "report/frame_00000007.jpg").is_file())

    def test_checkpoint_prefix_supports_tracker_bundles_and_raw_exports_strictly(self):
        from hmlib.aspen.plugins.detector_factory_plugin import DetectorFactoryPlugin

        checkpoint = self.root / "tracker.pth"
        checkpoint.touch()
        weights = torch.nn.Linear(2, 1).state_dict()
        for prefix in ("detector", "detector."):
            deployed = {
                "detector": {
                    "type": "TestDetector",
                    "init_cfg": {"type": "Pretrained", "checkpoint": "unused", "prefix": prefix},
                }
            }
            spec = resolve_detector("deployed", deployed, str(checkpoint))
            self.assertEqual(spec["checkpoint_prefix"], prefix)
            bundled = {"detector." + key: value for key, value in weights.items()}
            bundled["reid.ignored"] = torch.zeros(1)
            partial = dict(weights, **{"detector.weight": weights["weight"]})
            for saved, valid in (
                (bundled, True),
                (weights, True),
                (partial, False),
                ({"wrong": torch.zeros(1)}, False),
            ):
                with self.subTest(prefix=prefix, valid=valid, keys=list(saved)):
                    with (
                        patch("mmyolo.registry.MODELS.build", return_value=torch.nn.Linear(2, 1)),
                        patch(
                            "mmengine.runner.checkpoint._load_checkpoint",
                            return_value={"state_dict": saved},
                        ),
                    ):
                        factory = DetectorFactoryPlugin(
                            **spec, nms_backend="head", cuda_graph=False
                        )
                        if valid:
                            loaded = factory({"device": torch.device("cpu")})["detector_model"]
                            self.assertTrue(torch.equal(loaded.weight, weights["weight"]))
                        else:
                            with self.assertRaises(RuntimeError):
                                factory({"device": torch.device("cpu")})

    def test_batched_tracking_compares_labeled_frames_with_single_frame_timings(self):
        from hmlib.aspen.plugins.detector_compare_plugin import DetectorComparePlugin
        from hmlib.utils.gpu import unwrap_tensor

        class FakeModel:
            def __init__(self, x):
                self.x = x
                self.batch_sizes = []

            def predict(self, inputs, samples):
                self.batch_sizes.append(len(inputs))
                for sample in samples:
                    sample.pred_instances = InstanceData(
                        bboxes=torch.tensor([[self.x, 2.0, self.x + 10, 22.0]]),
                        scores=torch.tensor([0.9]),
                        labels=torch.tensor([0]),
                    )
                return samples

        labels = self.root / "labels.json"
        labels.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "frame_id": 12, "height": 100, "width": 100}],
                    "categories": [{"id": 1, "name": "person"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [10, 2, 10, 20],
                            "area": 200,
                            "iscrowd": 0,
                        }
                    ],
                }
            )
        )
        selected, alternative = FakeModel(10.0), FakeModel(30.0)
        with patch(
            "hmlib.aspen.plugins.detector_compare_plugin.DetectorFactoryPlugin.forward",
            return_value={"detector_model": alternative},
        ):
            plugin = DetectorComparePlugin(
                models={"deployed": {}, "trained": {}},
                selected="deployed",
                output_dir=str(self.root / "report"),
                annotations=str(labels),
                preview_frames=0,
                sample_every=30,
            )
            track = TrackDataSample(
                video_data_samples=[
                    DetDataSample(
                        metainfo={
                            "img_id": frame_id,
                            "ori_shape": (2, 3, 100, 100),
                            "scale_factor": (1.0, 1.0),
                        }
                    )
                    for frame_id in (11, 12)
                ]
            )
            plugin(
                {
                    "inputs": torch.zeros(2, 3, 100, 100),
                    "data_samples": track,
                    "detector_model": selected,
                    "device": torch.device("cpu"),
                }
            )
            self.assertEqual(selected.batch_sizes, [2, 1, 1])
            self.assertEqual(alternative.batch_sizes, [1, 1])
            for sample in track:
                self.assertEqual(float(unwrap_tensor(sample.pred_instances.bboxes)[0, 0]), 10.0)
            plugin.finalize()
        self.assertEqual(plugin.report.summary["frame_ids"], [12])
        self.assertAlmostEqual(plugin.report.summary["models"]["deployed"]["accuracy"]["mAP"], 1.0)

    def test_distilled_architecture_loads_raw_and_exported_checkpoint(self):
        from mmengine.config import ConfigDict
        from mmengine.registry import DefaultScope
        from mmyolo.registry import MODELS
        from mmyolo.utils import register_all_modules

        from hmlib.aspen.plugins.detector_factory_plugin import DetectorFactoryPlugin

        register_all_modules()
        checkpoint = self.root / "student.pth"
        checkpoint.touch()
        deployed = self.config["aspen"]["plugins"]["detector_factory"]["params"]
        spec = resolve_detector("distilled", deployed, str(checkpoint))
        with DefaultScope.overwrite_default_scope("mmyolo"):
            model = MODELS.build(ConfigDict(spec["detector"]))
        weights = model.state_dict()
        for prefix in ("", "student."):
            saved = {prefix + name: value for name, value in weights.items()}
            if prefix:
                saved["teacher.ignored"] = torch.zeros(1)
            torch.save({"state_dict": saved}, checkpoint)
            factory = DetectorFactoryPlugin(**spec, nms_backend="head", cuda_graph=False)
            loaded = factory({"device": torch.device("cpu")})["detector_model"]
            self.assertTrue(torch.equal(next(loaded.parameters()), next(model.parameters())))
        torch.save({"state_dict": {"wrong_key": torch.zeros(1)}}, checkpoint)
        factory = DetectorFactoryPlugin(**spec, nms_backend="head", cuda_graph=False)
        with self.assertRaises(RuntimeError):
            factory({"device": torch.device("cpu")})


if __name__ == "__main__":
    unittest.main()
