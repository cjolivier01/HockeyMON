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

            def predict(self, inputs, samples):
                if inputs.dtype != torch.float32:
                    raise AssertionError("Comparison inference must use FP32")
                for sample in samples:
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
            sample = DetDataSample(
                metainfo={"img_id": 7, "ori_shape": (100, 100), "scale_factor": (1.0, 1.0)}
            )
            track = TrackDataSample(video_data_samples=[sample])
            plugin(
                {
                    "inputs": torch.zeros(1, 3, 100, 100, dtype=torch.float16),
                    "fp16": True,
                    "data_samples": track,
                    "detector_model": selected,
                    "device": torch.device("cpu"),
                }
            )
            self.assertEqual(float(unwrap_tensor(track[0].pred_instances.bboxes)[0, 0]), 10.0)
            plugin.finalize()
        record = json.loads((self.root / "report/predictions.jsonl").read_text())
        self.assertEqual(record["predictions"]["trained"][0]["bbox"][0], 30.0)
        self.assertEqual(len(record["predictions"]["deployed"]), 1)
        self.assertTrue((self.root / "report/frame_00000007.jpg").is_file())

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
