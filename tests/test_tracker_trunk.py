import unittest

import torch

from hmlib.aspen.plugins.tracker_plugin import TrackerPlugin


class TrackerPluginConfigTest(unittest.TestCase):
    def test_default_tracker_class_instantiates_cpu_tracker(self):
        trunk = TrackerPlugin(enabled=True)
        trunk._ensure_tracker(image_size=torch.Size([1, 3, 720, 1280]))
        self.assertIsNotNone(trunk._hm_tracker)
        self.assertEqual(trunk.tracker_class, "hockeymon.core.HmTracker")
        self.assertEqual(trunk._hm_tracker.__class__.__name__, "HmTracker")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_tracker_class_instantiates_cuda_tracker(self):
        trunk = TrackerPlugin(
            enabled=True,
            tracker_class="hockeymon.core.HmByteTrackerCuda",
            tracker_kwargs={"device": "cuda:0"},
        )
        trunk._ensure_tracker(image_size=torch.Size([1, 3, 720, 1280]))
        self.assertIsNotNone(trunk._hm_tracker)
        self.assertEqual(trunk.tracker_class, "hockeymon.core.HmByteTrackerCuda")
        self.assertEqual(trunk._hm_tracker.__class__.__name__, "HmByteTrackerCuda")

    def test_prepare_tracker_inputs_with_static_limits(self):
        trunk = TrackerPlugin(enabled=True)
        trunk._static_tracker_max_detections = 4
        trunk._static_tracker_max_tracks = 8
        bboxes = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        labels = torch.arange(5, dtype=torch.long)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.8], dtype=torch.float32)
        payload = trunk._prepare_tracker_inputs(3, bboxes, labels, scores)
        self.assertIn("num_detections", payload)
        self.assertEqual(int(payload["num_detections"].item()), 4)
        self.assertEqual(payload["bboxes"].shape, (4, 4))
        self.assertEqual(payload["labels"].shape[0], 4)
        self.assertTrue(torch.all(payload["bboxes"][0] == bboxes[1]))
        self.assertTrue(torch.all(payload["bboxes"][3] == bboxes[4]))

    def test_prepare_tracker_inputs_trims_padded_non_static_detections(self):
        trunk = TrackerPlugin(enabled=True)
        bboxes = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        labels = torch.tensor([0, 0, 9876543210, 0, 0], dtype=torch.long)
        scores = torch.tensor([0.9, 0.8, -32768.0, -32768.0, -32768.0])
        payload = trunk._prepare_tracker_inputs(
            3,
            bboxes,
            labels,
            scores,
            num_detections=torch.tensor([2], dtype=torch.long),
        )

        self.assertNotIn("num_detections", payload)
        self.assertEqual(payload["bboxes"].shape, (2, 4))
        self.assertTrue(torch.equal(payload["labels"], torch.tensor([0, 0])))
        self.assertTrue(torch.equal(payload["scores"], torch.tensor([0.9, 0.8])))
        self.assertNotEqual(payload["labels"].data_ptr(), labels.data_ptr())

    def test_prepare_tracker_inputs_sanitizes_invalid_cpp_labels(self):
        trunk = TrackerPlugin(enabled=True)
        bboxes = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        labels = torch.tensor([2**40, 1], dtype=torch.long)
        scores = torch.tensor([0.9, 0.8])
        payload = trunk._prepare_tracker_inputs(
            3,
            bboxes,
            labels,
            scores,
            num_detections=torch.tensor([2], dtype=torch.long),
        )

        self.assertTrue(torch.equal(payload["labels"], torch.tensor([0, 1])))

    def test_prepare_tracker_inputs_drops_invalid_bboxes(self):
        trunk = TrackerPlugin(enabled=True)
        bboxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [4.0, 9.0, 12.0, 3.0],
                [20.0, 20.0, 40.0, 60.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        scores = torch.tensor([0.9, 0.8, 0.7])
        payload = trunk._prepare_tracker_inputs(
            3,
            bboxes,
            labels,
            scores,
            num_detections=torch.tensor([3], dtype=torch.long),
        )

        self.assertEqual(payload["bboxes"].shape, (2, 4))
        self.assertTrue(torch.equal(payload["labels"], torch.tensor([0, 2])))

    def test_ice_boundaries_sanitizes_outputs_before_tracker(self):
        from hmlib.aspen.plugins.ice_rink_boundaries_plugins import IceRinkSegmBoundariesPlugin

        bboxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 3.0, 7.0],
                [20.0, 20.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 4232601635951843710], dtype=torch.long)
        scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

        out_b, out_l, out_s, num_valid = IceRinkSegmBoundariesPlugin._sanitize_pruned_detections(
            bboxes,
            labels,
            scores,
            torch.tensor([3], dtype=torch.int32),
            max_items=3,
        )

        self.assertEqual(int(num_valid.item()), 2)
        self.assertEqual(tuple(out_b.shape), (3, 4))
        self.assertEqual(out_l.tolist(), [0, 0, 0])
        self.assertTrue(torch.equal(out_b[2:], torch.zeros_like(out_b[2:])))
        self.assertTrue(torch.equal(out_s[2:], torch.zeros_like(out_s[2:])))

    def test_native_hmtracker_clones_retained_label_memos(self):
        from hockeymon.core import HmByteTrackConfig, HmTracker

        config = HmByteTrackConfig()
        config.init_track_thr = 0.1
        config.obj_score_thrs_high = 0.1
        config.obj_score_thrs_low = 0.0
        config.match_iou_thrs_high = 0.1
        tracker = HmTracker(config)

        bboxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        labels = torch.tensor([0], dtype=torch.long)
        scores = torch.tensor([0.9], dtype=torch.float32)
        out0 = tracker.track(
            {
                "frame_id": torch.tensor([0], dtype=torch.long),
                "bboxes": bboxes,
                "labels": labels,
                "scores": scores,
            }
        )

        labels[0] = 123
        out1 = tracker.track(
            {
                "frame_id": torch.tensor([1], dtype=torch.long),
                "bboxes": bboxes.clone(),
                "labels": torch.tensor([0], dtype=torch.long),
                "scores": scores.clone(),
            }
        )

        self.assertEqual(int(out0["ids"][0].item()), int(out1["ids"][0].item()))


if __name__ == "__main__":
    unittest.main()
