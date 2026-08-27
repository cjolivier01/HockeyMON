import os
import unittest

try:
    import torch
except ImportError:
    raise RuntimeError("torch is required for the HockeyMON GPU runtime tests")

from hmlib.utils.torch_backend import torch_backend

if not torch.cuda.is_available():
    raise RuntimeError(
        "torch.cuda.is_available() is false; the HockeyMON GPU runtime tests require a usable GPU backend"
    )

EXPECTED_BACKEND = os.environ.get("HM_TEST_EXPECT_TORCH_BACKEND")
if EXPECTED_BACKEND is None:
    raise RuntimeError("HM_TEST_EXPECT_TORCH_BACKEND must be set for the GPU runtime tests")

ACTIVE_BACKEND = torch_backend()

if ACTIVE_BACKEND != EXPECTED_BACKEND:
    raise RuntimeError(f"expected torch backend {EXPECTED_BACKEND}, got {ACTIVE_BACKEND}")

GPU_TRACKING_RUNTIME_ERROR = None
try:
    torch.linalg.cholesky(torch.eye(2, dtype=torch.float32, device="cuda"))
except RuntimeError as exc:
    GPU_TRACKING_RUNTIME_ERROR = str(exc)

from hockeymon.core import HmByteTrackConfig, HmByteTrackerCuda, HmByteTrackerCudaStatic, HmTracker


def _make_data(device: torch.device, frame_id: int, boxes, labels, scores):
    return {
        "frame_id": torch.tensor([frame_id], dtype=torch.long, device=device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "scores": torch.tensor(scores, dtype=torch.float32, device=device),
    }


def _make_padded_data(
    device: torch.device,
    frame_id: int,
    boxes,
    labels,
    scores,
    max_detections: int,
):
    num_det = len(boxes)
    padded_boxes = torch.zeros((max_detections, 4), dtype=torch.float32, device=device)
    padded_labels = torch.zeros((max_detections,), dtype=torch.long, device=device)
    padded_scores = torch.zeros((max_detections,), dtype=torch.float32, device=device)
    if num_det:
        padded_boxes[:num_det] = torch.tensor(boxes, dtype=torch.float32, device=device)
        padded_labels[:num_det] = torch.tensor(labels, dtype=torch.long, device=device)
        padded_scores[:num_det] = torch.tensor(scores, dtype=torch.float32, device=device)
    return {
        "frame_id": torch.tensor([frame_id], dtype=torch.long, device=device),
        "bboxes": padded_boxes,
        "labels": padded_labels,
        "scores": padded_scores,
        "num_detections": torch.tensor([num_det], dtype=torch.long, device=device),
    }


class ByteTrackGpuBackendTest(unittest.TestCase):
    def test_expected_backend_matches_runtime(self):
        self.assertIn(ACTIVE_BACKEND, {"cuda", "rocm"})
        self.assertEqual(ACTIVE_BACKEND, EXPECTED_BACKEND)
        self.assertTrue(torch.cuda.is_available())

        if ACTIVE_BACKEND == "rocm":
            self.assertIsNotNone(getattr(torch.version, "hip", None))
            self.assertFalse(bool(getattr(torch.version, "cuda", None)))
        else:
            self.assertTrue(bool(getattr(torch.version, "cuda", None)))
            self.assertFalse(bool(getattr(torch.version, "hip", None)))

    def test_gpu_tracking_runtime_supports_cholesky(self):
        if GPU_TRACKING_RUNTIME_ERROR is not None:
            self.fail(GPU_TRACKING_RUNTIME_ERROR)

    @unittest.skipIf(
        GPU_TRACKING_RUNTIME_ERROR is not None,
        "torch GPU Cholesky unavailable in this runtime",
    )
    def test_gpu_tracker_matches_cpu(self):
        config = HmByteTrackConfig()
        cpu_tracker = HmTracker(config)
        gpu_tracker = HmByteTrackerCuda(config, device="cuda:0")

        frames = [
            (
                [[10.0, 10.0, 30.0, 40.0], [100.0, 100.0, 140.0, 160.0]],
                [1, 1],
                [0.9, 0.85],
            ),
            (
                [[12.0, 12.0, 32.0, 42.0], [103.0, 103.0, 143.0, 163.0]],
                [1, 1],
                [0.92, 0.8],
            ),
        ]

        for frame_id, (boxes, labels, scores) in enumerate(frames):
            cpu_data = _make_data(torch.device("cpu"), frame_id, boxes, labels, scores)
            gpu_data = _make_data(torch.device("cuda"), frame_id, boxes, labels, scores)

            cpu_res = cpu_tracker.track(cpu_data.copy())
            gpu_res = gpu_tracker.track(gpu_data)

            self.assertEqual(gpu_res["ids"].device.type, "cuda")
            self.assertEqual(gpu_res["bboxes"].device.type, "cuda")
            self.assertEqual(gpu_res["labels"].device.type, "cuda")
            self.assertEqual(gpu_res["scores"].device.type, "cuda")
            self.assertTrue(torch.equal(cpu_res["ids"], gpu_res["ids"].cpu()))
            self.assertTrue(torch.allclose(cpu_res["bboxes"], gpu_res["bboxes"].cpu(), atol=1e-3))
            self.assertTrue(torch.allclose(cpu_res["scores"], gpu_res["scores"].cpu(), atol=1e-3))


@unittest.skipIf(
    GPU_TRACKING_RUNTIME_ERROR is not None,
    "torch GPU Cholesky unavailable in this runtime",
)
class ByteTrackCudaStaticTest(unittest.TestCase):
    def test_static_tracker_pads_outputs(self):
        config = HmByteTrackConfig()
        dynamic_tracker = HmByteTrackerCuda(config, device="cuda:0")
        static_tracker = HmByteTrackerCudaStatic(
            config,
            max_detections=8,
            max_tracks=8,
            device="cuda:0",
        )

        frames = [
            (
                [[10.0, 10.0, 30.0, 40.0], [100.0, 100.0, 140.0, 160.0]],
                [1, 1],
                [0.9, 0.85],
            ),
            (
                [[12.0, 12.0, 32.0, 42.0], [103.0, 103.0, 143.0, 163.0]],
                [1, 1],
                [0.92, 0.8],
            ),
        ]

        for frame_id, (boxes, labels, scores) in enumerate(frames):
            dyn_data = _make_data(torch.device("cuda"), frame_id, boxes, labels, scores)
            dyn_res = dynamic_tracker.track(dyn_data)

            padded_data = _make_padded_data(
                torch.device("cuda"), frame_id, boxes, labels, scores, max_detections=8
            )
            static_res = static_tracker.track(padded_data)

            num_tracks = int(static_res["num_tracks"].item())
            self.assertEqual(static_res["ids"].device.type, "cuda")
            self.assertEqual(static_res["bboxes"].device.type, "cuda")
            self.assertEqual(static_res["labels"].device.type, "cuda")
            self.assertEqual(static_res["scores"].device.type, "cuda")
            self.assertEqual(num_tracks, dyn_res["ids"].shape[0])
            self.assertEqual(static_res["ids"].shape[0], 8)
            self.assertEqual(static_res["bboxes"].shape[0], 8)

            if num_tracks:
                self.assertTrue(torch.equal(static_res["ids"][:num_tracks], dyn_res["ids"]))
                self.assertTrue(
                    torch.allclose(
                        static_res["bboxes"][:num_tracks],
                        dyn_res["bboxes"],
                        atol=1e-4,
                    )
                )
                self.assertTrue(torch.equal(static_res["labels"][:num_tracks], dyn_res["labels"]))
                self.assertTrue(
                    torch.allclose(static_res["scores"][:num_tracks], dyn_res["scores"], atol=1e-4)
                )

            if num_tracks < 8:
                self.assertTrue(torch.all(static_res["ids"][num_tracks:] == -1))
                self.assertTrue(torch.all(static_res["bboxes"][num_tracks:] == 0))
                self.assertTrue(torch.all(static_res["labels"][num_tracks:] == 0))
                self.assertTrue(torch.all(static_res["scores"][num_tracks:] == 0))

            self.assertEqual(int(static_res["num_detections"].item()), len(boxes))

    def test_static_tracker_matches_cpu_tracker(self):
        config = HmByteTrackConfig()
        cpu_tracker = HmTracker(config)
        static_tracker = HmByteTrackerCudaStatic(
            config,
            max_detections=8,
            max_tracks=8,
            device="cuda:0",
        )

        frames = [
            (
                [[20.0, 20.0, 40.0, 60.0], [110.0, 90.0, 140.0, 150.0]],
                [1, 2],
                [0.95, 0.83],
            ),
            (
                [[22.0, 22.0, 42.0, 62.0], [115.0, 95.0, 145.0, 155.0]],
                [1, 2],
                [0.91, 0.8],
            ),
        ]

        for frame_id, (boxes, labels, scores) in enumerate(frames):
            cpu_data = _make_data(torch.device("cpu"), frame_id, boxes, labels, scores)
            cpu_res = cpu_tracker.track(cpu_data.copy())

            static_input = _make_padded_data(
                torch.device("cuda"), frame_id, boxes, labels, scores, max_detections=8
            )
            static_res = static_tracker.track(static_input)

            cpu_ids = cpu_res["ids"].cpu()
            cpu_bboxes = cpu_res["bboxes"].cpu()
            cpu_labels = cpu_res["labels"].cpu()
            cpu_scores = cpu_res["scores"].cpu()

            num_tracks = int(static_res["num_tracks"].item())
            self.assertEqual(num_tracks, cpu_ids.shape[0])

            self.assertTrue(torch.equal(static_res["ids"][:num_tracks].cpu(), cpu_ids))
            self.assertTrue(
                torch.allclose(static_res["bboxes"][:num_tracks].cpu(), cpu_bboxes, atol=1e-4)
            )
            self.assertTrue(torch.equal(static_res["labels"][:num_tracks].cpu(), cpu_labels))
            self.assertTrue(
                torch.allclose(static_res["scores"][:num_tracks].cpu(), cpu_scores, atol=1e-4)
            )


if __name__ == "__main__":
    unittest.main()
