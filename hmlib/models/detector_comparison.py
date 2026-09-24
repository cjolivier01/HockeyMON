"""Detection comparison reports; accuracy requires independently labeled boxes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def box_agreement(reference: list[dict], candidate: list[dict], threshold: float) -> dict:
    """Greedy, score-ordered IoU matching for diagnostics, not ground-truth accuracy."""
    reference = sorted(
        (p for p in reference if p["score"] >= threshold), key=lambda p: p["score"], reverse=True
    )
    candidate = [p for p in candidate if p["score"] >= threshold]
    available = set(range(len(candidate)))
    matched = 0
    for ref in reference:
        x, y, w, h = ref["bbox"]
        best_iou, best_index = 0.0, None
        for index in sorted(available):
            a, b, c, d = candidate[index]["bbox"]
            intersection = max(0, min(x + w, a + c) - max(x, a)) * max(
                0, min(y + h, b + d) - max(y, b)
            )
            iou = intersection / max(w * h + c * d - intersection, 1e-12)
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_iou >= 0.5:
            matched += 1
            available.remove(best_index)
    return dict(
        matched=matched,
        deployed_only=len(reference) - matched,
        candidate_only=len(candidate) - matched,
    )


class ComparisonReport:
    def __init__(
        self,
        output_dir: str,
        models: dict,
        annotations: str | None,
        threshold: float,
        metadata: dict | None = None,
    ):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / "predictions.jsonl"
        self.models = models
        self.metadata = dict(metadata or {})
        self.threshold = threshold
        self.ground_truth = None
        self.frames = {}
        if annotations:
            with open(annotations, encoding="utf-8") as stream:
                self.ground_truth = json.load(stream)
            categories = [c for c in self.ground_truth["categories"] if c["name"] == "person"]
            if len(categories) != 1:
                raise ValueError("Ground truth must contain exactly one category named 'person'")
            self.category_id = categories[0]["id"]
            for image in self.ground_truth["images"]:
                if "frame_id" not in image:
                    raise ValueError("Each COCO image requires an explicit hmtrack frame_id")
                frame_id = int(image["frame_id"])
                if frame_id in self.frames:
                    raise ValueError(f"Duplicate annotation frame_id: {frame_id}")
                self.frames[frame_id] = image
        self.stream = self.path.open("x", encoding="utf-8")
        self.seen = set()
        self.closed = False

    def record(
        self,
        frame_id: int,
        shape: tuple[int, int],
        predictions: dict,
        elapsed_ms: dict,
        input_shape: tuple[int, int],
    ) -> None:
        if frame_id in self.seen:
            raise ValueError(f"Duplicate compared frame: {frame_id}")
        if self.ground_truth is not None:
            image = self.frames[frame_id]
            if (int(image["height"]), int(image["width"])) != shape:
                raise ValueError(
                    f"Frame {frame_id}: ground-truth dimensions do not match {shape}. "
                    "Labels must use hmtrack's cropped, pre-resize image coordinates."
                )
        self.stream.write(
            json.dumps(
                dict(
                    frame_id=frame_id,
                    height=shape[0],
                    width=shape[1],
                    input_shape=input_shape,
                    predictions=predictions,
                    inference_ms=elapsed_ms,
                )
            )
            + "\n"
        )
        self.stream.flush()
        self.seen.add(frame_id)

    def finalize(self) -> dict:
        if self.closed:
            return self.summary
        self.stream.close()
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        if not rows:
            raise ValueError(
                "No frames were compared. Check clip range and annotation frame_id values."
            )
        summary = dict(
            metadata=self.metadata,
            frames=len(rows),
            frame_ids=[r["frame_id"] for r in rows],
            review_score_threshold=self.threshold,
            accuracy_available=self.ground_truth is not None,
            note="mAP uses person boxes before rink filtering, COCO maxDets=100. "
            "Agreement with deployed predictions is not accuracy. "
            "Inference timings are diagnostic, may include warmup, and exclude model loading.",
            models={},
        )
        for name, spec in self.models.items():
            counts = [
                sum(p["score"] >= self.threshold for p in r["predictions"][name]) for r in rows
            ]
            result = dict(
                spec=spec,
                mean_person_detections=float(np.mean(counts)),
                mean_inference_ms=float(np.mean([r["inference_ms"][name] for r in rows])),
            )
            if name != "deployed":
                matches = [
                    box_agreement(
                        r["predictions"]["deployed"], r["predictions"][name], self.threshold
                    )
                    for r in rows
                ]
                result["agreement_with_deployed"] = {
                    key: sum(m[key] for m in matches) for key in matches[0]
                }
            if self.ground_truth is not None:
                result["accuracy"] = self._accuracy(rows, name)
            summary["models"][name] = result
        if self.ground_truth is not None:
            summary["labeled_frames_outside_compared_clip"] = sorted(set(self.frames) - self.seen)
        self.summary = summary
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        lines = [
            f"Compared {len(rows)} frames. Score threshold for counts: {self.threshold}.",
            "model       mean persons   mean ms   mAP       AP50      AP75",
        ]
        for name, result in summary["models"].items():
            accuracy = result.get("accuracy", {})
            metrics = "  ".join(
                f"{accuracy[k]:.4f}" if k in accuracy else "n/a" for k in ("mAP", "AP50", "AP75")
            )
            lines.append(
                f"{name:10}  {result['mean_person_detections']:12.2f}  "
                f"{result['mean_inference_ms']:8.2f}  {metrics}"
            )
        lines.append(summary["note"])
        (self.output / "summary.txt").write_text("\n".join(lines) + "\n")
        self.closed = True
        return summary

    def _accuracy(self, rows: list[dict], name: str) -> dict:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        ground_truth = COCO()
        ground_truth.dataset = {**self.ground_truth, "info": self.ground_truth.get("info", {})}
        ground_truth.createIndex()
        predictions = [
            dict(p, image_id=self.frames[r["frame_id"]]["id"], category_id=self.category_id)
            for r in rows
            for p in r["predictions"][name]
        ]
        if predictions:
            detected = ground_truth.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {**ground_truth.dataset, "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(ground_truth, detected, "bbox")
        evaluator.params.imgIds = [self.frames[r["frame_id"]]["id"] for r in rows]
        evaluator.params.catIds = [self.category_id]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {
            key: float(evaluator.stats[index])
            for key, index in (("mAP", 0), ("AP50", 1), ("AP75", 2), ("AR100", 8))
        }
