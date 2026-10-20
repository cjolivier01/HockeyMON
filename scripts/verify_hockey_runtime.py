#!/usr/bin/env python3
"""Replay every exported reference through an existing native TensorRT bundle."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from hockey_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Choose a new immutable verification report")
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    exported = json.loads((args.export_dir / "export.json").read_text())
    for key in ("inputs", "outputs", "preprocessing", "topology_id", "model_id"):
        if manifest[key] != exported[key]:
            raise ValueError(f"Export and bundle disagree: {key}")
    if manifest["onnx"]["sha256"] != exported["onnx"]["sha256"]:
        raise ValueError("Bundle was built from a different ONNX graph")
    for section in ("engine", "onnx"):
        if digest(args.bundle / manifest[section]["file"]) != manifest[section]["sha256"]:
            raise ValueError(f"Bundle {section} checksum mismatch")
    cases = exported["validation"]["cases"]
    covered = {row for case in cases for row in case["input_indices"]}
    if covered != set(range(exported["validation"]["input_count"])) or not {1, 2, 8}.issubset(
        {case["batch"] for case in cases}
    ):
        raise ValueError("References must cover every row and batches 1, 2 and 8")
    reports = []
    for case in cases:
        input_path = args.export_dir / case["input"]
        if digest(input_path) != case["input_sha256"]:
            raise ValueError("Input checksum mismatch")
        with tempfile.TemporaryDirectory(prefix="hockey-native-") as temp:
            process = subprocess.run(
                [
                    str(args.builder.resolve()),
                    "--engine",
                    str(args.bundle / manifest["engine"]["file"]),
                    "--input",
                    str(input_path),
                    "--batch",
                    str(case["batch"]),
                    "--outputs",
                    temp,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(process.stdout)
            for key, value in result["runtime"].items():
                if manifest["engine"][key] != value:
                    raise ValueError(f"Native GPU/SDK identity changed: {key}")
            actual_outputs = {output["name"]: output for output in result["outputs"]}
            if set(actual_outputs) != {reference["name"] for reference in case["outputs"]}:
                raise ValueError("Native output names differ")
            for reference in case["outputs"]:
                ref_path = args.export_dir / reference["file"]
                if digest(ref_path) != reference["sha256"]:
                    raise ValueError("Reference checksum mismatch")
                output = actual_outputs[reference["name"]]
                if output["shape"] != reference["shape"] or output["dtype"] != "float32":
                    raise ValueError("Native output shape or dtype differs")
                expected = np.fromfile(ref_path, np.float32).reshape(reference["shape"])
                actual = np.fromfile(Path(temp) / output["file"], np.float32).reshape(
                    reference["shape"]
                )
                atol, rtol = (
                    (0.08, 0.03) if manifest["engine"]["precision"] == "fp16" else (1e-4, 5e-4)
                )
                if not np.isfinite(actual).all() or not np.allclose(
                    actual, expected, atol=atol, rtol=rtol
                ):
                    raise ValueError(
                        f"TensorRT parity failed: {case['input']}, {reference['name']}"
                    )
                ordered = np.sort(expected, axis=-1)
                equal = actual.argmax(-1) == expected.argmax(-1)
                if np.any((ordered[..., -1] - ordered[..., -2] > 2 * atol) & ~equal):
                    raise ValueError("TensorRT changed a high-margin decision")
                reports.append(
                    {
                        "batch": case["batch"],
                        "input": case["input"],
                        "output": reference["name"],
                        "max_abs_error": float(np.abs(actual - expected).max()),
                        "argmax_agreement": float(equal.mean()),
                    }
                )
    report = {
        "bundle_sha256": digest(args.bundle / "manifest.json"),
        "export_sha256": digest(args.export_dir / "export.json"),
        "runtime": result["runtime"],
        "results": reports,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Validated {len(cases)} native batches; max error {max(row['max_abs_error'] for row in reports):.6f}"
    )


if __name__ == "__main__":
    main()
