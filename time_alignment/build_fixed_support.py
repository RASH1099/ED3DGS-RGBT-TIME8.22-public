#!/usr/bin/env python3
"""Build a deterministic same-split fixed-support contract before training."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(names):
    return hashlib.sha256(
        ("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--shift", required=True, type=int)
    parser.add_argument("--clock-bound", default=24, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing existing support contract: {args.output}")
    if args.shift <= 0 or args.clock_bound < args.shift:
        raise ValueError("Expected a positive shift inside the clock bound")

    rgb = json.loads((args.dataset / "rgb" / "dataset.json").read_text())
    thermal = json.loads(
        (args.dataset / "thermal" / "dataset.json").read_text())
    frame_count = int(rgb["num_exemplars"])
    if int(thermal["num_exemplars"]) != frame_count:
        raise RuntimeError("RGB and Thermal frame counts differ")
    train_ids = list(rgb["train_ids"])
    test_ids = list(rgb["val_ids"])
    if len(train_ids) != frame_count or len(test_ids) != frame_count:
        raise RuntimeError("Unexpected native train/test split size")

    support_count = frame_count - args.shift - 2
    if support_count <= 0 or support_count % 2:
        raise RuntimeError("Boundary-safe support must be positive and even")
    training_names = [f"{name}.png" for name in train_ids[:support_count]]
    evaluation_names = [
        f"{name}.png" for name in test_ids[1:1 + support_count]]
    calibration_count = frame_count - 2 * args.clock_bound
    if calibration_count <= 0 or calibration_count % 2:
        raise RuntimeError("Clock-bound calibration support is invalid")

    report = {
        "schema": f"ed3dgs_rgbt_{args.scene.lower()}_shift{args.shift}_fixed_support",
        "scene": args.scene,
        "trajectory_frame_count": frame_count,
        "calibration_support_count": calibration_count,
        "support_count": support_count,
        "training_frame_start": 0,
        "training_frame_end_exclusive": support_count,
        "evaluation_frame_start": 1,
        "evaluation_frame_end_exclusive": 1 + support_count,
        "support_names_sha256": digest(evaluation_names),
        "training_support_names_sha256": digest(training_names),
        "support_names": evaluation_names,
        "training_support_names": training_names,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FIXED_SUPPORT_BUILD_PASS " + json.dumps({
        key: report[key] for key in (
            "scene", "trajectory_frame_count", "calibration_support_count",
            "support_count", "support_names_sha256",
            "training_support_names_sha256")
    }, sort_keys=True))


if __name__ == "__main__":
    main()
