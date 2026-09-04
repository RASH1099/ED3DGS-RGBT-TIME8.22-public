#!/usr/bin/env python3
"""Apply one audited training-only clock refinement to an isolated model copy."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--refinement-report", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    report_path = args.refinement_report.resolve()
    state_path = model / "point_cloud/iteration_30000/thermal_camera_state.pth"
    application_path = model / "clock_refinement_application.json"
    marker = model / "CLOCK_REFINEMENT_APPLIED"
    pending = state_path.with_suffix(".pth.pending")
    require(model.is_dir() and state_path.is_file(), "Missing isolated model state")
    require(report_path.is_file(), "Missing refinement report")
    require(not application_path.exists() and not marker.exists()
            and not pending.exists(), "Refusing existing refinement artifact")

    refinement = json.loads(report_path.read_text(encoding="utf-8"))
    require(refinement.get("status") == "PASS"
            and all(refinement.get("checks", {}).values()),
            "Refinement report did not pass")
    require(refinement.get("candidate_enumeration") is False
            and refinement.get("shift_truth_input_to_optimizer") is False,
            "Refinement violated the blind optimizer contract")
    offset = float(refinement["refined_offset_frames"])
    require(math.isfinite(offset), "Refined offset is non-finite")

    before_sha256 = sha256(state_path)
    payload = torch.load(state_path, map_location="cpu")
    temporal = payload.get("temporal_alignment", {})
    max_frames = float(temporal.get("max_frames", -1.0))
    require(temporal.get("enabled") is True and max_frames > 0.0,
            "Model has no scalar temporal alignment state")
    require(temporal.get("affine_enabled") is False,
            "Refinement supports only a scalar clock")
    require(abs(offset) < max_frames, "Refined offset reached clock boundary")
    previous = {
        "offset_raw": float(temporal["offset_raw"]),
        "offset_frames": float(temporal["offset_frames"]),
    }
    temporal["offset_raw"] = math.atanh(offset / max_frames)
    temporal["offset_frames"] = offset
    torch.save(payload, pending)
    require(pending.is_file(), "Refined state was not written")
    os.replace(pending, state_path)
    after_sha256 = sha256(state_path)
    require(before_sha256 != after_sha256, "Clock-state hash did not change")

    application = {
        "schema": "covers_training_only_clock_refinement_application",
        "status": "PASS",
        "model": str(model),
        "refinement_report": str(report_path),
        "refinement_report_sha256": sha256(report_path),
        "previous_clock": previous,
        "refined_clock": {
            "offset_raw": temporal["offset_raw"],
            "offset_frames": offset,
            "max_frames": max_frames,
        },
        "thermal_camera_state_sha256_before": before_sha256,
        "thermal_camera_state_sha256_after": after_sha256,
        "gaussian_or_pose_updated": False,
        "formal_test_cameras_used_for_refinement": False,
    }
    application_path.write_text(
        json.dumps(application, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    marker.write_text("PASS\n", encoding="utf-8")
    print(json.dumps(application, sort_keys=True))


if __name__ == "__main__":
    main()
