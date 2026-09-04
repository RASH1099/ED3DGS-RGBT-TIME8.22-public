#!/usr/bin/env python3
"""Lock the matched fixed-support GT manifest before method training."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def image_manifest(directory):
    rows = []
    for path in sorted(directory.glob("*.png")):
        rows.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    payload = "\n".join(f"{name} {digest}" for name, digest in rows) + "\n"
    return len(rows), hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--render-log", required=True, type=Path)
    parser.add_argument("--support-contract", required=True, type=Path)
    parser.add_argument("--expected-shift", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.expected_shift not in {0, 8, 20}:
        raise RuntimeError("The GT lock requires shift 0, 8, or 20")
    if args.output.exists():
        raise RuntimeError(f"Refusing existing GT lock: {args.output}")
    support = json.loads(args.support_contract.read_text())
    support_names = support.get("support_names", [])
    support_count = int(support.get("support_count", 0))
    trajectory_count = int(support.get("trajectory_frame_count", 266))
    support_digest = hashlib.sha256(
        ("\n".join(support_names) + "\n").encode("utf-8")).hexdigest()
    if (support_count <= 0
            or len(support_names) != support_count
            or support_digest != support.get("support_names_sha256")):
        raise RuntimeError("Unexpected matched support contract")

    contract = json.loads((args.eval_root / "render_contract.json").read_text())
    if (contract.get("synthetic_test_shift_frames") != args.expected_shift
            or contract.get("rendered_support_count") != support_count
            or contract.get("support_names_sha256") !=
            support["support_names_sha256"]):
        raise RuntimeError("GT-lock render contract mismatch")
    log = args.render_log.read_text(errors="replace")
    shifted = (0 if args.expected_shift == 0
               else trajectory_count - abs(args.expected_shift))
    boundary = (trajectory_count if args.expected_shift == 0
                else abs(args.expected_shift))
    if not re.search(
            rf"\[TemporalCorruption\] split=test "
            rf"requested_shift={args.expected_shift} "
            rf"effective_shift={args.expected_shift} .* "
            rf"shifted={shifted}/{trajectory_count} "
            rf"boundary_identity={boundary}", log):
        raise RuntimeError("GT-lock loader shift contract mismatch")

    base = args.eval_root / "test" / "ours_30000"
    rgb_count, rgb_digest = image_manifest(base / "test_rgb" / "gt")
    thermal_count, thermal_digest = image_manifest(base / "test_thermal" / "gt")
    if rgb_count != support_count or thermal_count != support_count:
        raise RuntimeError("GT-lock image count mismatch")
    report = {
        "schema": "covers_matched_ground_truth_lock",
        "status": "PASS",
        "created_before_method_training": True,
        "expected_shift_frames": args.expected_shift,
        "trajectory_frame_count": trajectory_count,
        "support_count": support_count,
        "support_names_sha256": support["support_names_sha256"],
        "rgb_gt_sha256": rgb_digest,
        "thermal_gt_sha256": thermal_digest,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("MATCHED_GT_LOCK_PASS " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
