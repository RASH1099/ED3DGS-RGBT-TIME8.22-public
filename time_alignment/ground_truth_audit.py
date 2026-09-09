import argparse
import hashlib
import json
import re
from pathlib import Path


def image_manifest(directory: Path):
    rows = []
    for path in sorted(directory.glob("*.png")):
        rows.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    payload = "\n".join(f"{name} {digest}" for name, digest in rows) + "\n"
    return len(rows), hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--render-log", required=True)
    parser.add_argument("--expected-shift", required=True, type=int)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()

    root = Path(args.eval_root)
    lock = json.loads(args.lock.read_text())
    support_count = int(lock.get("support_count", 0))
    trajectory_count = int(lock.get("trajectory_frame_count", 266))
    if (lock.get("status") != "PASS"
            or lock.get("created_before_method_training") is not True
            or lock.get("expected_shift_frames") != args.expected_shift
            or support_count <= 0):
        raise RuntimeError("Invalid preregistered matched GT lock")
    audit_path = root / "ground_truth_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"Refusing existing GT audit: {audit_path}")

    render_contract = json.loads((root / "render_contract.json").read_text())
    method = render_contract.get("method")
    if not isinstance(method, str) or not re.fullmatch(r"ours_[0-9]+", method):
        raise RuntimeError("Invalid render method in contract")
    method_root = root / "test" / method
    rgb_count, rgb_digest = image_manifest(method_root / "test_rgb" / "gt")
    thermal_count, thermal_digest = image_manifest(
        method_root / "test_thermal" / "gt")
    log_text = Path(args.render_log).read_text(errors="replace")
    shifted = (0 if args.expected_shift == 0
               else trajectory_count - abs(args.expected_shift))
    boundary = (trajectory_count if args.expected_shift == 0
                else abs(args.expected_shift))
    pattern = re.compile(
        rf"\[TemporalCorruption\] split=test "
        rf"requested_shift={args.expected_shift} "
        rf"effective_shift={args.expected_shift} .* "
        rf"shifted={shifted}/{trajectory_count} "
        rf"boundary_identity={boundary}")
    checks = {
        "loader_effective_shift_exact": bool(pattern.search(log_text)),
        "rgb_count_matches_support": rgb_count == support_count,
        "thermal_count_matches_support": thermal_count == support_count,
        "rgb_gt_matches_locked_reference": (
            rgb_digest == lock.get("rgb_gt_sha256")),
        "thermal_gt_matches_locked_reference": (
            thermal_digest == lock.get("thermal_gt_sha256")),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps({
            "checks": checks,
            "rgb_count": rgb_count,
            "thermal_count": thermal_count,
            "rgb_gt_sha256": rgb_digest,
            "thermal_gt_sha256": thermal_digest,
        }, sort_keys=True))

    audit = {
        "schema": "covers_matched_locked_gt_manifest_audit",
        "status": "PASS",
        "expected_shift_frames": args.expected_shift,
        "checks": checks,
        "rgb_gt_sha256": rgb_digest,
        "thermal_gt_sha256": thermal_digest,
        "support_count": support_count,
        "reference_protocol": "matched-pretraining-locked-fixed-support",
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print("R6_LOCKED_GT_MANIFEST_PASS " + json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
