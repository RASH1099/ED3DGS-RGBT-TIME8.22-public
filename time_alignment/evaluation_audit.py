#!/usr/bin/env python3
"""Fail-closed verifier for one arm of the matched Covers test panel."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from time_alignment import schedule


METRICS = ("PSNR", "SSIM", "LPIPS_VGG", "LPIPS_ALEX")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(path):
    for line in path.read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        filename = filename.lstrip(" *")
        if sha256(Path(filename)) != expected:
            raise RuntimeError(f"Model hash changed: {filename}")


parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True,
                    choices=("baseline", "time_only", "pose_only", "full"))
parser.add_argument("--eval-root", required=True, type=Path)
parser.add_argument("--model-manifest", required=True, type=Path)
parser.add_argument("--expected-shift", required=True, type=float)
parser.add_argument("--expected-iteration", type=int, default=30000)
parser.add_argument("--calibration-steps", type=int)
parser.add_argument("--scene-steps", type=int)
parser.add_argument("--support-contract", required=True, type=Path)
parser.add_argument("--baseline-gate", type=Path)
parser.add_argument("--strict-scene-freeze", action="store_true")
args = parser.parse_args()

strict_step_budget_v2 = (
    args.calibration_steps is not None or args.scene_steps is not None)
if ((args.calibration_steps is None)
        != (args.scene_steps is None)):
    raise RuntimeError(
        "calibration-steps and scene-steps must be provided together")
strict_counts = schedule.strict_block_step_counts(
    int(args.expected_iteration),
    calibration_steps=args.calibration_steps,
    scene_steps=args.scene_steps)

check_manifest(args.model_manifest)
support = json.loads(args.support_contract.read_text())
support_names = support.get("support_names", [])
support_count = int(support.get("support_count", 0))
support_digest = hashlib.sha256(
    ("\n".join(support_names) + "\n").encode("utf-8")).hexdigest()
if (support_count <= 0 or len(support_names) != support_count
        or support_digest != support.get("support_names_sha256")):
    raise RuntimeError("Invalid preregistered evaluation support")
baseline = (
    json.loads(args.baseline_gate.read_text())
    if args.baseline_gate is not None else None)
baseline_checks = (baseline or {}).get("checks") or {}
baseline_ignored_checks = {"clock_accuracy", "profile_consistency"}
baseline_valid = (
    baseline is not None
    and baseline.get("schema") == "covers_self_calibrating_global_clock_gate"
    and baseline.get("mechanical") is False
    and baseline.get("expected_shift_frames") == 0
    and bool(baseline.get("strict_scene_freeze"))
    == bool(args.strict_scene_freeze)
    and bool(baseline.get("strict_step_budget_v2"))
    == strict_step_budget_v2
    and (not strict_step_budget_v2
         or (baseline.get("expected_calibration_steps")
             == strict_counts["calibration"]
             and baseline.get("expected_scene_steps")
             == strict_counts["scene"]))
    and baseline.get("support_contract_sha256") == sha256(args.support_contract)
    and math.isfinite(float(baseline.get("final_offset_frames")))
    and set(baseline_checks) >= baseline_ignored_checks
    and all(value is True for name, value in baseline_checks.items()
            if name not in baseline_ignored_checks))
if baseline is not None and not baseline_valid:
    raise RuntimeError("Invalid zero-shift baseline Gate")
baseline_offset = (
    float(baseline["final_offset_frames"]) if baseline_valid else 0.0)
contract = json.loads((args.eval_root / "render_contract.json").read_text())
if (contract.get("schema") != "covers_r25_fourarm_eval_v2"
        or contract.get("status") != "PASS"
        or contract.get("arm") != args.arm):
    raise RuntimeError("Invalid render contract")
if (contract.get("synthetic_test_shift_frames") != args.expected_shift
        or contract.get("train_shift_frames") != 0
        or contract.get("rendered_support_count") != support_count
        or contract.get("support_names_sha256") != support_digest
        or not contract.get("truth_metadata_cleared_before_forward")
        or contract.get("renderer_reads_shift_truth")
        or contract.get("test_time_pose_or_fov_optimization")):
    raise RuntimeError("Evaluation protocol mismatch")
time_enabled = args.arm in {"time_only", "full"}
expected_offset = float(contract["applied_offset_frames"])
target_offset = float(args.expected_shift) + baseline_offset
offset_ok = (
    abs(expected_offset - target_offset) <= 0.25
    if time_enabled else abs(expected_offset) <= 1.0e-8)
if (not offset_ok
        or bool(contract.get("renderer_temporal_branch_applied_all")) != time_enabled
        or abs(float(contract["renderer_actual_delta_min_frames"])
               - expected_offset) > 5e-4
        or abs(float(contract["renderer_actual_delta_max_frames"])
               - expected_offset) > 5e-4
        or not (args.eval_root / "renderer_time_trace.json").is_file()):
    raise RuntimeError("Renderer-consumed clock audit failed")

method = contract["method"]
metrics = {}
counts = {}
names = None
for modality in ("rgb", "thermal"):
    base = args.eval_root / "test" / method / f"test_{modality}"
    render_names = sorted(path.name for path in (base / "renders").glob("*.png"))
    gt_names = sorted(path.name for path in (base / "gt").glob("*.png"))
    if render_names != gt_names or len(render_names) != support_count:
        raise RuntimeError(f"Image contract failed for {modality}")
    if names is not None and render_names != names:
        raise RuntimeError("RGB/Thermal output names differ")
    names = render_names
    counts[f"{modality}_renders"] = len(render_names)
    counts[f"{modality}_gt"] = len(gt_names)
    payload = json.loads((args.eval_root / f"summary_{modality}.json").read_text())
    values = payload[method]
    if set(values) != set(METRICS) or not all(
            math.isfinite(float(values[key])) for key in METRICS):
        raise RuntimeError(f"Invalid metrics for {modality}")
    metrics[modality] = {key: float(values[key]) for key in METRICS}

check_manifest(args.model_manifest)
audit = {
    "schema": "covers_r25_fourarm_eval_audit_v2",
    "status": "PASS",
    "arm": args.arm,
    "strict_scene_freeze": bool(args.strict_scene_freeze),
    "strict_step_budget_v2": strict_step_budget_v2,
    "expected_calibration_steps": (
        strict_counts["calibration"] if strict_step_budget_v2 else None),
    "expected_scene_steps": (
        strict_counts["scene"] if strict_step_budget_v2 else None),
    "expected_iteration": int(args.expected_iteration),
    "support_count": support_count,
    "support_names_sha256": contract["support_names_sha256"],
    "test_time_optimization": False,
    "model_hashes_unchanged": True,
    "clock_accuracy_mode": (
        "baseline_residual" if baseline is not None else "absolute"),
    "baseline_gate": ({
        "path": str(args.baseline_gate),
        "sha256": sha256(args.baseline_gate),
        "final_offset_frames": baseline_offset,
    } if baseline is not None else None),
    "expected_effective_offset_frames": target_offset,
    "contract": contract,
    "counts": counts,
    "metrics": metrics,
}
path = args.eval_root / "evaluation_audit.json"
if path.exists():
    raise RuntimeError("Refusing existing evaluation audit")
path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print("R25_FOURARM_EVAL_V1_PASS " + json.dumps(audit, sort_keys=True))
