#!/usr/bin/env python3
"""Fail-closed Gate audit for unified self-calibrating affine clocks."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from time_alignment import schedule


MAX_GAUSSIANS = 155000


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def rows(log, name):
    pattern = (r"(?:^|[\r\n])(?:Training progress:[^\r\n]*\])?"
               + re.escape(name) + r" (\{[^\r\n]*\})")
    return [json.loads(match.group(1)) for match in re.finditer(pattern, log)]


def ply_vertex_count(path):
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").strip()
            match = re.fullmatch(r"element vertex (\d+)", line)
            if match:
                return int(match.group(1))
            if line == "end_header":
                break
    raise RuntimeError(f"Missing PLY vertex count: {path}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deformation_optimizer_audit(path):
    capture, iteration = torch.load(path, map_location="cpu")
    optimizer = capture[21]
    groups = [group for group in optimizer["param_groups"]
              if group.get("name") == "deformation"]
    if len(groups) != 1:
        return {"iteration": iteration, "parameter_count": 0,
                "state_count": 0, "finite": False, "nonzero": False}
    parameter_ids = groups[0]["params"]
    states = [optimizer["state"].get(parameter_id)
              for parameter_id in parameter_ids]
    tensors = [value for state in states if state is not None
               for value in state.values() if torch.is_tensor(value)]
    moments = [state.get("exp_avg") for state in states if state is not None
               and torch.is_tensor(state.get("exp_avg"))]
    return {
        "iteration": iteration,
        "parameter_count": len(parameter_ids),
        "state_count": sum(state is not None for state in states),
        "finite": bool(tensors) and all(
            bool(torch.isfinite(value).all()) for value in tensors),
        "nonzero": bool(moments) and any(
            bool(torch.count_nonzero(value)) for value in moments),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--arm", choices=("full", "time_only"),
                        default="full")
    parser.add_argument("--expected-shift", required=True, type=float)
    parser.add_argument("--expected-iterations", required=True, type=int)
    parser.add_argument("--calibration-steps", type=int)
    parser.add_argument("--scene-steps", type=int)
    parser.add_argument("--support-contract", required=True, type=Path)
    parser.add_argument("--baseline-gate", type=Path)
    parser.add_argument("--mechanical", action="store_true")
    parser.add_argument("--strict-scene-freeze", action="store_true")
    args = parser.parse_args()

    expected_shift = float(args.expected_shift)
    expected_iterations = int(args.expected_iterations)
    strict_step_budget_v2 = (
        args.calibration_steps is not None or args.scene_steps is not None)
    if ((args.calibration_steps is None)
            != (args.scene_steps is None)):
        raise RuntimeError(
            "calibration-steps and scene-steps must be provided together")
    strict_counts = schedule.strict_block_step_counts(
        expected_iterations,
        calibration_steps=args.calibration_steps,
        scene_steps=args.scene_steps)
    expected_clock_steps = (
        strict_counts["calibration"]
        if args.strict_scene_freeze else expected_iterations)
    expected_scene_steps = (
        strict_counts["scene"]
        if args.strict_scene_freeze else expected_iterations)
    expected_last_clock_step = expected_iterations
    if args.strict_scene_freeze:
        expected_last_clock_step = max(
            iteration for iteration in range(1, expected_iterations + 1)
            if schedule.phase_for_iteration(
                iteration,
                calibration_steps=(args.calibration_steps
                                   if strict_step_budget_v2 else None),
                scene_steps=(args.scene_steps
                             if strict_step_budget_v2 else None))
            == "calibration")
    expected_pose = args.arm == "full"
    expected_pose_count = 2 if expected_pose else 0
    expected_ablation_mode = None if expected_pose else "frozen_pose"
    support_contract = json.loads(args.support_contract.read_text())
    support_names = list(support_contract.get("support_names", []))
    training_support_names = list(
        support_contract.get("training_support_names", []))
    expected_support_count = int(support_contract.get("support_count", 0))
    expected_trajectory_count = int(support_contract.get(
        "trajectory_frame_count", 266))
    expected_calibration_count = int(support_contract.get(
        "calibration_support_count", 218))
    expected_frame_start = int(support_contract.get(
        "training_frame_start", 0))
    expected_frame_end = int(support_contract.get(
        "training_frame_end_exclusive",
        expected_frame_start + expected_support_count))
    expected_evaluation_frame_start = int(support_contract.get(
        "evaluation_frame_start", 0))
    expected_evaluation_frame_end = int(support_contract.get(
        "evaluation_frame_end_exclusive",
        expected_evaluation_frame_start + expected_support_count))
    expected_names_digest = hashlib.sha256(
        ("\n".join(support_names) + "\n").encode("utf-8")).hexdigest()
    expected_training_names_digest = hashlib.sha256(
        ("\n".join(training_support_names) + "\n").encode(
            "utf-8")).hexdigest()
    if (expected_support_count <= expected_calibration_count
            or expected_support_count % 2 != 0
            or len(support_names) != expected_support_count
            or len(training_support_names) != expected_support_count
            or expected_frame_end - expected_frame_start
            != expected_support_count
            or expected_evaluation_frame_end - expected_evaluation_frame_start
            != expected_support_count
            or expected_names_digest
            != support_contract.get("support_names_sha256")
            or expected_training_names_digest
            != support_contract.get("training_support_names_sha256")):
        raise RuntimeError("Invalid preregistered reconstruction support")
    expected_per_side = expected_support_count // 2
    expected_selection_policy = (
        f"pre_registered_train_frames_{expected_frame_start}_"
        f"{expected_frame_end - 1}")
    report = json.loads(
        (args.output / "stage2_training_result.json").read_text())
    log = args.log.read_text(errors="replace")
    contract = report.get("joint_clock_contract") or {}
    internal = report.get("in_process_internal_clock") or {}
    counts = report.get("optimizer_update_counts") or {}
    consensus = counts.get("temporal_consensus_contract") or {}
    profiles = consensus.get("profiles") or {}
    route = rows(log, "MODALITY_ROUTE_STATS")
    grads = rows(log, "MODALITY_GRADIENT_STATS")
    capacity = rows(log, "GAUSSIAN_CAPACITY_CONFIG")
    fastpath = rows(log, "A5000_FASTPATH_CONFIG")
    nuisance = rows(log, "CLOCK_STEP_NUISANCE_AUDIT")
    preflights = rows(log, "STAGE2_SUPPORT_TRANSITION_PREFLIGHT")
    strict_contracts = rows(log, "STRICT_SCENE_FREEZE_CONTRACT")

    point_cloud = (args.output / "point_cloud"
                   / f"iteration_{expected_iterations}" / "point_cloud.ply")
    deformation = (args.output / "point_cloud"
                   / f"iteration_{expected_iterations}" / "deformation.pth")
    checkpoint = args.output / f"chkpnt{expected_iterations}.pth"
    point_count = ply_vertex_count(point_cloud) if point_cloud.is_file() else None
    deformation_hash = sha256_file(deformation) if deformation.is_file() else None
    teacher_deformation_hash = (report.get("teacher_hashes_before") or {}).get(
        "deformation.pth")
    deformation_optimizer = (
        deformation_optimizer_audit(checkpoint)
        if checkpoint.is_file() else {})

    baseline = (
        json.loads(args.baseline_gate.read_text())
        if args.baseline_gate is not None else None)
    baseline_checks = (baseline or {}).get("checks") or {}
    baseline_ignored_checks = {"clock_accuracy", "profile_consistency"}
    baseline_valid = (
        baseline is not None
        and baseline.get("schema")
        == "covers_self_calibrating_global_clock_gate"
        and baseline.get("mechanical") is False
        and baseline.get("expected_shift_frames") == 0
        and bool(baseline.get("strict_scene_freeze"))
        == bool(args.strict_scene_freeze)
        and bool(baseline.get("strict_step_budget_v2"))
        == strict_step_budget_v2
        and baseline.get("expected_calibration_steps")
        == strict_counts["calibration"]
        and baseline.get("expected_scene_steps") == strict_counts["scene"]
        and baseline.get("support_contract_sha256")
        == sha256_file(args.support_contract)
        and baseline.get("teacher_deformation_sha256")
        == teacher_deformation_hash
        and finite(baseline.get("final_offset_frames"))
        and len(baseline.get("final_endpoint_offsets_frames") or []) == 2
        and all(finite(value) for value in
                baseline.get("final_endpoint_offsets_frames") or [])
        and set(baseline_checks) >= baseline_ignored_checks
        and all(value is True for name, value in baseline_checks.items()
                if name not in baseline_ignored_checks))

    profile_bests = {
        name: value.get("best_integer_offset_frames")
        for name, value in profiles.items()
        if isinstance(value, dict)
    }
    required_profiles = {"full", "left", "right", "early", "late"}
    baseline_profiles = (baseline or {}).get("profile_best_offsets_frames") or {}
    profile_values = (
        {name: float(profile_bests[name]) - float(baseline_profiles[name])
         for name in required_profiles}
        if baseline_valid
        and set(profile_bests) == required_profiles
        and set(baseline_profiles) == required_profiles
        and all(finite(value) for value in baseline_profiles.values())
        else profile_bests)
    agreeing_profiles = sum(
        abs(float(value) - expected_shift) <= 2.0
        for value in profile_values.values() if finite(value))
    profile_consistent = (
        set(profile_bests) == required_profiles
        and all(finite(value) for value in profile_bests.values())
        and (baseline is None or baseline_valid)
        and abs(float(profile_values["full"]) - expected_shift) <= 1.0
        and agreeing_profiles >= 3)

    endpoints = report.get("final_endpoint_offsets_frames") or []
    baseline_endpoints = (
        baseline.get("final_endpoint_offsets_frames")
        if baseline_valid else [0.0, 0.0])
    baseline_offset = (
        float(baseline["final_offset_frames"])
        if baseline_valid else 0.0)
    endpoint_accuracy = (
        len(endpoints) == 2
        and all(finite(value) for value in endpoints)
        and (baseline is None or baseline_valid)
        and all(abs(float(value) - float(reference) - expected_shift) <= 0.25
                for value, reference in zip(endpoints, baseline_endpoints)))
    learned_accuracy = (
        finite(report.get("final_offset_frames"))
        and (baseline is None or baseline_valid)
        and abs(float(report["final_offset_frames"]) - baseline_offset
                - expected_shift) <= 0.25
        and finite(internal.get("learned_drift_frames"))
        and abs(float(internal["learned_drift_frames"])) <= 0.25
        and endpoint_accuracy)

    route_ok = bool(route) and all(
        row.get("stable") is True
        and all(finite(row.get(key)) for key in (
            "shared_fraction", "rgb_fraction", "thermal_fraction"))
        and abs(sum(float(row[key]) for key in (
            "shared_fraction", "rgb_fraction", "thermal_fraction")) - 1.0)
        <= 1.0e-6 for row in route)
    modal_grad_ok = any(
        float(row.get("rgb_feature_grad_norm", 0.0)) > 0.0
        and float(row.get("thermal_feature_grad_norm", 0.0)) > 0.0
        for row in grads)
    capacity_ok = (
        len(capacity) == 1
        and capacity[0].get("enabled") is True
        and capacity[0].get("max_gaussians") == MAX_GAUSSIANS
        and point_count is not None and point_count <= MAX_GAUSSIANS)
    fastpath_ok = (
        len(fastpath) == 1
        and fastpath[0] == {
            "deformation_checkpoint": False,
            "memory_safe_backward": False,
        }
        and contract.get("memory_safe_backward") is False
        and "MEMORY_SAFE_CACHE_RELEASE" not in log
        and "MEMORY_SAFE_CLOCK_DETACH" not in log)

    checks = {
        "arm_contract": (
            report.get("arm") == args.arm
            and report.get("ablation_mode") == expected_ablation_mode
            and report.get("temporal_alignment_enabled") is True
            and report.get("thermal_pose_enabled") is expected_pose
            and report.get("thermal_pose_unique_rotation_parameters")
            == expected_pose_count
            and report.get("thermal_pose_unique_translation_parameters")
            == expected_pose_count),
        "iterations": report.get("iterations") == expected_iterations,
        "raw_zero": (
            internal.get("initial_raw") == 0.0
            and internal.get("initial_offset_frames") == 0.0
            and internal.get("initial_drift_raw") == 0.0
            and internal.get("initial_drift_frames") == 0.0
            and internal.get("shift_truth_input") is False
            and internal.get("pretraining_offset_assignment") is False
            and internal.get("pre_reconstruction_clock_phase") is False),
        "shift_protocol": (
            report.get("observed_dataset_frame_shift") == expected_shift),
        "teacher_immutable": (
            report.get("teacher_hashes_before")
            == report.get("teacher_hashes_after")),
        "global_clock": (
            report.get("self_calibrating_clock") is True
            and report.get("affine_clock_enabled") is False
            and internal.get("schema")
            == "self_calibrating_global_clock_contract"
            and internal.get("unified_main_training") is True
            and internal.get("candidate_enumeration") is False),
        "gradient_isolation": (
            contract.get("enabled") is True
            and contract.get("alignment_gradient_owners") == ["scene_clock"]
            and contract.get("reconstruction_backward_excludes_alignment") is True
            and contract.get("reconstruction_gradient_to_clock") is False
            and contract.get("clock_gradient_mode") == "alignment_only"
            and len(nuisance) == 2
            and {row.get("iteration") for row in nuisance}
            == {1, expected_last_clock_step}
            and all(row.get("parameter_versions_unchanged") is True
                    and int(row.get("nuisance_parameter_count", 0)) > 0
                    for row in nuisance)),
        "clock_observable": (
            consensus.get("schema")
            == "covers_continuous_global_motion_cost_volume_clock"
            and consensus.get("candidate_lag_bank_in_optimizer") is False
            and consensus.get("clock_initialized_from_profile") is False
            and consensus.get("profile_scan_used_only_for_audit") is True
            and consensus.get("shift_truth_input") is False
            and consensus.get("camera_shift_metadata_read") is False
            and consensus.get("cost_volume_detached") is True
            and consensus.get("cost_volume_built_before_clock_updates") is True
            and consensus.get("affine_clock") is False
            and consensus.get("global_offset_only") is True
            and (consensus.get("aggregation") or {}).get("schema")
            == "global_mean_all_supported_transitions"
            and consensus.get("continuous_query")
            == "gaussian_continuation_then_linear"
            and consensus.get("linear_start_iteration") == 301
            and len(consensus.get("block_profiles") or {}) == 8),
        "profile_consistency": True if args.mechanical else profile_consistent,
        "clock_optimizer": (
            counts.get("temporal_offset_optimizer_steps")
            == expected_clock_steps
            and counts.get("joint_clock_nonzero_offset_gradients", 0) > 0
            and counts.get("joint_clock_nonzero_drift_gradients", 0) == 0
            and internal.get("optimizer_steps") == expected_clock_steps),
        "clock_accuracy": True if args.mechanical else learned_accuracy,
        "route_contract": route_ok,
        "both_modal_gradients": modal_grad_ok,
        "deformation_trainable": (
            deformation_hash is not None
            and teacher_deformation_hash is not None
            and deformation_hash != teacher_deformation_hash
            and deformation_optimizer.get("iteration") == expected_iterations
            and deformation_optimizer.get("parameter_count", 0) > 0
            and deformation_optimizer.get("state_count")
            == deformation_optimizer.get("parameter_count")
            and deformation_optimizer.get("finite") is True
            and deformation_optimizer.get("nonzero") is True),
        "optimizer_counts": (
            counts.get("gaussian_optimizer_steps") == expected_scene_steps
            and counts.get("target_gaussian_optimizer_steps")
            == expected_scene_steps),
        "clock_capture": (
            finite(report.get("final_offset_frames"))
            and report.get("final_drift_raw") is None
            and internal.get("learned_drift_frames") == 0.0
            and len(endpoints) == 2
            and report.get("candidate_enumeration") is False
            and report.get("candidate_lag_bank_in_observable") is False
            and report.get("temporal_raw_same_object_end_to_end") is True),
        "capacity_budget": capacity_ok,
        "fastpath": fastpath_ok,
        "support_transition_preflight": (
            len(preflights) == 1
            and preflights[0].get("schema")
            == "covers_stage2_support_transition_preflight"
            and preflights[0].get("immutable_camera_snapshot") is True
            and preflights[0].get("all_trajectory_camera_count")
            == expected_trajectory_count
            and preflights[0].get("calibration_camera_count")
            == expected_calibration_count
            and preflights[0].get("reconstruction_camera_count")
            == expected_support_count
            and preflights[0].get("reconstruction_cameras_by_side")
            == {"left": expected_per_side, "right": expected_per_side}
            and preflights[0].get("training_frame_start")
            == expected_frame_start
            and preflights[0].get("training_frame_end_exclusive")
            == expected_frame_end
            and preflights[0].get("evaluation_frame_start")
            == expected_evaluation_frame_start
            and preflights[0].get("evaluation_frame_end_exclusive")
            == expected_evaluation_frame_end
            and preflights[0].get("selection_policy")
            == expected_selection_policy
            and preflights[0].get("support_contract_sha256")
            == sha256_file(args.support_contract)
            and preflights[0].get("support_contract_names_sha256")
            == expected_training_names_digest
            and preflights[0].get("selected_names_sha256")
            == expected_training_names_digest
            and preflights[0].get("shift_truth_used_for_selection") is False
            and preflights[0].get("learned_clock_used_for_selection") is False
            and report.get("support_transition_preflight") == preflights[0]),
        "clean_log": not any(token in log for token in (
            "Traceback", "RuntimeError", "CUDA error", "out of memory",
            "Killed", "Non-finite")),
    }
    if args.strict_scene_freeze:
        expected_pose_steps = expected_clock_steps if expected_pose else 0
        phase_counts = counts.get("optimizer_phase_iterations") or {}
        scene_backward = counts.get("scene_backward_phase_counts") or {}
        densification = counts.get("densification_phase_counts") or {}
        sh_updates = counts.get("sh_degree_update_phase_counts") or {}
        checks["strict_scene_freeze"] = (
            report.get("strict_scene_freeze") is True
            and report.get("scene_optimizer_every_iteration") is False
            and len(strict_contracts) == 1
            and strict_contracts[0].get("calibration_scene_gradients") is False
            and strict_contracts[0].get("calibration_scene_mutations") is False
            and strict_contracts[0].get("reconstruction_clock_gradients") is False
            and strict_contracts[0].get("reconstruction_pose_gradients") is False
            and strict_contracts[0].get(
                "temporal_forward_uses_detached_state") is True
            and strict_contracts[0].get("expected_scene_optimizer_steps")
            == expected_scene_steps
            and strict_contracts[0].get(
                "expected_calibration_optimizer_steps")
            == expected_clock_steps
            and counts.get("strict_step_budget_v2")
            is strict_step_budget_v2
            and counts.get("expected_outer_iterations")
            == expected_iterations
            and counts.get("target_clock_optimizer_steps")
            == expected_clock_steps
            and counts.get("target_pose_optimizer_steps")
            == expected_clock_steps
            and counts.get("thermal_pose_optimizer_steps")
            == expected_pose_steps
            and scene_backward.get("calibration") == 0
            and scene_backward.get("reconstruction")
            == phase_counts.get("reconstruction")
            and scene_backward.get("scene_tail", 0)
            == strict_counts.get("scene_tail", 0)
            and phase_counts.get("calibration") == expected_clock_steps
            and phase_counts.get("reconstruction")
            == strict_counts.get("reconstruction", expected_clock_steps)
            and phase_counts.get("scene_tail", 0)
            == strict_counts.get("scene_tail", 0)
            and densification.get("calibration") == 0
            and sh_updates.get("calibration") == 0)
    if baseline is not None:
        checks["baseline_reference"] = baseline_valid

    audit = {
        "schema": "covers_self_calibrating_global_clock_gate",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "arm": args.arm,
        "checks": checks,
        "mechanical": bool(args.mechanical),
        "strict_scene_freeze": bool(args.strict_scene_freeze),
        "strict_step_budget_v2": strict_step_budget_v2,
        "expected_calibration_steps": strict_counts["calibration"],
        "expected_scene_steps": strict_counts["scene"],
        "iterations": report.get("iterations"),
        "expected_shift_frames": expected_shift,
        "clock_accuracy_mode": (
            "baseline_residual" if baseline is not None else "absolute"),
        "baseline_gate": ({
            "path": str(args.baseline_gate),
            "sha256": sha256_file(args.baseline_gate),
            "final_offset_frames": baseline_offset,
        } if baseline is not None else None),
        "final_offset_frames": report.get("final_offset_frames"),
        "residual_offset_frames": (
            float(report["final_offset_frames"]) - baseline_offset
            if finite(report.get("final_offset_frames")) else None),
        "final_drift_frames": internal.get("learned_drift_frames"),
        "final_endpoint_offsets_frames": endpoints,
        "profile_best_offsets_frames": profile_bests,
        "profile_residual_offsets_frames": profile_values,
        "agreeing_profile_count": agreeing_profiles,
        "final_gaussian_count": point_count,
        "latest_route": route[-1] if route else None,
        "deformation_sha256": deformation_hash,
        "teacher_deformation_sha256": teacher_deformation_hash,
        "deformation_optimizer": deformation_optimizer,
        "support_contract_sha256": sha256_file(args.support_contract),
    }
    path = args.output / "gate_audit.json"
    if path.exists():
        raise RuntimeError(f"Refusing existing audit: {path}")
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print("SELF_CALIBRATING_CLOCK_GATE_" + audit["status"] + " "
          + json.dumps(audit, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(9)


if __name__ == "__main__":
    main()
