#!/usr/bin/env python3
"""Fail-closed Gate and 30k audit for the locked shift20 ablations."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch


MODES = {
    "fixed_clock": {"arm": "pose_only", "clock": False, "pose": True},
    "total_gradient": {"arm": "full", "clock": True, "pose": True},
    "frozen_pose": {"arm": "time_only", "clock": True, "pose": False},
}
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--mode", required=True, choices=tuple(MODES))
    parser.add_argument("--stage", required=True, choices=("gate", "training"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--launcher-log", required=True, type=Path)
    parser.add_argument("--expected-iterations", required=True, type=int)
    parser.add_argument("--support-contract", required=True, type=Path)
    args = parser.parse_args()

    expected = MODES[args.mode]
    expected_iterations = int(args.expected_iterations)
    if ((args.stage == "gate" and expected_iterations != 1000)
            or (args.stage == "training" and expected_iterations != 30000)):
        raise RuntimeError("Ablation audit iteration contract mismatch")

    support_contract = json.loads(args.support_contract.read_text())
    support_names = list(support_contract.get("support_names", []))
    training_support_names = list(
        support_contract.get("training_support_names", []))
    support_count = int(support_contract.get("support_count", 0))
    evaluation_frame_start = int(support_contract.get(
        "evaluation_frame_start", -1))
    evaluation_frame_end = int(support_contract.get(
        "evaluation_frame_end_exclusive", -1))
    support_digest = hashlib.sha256(
        ("\n".join(support_names) + "\n").encode("utf-8")).hexdigest()
    training_support_digest = hashlib.sha256(
        ("\n".join(training_support_names) + "\n").encode(
            "utf-8")).hexdigest()
    if (support_count != 244 or len(support_names) != support_count
            or len(training_support_names) != support_count
            or evaluation_frame_start != 1 or evaluation_frame_end != 245
            or support_digest != support_contract.get("support_names_sha256")
            or training_support_digest
            != support_contract.get("training_support_names_sha256")):
        raise RuntimeError("Invalid locked shift20 support contract")

    report = json.loads(
        (args.output / "stage2_training_result.json").read_text())
    log = args.log.read_text(errors="replace")
    launcher_log = args.launcher_log.read_text(errors="replace")
    contract = report.get("joint_clock_contract") or {}
    internal = report.get("in_process_internal_clock") or {}
    counts = report.get("optimizer_update_counts") or {}
    support = report.get("dual_support") or {}

    ablation_rows = rows(log, "SHIFT20_ABLATION_CONTRACT")
    support_states = rows(log, "STAGE2_RECONSTRUCTION_SUPPORT_STATE")
    preflights = rows(log, "STAGE2_SUPPORT_TRANSITION_PREFLIGHT")
    switches = rows(log, "STAGE2_RECONSTRUCTION_SUPPORT_SWITCH")
    freezes = rows(log, "SELF_CALIBRATING_CLOCK_FREEZE")
    temporal_steps = rows(log, "STAGE2_TEMPORAL_STEP")
    route = rows(log, "MODALITY_ROUTE_STATS")
    gradients = rows(log, "MODALITY_GRADIENT_STATS")
    capacity = rows(log, "GAUSSIAN_CAPACITY_CONFIG")
    fastpath = rows(launcher_log, "A5000_FASTPATH_CONFIG")

    point_cloud = (args.output / "point_cloud"
                   / f"iteration_{expected_iterations}" / "point_cloud.ply")
    deformation = (args.output / "point_cloud"
                   / f"iteration_{expected_iterations}" / "deformation.pth")
    checkpoint = args.output / f"chkpnt{expected_iterations}.pth"
    point_count = ply_vertex_count(point_cloud) if point_cloud.is_file() else None
    deformation_hash = sha256(deformation) if deformation.is_file() else None
    teacher_deformation_hash = (report.get("teacher_hashes_before") or {}).get(
        "deformation.pth")
    deformation_optimizer = (
        deformation_optimizer_audit(checkpoint)
        if checkpoint.is_file() else {})

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
        for row in gradients)
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
        and "MEMORY_SAFE_CACHE_RELEASE" not in log
        and "MEMORY_SAFE_CLOCK_DETACH" not in log)

    expected_pose_count = 2 if expected["pose"] else 0
    common_checks = {
        "ablation_contract": (
            len(ablation_rows) == 1
            and ablation_rows[0].get("mode") == args.mode
            and ablation_rows[0].get("arm") == expected["arm"]
            and ablation_rows[0].get("clock_enabled") is expected["clock"]
            and ablation_rows[0].get("thermal_pose_enabled") is expected["pose"]
            and ablation_rows[0].get("shift_truth_used_for_optimization") is False),
        "configuration": (
            report.get("ablation_mode") == args.mode
            and report.get("arm") == expected["arm"]
            and report.get("observed_dataset_frame_shift") == 20
            and report.get("temporal_alignment_enabled") is expected["clock"]
            and report.get("thermal_pose_enabled") is expected["pose"]
            and report.get("thermal_pose_unique_rotation_parameters")
            == expected_pose_count
            and report.get("thermal_pose_unique_translation_parameters")
            == expected_pose_count),
        "iterations": report.get("iterations") == expected_iterations,
        "teacher_immutable": (
            report.get("teacher_hashes_before")
            == report.get("teacher_hashes_after")),
        "optimizer_budget": (
            counts.get("gaussian_optimizer_steps") == expected_iterations
            and counts.get("target_gaussian_optimizer_steps")
            == expected_iterations
            and counts.get("last_gaussian_step_iteration")
            == expected_iterations),
        "deformation_trainable": (
            deformation_hash is not None
            and teacher_deformation_hash is not None
            and deformation_hash != teacher_deformation_hash
            and deformation_optimizer.get("iteration") == expected_iterations
            and deformation_optimizer.get("parameter_count") == 65
            and deformation_optimizer.get("state_count") == 65
            and deformation_optimizer.get("finite") is True
            and deformation_optimizer.get("nonzero") is True),
        "support_initial_state": (
            len(support_states) == 1
            and support_states[0].get("configured") is False
            and support_states[0].get("current_camera_count") == 218
            and support_states[0].get(
                "expected_reconstruction_camera_count") == support_count
            and support_states[0].get("ablation_mode") == args.mode),
        "support_transition_preflight": (
            len(preflights) == 1
            and preflights[0].get("schema")
            == "covers_stage2_support_transition_preflight"
            and preflights[0].get("immutable_camera_snapshot") is True
            and preflights[0].get("all_trajectory_camera_count") == 266
            and preflights[0].get("calibration_camera_count") == 218
            and preflights[0].get("reconstruction_camera_count")
            == support_count
            and preflights[0].get("reconstruction_cameras_by_side")
            == {"left": 122, "right": 122}
            and preflights[0].get("training_frame_start") == 0
            and preflights[0].get("training_frame_end_exclusive") == 244
            and preflights[0].get("evaluation_frame_start") == 1
            and preflights[0].get("evaluation_frame_end_exclusive") == 245
            and preflights[0].get("selection_policy")
            == "pre_registered_train_frames_0_243"
            and preflights[0].get("support_contract_sha256")
            == sha256(args.support_contract)
            and preflights[0].get("support_contract_names_sha256")
            == support_contract.get("training_support_names_sha256")
            and preflights[0].get("selected_names_sha256")
            == support_contract.get("training_support_names_sha256")
            and preflights[0].get("shift_truth_used_for_selection") is False
            and preflights[0].get("learned_clock_used_for_selection") is False
            and report.get("support_transition_preflight") == preflights[0]),
        "route_contract": route_ok,
        "both_modal_gradients": modal_grad_ok,
        "capacity_budget": capacity_ok,
        "fastpath": fastpath_ok,
        "clean_log": (
            not any(token in log for token in (
                "Traceback", "RuntimeError", "CUDA error", "out of memory",
                "Killed", "Non-finite"))
            and re.search(
                r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])", log,
                flags=re.IGNORECASE) is None),
    }

    if expected["clock"]:
        expected_clock_steps = min(expected_iterations, 15000)
        gradient_routing_ok = bool(temporal_steps)
        for row in temporal_steps:
            raw = row.get("raw_gradient")
            alignment = row.get("joint_alignment_gradient")
            reconstruction = row.get("reconstruction_gradient")
            if not all(finite(value) for value in (
                    raw, alignment, reconstruction)):
                gradient_routing_ok = False
                break
            expected_raw = (float(alignment) + float(reconstruction)
                            if args.mode == "total_gradient"
                            else float(alignment))
            if abs(float(raw) - expected_raw) > 1.0e-6:
                gradient_routing_ok = False
                break
        if args.mode == "total_gradient":
            gradient_routing_ok = gradient_routing_ok and any(
                abs(float(row.get("reconstruction_gradient", 0.0))) > 1.0e-8
                for row in temporal_steps)
        clock_checks = {
            "raw_zero": (
                internal.get("initial_raw") == 0.0
                and internal.get("initial_offset_frames") == 0.0
                and internal.get("shift_truth_input") is False
                and internal.get("pretraining_offset_assignment") is False
                and internal.get("pre_reconstruction_clock_phase") is False),
            "clock_finite": finite(report.get("final_offset_frames")),
            "clock_optimizer": (
                counts.get("joint_clock_optimizer_steps")
                == expected_clock_steps
                and counts.get("joint_clock_nonzero_offset_gradients", 0) > 0
                and internal.get("optimizer_steps") == expected_clock_steps),
            "clock_gradient_mode": (
                contract.get("reconstruction_gradient_to_clock")
                is (args.mode == "total_gradient")
                and contract.get("clock_gradient_mode")
                == ("total_loss_routed" if args.mode == "total_gradient"
                    else "alignment_only")
                and gradient_routing_ok),
            "clock_accuracy": (
                True if args.mode == "total_gradient" else
                abs(float(report.get("final_offset_frames")) - 20.0) <= 0.25),
        }
    else:
        clock_checks = {
            "clock_disabled": (
                not internal and not contract
                and counts.get("joint_clock_optimizer_steps", 0) == 0
                and report.get("temporal_alignment_enabled") is False),
        }

    if args.stage == "gate":
        stage_checks = {
            "no_early_support_switch": not switches,
            "no_early_clock_freeze": not freezes,
            "calibration_support": (
                report.get("train_camera_count") == 218
                and report.get("reconstruction_camera_count", 218) == 218),
        }
    else:
        expected_support_schema = (
            "covers_fixed_clock_dual_support"
            if args.mode == "fixed_clock"
            else "covers_blind_global_dual_support")
        stage_checks = {
            "support_switch": (
                len(switches) == 1
                and switches[0].get("iteration") == 15001
                and switches[0].get("calibration_camera_count") == 218
                and switches[0].get("reconstruction_camera_count")
                == support_count
                and switches[0].get("ablation_mode") == args.mode),
            "support_contract": (
                report.get("train_camera_count") == 218
                and report.get("reconstruction_camera_count") == support_count
                and support.get("schema") == expected_support_schema
                and support.get("calibration_camera_count") == 218
                and support.get("reconstruction_camera_count") == support_count
                and support.get("reconstruction_cameras_by_side")
                == {"left": 122, "right": 122}
                and support.get("selection_policy")
                == "pre_registered_train_frames_0_243"
                and support.get("evaluation_frame_start") == 1
                and support.get("evaluation_frame_end_exclusive") == 245
                and support.get("shift_truth_used_for_selection") is False
                and support.get("learned_clock_used_for_selection") is False
                and support.get("support_contract_sha256")
                == sha256(args.support_contract)
                and support.get("support_contract_names_sha256")
                == support_contract.get("training_support_names_sha256")
                and support.get("selected_names_sha256")
                == support_contract.get("training_support_names_sha256")),
            "clock_freeze": (
                len(freezes) == 1
                and freezes[0].get("freeze_after_iteration") == 15000
                and freezes[0].get("requires_grad") is False)
            if expected["clock"] else not freezes,
        }

    checks = {**common_checks, **clock_checks, **stage_checks}
    audit = {
        "schema": "covers_shift20_ablation_audit_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": args.mode,
        "stage": args.stage,
        "checks": checks,
        "iterations": expected_iterations,
        "final_offset_frames": report.get("final_offset_frames"),
        "final_gaussian_count": point_count,
        "deformation_sha256": deformation_hash,
        "teacher_deformation_sha256": teacher_deformation_hash,
        "deformation_optimizer": deformation_optimizer,
        "support_count": support_count,
    }
    filename = "gate_audit.json" if args.stage == "gate" else "training_audit.json"
    path = args.output / filename
    if path.exists():
        raise RuntimeError(f"Refusing existing audit: {path}")
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print("SHIFT20_ABLATION_" + args.stage.upper() + "_" + audit["status"]
          + " " + json.dumps(audit, sort_keys=True))
    if audit["status"] != "PASS":
        raise SystemExit(9)


if __name__ == "__main__":
    main()
