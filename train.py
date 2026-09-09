#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import hashlib
import json
import numpy as np
import random
import os
import torch
from pathlib import Path
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
from utils.extra_utils import o3d_knn, weighted_l2_loss_v2, image_sampler, calculate_distances, sample_camera
from utils.modality_routing import compute_modality_routing
from time_alignment import clock_loss
from time_alignment import clock_refinement
from time_alignment import motion_clock
from time_alignment import motion_cost_volume
from time_alignment import pose_calibration
from time_alignment import pose_math
from time_alignment import schedule
from time_alignment import temporal_consensus as temporal_consensus_module

# import lpips
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def configure_blind_reconstruction_support(scene, learned_clock_report):
    """Validate and activate a preregistered reconstruction support."""
    if os.environ.get("ED3DGS_R25_DUAL_SUPPORT") != "1":
        return None
    if not scene.temporal_alignment_enabled:
        raise RuntimeError("R25 dual support requires a learned Scene clock")
    learned_offset = float(scene.temporal_offset_frames().detach().item())
    learned_drift = float(scene.temporal_drift_frames().detach().item())
    learned_endpoints = [
        float(value.detach().item())
        for value in scene.temporal_endpoint_offsets()
    ]
    reported_offset = learned_clock_report.get("learned_offset_frames")
    if reported_offset is not None:
        reported_offset = float(reported_offset)
    if (not math.isfinite(learned_offset)
            or (reported_offset is not None
                and abs(learned_offset - reported_offset) > 1e-6)):
        raise RuntimeError("Dual-support clock state does not match Stage-A")

    contract_path_value = os.environ.get(
        "ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT", "")
    if not contract_path_value:
        raise RuntimeError("Missing preregistered reconstruction support contract")
    contract_path = Path(contract_path_value)
    contract = json.loads(contract_path.read_text())
    expected_names = list(contract.get("training_support_names", []))
    support_count = int(contract.get("support_count", 0))
    frame_start = int(contract.get("training_frame_start", 0))
    frame_end = int(contract.get(
        "training_frame_end_exclusive", frame_start + support_count))
    evaluation_frame_start = int(contract.get("evaluation_frame_start", 0))
    evaluation_frame_end = int(contract.get(
        "evaluation_frame_end_exclusive",
        evaluation_frame_start + support_count))
    expected_names_digest = hashlib.sha256(
        ("\n".join(expected_names) + "\n").encode("utf-8")).hexdigest()
    if (support_count <= 0 or support_count % 2 != 0
            or len(expected_names) != support_count
            or frame_end - frame_start != support_count
            or evaluation_frame_end - evaluation_frame_start != support_count
            or expected_names_digest
            != contract.get("training_support_names_sha256")):
        raise RuntimeError("Invalid preregistered reconstruction support contract")
    expected_per_side = support_count // 2

    bounded_candidates_by_side = {"left": [], "right": []}
    fixed_candidates_by_side = {"left": [], "right": []}
    duration_max = float(scene.maxtime) - 1.0
    camera_pool = list(getattr(
        scene, "stage2_all_train_cameras", scene.getTrainCameras()))
    for camera in camera_pool:
        side, frame = scene._camera_state_side_frame(camera)
        if side not in bounded_candidates_by_side or frame is None:
            continue
        if frame_start <= frame < frame_end:
            fixed_candidates_by_side[side].append(camera)
        query_frame = float(frame) + float(
            camera.effective_temporal_offset_frames().detach().item())
        pose_min = float(camera.temporal_pose_frames[0].detach().item())
        pose_max = float(camera.temporal_pose_frames[-1].detach().item())
        if (0.0 <= query_frame <= duration_max
                and pose_min <= query_frame <= pose_max):
            bounded_candidates_by_side[side].append(camera)

    bounded_names = {
        camera.image_name
        for cameras in bounded_candidates_by_side.values()
        for camera in cameras
    }
    selected = []
    side_counts = {"left": 0, "right": 0}
    candidate_counts = {
        side: len(fixed_candidates_by_side[side]) for side in side_counts}
    bounded_counts = {
        side: len(bounded_candidates_by_side[side]) for side in side_counts}
    for side in ("left", "right"):
        candidates = sorted(
            fixed_candidates_by_side[side],
            key=lambda camera: scene._camera_state_side_frame(camera)[1])
        if len(candidates) != expected_per_side:
            raise RuntimeError(
                "Preregistered reconstruction support does not match fixed frames "
                f"on {side}: {len(candidates)}")
        invalid = [camera.image_name for camera in candidates
                   if camera.image_name not in bounded_names]
        if invalid:
            raise RuntimeError(
                "Learned clock leaves the preregistered support domain: "
                f"{invalid[:3]}")
        selected.extend(candidates)
        side_counts[side] = expected_per_side
    trimmed_counts = {
        side: bounded_counts[side] - candidate_counts[side]
        for side in side_counts}
    if (sum(candidate_counts.values()) != support_count
            or any(count < 0 for count in trimmed_counts.values())):
        raise RuntimeError(
            "Preregistered reconstruction support is not valid: "
            f"candidates={candidate_counts}")
    selected.sort(
        key=lambda camera: scene._camera_state_side_frame(camera)[1])
    selection_policy = (
        f"pre_registered_train_frames_{frame_start}_{frame_end - 1}")

    calibration_cameras = list(scene.stage2_training_cameras)
    if (len(selected) != support_count
            or side_counts != {
                "left": expected_per_side, "right": expected_per_side}
            or len(selected) <= len(calibration_cameras)):
        raise RuntimeError(
            "Blind reconstruction support does not match the locked full support: "
            f"selected={len(selected)} sides={side_counts}")
    expected_shift = int(os.environ.get("ED3DGS_THERMAL_FRAME_SHIFT", "0"))
    observed_shifts = sorted({int(camera.thermal_frame_shift)
                              for camera in selected})
    if observed_shifts != [expected_shift]:
        raise RuntimeError(
            "Blindly selected support includes synthetic boundary fallback: "
            f"{observed_shifts}")

    selected_names = [
        Path(str(camera.image_name)).stem + ".png" for camera in selected]
    if selected_names != expected_names:
        raise RuntimeError(
            "Selected reconstruction cameras differ from the preregistered names")
    report = {
        "schema": "covers_blind_global_dual_support",
        "selection_inputs": [
            "pre_registered_support_contract", "frame_coordinate"],
        "shift_truth_used_for_selection": False,
        "learned_clock_used_for_selection": False,
        "learned_clock_used_for_validity_only": True,
        "support_contract_path": str(contract_path),
        "support_contract_sha256": hashlib.sha256(
            contract_path.read_bytes()).hexdigest(),
        "support_contract_names_sha256": contract.get(
            "training_support_names_sha256"),
        "evaluation_support_names_sha256": contract.get(
            "support_names_sha256"),
        "training_frame_start": frame_start,
        "training_frame_end_exclusive": frame_end,
        "evaluation_frame_start": evaluation_frame_start,
        "evaluation_frame_end_exclusive": evaluation_frame_end,
        "learned_offset_frames": learned_offset,
        "learned_drift_frames": learned_drift,
        "learned_endpoint_offsets_frames": learned_endpoints,
        "calibration_camera_count": len(calibration_cameras),
        "reconstruction_camera_count": len(selected),
        "reconstruction_cameras_by_side": side_counts,
        "candidate_camera_count": sum(candidate_counts.values()),
        "candidate_cameras_by_side": candidate_counts,
        "bounded_candidate_camera_count": sum(bounded_counts.values()),
        "bounded_candidates_by_side": bounded_counts,
        "trimmed_candidate_cameras_by_side": trimmed_counts,
        "selection_policy": selection_policy,
        "selected_observation_shifts_audit_only": observed_shifts,
        "selected_names": selected_names,
        "selected_names_sha256": expected_names_digest,
    }
    scene.stage2_reconstruction_cameras = selected
    scene.stage2_reconstruction_support_report = report
    print("STAGE2_BLIND_DUAL_SUPPORT " + json.dumps(report, sort_keys=True))
    return report


def configure_fixed_clock_reconstruction_support(scene):
    """Activate the same preregistered support with the clock fixed at zero."""
    if os.environ.get("ED3DGS_R25_DUAL_SUPPORT") != "1":
        return None
    if scene.temporal_alignment_enabled:
        raise RuntimeError(
            "Fixed-clock support requires temporal alignment to be disabled")
    contract_path_value = os.environ.get(
        "ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT", "")
    if not contract_path_value:
        raise RuntimeError("Missing preregistered reconstruction support contract")
    contract_path = Path(contract_path_value)
    contract = json.loads(contract_path.read_text())
    expected_names = list(contract.get("training_support_names", []))
    support_count = int(contract.get("support_count", 0))
    frame_start = int(contract.get("training_frame_start", 0))
    frame_end = int(contract.get(
        "training_frame_end_exclusive", frame_start + support_count))
    evaluation_frame_start = int(contract.get("evaluation_frame_start", 0))
    evaluation_frame_end = int(contract.get(
        "evaluation_frame_end_exclusive",
        evaluation_frame_start + support_count))
    expected_names_digest = hashlib.sha256(
        ("\n".join(expected_names) + "\n").encode("utf-8")).hexdigest()
    if (support_count <= 0 or support_count % 2 != 0
            or len(expected_names) != support_count
            or frame_end - frame_start != support_count
            or evaluation_frame_end - evaluation_frame_start != support_count
            or expected_names_digest
            != contract.get("training_support_names_sha256")):
        raise RuntimeError("Invalid preregistered fixed-clock support contract")

    camera_pool = list(getattr(
        scene, "stage2_all_train_cameras", scene.getTrainCameras()))
    by_name = {
        Path(str(camera.image_name)).stem + ".png": camera
        for camera in camera_pool}
    if len(by_name) != len(camera_pool):
        raise RuntimeError("Duplicate training camera names")
    missing = [name for name in expected_names if name not in by_name]
    if missing:
        raise RuntimeError(
            f"Preregistered fixed-clock cameras are missing: {missing[:3]}")
    selected = [by_name[name] for name in expected_names]
    side_counts = {"left": 0, "right": 0}
    selected_frames = []
    for camera in selected:
        side, frame = scene._camera_state_side_frame(camera)
        if side not in side_counts or frame is None:
            raise RuntimeError(
                f"Invalid preregistered fixed-clock camera: {camera.image_name}")
        side_counts[side] += 1
        selected_frames.append(int(frame))
    expected_per_side = support_count // 2
    if (side_counts != {"left": expected_per_side, "right": expected_per_side}
            or min(selected_frames) != frame_start
            or max(selected_frames) != frame_end - 1
            or sorted(selected_frames) != list(range(frame_start, frame_end))):
        raise RuntimeError(
            "Fixed-clock support does not match the preregistered frame range")

    calibration_cameras = list(scene.stage2_training_cameras)
    if len(selected) <= len(calibration_cameras):
        raise RuntimeError(
            "Fixed-clock reconstruction support must exceed calibration support")
    observed_shifts = sorted({
        int(camera.thermal_frame_shift) for camera in selected})
    expected_shift = int(os.environ.get("ED3DGS_THERMAL_FRAME_SHIFT", "0"))
    if observed_shifts != [expected_shift]:
        raise RuntimeError(
            "Fixed-clock support includes a synthetic boundary fallback")

    selection_policy = (
        f"pre_registered_train_frames_{frame_start}_{frame_end - 1}")
    report = {
        "schema": "covers_fixed_clock_dual_support",
        "selection_inputs": ["pre_registered_support_contract"],
        "shift_truth_used_for_selection": False,
        "learned_clock_used_for_selection": False,
        "learned_clock_used_for_validity_only": False,
        "fixed_offset_frames": 0.0,
        "support_contract_path": str(contract_path),
        "support_contract_sha256": hashlib.sha256(
            contract_path.read_bytes()).hexdigest(),
        "support_contract_names_sha256": contract.get(
            "training_support_names_sha256"),
        "evaluation_support_names_sha256": contract.get(
            "support_names_sha256"),
        "training_frame_start": frame_start,
        "training_frame_end_exclusive": frame_end,
        "evaluation_frame_start": evaluation_frame_start,
        "evaluation_frame_end_exclusive": evaluation_frame_end,
        "calibration_camera_count": len(calibration_cameras),
        "reconstruction_camera_count": len(selected),
        "reconstruction_cameras_by_side": side_counts,
        "candidate_camera_count": len(selected),
        "candidate_cameras_by_side": side_counts,
        "selection_policy": selection_policy,
        "selected_observation_shifts_audit_only": observed_shifts,
        "selected_names": expected_names,
        "selected_names_sha256": hashlib.sha256(
            ("\n".join(expected_names) + "\n").encode("utf-8")).hexdigest(),
    }
    scene.stage2_reconstruction_cameras = selected
    scene.stage2_reconstruction_support_report = report
    print("STAGE2_FIXED_CLOCK_DUAL_SUPPORT "
          + json.dumps(report, sort_keys=True))
    return report


def preflight_reconstruction_support_snapshot(scene):
    """Validate the delayed Stage2 support switch before iteration one."""
    if os.environ.get("ED3DGS_R25_DUAL_SUPPORT") != "1":
        return None
    contract_path_value = os.environ.get(
        "ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT", "")
    if not contract_path_value:
        raise RuntimeError("Missing preregistered reconstruction support contract")
    contract_path = Path(contract_path_value)
    contract = json.loads(contract_path.read_text())
    expected_trajectory_count = int(contract.get(
        "trajectory_frame_count", 266))
    expected_calibration_count = int(contract.get(
        "calibration_support_count", 218))
    camera_snapshot = getattr(scene, "stage2_all_train_cameras", None)
    if not isinstance(camera_snapshot, tuple):
        raise RuntimeError(
            "Dual-support preflight requires an immutable full-camera snapshot")
    camera_pool = list(camera_snapshot)
    if len(camera_pool) != expected_trajectory_count:
        raise RuntimeError(
            "Dual-support preflight trajectory-camera mismatch: "
            f"expected={expected_trajectory_count} actual={len(camera_pool)}")
    calibration_cameras = list(getattr(
        scene, "stage2_training_cameras", scene.getTrainCameras()))
    if len(calibration_cameras) != expected_calibration_count:
        raise RuntimeError(
            "Dual-support preflight calibration-camera mismatch: "
            f"expected={expected_calibration_count} "
            f"actual={len(calibration_cameras)}")
    support_names = list(contract.get("training_support_names", []))
    support_count = int(contract.get("support_count", 0))
    frame_start = int(contract.get("training_frame_start", 0))
    frame_end = int(contract.get(
        "training_frame_end_exclusive", frame_start + support_count))
    evaluation_frame_start = int(contract.get("evaluation_frame_start", 0))
    evaluation_frame_end = int(contract.get(
        "evaluation_frame_end_exclusive",
        evaluation_frame_start + support_count))
    support_names_digest = hashlib.sha256(
        ("\n".join(support_names) + "\n").encode("utf-8")).hexdigest()
    if (support_count <= len(calibration_cameras)
            or support_count % 2 != 0
            or len(support_names) != support_count
            or frame_end - frame_start != support_count
            or evaluation_frame_end - evaluation_frame_start != support_count
            or support_names_digest
            != contract.get("training_support_names_sha256")):
        raise RuntimeError("Invalid preregistered reconstruction support contract")

    cameras_by_name = {
        Path(str(camera.image_name)).stem + ".png": camera
        for camera in camera_pool}
    if len(cameras_by_name) != len(camera_pool):
        raise RuntimeError("Duplicate camera names in the full-camera snapshot")
    missing = [name for name in support_names if name not in cameras_by_name]
    if missing:
        raise RuntimeError(
            "Preregistered reconstruction cameras are absent at startup: "
            f"{missing[:3]}")
    selected = [cameras_by_name[name] for name in support_names]
    side_counts = {"left": 0, "right": 0}
    selected_frames = []
    for camera in selected:
        side, frame = scene._camera_state_side_frame(camera)
        if side not in side_counts or frame is None:
            raise RuntimeError(
                f"Invalid preregistered camera at startup: {camera.image_name}")
        side_counts[side] += 1
        selected_frames.append(int(frame))
    expected_per_side = support_count // 2
    if (side_counts != {
            "left": expected_per_side, "right": expected_per_side}
            or selected_frames != list(range(frame_start, frame_end))):
        raise RuntimeError(
            "Preregistered reconstruction support does not match its frame range")

    report = {
        "schema": "covers_stage2_support_transition_preflight",
        "immutable_camera_snapshot": True,
        "all_trajectory_camera_count": len(camera_pool),
        "calibration_camera_count": len(calibration_cameras),
        "reconstruction_camera_count": len(selected),
        "reconstruction_cameras_by_side": side_counts,
        "training_frame_start": frame_start,
        "training_frame_end_exclusive": frame_end,
        "evaluation_frame_start": evaluation_frame_start,
        "evaluation_frame_end_exclusive": evaluation_frame_end,
        "selection_policy": (
            f"pre_registered_train_frames_{frame_start}_{frame_end - 1}"),
        "support_contract_sha256": hashlib.sha256(
            contract_path.read_bytes()).hexdigest(),
        "support_contract_names_sha256": contract.get(
            "training_support_names_sha256"),
        "evaluation_support_names_sha256": contract.get(
            "support_names_sha256"),
        "selected_names_sha256": support_names_digest,
        "shift_truth_used_for_selection": False,
        "learned_clock_used_for_selection": False,
    }
    scene.stage2_support_transition_preflight = report
    print("STAGE2_SUPPORT_TRANSITION_PREFLIGHT "
          + json.dumps(report, sort_keys=True))
    return report


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter,timer, start_time):
    first_iter = 0
    ablation_mode = os.environ.get("ED3DGS_SHIFT20_ABLATION_MODE", "")
    if ablation_mode not in {"", "fixed_clock", "total_gradient", "frozen_pose"}:
        raise ValueError(f"Invalid shift20 ablation mode: {ablation_mode!r}")
    fixed_clock_ablation = ablation_mode == "fixed_clock"
    total_gradient_ablation = ablation_mode == "total_gradient"
    frozen_pose_ablation = ablation_mode == "frozen_pose"
    v34_block_calibration = (
        os.environ.get("ED3DGS_BLOCK_CALIBRATION_V34") == "1")
    strict_scene_freeze = (
        os.environ.get("ED3DGS_STRICT_SCENE_FREEZE_V1") == "1")
    strict_step_budget_v2 = bool(getattr(
        opt, "strict_step_budget_v2", False)) or (
            os.environ.get("ED3DGS_STRICT_STEP_BUDGET_V2", "0") == "1")
    clock_target_steps = int(getattr(
        opt, "temporal_offset_target_steps", -1))
    if fixed_clock_ablation:
        clock_target_steps = 0
    pose_target_steps = int(getattr(
        opt, "thermal_pose_target_steps", -1))
    geoflow_clock = (
        os.environ.get("ED3DGS_GEOFLOW_SOFTVOLUME_V1") == "1")
    zero_init_routed_clock = (
        os.environ.get("ED3DGS_ZERO_INIT_ROUTED_CLOCK_V1") == "1")
    stable_modality_routing = (
        os.environ.get("ED3DGS_MODALITY_ROUTING_STABLE_V1") == "1")
    clock_total_gradient = (
        os.environ.get("ED3DGS_CLOCK_TOTAL_GRAD_V1") == "1")
    ema_modal_loss = (
        os.environ.get("ED3DGS_EMA_MODAL_LOSS_V1") == "1")
    memory_safe_backward = (
        os.environ.get("ED3DGS_MEMORY_SAFE_BACKWARD_V1") == "1")
    memory_safe_gaussian_threshold = int(os.environ.get(
        "ED3DGS_MEMORY_SAFE_GAUSSIAN_THRESHOLD_V1", "160000"))
    if memory_safe_gaussian_threshold < 1:
        raise ValueError("memory-safe Gaussian threshold must be positive")
    temporal_consensus_enabled = (
        os.environ.get("ED3DGS_TEMPORAL_CONSENSUS_V1") == "1")
    self_calibrating_clock = (
        os.environ.get("ED3DGS_SELF_CALIBRATING_CLOCK_STUDY") == "1")
    zero_start_clock_phase_enabled = (
        os.environ.get("ED3DGS_ZERO_START_CLOCK_PHASE") == "1")
    motion_clock_continuation = (
        os.environ.get("ED3DGS_MOTION_CLOCK_CONTINUATION") == "1")
    zero_start_clock_phase_steps = int(os.environ.get(
        "ED3DGS_ZERO_START_CLOCK_PHASE_STEPS", "1000"))
    if motion_clock_continuation and not zero_start_clock_phase_enabled:
        raise ValueError(
            "Motion-clock continuation requires the zero-start phase")
    if motion_clock_continuation and temporal_consensus_enabled:
        raise ValueError(
            "Motion-clock continuation and temporal consensus are exclusive")
    if self_calibrating_clock:
        if motion_clock_continuation or zero_start_clock_phase_enabled:
            raise ValueError(
                "Self-calibrating clock forbids a pre-reconstruction clock phase")
        if not temporal_consensus_enabled:
            raise ValueError(
                "Self-calibrating clock requires frozen-teacher temporal consensus")
        if scene.temporal_affine_clock_enabled:
            raise ValueError(
                "Covers self-calibration supports one identifiable global offset")
        if clock_total_gradient and not total_gradient_ablation:
            raise ValueError(
                "Self-calibrating clock forbids reconstruction gradients to clock")
    if total_gradient_ablation and not clock_total_gradient:
        raise ValueError(
            "Total-gradient ablation requires reconstruction gradients to clock")
    if fixed_clock_ablation and (
            scene.temporal_alignment_enabled or not scene.has_thermal_pose):
        raise ValueError(
            "Fixed-clock ablation requires clock disabled and Thermal pose enabled")
    if frozen_pose_ablation and (
            not scene.temporal_alignment_enabled or scene.has_thermal_pose):
        raise ValueError(
            "Frozen-pose ablation requires clock enabled and Thermal pose disabled")
    if (zero_start_clock_phase_enabled
            and not motion_clock_continuation
            and zero_start_clock_phase_steps != 1000):
        raise ValueError(
            "The zero-start clock phase requires exactly 1000 optimizer steps")
    temporal_consensus_mode = os.environ.get(
        "ED3DGS_TEMPORAL_CONSENSUS_MODE", "consensus_only")
    if temporal_consensus_mode not in {"consensus_only", "consensus_ngf"}:
        raise ValueError(
            f"Invalid temporal consensus mode: {temporal_consensus_mode!r}")
    joint_clock_loss_enabled = bool(getattr(
        opt, "joint_clock_loss_enabled", False))
    joint_clock_loss_weight = float(getattr(
        opt, "joint_clock_loss_weight", 0.0))
    joint_clock_loss_start_iter = int(getattr(
        opt, "joint_clock_loss_start_iter", 0))
    joint_clock_loss_ramp_iters = int(getattr(
        opt, "joint_clock_loss_ramp_iters", 0))
    joint_clock_loss_interval = int(getattr(
        opt, "joint_clock_loss_interval", 1))
    matched_arm = os.environ.get("ED3DGS_R25_MATCHED_ARM", "")
    fourarm = os.environ.get("ED3DGS_R25_FOURARM", "")
    valid_fourarms = {"baseline", "time_only", "pose_only", "full"}
    if fourarm and fourarm not in valid_fourarms:
        raise ValueError(f"Invalid ED3DGS_R25_FOURARM: {fourarm!r}")
    matched_clock_only_block = (
        (matched_arm == "shifted_r25_clock" or fourarm == "time_only")
        and v34_block_calibration)
    full_scene_steps = bool(getattr(
        opt, "scene_optimizer_every_iteration", False))
    scene_optimizer_target_steps = int(getattr(
        opt, "scene_optimizer_target_steps", 0))
    strict_step_budget_v2 = bool(getattr(
        opt, "strict_step_budget_v2", False)) or (
            os.environ.get("ED3DGS_STRICT_STEP_BUDGET_V2", "0") == "1")
    strict_calibration_steps = int(getattr(
        opt, "strict_calibration_steps", -1))
    clock_target_steps = int(getattr(
        opt, "temporal_offset_target_steps", -1))
    if fixed_clock_ablation:
        clock_target_steps = 0
    pose_target_steps = int(getattr(
        opt, "thermal_pose_target_steps", -1))
    if strict_step_budget_v2 and not strict_scene_freeze:
        raise ValueError("Strict step-budget v2 requires strict scene freeze")
    if strict_step_budget_v2 and (strict_calibration_steps <= 0
                                   or scene_optimizer_target_steps <= 0):
        raise ValueError("Strict step-budget v2 requires explicit optimizer budgets")
    if strict_scene_freeze:
        if not v34_block_calibration:
            raise ValueError(
                "Strict scene freeze requires v34 block calibration")
        if full_scene_steps:
            raise ValueError(
                "Strict scene freeze forbids every-iteration Scene steps")
    if full_scene_steps:
        if not (matched_clock_only_block or fourarm in valid_fourarms):
            raise ValueError(
                "full Scene-step schedule requires a released R25 arm")
        if scene_optimizer_target_steps != int(train_iter):
            raise ValueError(
                "scene_optimizer_target_steps must equal the main iteration budget")
    if joint_clock_loss_enabled:
        valid_clock_arm = (
            fourarm == "full"
            or (fourarm == "time_only" and frozen_pose_ablation))
        if not (geoflow_clock and valid_clock_arm
                and v34_block_calibration
                and scene.temporal_alignment_enabled):
            raise ValueError(
                "joint clock loss requires the released GeoFlow full arm")
        if joint_clock_loss_weight <= 0.0:
            raise ValueError("joint_clock_loss_weight must be positive")
        if joint_clock_loss_start_iter < 1:
            raise ValueError("joint_clock_loss_start_iter must be positive")
        if joint_clock_loss_ramp_iters < 1:
            raise ValueError("joint_clock_loss_ramp_iters must be positive")
        if joint_clock_loss_interval != 1:
            raise ValueError("joint clock loss must run on every training forward")
        if zero_init_routed_clock and clock_loss.loss_name() != (
                "multiscale_observable_polarity_invariant_NGF"):
            raise ValueError(
                "zero-init routed clock requires the routed_ngf loss")
        if temporal_consensus_enabled and not zero_init_routed_clock:
            raise ValueError(
                "temporal consensus requires the zero-init routed clock")
        alignment_name = (
            "continuous_global_motion_cost_volume"
            if self_calibrating_clock else
            "training_sequence_motion_correlation_continuation"
            if motion_clock_continuation
            else "temporal_activity_consensus_v1"
            if temporal_consensus_enabled
               and temporal_consensus_mode == "consensus_only"
            else "0.5*temporal_activity_consensus_v1+0.5*"
                 "multiscale_observable_polarity_invariant_NGF"
            if temporal_consensus_enabled
            else clock_loss.loss_name())
        print("JOINT_CLOCK_LOSS_CONTRACT " + json.dumps({
            "gradient_owners": ["scene_clock"],
            "initialization": (
                "raw_zero_no_pretraining_estimate"
                if zero_init_routed_clock else "geoflow_softvolume_v1"),
            "interval": joint_clock_loss_interval,
            "loss": alignment_name,
            "ramp_iters": joint_clock_loss_ramp_iters,
            "reconstruction_gradient_to_clock": bool(
                clock_total_gradient or not zero_init_routed_clock),
            "clock_gradient_mode": (
                "pre_reconstruction_motion_continuation"
                if motion_clock_continuation
                else "total_loss_routed" if clock_total_gradient
                else "alignment_only" if zero_init_routed_clock
                else "legacy_total"),
            "stable_modality_routing": stable_modality_routing,
            "ema_modal_loss": ema_modal_loss,
            "memory_safe_backward": memory_safe_backward,
            "memory_safe_gaussian_threshold": memory_safe_gaussian_threshold,
            "start_iteration": joint_clock_loss_start_iter,
            "weight": joint_clock_loss_weight,
            "temporal_consensus_enabled": temporal_consensus_enabled,
            "temporal_consensus_mode": (
                temporal_consensus_mode if temporal_consensus_enabled else None),
            "motion_clock_continuation": motion_clock_continuation,
            "self_calibrating_clock": self_calibrating_clock,
            "clock_parameters": (["offset_raw", "drift_raw"]
                                 if scene.temporal_affine_clock_enabled
                                 else ["offset_raw"]),
            "ablation_mode": ablation_mode or None,
        }, sort_keys=True))
    elif zero_init_routed_clock:
        raise ValueError("zero-init routed clock requires joint clock loss")

    if opt.thermal_only and opt.rgb_only_teacher:
        raise ValueError("thermal_only and rgb_only_teacher are mutually exclusive")
    if opt.rgb_only_teacher:
        for name, parameter in gaussians._deformation.named_parameters():
            if "thermal" in name.lower():
                parameter.requires_grad_(False)
    gaussians.training_setup(opt)
    if opt.rgb_only_teacher:
        required = {
            "no_thermal_pose_opt": bool(opt.no_thermal_pose_opt),
            "enable_modality_densify_disabled": not bool(opt.enable_modality_densify),
            "temporal_alignment_disabled": not bool(opt.temporal_alignment_enabled),
            "thermal_intrinsic_prior_disabled": opt.thermal_intrinsic_prior_weight == 0,
            "thermal_loss_weight_zero": opt.thermal_loss_weight == 0,
            "thermal_feature_lr_zero": opt.thermal_feature_lr == 0,
            "thermal_opacity_lr_zero": opt.thermal_opacity_lr == 0,
            "modality_lr_zero": opt.modality_lr == 0,
            "thermal_geometry_change_disabled": not bool(hyper.change_thermal_geo),
        }
        failed = [name for name, passed in required.items() if not passed]
        if failed or scene.has_thermal_pose:
            raise RuntimeError(
                f"invalid RGB-only teacher contract: failed={failed}, "
                f"scene.has_thermal_pose={scene.has_thermal_pose}")
    # Thermal intrinsics/pose now handled by Scene optimizers (Self-Cali-GS style)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    bg_thermal = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    background_thermal = torch.tensor(bg_thermal, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    train_cams = getattr(
        scene, "stage2_reconstruction_cameras",
        getattr(scene, "stage2_training_cameras", scene.getTrainCameras()))
    video_cams = None

    num_traincams = 1
    if dataset.loader != 'nerfies': # for multi-view setting
        num_traincams = int(len(train_cams) / scene.maxtime)
    
        camera_centers = []
        for i in range(num_traincams):
            camera_centers.append(train_cams[i*scene.maxtime].camera_center.cpu().numpy())
        camera_centers = np.array(camera_centers)
        cam_dists = calculate_distances(camera_centers)
        sorted_dists = np.unique(cam_dists)
        min_dist = sorted_dists[int(sorted_dists.shape[0] * 0.5)]

        last_camera_index = 0
    
    cam_no_list = list(set(c.cam_no for c in train_cams))
    print("train cameras:", cam_no_list)
    if dataset.loader in ['nerfies']:  # single-view
        loss_list = np.zeros([num_traincams, scene.maxtime]) + 100  # pick frames that have not yet been sampled
    else:  # n3v, technicolor, etc.
        loss_list = np.zeros([max(cam_no_list) + 1, scene.maxtime])
        for c in cam_no_list:
            loss_list[c] = 100

    ssim_cnt = 0
    sampled_frame_no = None
    prev_num_pts = 0
    scene_optimizer_step_count = 0
    scene_optimizer_last_step_iteration = 0
    clock_stage2_update_count = 0
    temporal_offset_optimizer_step_count = 0
    joint_clock_optimizer_step_count = 0
    joint_clock_nonzero_alignment_gradient_count = 0
    joint_clock_nonzero_offset_gradient_count = 0
    joint_clock_nonzero_drift_gradient_count = 0
    thermal_pose_optimizer_step_count = 0
    optimizer_phase_counts = {
        "calibration": 0,
        "reconstruction": 0,
        "scene_tail": 0,
        "warmup": 0,
        "joint": 0,
    }
    scene_save_step_counts = {}
    checkpoint_step_counts = {}
    scene_backward_phase_counts = {
        "calibration": 0,
        "reconstruction": 0,
        "scene_tail": 0,
        "warmup": 0,
        "joint": 0,
    }
    densification_phase_counts = dict(scene_backward_phase_counts)
    sh_degree_update_phase_counts = dict(scene_backward_phase_counts)

    # We sort training images to sample image of the desired camera number and frame.
    if dataset.loader not in ['nerfies']:
        train_cams = sorted(train_cams, key=lambda x: (x.cam_no, x.frame_no))

    viewpoint_stack = train_cams
    reconstruction_support_configured = False
    baseline_arm = fourarm == "baseline"
    if zero_init_routed_clock or fixed_clock_ablation or baseline_arm:
        support_contract_path = os.environ.get(
            "ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT", "")
        if not support_contract_path:
            raise RuntimeError(
                "Missing preregistered reconstruction support contract")
        support_contract = json.loads(Path(support_contract_path).read_text())
        expected_reconstruction_count = int(
            support_contract.get("support_count", 0))
        if expected_reconstruction_count <= 0:
            raise RuntimeError(
                "Invalid preregistered reconstruction support count")
        initial_reconstruction_count = len(
            getattr(scene, "stage2_reconstruction_cameras", []))
        reconstruction_support_configured = (
            initial_reconstruction_count == expected_reconstruction_count)
        print("STAGE2_RECONSTRUCTION_SUPPORT_STATE " + json.dumps({
            "configured": reconstruction_support_configured,
            "current_camera_count": initial_reconstruction_count,
            "expected_reconstruction_camera_count":
                expected_reconstruction_count,
            "ablation_mode": ablation_mode or None,
        }, sort_keys=True))
        if baseline_arm and not reconstruction_support_configured:
            support_report = configure_fixed_clock_reconstruction_support(scene)
            train_cams = list(scene.stage2_reconstruction_cameras)
            if dataset.loader != "nerfies":
                train_cams = sorted(
                    train_cams,
                    key=lambda camera: (camera.cam_no, camera.frame_no))
            viewpoint_stack = train_cams
            reconstruction_support_configured = True
            print("STAGE2_RECONSTRUCTION_SUPPORT_SWITCH " + json.dumps({
                "iteration": 0,
                "calibration_camera_count": len(
                    scene.stage2_training_cameras),
                "reconstruction_camera_count": len(train_cams),
                "fixed_offset_frames": 0.0,
                "ablation_mode": None,
            }, sort_keys=True))
    method = None

    temporal_consensus = None
    temporal_consensus_initial_gradient = None
    zero_start_clock_phase = None
    if motion_clock_continuation:
        zero_start_report = os.environ.get(
            "ED3DGS_ZERO_START_CLOCK_PHASE_REPORT", "")
        if not zero_start_report:
            raise ValueError(
                "Motion-clock continuation requires a report path")
        if (float(scene.temporal_offset_raw.detach().item()) != 0.0
                or float(
                    scene.temporal_offset_frames().detach().item()) != 0.0):
            raise RuntimeError(
                "Motion-clock continuation must begin at exact raw=0, offset=0")
        if scene.temporal_affine_clock_enabled:
            raise RuntimeError(
                "Motion-clock continuation requires one scalar clock")
        if len(scene.optimizer_temporal_offset.state) != 0:
            raise RuntimeError(
                "Motion-clock optimizer must have no prior state")
        phase_path = Path(zero_start_report).resolve()
        if phase_path.exists():
            raise RuntimeError(
                f"Refusing existing zero-start report: {phase_path}")
        motion_path = phase_path.with_name("motion_clock.json")
        print("ZERO_START_CLOCK_PHASE_BEGIN " + json.dumps({
            "candidate_lag_bank_in_observable": True,
            "initial_offset_frames": 0.0,
            "initial_raw": 0.0,
            "preloaded_offset": False,
            "shift_truth_input_to_observable": False,
            "trainable_candidate_logits": False,
            "trainable_state": "scene_temporal_offset_raw",
        }, sort_keys=True), flush=True)
        motion = motion_clock.run(
            motion_path, list(scene.getTrainCameras()),
            scene.temporal_offset_raw, scene.optimizer_temporal_offset,
            scene.temporal_offset_max_frames,
            lambda: os.environ.get("ED3DGS_THERMAL_FRAME_SHIFT", "nan"))
        if (motion.get("status") != "PASS"
                or not all(motion.get("checks", {}).values())):
            raise RuntimeError(
                "Motion-clock continuation failed its scientific audit")
        applied_offset = float(
            scene.temporal_offset_frames().detach().item())
        applied_raw = float(scene.temporal_offset_raw.detach().item())
        if (abs(applied_offset - float(motion["final_offset_frames"])) > 1.0e-5
                or abs(applied_raw - float(motion["final_raw"])) > 1.0e-6):
            raise RuntimeError(
                "Scene clock does not match the optimized motion clock")
        scene.temporal_offset_raw.requires_grad_(False)
        joint_clock_loss_enabled = False
        zero_start_clock_phase = {
            "schema": "covers_zero_start_motion_continuation_learning",
            "status": "PASS",
            "initial_offset_frames": 0.0,
            "initial_raw": 0.0,
            "preloaded_offset": False,
            "candidate_lag_bank_in_observable": True,
            "trainable_candidate_logits": False,
            "trainable_state": "one_scene_raw_clock_scalar",
            "shift_truth_input_to_observable": False,
            "expected_shift_used_only_after_optimization_for_accuracy_audit": True,
            "optimizer": "adam",
            "optimizer_steps": int(motion["optimization"]["optimizer_steps"]),
            "nonzero_gradients": int(
                motion["optimization"]["nonzero_gradient_count"]),
            "finite_gradients": int(
                motion["optimization"]["finite_gradient_count"]),
            "scene_raw_object_id": id(scene.temporal_offset_raw),
            "scene_optimizer_object_id": id(scene.optimizer_temporal_offset),
            "motion_clock": {
                "status": motion["status"],
                "report_path": str(motion_path),
                "report_sha256": hashlib.sha256(
                    motion_path.read_bytes()).hexdigest(),
                "checks": motion["checks"],
                "candidate_offsets_frames": motion[
                    "candidate_offsets_frames"],
                "full_profile_peak_offset_frames": motion[
                    "profiles"]["all"]["joint_best_offset_frames"],
                "left_profile_peak_offset_frames": motion[
                    "profiles"]["all"]["left_best_offset_frames"],
                "right_profile_peak_offset_frames": motion[
                    "profiles"]["all"]["right_best_offset_frames"],
                "early_profile_peak_offset_frames": motion[
                    "profiles"]["early"]["joint_best_offset_frames"],
                "late_profile_peak_offset_frames": motion[
                    "profiles"]["late"]["joint_best_offset_frames"],
                "expected_shift_error_frames": motion[
                    "expected_shift_error_frames"],
                "final_offset_frames": motion["final_offset_frames"],
                "final_raw": motion["final_raw"],
            },
            "final_offset_frames": applied_offset,
            "final_raw": applied_raw,
            "main_training_clock_frozen": True,
        }
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        phase_path.write_text(
            json.dumps(
                zero_start_clock_phase, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print("ZERO_START_CLOCK_PHASE_PASS " + json.dumps(
            zero_start_clock_phase, sort_keys=True), flush=True)
    elif temporal_consensus_enabled:
        if self_calibrating_clock:
            temporal_consensus = motion_cost_volume.MotionCostVolumeLoss(
                list(scene.getTrainCameras()),
                scene.temporal_offset_max_frames,
                0.0,
                scene.maxtime, device="cuda")
            consensus_probe, consensus_probe_sides = temporal_consensus.loss(
                scene.temporal_offset_frames(), scene.temporal_drift_frames(),
                iteration=1)
        else:
            temporal_consensus = temporal_consensus_module.TemporalConsensusLoss(
                train_cams, scene.temporal_offset_max_frames,
                scene.temporal_drift_max_endpoint_frames,
                scene.maxtime,
                gaussians, pipe, hyper, background,
                teacher_iteration=30000, device="cuda")
            consensus_probe, consensus_probe_sides = temporal_consensus.loss(
                scene.temporal_offset_frames(), scene.temporal_drift_frames())
        clock_parameters = [scene.temporal_offset_raw]
        if scene.temporal_affine_clock_enabled:
            clock_parameters.append(scene.temporal_drift_raw)
        temporal_consensus_initial_gradient = tuple(
            value.detach() for value in torch.autograd.grad(
                consensus_probe, tuple(clock_parameters),
                retain_graph=False, allow_unused=False))
        if not all(bool(torch.isfinite(value).all())
                   for value in temporal_consensus_initial_gradient):
            raise FloatingPointError(
                "Non-finite initial temporal-consensus gradient")
        print("TEMPORAL_CONSENSUS_INITIAL_GRADIENT " + json.dumps({
            "loss": float(consensus_probe.detach().item()),
            "offset_frames": float(
                scene.temporal_offset_frames().detach().item()),
            "offset_raw_gradient": float(
                temporal_consensus_initial_gradient[0].item()),
            "drift_raw_gradient": (
                None if len(temporal_consensus_initial_gradient) == 1
                else float(temporal_consensus_initial_gradient[1].item())),
            "side_losses": {
                side: float(value.detach().item())
                for side, value in consensus_probe_sides.items()
            },
        }, sort_keys=True))
        refinement_report = os.environ.get(
            "ED3DGS_CLOCK_REFINEMENT_GATE_REPORT", "")
        optimization_report = os.environ.get(
            "ED3DGS_CLOCK_REFINEMENT_OPTIMIZATION_REPORT", "")
        zero_start_report = os.environ.get(
            "ED3DGS_ZERO_START_CLOCK_PHASE_REPORT", "")
        if sum(bool(path) for path in (
                refinement_report, optimization_report,
                zero_start_report)) > 1:
            raise ValueError(
                "Standalone clock diagnostics and zero-start learning are "
                "mutually exclusive")
        if optimization_report:
            coarse_offset = float(os.environ.get(
                "ED3DGS_CLOCK_REFINEMENT_COARSE_OFFSET", "nan"))
            expected_shift = float(os.environ.get(
                "ED3DGS_THERMAL_FRAME_SHIFT", "nan"))
            if not math.isfinite(coarse_offset):
                raise ValueError(
                    "Clock refinement requires a finite blind coarse clock")
            if not math.isfinite(expected_shift):
                raise ValueError(
                    "Clock refinement requires an audit-only shift value")
            clock_refinement.run_refinement(
                optimization_report, train_cams, coarse_offset,
                expected_shift, scene.temporal_offset_max_frames, gaussians,
                pipe, hyper, background, teacher_iteration=30000)
            raise SystemExit(0)
        if refinement_report:
            coarse_offset = float(os.environ.get(
                "ED3DGS_CLOCK_REFINEMENT_COARSE_OFFSET", "nan"))
            expected_shift = float(os.environ.get(
                "ED3DGS_THERMAL_FRAME_SHIFT", "nan"))
            if not math.isfinite(coarse_offset):
                raise ValueError(
                    "Clock-refinement Gate requires a finite blind coarse clock")
            if not math.isfinite(expected_shift):
                raise ValueError(
                    "Clock-refinement Gate requires an audit-only shift value")
            clock_refinement.run_gate(
                refinement_report, train_cams, temporal_consensus,
                coarse_offset, expected_shift,
                scene.temporal_offset_max_frames, gaussians, pipe, hyper,
                background, teacher_iteration=30000)
            raise SystemExit(0)
        if zero_start_clock_phase_enabled:
            if not zero_start_report:
                raise ValueError(
                    "Zero-start clock learning requires a report path")
            if (float(scene.temporal_offset_raw.detach().item()) != 0.0
                    or float(
                        scene.temporal_offset_frames().detach().item()) != 0.0):
                raise RuntimeError(
                    "Zero-start clock learning must begin at exact raw=0, "
                    "offset=0")
            if scene.temporal_affine_clock_enabled:
                raise RuntimeError(
                    "Zero-start clock learning requires one scalar clock")
            if len(scene.optimizer_temporal_offset.state) != 0:
                raise RuntimeError(
                    "Zero-start clock optimizer must have no prior state")
            phase_path = Path(zero_start_report).resolve()
            if phase_path.exists():
                raise RuntimeError(
                    f"Refusing existing zero-start report: {phase_path}")
            print("ZERO_START_CLOCK_PHASE_BEGIN " + json.dumps({
                "candidate_enumeration": False,
                "initial_offset_frames": 0.0,
                "initial_raw": 0.0,
                "optimizer_steps": zero_start_clock_phase_steps,
                "preloaded_offset": False,
                "shift_truth_input_to_optimizer": False,
            }, sort_keys=True), flush=True)
            coarse_nonzero_gradients = 0
            coarse_first_loss = None
            coarse_last_loss = None
            for clock_step in range(1, zero_start_clock_phase_steps + 1):
                scene.optimizer_temporal_offset.zero_grad(set_to_none=True)
                coarse_loss, coarse_sides = temporal_consensus.loss(
                    scene.temporal_offset_frames())
                raw_gradient = torch.autograd.grad(
                    coarse_loss, scene.temporal_offset_raw,
                    retain_graph=False, allow_unused=False)[0].detach()
                if not bool(torch.isfinite(raw_gradient).all()):
                    raise FloatingPointError(
                        "Non-finite zero-start clock gradient")
                if float(raw_gradient.abs().item()) > 0.0:
                    coarse_nonzero_gradients += 1
                scene.temporal_offset_raw.grad = raw_gradient.clone()
                scene.optimizer_temporal_offset.step()
                coarse_last_loss = float(coarse_loss.detach().item())
                if coarse_first_loss is None:
                    coarse_first_loss = coarse_last_loss
                if (clock_step == 1 or clock_step % 100 == 0
                        or clock_step == zero_start_clock_phase_steps):
                    print("ZERO_START_CLOCK_PHASE_STEP " + json.dumps({
                        "loss": coarse_last_loss,
                        "offset_frames": float(
                            scene.temporal_offset_frames().detach().item()),
                        "raw": float(
                            scene.temporal_offset_raw.detach().item()),
                        "raw_gradient": float(raw_gradient.item()),
                        "side_losses": {
                            side: float(value.detach().item())
                            for side, value in coarse_sides.items()
                        },
                        "step": clock_step,
                    }, sort_keys=True), flush=True)
            scene.optimizer_temporal_offset.zero_grad(set_to_none=True)
            coarse_offset = float(
                scene.temporal_offset_frames().detach().item())
            coarse_raw = float(scene.temporal_offset_raw.detach().item())
            if (coarse_nonzero_gradients != zero_start_clock_phase_steps
                    or not math.isfinite(coarse_offset)
                    or coarse_first_loss is None
                    or coarse_last_loss is None
                    or coarse_last_loss >= coarse_first_loss):
                raise RuntimeError(
                    "Zero-start coarse clock learning did not converge")
            expected_shift = float(os.environ.get(
                "ED3DGS_THERMAL_FRAME_SHIFT", "nan"))
            if not math.isfinite(expected_shift):
                raise ValueError(
                    "Zero-start clock learning requires an audit-only shift")
            regional_path = phase_path.with_name(
                "regional_clock_refinement.json")
            regional = clock_refinement.run_refinement(
                regional_path, train_cams, coarse_offset, expected_shift,
                scene.temporal_offset_max_frames, gaussians, pipe, hyper,
                background, teacher_iteration=30000)
            if (regional.get("status") != "PASS"
                    or not all(regional.get("checks", {}).values())):
                raise RuntimeError(
                    "Regional clock refinement failed its scientific audit")
            refined_raw = float(regional["refined_raw"])
            refined_offset = float(regional["refined_offset_frames"])
            with torch.no_grad():
                scene.temporal_offset_raw.copy_(
                    torch.as_tensor(
                        refined_raw,
                        dtype=scene.temporal_offset_raw.dtype,
                        device=scene.temporal_offset_raw.device))
            applied_offset = float(
                scene.temporal_offset_frames().detach().item())
            if abs(applied_offset - refined_offset) > 1.0e-5:
                raise RuntimeError(
                    "Applied refined clock does not match the blind report")
            scene.temporal_offset_raw.requires_grad_(False)
            if scene.temporal_affine_clock_enabled:
                scene.temporal_drift_raw.requires_grad_(False)
            joint_clock_loss_enabled = False
            zero_start_clock_phase = {
                "schema": "covers_zero_start_clock_learning",
                "status": "PASS",
                "candidate_enumeration": False,
                "initial_offset_frames": 0.0,
                "initial_raw": 0.0,
                "preloaded_offset": False,
                "shift_truth_input_to_optimizer": False,
                "expected_shift_used_only_for_accuracy_audit": True,
                "coarse_optimizer": "adam",
                "coarse_optimizer_steps": zero_start_clock_phase_steps,
                "coarse_nonzero_gradients": coarse_nonzero_gradients,
                "coarse_first_loss": coarse_first_loss,
                "coarse_last_loss": coarse_last_loss,
                "coarse_offset_frames": coarse_offset,
                "coarse_raw": coarse_raw,
                "regional_refinement": {
                    "status": regional["status"],
                    "report_path": str(regional_path),
                    "report_sha256": hashlib.sha256(
                        regional_path.read_bytes()).hexdigest(),
                    "checks": regional["checks"],
                    "candidate_enumeration": regional[
                        "candidate_enumeration"],
                    "shift_truth_input_to_optimizer": regional[
                        "shift_truth_input_to_optimizer"],
                    "refined_offset_frames": refined_offset,
                    "refined_raw": refined_raw,
                    "expected_shift_error_frames": regional[
                        "expected_shift_error_frames"],
                },
                "final_offset_frames": applied_offset,
                "final_raw": float(
                    scene.temporal_offset_raw.detach().item()),
                "main_training_clock_frozen": True,
            }
            phase_path.parent.mkdir(parents=True, exist_ok=True)
            phase_path.write_text(
                json.dumps(
                    zero_start_clock_phase, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            print("ZERO_START_CLOCK_PHASE_PASS " + json.dumps(
                zero_start_clock_phase, sort_keys=True), flush=True)
    elif zero_start_clock_phase_enabled:
        raise ValueError(
            "Zero-start clock learning requires temporal consensus")

    thermal_luma_weights = torch.tensor([0.299, 0.587, 0.114], device="cuda").view(1, 3, 1, 1)

    def _thermal_gray_l1(pred, gt):
        pred_gray = (pred * thermal_luma_weights).sum(dim=1, keepdim=True)
        gt_gray = (gt * thermal_luma_weights).sum(dim=1, keepdim=True)
        return torch.abs(pred_gray - gt_gray).mean() * opt.thermal_pose_gray_loss_weight

    def _get_thermal_pose_params():
        params = []
        seen = set()
        for cam in train_cams:
            if not (hasattr(cam, "has_thermal") and cam.has_thermal):
                continue
            names = ("thermal_delta_quaternion", "thermal_delta_translation")
            if not v34_block_calibration:
                names += ("learnable_tfovx", "learnable_tfovy")
            for name in names:
                param = getattr(cam, name, None)
                if param is not None and param.requires_grad and id(param) not in seen:
                    seen.add(id(param))
                    params.append(param)
        return params

    def _thermal_intrinsic_prior():
        terms = []
        seen_x = set()
        seen_y = set()
        for cam in train_cams:
            if not (hasattr(cam, "has_thermal") and cam.has_thermal):
                continue
            fovx, fovy = cam.get_thermal_fovs()
            if id(cam.learnable_tfovx) not in seen_x:
                seen_x.add(id(cam.learnable_tfovx))
                init = torch.as_tensor(cam.TFoVx, device=fovx.device, dtype=fovx.dtype)
                log_scale = torch.log(torch.tan(0.5 * init) / torch.tan(0.5 * fovx))
                terms.append(log_scale.square())
            if (not cam.thermal_intrinsic_tied_aspect
                    and id(cam.learnable_tfovy) not in seen_y):
                seen_y.add(id(cam.learnable_tfovy))
                init = torch.as_tensor(cam.TFoVy, device=fovy.device, dtype=fovy.dtype)
                log_scale = torch.log(torch.tan(0.5 * init) / torch.tan(0.5 * fovy))
                terms.append(log_scale.square())
        if not terms:
            return torch.tensor(0.0, device="cuda")
        return torch.stack(terms).mean()

    thermal_pose_params = [] if opt.rgb_only_teacher else _get_thermal_pose_params()
    calibration_reconstruction_alternation = bool(getattr(
        opt, "calibration_reconstruction_alternation", False))
    calibration_reconstruction_phase_length = int(getattr(
        opt, "calibration_reconstruction_phase_length", 8))
    calibration_reconstruction_start_iter = int(opt.temporal_offset_start_iter)
    if scene.has_thermal_pose:
        calibration_reconstruction_start_iter = max(
            int(opt.thermal_pose_start_iter), int(opt.temporal_offset_start_iter))
    if calibration_reconstruction_alternation:
        if calibration_reconstruction_phase_length <= 0:
            raise ValueError(
                "calibration_reconstruction_phase_length must be positive")
        if not (scene.temporal_alignment_enabled or scene.has_thermal_pose):
            raise ValueError(
                "calibration/reconstruction alternation requires temporal "
                "alignment or Thermal pose")
        if (scene.has_thermal_pose
                and opt.thermal_pose_start_iter != opt.temporal_offset_start_iter):
            raise ValueError(
                "alternating Thermal pose and temporal scalar must share a "
                "start iteration")
        print("STAGE2_OPTIMIZER_ALTERNATION " + json.dumps({
            "calibration_updates": (
                ["temporal_offset"] if scene.temporal_alignment_enabled else [])
            + (["thermal_pose"] if scene.has_thermal_pose else []),
            "phase_length": calibration_reconstruction_phase_length,
            "reconstruction_updates": ["gaussian", "deformation", "appearance"],
            "start_iteration": calibration_reconstruction_start_iter,
        }, sort_keys=True))
    if strict_scene_freeze:
        expected_strict_steps = schedule.strict_block_step_counts(
            train_iter,
            phase_length=calibration_reconstruction_phase_length,
            start_iteration=calibration_reconstruction_start_iter,
            calibration_steps=(strict_calibration_steps
                               if strict_step_budget_v2 else None),
            scene_steps=(scene_optimizer_target_steps
                         if strict_step_budget_v2 else None))
        if scene_optimizer_target_steps != expected_strict_steps["scene"]:
            raise ValueError(
                "Strict Scene optimizer target mismatch: "
                f"configured={scene_optimizer_target_steps} "
                f"expected={expected_strict_steps['scene']}")
        print("STRICT_SCENE_FREEZE_CONTRACT " + json.dumps({
            "strict_step_budget_v2": bool(strict_step_budget_v2),
            "expected_outer_iterations": expected_strict_steps["outer"],
            "calibration_backward": ["clock", "thermal_pose"],
            "calibration_scene_gradients": False,
            "calibration_scene_mutations": False,
            "expected_calibration_optimizer_steps":
                expected_strict_steps["calibration"],
            "expected_scene_optimizer_steps": expected_strict_steps["scene"],
            "expected_reconstruction_optimizer_steps": (
                expected_strict_steps["reconstruction"]),
            "expected_scene_tail_optimizer_steps": (
                expected_strict_steps["scene_tail"]),
            "clock_target_steps": clock_target_steps,
            "pose_target_steps": pose_target_steps,
            "freeze_after_clock_steps": clock_target_steps,
            "forward_state_coupling": True,
            "temporal_forward_uses_detached_state": True,
            "reconstruction_backward": [
                "gaussian", "deformation", "appearance", "routing"],
            "reconstruction_clock_gradients": False,
            "reconstruction_pose_gradients": False,
        }, sort_keys=True))
    if v34_block_calibration:
        if not (calibration_reconstruction_alternation
                and (scene.has_thermal_pose or matched_clock_only_block)
                and (scene.temporal_alignment_enabled or scene.has_thermal_pose)):
            raise ValueError(
                "v34 block calibration requires A/B phases and at least one "
                "enabled clock/pose calibration parameter")
        if float(opt.thermal_intrinsic_lr) != 0.0:
            raise ValueError("v34 requires frozen Thermal intrinsics")
        if calibration_reconstruction_phase_length != 8:
            raise ValueError("v34 requires the locked 8/8 A/B phase length")
        print("V34_BLOCK_CALIBRATION_CONTRACT " + json.dumps({
            "clock_update_interval": 32,
            "clock_window_centers_per_side": 9,
            "pose_loss": (("thermal_gray_L1" if fourarm else
                           "0.5*MIND+0.5*polarity_invariant_NGF")
                          if scene.has_thermal_pose else "disabled"),
            "pose_updates": ("every_calibration_iteration"
                             if scene.has_thermal_pose else "disabled"),
            "scene_updates": (
                "every_main_iteration"
                if full_scene_steps else "reconstruction_iterations_only"),
            "thermal_intrinsics_frozen": True,
        }, sort_keys=True))
        if scene.has_thermal_pose:
            v34_pose_views_by_side = {"left": [], "right": []}
            for camera in train_cams:
                side, frame = scene._camera_state_side_frame(camera)
                if side in v34_pose_views_by_side and frame is not None:
                    v34_pose_views_by_side[side].append((int(frame), camera))
            for side in ("left", "right"):
                v34_pose_views_by_side[side].sort(key=lambda row: row[0])
            pose_frames = {
                side: [frame for frame, _ in rows]
                for side, rows in v34_pose_views_by_side.items()
            }
            if (not pose_frames["left"]
                    or len(pose_frames["left"]) != len(pose_frames["right"])):
                raise RuntimeError(
                    "v34 pose batches require balanced left/right support")

            def v34_pose_batch(current_iteration):
                relative = int(current_iteration) - calibration_reconstruction_start_iter
                cycle_length = 2 * calibration_reconstruction_phase_length
                within_cycle = relative % cycle_length
                if relative < 0 or within_cycle >= calibration_reconstruction_phase_length:
                    raise RuntimeError("v34 pose batch requested outside calibration phase")
                update_index = ((relative // cycle_length)
                                * calibration_reconstruction_phase_length
                                + within_cycle)
                frame_index = update_index % len(pose_frames["left"])
                return [
                    v34_pose_views_by_side[side][frame_index][1]
                    for side in ("left", "right")
                ], {
                    "frames": {
                        side: pose_frames[side][frame_index]
                        for side in ("left", "right")
                    },
                    "frame_index": frame_index,
                    "update_index": update_index,
                    "sides": ["left", "right"],
                }
        else:
            v34_pose_views_by_side = None
    else:
        v34_pose_views_by_side = None

    if opt.rgb_only_teacher:
        print("=" * 60)
        print("[RGBTeacher] RGB-ONLY TEACHER MODE ACTIVE")
        print("[RGBTeacher] Thermal rendering: DISABLED")
        print("[RGBTeacher] Thermal loss/pose/intrinsics/temporal alignment: DISABLED")
        print("[RGBTeacher] Densification: RGB gradient only")
        print("[RGBTeacher] Routing: forced all-shared")
        print("=" * 60)

    # Thermal-only mode diagnostic
    if opt.thermal_only:
        print("=" * 60)
        print("[ThermalOnly] THERMAL-ONLY MODE ACTIVE")
        print("[ThermalOnly] RGB rendering: DISABLED")
        print("[ThermalOnly] RGB loss: DISABLED")
        print("[ThermalOnly] All geometry (xyz, scales, rotations): learned from thermal")
        print("[ThermalOnly] All appearance (thermal SH, thermal opacity): learned from thermal")
        print("[ThermalOnly] Densification: thermal gradient only")
        print("[ThermalOnly] Routing: forced all-shared")
        print("=" * 60)

    # Diagnostic: verify thermal pose is trainable (Camera-level params)
    if scene.has_thermal_pose:
        cam0 = scene.getTrainCameras()[0]
        dq_req = cam0.thermal_delta_quaternion.requires_grad
        dt_req = cam0.thermal_delta_translation.requires_grad
        print(f"[ThermalPose] Camera-level delta: quat.requires_grad={dq_req}, trans.requires_grad={dt_req}")
        print(f"[ThermalPose] delta_quaternion init: {cam0.thermal_delta_quaternion.data}")
        print(f"[ThermalPose] delta_translation init: {cam0.thermal_delta_translation.data}")
        print(f"[ThermalPose] optimizers: rotation={scene.optimizer_thermal_rotation is not None}, "
              f"translation={scene.optimizer_thermal_translation is not None}, "
              f"fovx={scene.optimizer_thermal_fovx is not None}, "
              f"fovy={scene.optimizer_thermal_fovy is not None}")

    # Diagnostic: verify 3dgs-pose rasterizer supports pose gradients
    import diff_gaussian_rasterization as dgr
    import inspect
    bw_src = inspect.getsource(dgr._RasterizeGaussians.backward)
    if 'grad_world_view' in bw_src:
        print("[Rasterizer] 3dgs-pose detected: grad_world_view supported")
    else:
        print("[Rasterizer] WARNING: old rasterizer detected, pose gradients NOT supported!")

    print("MEMORY_SAFE_BACKWARD_CONFIG " + json.dumps({
        "enabled": bool(memory_safe_backward),
        "empty_cache_before_backward": bool(memory_safe_backward),
        "gaussian_threshold": int(memory_safe_gaussian_threshold),
        "detach_clock_after_freeze": bool(memory_safe_backward),
    }, sort_keys=True))
    ema_loss_rgb = None
    ema_loss_thermal = None
    memory_safe_cache_release_count = 0
    memory_safe_clock_detached = False
    start_time = time()
    for iteration in range(first_iter, final_iter+1):
        iter_start.record()

        if strict_step_budget_v2:
            support_switch_iteration = schedule.support_switch_iteration(
                strict_calibration_steps, calibration_reconstruction_start_iter)
            clock_step_candidate = temporal_offset_optimizer_step_count + 1
            pose_step_candidate = thermal_pose_optimizer_step_count + 1
        else:
            support_switch_iteration = 15001
            clock_step_candidate = iteration
            pose_step_candidate = iteration

        if (self_calibrating_clock
                and scene.temporal_alignment_enabled
                and not strict_step_budget_v2
                and opt.temporal_offset_freeze_after >= 0
                and iteration == opt.temporal_offset_freeze_after + 1
                and scene.temporal_offset_raw.requires_grad):
            scene.temporal_offset_raw.requires_grad_(False)
            print("SELF_CALIBRATING_CLOCK_FREEZE " + json.dumps({
                "freeze_after_iteration": int(
                    opt.temporal_offset_freeze_after),
                "offset_frames": float(
                    scene.temporal_offset_frames().detach().item()),
                "requires_grad": False,
            }, sort_keys=True))

        if (memory_safe_backward
                and scene.temporal_alignment_enabled
                and not strict_step_budget_v2
                and opt.temporal_offset_freeze_after >= 0
                and iteration == opt.temporal_offset_freeze_after + 1
                and scene.temporal_offset_raw.requires_grad):
            scene.temporal_offset_raw.requires_grad_(False)
            memory_safe_clock_detached = True
            print("MEMORY_SAFE_CLOCK_DETACH " + json.dumps({
                "iteration": int(iteration),
                "offset_frames": float(
                    scene.temporal_offset_frames().detach().item()),
            }, sort_keys=True))

        if (zero_init_routed_clock
                and not reconstruction_support_configured
                and scene.temporal_alignment_enabled
                and ((strict_step_budget_v2
                      and iteration == support_switch_iteration)
                     or (not strict_step_budget_v2
                         and opt.temporal_offset_freeze_after >= 0
                         and iteration == opt.temporal_offset_freeze_after + 1))):
            support_report = configure_blind_reconstruction_support(
                scene, {"learned_offset_frames": float(
                    scene.temporal_offset_frames().detach().item())})
            if support_report is None:
                raise RuntimeError(
                    "Zero-init full reconstruction requires blind dual support")
            train_cams = list(scene.stage2_reconstruction_cameras)
            if dataset.loader != "nerfies":
                train_cams = sorted(
                    train_cams, key=lambda camera: (camera.cam_no, camera.frame_no))
            viewpoint_stack = train_cams
            reconstruction_support_configured = True
            print("STAGE2_RECONSTRUCTION_SUPPORT_SWITCH " + json.dumps({
                "iteration": int(iteration),
                "calibration_camera_count": len(scene.stage2_training_cameras),
                "reconstruction_camera_count": len(train_cams),
                "learned_offset_frames": support_report[
                    "learned_offset_frames"],
                "ablation_mode": ablation_mode or None,
            }, sort_keys=True))

        if (fixed_clock_ablation
                and not reconstruction_support_configured
                and iteration == support_switch_iteration):
            support_report = configure_fixed_clock_reconstruction_support(scene)
            if support_report is None:
                raise RuntimeError(
                    "Fixed-clock ablation requires preregistered dual support")
            train_cams = list(scene.stage2_reconstruction_cameras)
            if dataset.loader != "nerfies":
                train_cams = sorted(
                    train_cams, key=lambda camera: (camera.cam_no, camera.frame_no))
            viewpoint_stack = train_cams
            reconstruction_support_configured = True
            print("STAGE2_RECONSTRUCTION_SUPPORT_SWITCH " + json.dumps({
                "iteration": int(iteration),
                "calibration_camera_count": len(scene.stage2_training_cameras),
                "reconstruction_camera_count": len(train_cams),
                "fixed_offset_frames": 0.0,
                "ablation_mode": ablation_mode,
            }, sort_keys=True))

        if (calibration_reconstruction_alternation
                and iteration >= calibration_reconstruction_start_iter):
            optimizer_phase = schedule.phase_for_iteration(
                iteration,
                phase_length=calibration_reconstruction_phase_length,
                start_iteration=calibration_reconstruction_start_iter,
                calibration_steps=(strict_calibration_steps
                                   if strict_step_budget_v2 else None),
                scene_steps=(scene_optimizer_target_steps
                             if strict_step_budget_v2 else None))
        elif calibration_reconstruction_alternation:
            optimizer_phase = "warmup"
        else:
            optimizer_phase = "joint"
        optimizer_phase_counts[optimizer_phase] += 1
        step_scene_optimizer = schedule.should_step_scene(
            full_scene_steps, optimizer_phase)
        step_calibration_optimizers = optimizer_phase in {"joint", "calibration"}
        # Reconstruction schedules advance with actual scene updates. During
        # calibration the frozen scene remains at its last committed step.
        scene_forward_step = (
            scene_optimizer_step_count + (1 if step_scene_optimizer else 0)
            if strict_step_budget_v2 else iteration)

        if strict_scene_freeze:
            for group in gaussians.optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.requires_grad_(step_scene_optimizer)
            pose_trainable = (
                step_calibration_optimizers
                and iteration >= opt.thermal_pose_start_iter
                and (pose_target_steps < 0
                     or thermal_pose_optimizer_step_count < pose_target_steps)
                and not (opt.thermal_pose_freeze_after >= 0
                         and iteration > opt.thermal_pose_freeze_after))
            for parameter in thermal_pose_params:
                parameter.requires_grad_(pose_trainable)
            if scene.temporal_alignment_enabled:
                clock_trainable = (
                    step_calibration_optimizers
                    and iteration >= opt.temporal_offset_start_iter
                    and (clock_target_steps < 0
                         or temporal_offset_optimizer_step_count < clock_target_steps)
                    and not (opt.temporal_offset_freeze_after >= 0
                             and iteration > opt.temporal_offset_freeze_after))
                scene.temporal_offset_raw.requires_grad_(clock_trainable)
                if scene.temporal_affine_clock_enabled:
                    scene.temporal_drift_raw.requires_grad_(clock_trainable)

        if step_scene_optimizer:
            scene_step_candidate = scene_optimizer_step_count + 1
            gaussians.update_learning_rate(
                scene_step_candidate if strict_step_budget_v2 else iteration)
            gaussians.update_pose(
                scene_step_candidate if strict_step_budget_v2 else iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if ((scene_step_candidate if strict_step_budget_v2 else iteration)
                    % 1000 == 0):
                gaussians.oneupSHdegree()
                sh_degree_update_phase_counts[optimizer_phase] += 1

        # opt.batch_size = 2
        ### Instead of the complex process below, simply training on random frames will also work well. If you follow this, comment out the `train_cams` sorting process above.
        if dataset.loader == 'nerfies':
            frame_set = np.random.choice(range(math.ceil(len(viewpoint_stack) / 2)), size=max(opt.batch_size // 2, 1))
            viewpoint_cams = [viewpoint_stack[(f*2) % scene.maxtime] for f in frame_set] + \
                             [viewpoint_stack[(f*2+1) % scene.maxtime] for f in frame_set]
        else:
            # Pick camera
            method = "random" if iteration < opt.random_until or iteration % 2 == 1 else "by_error"

            cam_no = []
            for _ in range(opt.batch_size):
                last_camera_index = sample_camera(cam_dists, last_camera_index, min_dist)
                cam_no.append(last_camera_index)
            
            viewpoint_cams, sampled_cam_no, sampled_frame_no = image_sampler(method=method, loader=viewpoint_stack, loss_list=loss_list, batch_size=opt.batch_size, \
                cam_no=cam_no, frame_no=sampled_frame_no, total_num_frames=scene.maxtime)
            if iteration >= opt.random_until and opt.num_multiview_ssim > 0 and iteration % 50 < opt.num_multiview_ssim:
                sampled_frame_no = sampled_frame_no  # reuse sampled frame (num_multiview_ssim) times
            else:
                sampled_frame_no = None
        ###

        # Routing stage control (A: force shared, B/C: gumbel + prior, D: freeze)
        if opt.thermal_only or opt.rgb_only_teacher:
            routing_stage = "A"  # single-modality modes use shared geometry only
        elif (scene_optimizer_step_count + (1 if step_scene_optimizer else 0)
              if strict_step_budget_v2 else iteration) < opt.modality_stage_a_until:
            routing_stage = "A"
        elif (scene_optimizer_step_count + (1 if step_scene_optimizer else 0)
              if strict_step_budget_v2 else iteration) < opt.modality_stage_b_until:
            routing_stage = "B"
        elif (scene_optimizer_step_count + (1 if step_scene_optimizer else 0)
              if strict_step_budget_v2 else iteration) < opt.modality_stage_c_until:
            routing_stage = "C"
        else:
            routing_stage = "D"

        if routing_stage == "A":
            s_hard = torch.zeros((gaussians.get_xyz.shape[0], 3), device="cuda")
            s_hard[:, 0] = 1.0
            s_soft = s_hard
        elif routing_stage == "D":
            s_soft = torch.softmax(gaussians._logit_modality, dim=-1)
            s_hard = torch.zeros_like(s_soft)
            s_hard.scatter_(1, torch.argmax(s_soft, dim=-1, keepdim=True), 1.0)
        else:
            s_hard, s_soft = compute_modality_routing(
                gaussians._logit_modality,
                tau=opt.modality_tau,
                hard=True,
                stochastic=not stable_modality_routing,
            )

        if routing_stage == "D":
            gaussians._logit_modality.requires_grad_(False)
            for param_group in gaussians.optimizer.param_groups:
                if param_group["name"] == "logit_modality":
                    param_group["lr"] = 0.0

        if iteration == 1 or iteration % 500 == 0:
            route_idx = torch.argmax(s_hard.detach(), dim=-1)
            route_counts = torch.bincount(route_idx, minlength=3)
            total_routes = max(1, int(route_idx.numel()))
            print("MODALITY_ROUTE_STATS " + json.dumps({
                "iteration": int(iteration),
                "stage": routing_stage,
                "stable": bool(stable_modality_routing),
                "shared": int(route_counts[0].item()),
                "rgb_only": int(route_counts[1].item()),
                "thermal_only": int(route_counts[2].item()),
                "shared_fraction": float(route_counts[0].item() / total_routes),
                "rgb_fraction": float(route_counts[1].item() / total_routes),
                "thermal_fraction": float(route_counts[2].item() / total_routes),
            }, sort_keys=True))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # images = []
        images_rgb = []
        images_thermal = []
        # gt_images = []
        gt_images_rgb = []
        gt_images_thermal = []
        radii_rgb_list = []
        radii_th_list = []
        visibility_filter_rgb_list = []
        visibility_filter_th_list = []
        viewspace_point_tensor_rgb_list = []
        viewspace_point_tensor_th_list = []
        cam_no_list, frame_no_list = [], []

        for viewpoint_cam in viewpoint_cams:
            if type(viewpoint_cam.original_image) == type(None):
                viewpoint_cam.load_image()  # for lazy loading (to avoid OOM issue)

            if (strict_scene_freeze
                    and viewpoint_cam.temporal_observation_correction_enabled):
                viewpoint_cam.nctc_temporal_offset_raw_override = (
                    scene.temporal_offset_raw.detach())
                if scene.temporal_affine_clock_enabled:
                    viewpoint_cam.nctc_temporal_drift_raw_override = (
                        scene.temporal_drift_raw.detach())

            cam_no = viewpoint_cam.cam_no
            frame_no = viewpoint_cam.frame_no
            cam_no_list.append(cam_no)
            frame_no_list.append(frame_no)

            # 渲染 RGB 与 Thermal 两种模态

            # A frozen-pose clock step only differentiates the cached clock
            # objective. Keep rendering for EMA/sampling, without a render graph.
            clock_only_render = (
                strict_scene_freeze and frozen_pose_ablation
                and optimizer_phase == "calibration" and not step_scene_optimizer
                and not scene.has_thermal_pose and opt.thermal_intrinsic_lr == 0
                and temporal_consensus_enabled
                and temporal_consensus_mode == "consensus_only")
            with torch.set_grad_enabled(torch.is_grad_enabled() and not clock_only_render):
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, background_thermal,
                                   cam_no=cam_no, iter=scene_forward_step,
                                   num_down_emb_c=hyper.min_embeddings, num_down_emb_f=hyper.min_embeddings,
                                   modality_routing=s_hard, modality_tau=opt.modality_tau, modality_stage=routing_stage,
                                   thermal_only=opt.thermal_only,
                                   rgb_only_teacher=opt.rgb_only_teacher)
            if bool(render_pkg.get("rgb_only_teacher", False)) != bool(opt.rgb_only_teacher):
                raise RuntimeError("renderer RGB-only contract mismatch")

            image_rgb = render_pkg["render_rgb"]
            image_thermal = render_pkg["render_thermal"]
            viewspace_point_tensor_rgb = render_pkg.get(
                "viewspace_points_rgb", render_pkg["viewspace_points"])
            viewspace_point_tensor_th = render_pkg.get(
                "viewspace_points_th", render_pkg["viewspace_points"])
            visibility_filter_rgb = render_pkg.get(
                "visibility_filter_rgb", render_pkg["visibility_filter"])
            visibility_filter_th = render_pkg.get(
                "visibility_filter_th", render_pkg["visibility_filter"])
            radii_rgb = render_pkg.get("radii_rgb", render_pkg["radii"])
            radii_th = render_pkg.get("radii_th", render_pkg["radii"])

            images_rgb.append(image_rgb.unsqueeze(0))
            if not opt.rgb_only_teacher:
                images_thermal.append(image_thermal.unsqueeze(0))

            gt_rgb = viewpoint_cam.original_image.cuda()

            gt_images_rgb.append(gt_rgb.unsqueeze(0))
            if not opt.rgb_only_teacher:
                gt_thermal = viewpoint_cam.thermal_image.cuda()
                gt_images_thermal.append(gt_thermal.unsqueeze(0))

            del image_rgb, image_thermal, gt_rgb
            if not opt.rgb_only_teacher:
                del gt_thermal

            radii_rgb_list.append(radii_rgb.unsqueeze(0))
            visibility_filter_rgb_list.append(visibility_filter_rgb.unsqueeze(0))
            viewspace_point_tensor_rgb_list.append(viewspace_point_tensor_rgb)
            if not opt.rgb_only_teacher:
                radii_th_list.append(radii_th.unsqueeze(0))
                visibility_filter_th_list.append(visibility_filter_th.unsqueeze(0))
                viewspace_point_tensor_th_list.append(viewspace_point_tensor_th)

        radii_rgb = torch.cat(radii_rgb_list,0).max(dim=0).values
        visibility_filter_rgb = torch.cat(visibility_filter_rgb_list).any(dim=0)
        if opt.rgb_only_teacher:
            radii_th = torch.zeros_like(radii_rgb)
            radii = radii_rgb
            visibility_filter_th = torch.zeros_like(visibility_filter_rgb)
            visibility_filter = visibility_filter_rgb
        else:
            radii_th = torch.cat(radii_th_list,0).max(dim=0).values
            radii = torch.maximum(radii_rgb, radii_th)
            visibility_filter_th = torch.cat(visibility_filter_th_list).any(dim=0)
            visibility_filter = torch.logical_or(visibility_filter_rgb, visibility_filter_th)
        # image_tensor = torch.cat(images,0)
        # gt_image_tensor = torch.cat(gt_images,0)
        image_tensor_rgb = torch.cat(images_rgb,0)
        gt_image_tensor_rgb = torch.cat(gt_images_rgb,0)
        if not opt.rgb_only_teacher:
            image_tensor_thermal = torch.cat(images_thermal,0)
            gt_image_tensor_thermal = torch.cat(gt_images_thermal,0)

        # print(gt_image_tensor_thermal.mean(), image_tensor_thermal.mean())

        thermal_pose_gray_loss = torch.tensor(0.0, device="cuda")
        v34_pose_loss = torch.tensor(0.0, device="cuda")
        v34_pose_components = None
        v34_pose_batch_report = None
        if (v34_block_calibration and not fourarm
                and optimizer_phase == "calibration"
                and scene.has_thermal_pose
                and iteration >= opt.thermal_pose_start_iter):
            v34_pose_views, v34_pose_batch_report = v34_pose_batch(iteration)
            if not hasattr(scene, "v34_pose_derivative_audit"):
                scene.v34_pose_derivative_audit = (
                    pose_calibration.directional_derivative_audit(
                        v34_pose_views[0], gaussians, pipe, hyper, background,
                        scene_forward_step, scene.cameras_extent))
                print("V34_POSE_DERIVATIVE_AUDIT " + json.dumps(
                    scene.v34_pose_derivative_audit, sort_keys=True))
            v34_pose_loss, v34_pose_components = (
                pose_calibration.batch_pose_loss(
                    v34_pose_views, gaussians, pipe, hyper, background,
                    scene_forward_step))

        # Ll1 = l1_loss(image_tensor, gt_image_tensor, keepdim=True)
        # Ll1_items = Ll1.detach()
        # Ll1 = Ll1.mean()
        # if opt.lambda_dssim > 0. and sampled_frame_no != None or (method == "by_error" and (iteration % 10 == 0) and opt.num_multiview_ssim==0):
        #     ssim_value, ssim_map = ssim(image_tensor, gt_image_tensor)
        #     Lssim = (1 - ssim_value) / 2
        #     loss = Ll1 + opt.lambda_dssim * Lssim
        # else:
        #     loss = Ll1

        if opt.rgb_only_teacher:
            Ll1_rgb = l1_loss(image_tensor_rgb, gt_image_tensor_rgb, keepdim=True)
            Ll1_rgb_items = Ll1_rgb.detach()
            Ll1_rgb = Ll1_rgb.mean()
            if opt.lambda_dssim > 0. and sampled_frame_no != None or (method == "by_error" and (iteration % 10 == 0) and opt.num_multiview_ssim==0):
                ssim_value_rgb, _ = ssim(image_tensor_rgb, gt_image_tensor_rgb)
                Lssim_rgb = (1 - ssim_value_rgb) / 2
                loss_rgb = Ll1_rgb + opt.lambda_dssim * Lssim_rgb
            else:
                loss_rgb = Ll1_rgb
            Ll1_thermal = torch.tensor(0.0, device="cuda")
            loss_thermal = torch.tensor(0.0, device="cuda")
            loss = loss_rgb
            loss_prior = torch.tensor(0.0, device=loss.device)
            entropy = torch.tensor(0.0, device=loss.device)
            p = torch.tensor([1.0, 0.0, 0.0], device=loss.device)
        elif opt.thermal_only:
            # ---- Thermal-only: single modality loss ----
            Ll1_rgb = torch.tensor(0.0, device="cuda")
            Ll1_rgb_items = torch.zeros(1, 1, device="cuda")
            loss_rgb = torch.tensor(0.0, device="cuda")

            Ll1_thermal = l1_loss(image_tensor_thermal, gt_image_tensor_thermal, keepdim=True)
            Ll1_thermal_items = Ll1_thermal.detach()
            Ll1_thermal = Ll1_thermal.mean()
            loss_thermal = Ll1_thermal
            loss = loss_thermal  # only thermal loss, no RGB, no routing

            loss_prior = torch.tensor(0.0, device=loss.device)
            entropy = torch.tensor(0.0, device=loss.device)
            p = torch.tensor([1.0, 0.0, 0.0], device=loss.device)  # all shared
        else:
            # ---- Normal dual-modal loss ----
            Ll1_rgb = l1_loss(image_tensor_rgb, gt_image_tensor_rgb, keepdim=True)
            Ll1_rgb_items = Ll1_rgb.detach()
            Ll1_rgb = Ll1_rgb.mean()

            if opt.lambda_dssim > 0. and sampled_frame_no != None or (method == "by_error" and (iteration % 10 == 0) and opt.num_multiview_ssim==0):
                ssim_value_rgb, _ = ssim(image_tensor_rgb, gt_image_tensor_rgb)
                Lssim_rgb = (1 - ssim_value_rgb) / 2
                loss_rgb = Ll1_rgb + opt.lambda_dssim * Lssim_rgb
            else:
                loss_rgb = Ll1_rgb

            Ll1_thermal = l1_loss(image_tensor_thermal, gt_image_tensor_thermal, keepdim=True)
            Ll1_thermal = Ll1_thermal.mean()
            if opt.thermal_pose_gray_grad_only and scene.has_thermal_pose:
                thermal_pose_gray_loss = _thermal_gray_l1(image_tensor_thermal, gt_image_tensor_thermal)

            lambda_th = 1.0
            lambda_rgb = 1.0
            if iteration > 500:
                if ema_modal_loss:
                    beta = 0.99
                    if ema_loss_rgb is None:
                        ema_loss_rgb = loss_rgb.detach()
                        ema_loss_thermal = Ll1_thermal.detach()
                    else:
                        ema_loss_rgb = beta * ema_loss_rgb + (1.0 - beta) * loss_rgb.detach()
                        ema_loss_thermal = beta * ema_loss_thermal + (1.0 - beta) * Ll1_thermal.detach()
                    s = ema_loss_rgb + ema_loss_thermal
                    lambda_th = s / torch.clamp(ema_loss_thermal, min=1.0e-8)
                    lambda_rgb = s / torch.clamp(ema_loss_rgb, min=1.0e-8)
                else:
                    s = (loss_rgb.detach() + Ll1_thermal.detach())
                    lambda_th  = (s / Ll1_thermal.detach())
                    lambda_rgb = (s / loss_rgb.detach())
            loss_thermal = lambda_th * Ll1_thermal
            loss_rgb_w = lambda_rgb * loss_rgb
            loss = 0.5 * (loss_rgb_w + loss_thermal)

            p = s_soft.mean(dim=0)
            loss_prior = torch.tensor(0.0, device=loss.device)
            entropy = torch.tensor(0.0, device=loss.device)

            # Routing prior loss uses soft probabilities to avoid hard collapse.
            if routing_stage in {"B", "C"}:
                if routing_stage == "B":
                    p0 = torch.tensor([opt.modality_prior_b_shared, opt.modality_prior_b_rgb, opt.modality_prior_b_thermal], device=p.device)
                else:
                    t0 = opt.modality_stage_b_until
                    t1 = opt.modality_stage_c_until
                    alpha = (iteration - t0) / max(1, (t1 - t0))
                    alpha = float(max(0.0, min(1.0, alpha)))
                    p0_start = torch.tensor([opt.modality_prior_c_start_shared,
                                            opt.modality_prior_c_start_rgb,
                                            opt.modality_prior_c_start_thermal], device=p.device)
                    p0_end   = torch.tensor([opt.modality_prior_c_end_shared,
                                            opt.modality_prior_c_end_rgb,
                                            opt.modality_prior_c_end_thermal], device=p.device)
                    p0 = (1 - alpha) * p0_start + alpha * p0_end
                p = torch.clamp(p, min=1e-8)
                p0 = torch.clamp(p0, min=1e-8)
                loss_prior = torch.sum(p * torch.log(p / p0))
                loss = loss + opt.modality_prior_weight * loss_prior

                if opt.modality_entropy_weight > 0:
                    if routing_stage == "B":
                        entropy = -torch.mean(torch.sum(s_soft * torch.log(torch.clamp(s_soft, min=1e-8)), dim=-1))
                        loss = loss + opt.modality_entropy_weight * entropy
                    elif routing_stage == "C":
                        entropy = -torch.mean(torch.sum(s_soft * torch.log(torch.clamp(s_soft, min=1e-8)), dim=-1))
                        loss = loss + (0.1 * opt.modality_entropy_weight) * entropy
        intrinsic_prior = torch.tensor(0.0, device=loss.device)
        if opt.thermal_intrinsic_prior_weight > 0:
            intrinsic_prior = _thermal_intrinsic_prior()
            loss = loss + opt.thermal_intrinsic_prior_weight * intrinsic_prior
        temporal_prior = torch.tensor(0.0, device=loss.device)
        if scene.temporal_alignment_enabled:
            temporal_prior = (
                scene.temporal_offset_frames() / scene.temporal_offset_max_frames
            ).square()
            loss = loss + opt.temporal_offset_prior_weight * temporal_prior
        joint_clock_alignment_loss = torch.zeros((), device=loss.device)
        joint_clock_alignment_term = torch.zeros((), device=loss.device)
        joint_clock_components = {
            "mind": torch.zeros((), device=loss.device),
            "ngf": torch.zeros((), device=loss.device),
            "routed_ngf": torch.zeros((), device=loss.device),
            "temporal_consensus": torch.zeros((), device=loss.device),
            "temporal_consensus_left": torch.zeros((), device=loss.device),
            "temporal_consensus_right": torch.zeros((), device=loss.device),
        }
        joint_clock_weight = 0.0
        clock_loss_iteration = (
            clock_step_candidate if strict_step_budget_v2 else iteration)
        if (joint_clock_loss_enabled
                and step_calibration_optimizers
                and (clock_target_steps < 0
                     or temporal_offset_optimizer_step_count < clock_target_steps)
                and clock_loss_iteration >= joint_clock_loss_start_iter
                and clock_loss_iteration % joint_clock_loss_interval == 0):
            ramp_progress = min(
                1.0,
                (clock_loss_iteration - joint_clock_loss_start_iter + 1)
                / joint_clock_loss_ramp_iters)
            joint_clock_weight = joint_clock_loss_weight * ramp_progress
            # consensus_only optimizes the cached temporal objective. Spatial
            # image diagnostics do not contribute to its loss or gradients.
            if not (temporal_consensus_enabled
                    and temporal_consensus_mode == "consensus_only"):
                spatial_alignment_loss, spatial_components = (
                    clock_loss.batch_image_loss(
                        image_tensor_thermal, gt_image_tensor_thermal))
                joint_clock_components.update(spatial_components)
            if temporal_consensus_enabled:
                if self_calibrating_clock:
                    consensus_loss, consensus_sides = temporal_consensus.loss(
                        scene.temporal_offset_frames(),
                        scene.temporal_drift_frames(), clock_loss_iteration)
                else:
                    consensus_loss, consensus_sides = temporal_consensus.loss(
                        scene.temporal_offset_frames(),
                        scene.temporal_drift_frames())
                joint_clock_components["temporal_consensus"] = consensus_loss
                joint_clock_components["temporal_consensus_left"] = (
                    consensus_sides["left"])
                joint_clock_components["temporal_consensus_right"] = (
                    consensus_sides["right"])
                if temporal_consensus_mode == "consensus_only":
                    joint_clock_alignment_loss = consensus_loss
                else:
                    joint_clock_alignment_loss = 0.5 * (
                        consensus_loss + spatial_alignment_loss)
            else:
                joint_clock_alignment_loss = spatial_alignment_loss
            joint_clock_alignment_term = (
                joint_clock_weight * joint_clock_alignment_loss)
        psnr_rgb = psnr(image_tensor_rgb, gt_image_tensor_rgb).mean().double()
        psnr_thermal = (torch.tensor(0.0, device="cuda") if opt.rgb_only_teacher
                        else psnr(image_tensor_thermal, gt_image_tensor_thermal).mean().double())
        
        del image_tensor_rgb, gt_image_tensor_rgb
        if not opt.rgb_only_teacher:
            del image_tensor_thermal, gt_image_tensor_thermal

        if tb_writer:
            tb_writer.add_scalar(
                "loss/total",
                (loss + joint_clock_alignment_term).item(), iteration)
            if not opt.thermal_only:
                tb_writer.add_scalar("loss/rgb", loss_rgb.item(), iteration)
            if not opt.rgb_only_teacher:
                tb_writer.add_scalar("loss/thermal", loss_thermal.item(), iteration)
            if opt.thermal_intrinsic_prior_weight > 0:
                tb_writer.add_scalar("thermal_intrinsic/prior", intrinsic_prior.item(), iteration)
            if scene.temporal_alignment_enabled:
                tb_writer.add_scalar(
                    "temporal_alignment/offset_frames",
                    scene.temporal_offset_frames().detach().item(), iteration)
                tb_writer.add_scalar(
                    "temporal_alignment/prior", temporal_prior.item(), iteration)
            if joint_clock_loss_enabled:
                tb_writer.add_scalar(
                    "joint_clock/weight", joint_clock_weight, iteration)
                tb_writer.add_scalar(
                    "joint_clock/loss", joint_clock_alignment_loss.item(), iteration)
                if not (temporal_consensus_enabled
                        and temporal_consensus_mode == "consensus_only"):
                    tb_writer.add_scalar(
                        "joint_clock/mind", joint_clock_components["mind"].item(), iteration)
                    tb_writer.add_scalar(
                        "joint_clock/ngf", joint_clock_components["ngf"].item(), iteration)
                    tb_writer.add_scalar(
                        "joint_clock/routed_ngf",
                        joint_clock_components["routed_ngf"].item(), iteration)
                if temporal_consensus_enabled:
                    tb_writer.add_scalar(
                        "joint_clock/temporal_consensus",
                        joint_clock_components["temporal_consensus"].item(),
                        iteration)
                    tb_writer.add_scalar(
                        "joint_clock/temporal_consensus_left",
                        joint_clock_components[
                            "temporal_consensus_left"].item(), iteration)
                    tb_writer.add_scalar(
                        "joint_clock/temporal_consensus_right",
                        joint_clock_components[
                            "temporal_consensus_right"].item(), iteration)
            if not opt.thermal_only:
                if opt.thermal_pose_gray_grad_only:
                    tb_writer.add_scalar("loss/thermal_pose_gray", thermal_pose_gray_loss.item(), iteration)
                tb_writer.add_scalar("psnr/rgb", psnr_rgb.item(), iteration)
            if not opt.rgb_only_teacher:
                tb_writer.add_scalar("psnr/thermal", psnr_thermal.item(), iteration)
            if not opt.thermal_only:
                tb_writer.add_scalar("routing/entropy", entropy.item(), iteration)
                tb_writer.add_scalar("routing/prior_kl", loss_prior.item(), iteration)
                tb_writer.add_scalar("routing/p_shared", p[0].item(), iteration)
                tb_writer.add_scalar("routing/p_rgb", p[1].item(), iteration)
                tb_writer.add_scalar("routing/p_thermal", p[2].item(), iteration)

            # Thermal pose logging (Camera-level params)
            if scene.has_thermal_pose:
                cam0 = scene.getTrainCameras()[0]
                dq_norm = cam0.thermal_delta_quaternion.norm().item()
                dt_norm = cam0.thermal_delta_translation.norm().item()
                tb_writer.add_scalar("thermal_pose/delta_quat_norm", dq_norm, iteration)
                tb_writer.add_scalar("thermal_pose/delta_trans_norm", dt_norm, iteration)
                # Log first camera thermal FoV
                if hasattr(cam0, 'learnable_tfovx'):
                    effective_fovx, effective_fovy = cam0.get_thermal_fovs()
                    tb_writer.add_scalar("thermal_intrinsic/tfovx", effective_fovx.item(), iteration)
                    tb_writer.add_scalar("thermal_intrinsic/tfovy", effective_fovy.item(), iteration)
                    focal_scale = math.tan(0.5 * cam0.TFoVx) / math.tan(0.5 * effective_fovx.item())
                    tb_writer.add_scalar("thermal_intrinsic/focal_scale", focal_scale, iteration)

        # for i in range(len(Ll1_items)):
        #     loss_list[cam_no_list[i], frame_no_list[i]] = Ll1_items[i].item()

        if opt.thermal_only:
            for i in range(len(Ll1_thermal_items)):
                loss_list[cam_no_list[i], frame_no_list[i]] = Ll1_thermal_items[i].item()
        else:
            for i in range(len(Ll1_rgb_items)):
                loss_list[cam_no_list[i], frame_no_list[i]] = Ll1_rgb_items[i].item()

        sample_names = [camera.image_name for camera in viewpoint_cams]

        # use l1 instead of opacity reset
        loss_opacity = torch.zeros((), device=loss.device)
        if opt.opacity_l1_coef_fine > 0.:
            loss_opacity = opt.opacity_l1_coef_fine * torch.sigmoid(
                gaussians._opacity.mean())
            if not torch.isfinite(loss_opacity):
                raise FloatingPointError(
                    f"Non-finite opacity regularizer at iteration {iteration}; "
                    f"cameras={sample_names}"
                )
            loss += loss_opacity

        # embedding reg using knn (https://github.com/JonathonLuiten/Dynamic3DGaussians)
        if prev_num_pts != gaussians._xyz.shape[0]:
            neighbor_sq_dist, neighbor_indices = o3d_knn(gaussians._xyz.detach().cpu().numpy(), 20)
            neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
            neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
            neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
            prev_num_pts = gaussians._xyz.shape[0]
        
        # Broadcast the source embedding instead of materializing a 20x copy.
        # This is value- and gradient-identical to repeat(1, 20, 1).
        loss_reg = weighted_l2_loss_v2(
            gaussians._embedding[:, None, :],
            gaussians._embedding[neighbor_indices],
            neighbor_weight)
        if not torch.isfinite(loss_reg):
            bad_embedding = int(
                (~torch.isfinite(gaussians._embedding)).sum().item())
            bad_xyz = int((~torch.isfinite(gaussians._xyz)).sum().item())
            bad_neighbor_weight = int(
                (~torch.isfinite(neighbor_weight)).sum().item())
            zero_neighbor_weight = int((neighbor_weight == 0).sum().item())
            raise FloatingPointError(
                f"Non-finite embedding regularizer at iteration {iteration}; "
                f"cameras={sample_names}; bad_embedding={bad_embedding}; "
                f"bad_xyz={bad_xyz}; bad_neighbor_weight={bad_neighbor_weight}; "
                f"zero_neighbor_weight={zero_neighbor_weight}"
            )
        loss += opt.reg_coef * loss_reg

        # smoothness reg on temporal embeddings
        loss_temporal_tv = torch.zeros((), device=loss.device)
        if opt.coef_tv_temporal_embedding > 0:
            weights = gaussians._deformation.weight
            N, C = weights.shape
            first_difference = weights[1:,:] - weights[N-1,:]
            second_difference = first_difference[1:,:] - first_difference[N-2,:]
            loss_temporal_tv = (
                opt.coef_tv_temporal_embedding
                * torch.square(second_difference).mean()
            )
            if not torch.isfinite(loss_temporal_tv):
                bad_weights = int((~torch.isfinite(weights)).sum().item())
                raise FloatingPointError(
                    f"Non-finite temporal TV at iteration {iteration}; "
                    f"cameras={sample_names}; bad_weights={bad_weights}"
                )
            loss += loss_temporal_tv

        if tb_writer:
            tb_writer.add_scalar("loss/opacity_l1", loss_opacity.item(), iteration)
            tb_writer.add_scalar("loss/embedding_reg", loss_reg.item(), iteration)
            tb_writer.add_scalar(
                "loss/temporal_tv", loss_temporal_tv.item(), iteration)

        reconstruction_loss = loss
        loss = reconstruction_loss + joint_clock_alignment_term

        need_modality_densify_grads = (
            (opt.thermal_only or opt.enable_modality_densify)
            and (opt.thermal_only or routing_stage in {"C", "D"})
            and step_scene_optimizer
            and (scene_optimizer_step_count + (1 if strict_step_budget_v2 else 0)
                 if strict_step_budget_v2 else iteration) < opt.densify_until_iter
            and (scene_optimizer_step_count + (1 if strict_step_budget_v2 else 0)
                 if strict_step_budget_v2 else iteration) > opt.densify_from_iter
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at iteration {iteration}: {loss.detach().item()}"
            )

        joint_clock_alignment_raw_gradient = None
        joint_clock_alignment_drift_raw_gradient = None
        if (joint_clock_loss_enabled and joint_clock_weight > 0.0
                and iteration >= opt.temporal_offset_start_iter
                and (not strict_scene_freeze
                     or step_calibration_optimizers)
                and (clock_target_steps < 0
                     or temporal_offset_optimizer_step_count < clock_target_steps)
                and (strict_step_budget_v2
                     or not (opt.temporal_offset_freeze_after >= 0
                             and iteration > opt.temporal_offset_freeze_after))):
            clock_parameters = [scene.temporal_offset_raw]
            if scene.temporal_affine_clock_enabled:
                clock_parameters.append(scene.temporal_drift_raw)
            clock_gradients = torch.autograd.grad(
                joint_clock_alignment_term,
                tuple(clock_parameters),
                retain_graph=True,
                allow_unused=False,
            )
            joint_clock_alignment_raw_gradient = clock_gradients[0]
            if scene.temporal_affine_clock_enabled:
                joint_clock_alignment_drift_raw_gradient = clock_gradients[1]
            if not all(bool(torch.isfinite(gradient).all())
                       for gradient in clock_gradients):
                raise FloatingPointError(
                    "Non-finite direct joint clock alignment gradient")
        scene_loss_for_backward = (
            reconstruction_loss if joint_clock_loss_enabled else loss)

        if (memory_safe_backward
                and gaussians.get_xyz.shape[0]
                >= memory_safe_gaussian_threshold):
            torch.cuda.empty_cache()
            memory_safe_cache_release_count += 1
            if iteration % 500 == 0:
                print("MEMORY_SAFE_CACHE_RELEASE " + json.dumps({
                    "iteration": int(iteration),
                    "gaussian_count": int(gaussians.get_xyz.shape[0]),
                    "release_count": int(memory_safe_cache_release_count),
                }, sort_keys=True))

        if (v34_block_calibration and not fourarm and scene.has_thermal_pose
                and thermal_pose_params):
            v34_pose_grads = None
            if step_calibration_optimizers:
                v34_pose_grads = torch.autograd.grad(
                    v34_pose_loss, thermal_pose_params, allow_unused=False)
            if (step_scene_optimizer
                    and (strict_scene_freeze
                         or optimizer_phase != "calibration")):
                scene_loss_for_backward.backward()
                scene_backward_phase_counts[optimizer_phase] += 1
            for parameter in thermal_pose_params:
                parameter.grad = None
            if v34_pose_grads is not None:
                for parameter, gradient in zip(
                        thermal_pose_params, v34_pose_grads):
                    if not bool(torch.isfinite(gradient).all()):
                        raise FloatingPointError(
                            "Non-finite v34 structural pose gradient")
                    parameter.grad = gradient.detach().clone()
            if scene.temporal_alignment_enabled:
                scene.temporal_offset_raw.grad = None
                if scene.temporal_affine_clock_enabled:
                    scene.temporal_drift_raw.grad = None
            if iteration == opt.thermal_pose_start_iter or iteration % 500 == 0:
                print("V34_POSE_STRUCTURAL_STEP " + json.dumps({
                    "iteration": int(iteration),
                    "active": optimizer_phase == "calibration",
                    "loss": float(v34_pose_loss.detach().item()),
                    "mind": (None if v34_pose_components is None else float(
                        v34_pose_components["mind"].detach().item())),
                    "ngf": (None if v34_pose_components is None else float(
                        v34_pose_components["ngf"].detach().item())),
                    "pose_batch": v34_pose_batch_report,
                }, sort_keys=True))
        elif opt.thermal_pose_gray_grad_only and scene.has_thermal_pose and thermal_pose_params:
            thermal_pose_grads = None
            if (step_calibration_optimizers
                    and thermal_pose_gray_loss.requires_grad):
                thermal_pose_grads = torch.autograd.grad(
                    thermal_pose_gray_loss,
                    thermal_pose_params,
                    retain_graph=step_scene_optimizer,
                    allow_unused=True,
                )
            if step_scene_optimizer:
                scene_loss_for_backward.backward()
                scene_backward_phase_counts[optimizer_phase] += 1
            if thermal_pose_grads is not None:
                for param, grad in zip(thermal_pose_params, thermal_pose_grads):
                    param.grad = None if grad is None else grad.detach().clone()
        else:
            if step_scene_optimizer:
                scene_loss_for_backward.backward()
                scene_backward_phase_counts[optimizer_phase] += 1

        joint_clock_reconstruction_raw_gradient = None
        joint_clock_reconstruction_drift_raw_gradient = None
        if joint_clock_loss_enabled:
            if scene.temporal_offset_raw.grad is not None:
                joint_clock_reconstruction_raw_gradient = (
                    scene.temporal_offset_raw.grad.detach().clone())
            if (scene.temporal_affine_clock_enabled
                    and scene.temporal_drift_raw.grad is not None):
                joint_clock_reconstruction_drift_raw_gradient = (
                    scene.temporal_drift_raw.grad.detach().clone())
            if zero_init_routed_clock and not clock_total_gradient:
                scene.temporal_offset_raw.grad = (
                    None if joint_clock_alignment_raw_gradient is None
                    else joint_clock_alignment_raw_gradient.detach().clone())
                if scene.temporal_affine_clock_enabled:
                    scene.temporal_drift_raw.grad = (
                        None if joint_clock_alignment_drift_raw_gradient is None
                        else joint_clock_alignment_drift_raw_gradient.detach().clone())
            elif joint_clock_alignment_raw_gradient is not None:
                if joint_clock_reconstruction_raw_gradient is None:
                    joint_clock_reconstruction_raw_gradient = torch.zeros_like(
                        joint_clock_alignment_raw_gradient)
                scene.temporal_offset_raw.grad = (
                    joint_clock_reconstruction_raw_gradient
                    + joint_clock_alignment_raw_gradient.detach())
                if scene.temporal_affine_clock_enabled:
                    if joint_clock_reconstruction_drift_raw_gradient is None:
                        joint_clock_reconstruction_drift_raw_gradient = (
                            torch.zeros_like(
                                joint_clock_alignment_drift_raw_gradient))
                    scene.temporal_drift_raw.grad = (
                        joint_clock_reconstruction_drift_raw_gradient
                        + joint_clock_alignment_drift_raw_gradient.detach())

        audit_rgb_teacher = bool(
            opt.rgb_only_teacher
            and (iteration == 1 or iteration == final_iter or iteration % 1000 == 0)
        )
        if audit_rgb_teacher:
            leaked = []
            thermal_groups = {
                "thermal_dc", "thermal_rest", "thermal_opacity",
                "thermal_embedding", "logit_modality",
            }
            for group in gaussians.optimizer.param_groups:
                group_name = group.get("name", "?")
                if group_name not in thermal_groups:
                    continue
                if group["lr"] != 0:
                    leaked.append(f"optimizer_lr:{group_name}={group['lr']}")
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        if torch.count_nonzero(parameter.grad).item() != 0:
                            leaked.append(f"optimizer_grad:{group_name}")
            for name, parameter in gaussians._deformation.named_parameters():
                if "thermal" in name.lower():
                    if parameter.requires_grad:
                        leaked.append(f"deformation_requires_grad:{name}")
                    if parameter.grad is not None:
                        if torch.count_nonzero(parameter.grad).item() != 0:
                            leaked.append(f"deformation_grad:{name}")
            for cam_index, cam in enumerate(train_cams):
                for name in ("thermal_delta_quaternion", "thermal_delta_translation",
                             "learnable_tfovx", "learnable_tfovy"):
                    parameter = getattr(cam, name, None)
                    if parameter is not None and parameter.grad is not None:
                        if torch.count_nonzero(parameter.grad).item() != 0:
                            leaked.append(f"camera[{cam_index}].{name}")
            if leaked:
                raise RuntimeError(f"RGB-only teacher Thermal gradient leakage: {leaked}")
            print(f"[RGBTeacher] RGB_ONLY_GRAD_AUDIT_PASS iteration={iteration}")

        bad_gradient_groups = []
        gradient_max_groups = {}
        for group in gaussians.optimizer.param_groups:
            bad_count = 0
            finite_max = 0.0
            for parameter in group["params"]:
                if parameter.grad is not None:
                    finite = torch.isfinite(parameter.grad)
                    bad_count += int((~finite).sum().item())
                    if finite.any():
                        finite_max = max(
                            finite_max,
                            parameter.grad[finite].abs().max().item(),
                        )
            gradient_max_groups[group.get("name", "?")] = finite_max
            if bad_count:
                bad_gradient_groups.append(f"{group.get('name', '?')}:{bad_count}")
        if bad_gradient_groups:
            raise FloatingPointError(
                f"Non-finite Gaussian gradients at iteration {iteration}; "
                f"cameras={sample_names}; groups={','.join(bad_gradient_groups)}; "
                f"finite_absmax={gradient_max_groups}"
            )

        if iteration == 1 or iteration == final_iter or iteration % 500 == 0:
            def _grad_norm(parameter):
                if parameter.grad is None:
                    return 0.0
                return float(parameter.grad.detach().norm().item())
            print("MODALITY_GRADIENT_STATS " + json.dumps({
                "iteration": int(iteration),
                "rgb_feature_grad_norm": _grad_norm(gaussians._features_dc),
                "thermal_feature_grad_norm": _grad_norm(gaussians._thermal_dc),
                "xyz_grad_norm": _grad_norm(gaussians._xyz),
                "modality_logit_grad_norm": _grad_norm(gaussians._logit_modality),
            }, sort_keys=True))

        if tb_writer and scene.has_thermal_pose:
            cam0 = scene.getTrainCameras()[0]
            if cam0.learnable_tfovx.grad is not None:
                tb_writer.add_scalar(
                    "thermal_intrinsic/grad_tfovx",
                    cam0.learnable_tfovx.grad.detach().abs().item(),
                    iteration,
                )
            if cam0.learnable_tfovy.grad is not None:
                tb_writer.add_scalar(
                    "thermal_intrinsic/grad_tfovy",
                    cam0.learnable_tfovy.grad.detach().abs().item(),
                    iteration,
                )
        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_rgb_list[0])
        viewspace_grad_rgb = None
        viewspace_grad_th = None
        for viewspace_point_tensor_rgb in viewspace_point_tensor_rgb_list:
            if viewspace_point_tensor_rgb.grad is not None:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_rgb.grad
        if need_modality_densify_grads:
            if opt.thermal_only:
                viewspace_grad_th = torch.zeros_like(viewspace_point_tensor_th_list[0])
                for viewspace_point_tensor_th in viewspace_point_tensor_th_list:
                    if viewspace_point_tensor_th.grad is not None:
                        viewspace_grad_th += viewspace_point_tensor_th.grad
            else:
                viewspace_grad_rgb = torch.zeros_like(viewspace_point_tensor_rgb_list[0])
                viewspace_grad_th = torch.zeros_like(viewspace_point_tensor_th_list[0])
                for viewspace_point_tensor_rgb, viewspace_point_tensor_th in zip(
                    viewspace_point_tensor_rgb_list, viewspace_point_tensor_th_list
                ):
                    if viewspace_point_tensor_rgb.grad is not None:
                        viewspace_grad_rgb += viewspace_point_tensor_rgb.grad
                    if viewspace_point_tensor_th.grad is not None:
                        viewspace_grad_th += viewspace_point_tensor_th.grad
                rgb_grad_scale = 2.0 / (lambda_rgb.detach() if torch.is_tensor(lambda_rgb) else lambda_rgb)
                viewspace_grad_rgb *= rgb_grad_scale
                viewspace_grad_th *= 2.0

        iter_end.record()

        if iteration in saving_iterations:
            elapsed_time = time()
            
            total_time_seconds = elapsed_time - start_time
            hours, remainder = divmod(total_time_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            with open(os.path.join(args.model_path, 'training_time.txt'), 'a') as file:
                file.write(f'Iteration {iteration}: {total_time_seconds} seconds ... {int(hours)}h {int(minutes)}m {seconds}sec  points: {gaussians._xyz.shape[0]}\n')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            if opt.thermal_only:
                ema_psnr_for_log = 0.4 * psnr_thermal + 0.6 * ema_psnr_for_log
            else:
                ema_psnr_for_log = 0.4 * psnr_rgb + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                # Line 1: tqdm bar — core metrics
                if opt.thermal_only:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "th": f"{loss_thermal:.{3}f}",
                        "P_th": f"{psnr_thermal:.1f}",
                        "pts": f"{total_point}",
                    })
                elif opt.rgb_only_teacher:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "rgb": f"{loss_rgb:.{3}f}",
                        "P_rgb": f"{psnr_rgb:.1f}",
                        "pts": f"{total_point}",
                    })
                else:
                    progress_bar.set_postfix({
                        "Loss": f"{ema_loss_for_log:.{4}f}",
                        "rgb": f"{loss_rgb:.{3}f}",
                    "th": f"{loss_thermal:.{3}f}",
                    "P_rgb": f"{psnr_rgb:.1f}",
                    "P_th": f"{psnr_thermal:.1f}",
                    "pts": f"{total_point}",
                })
                # Line 2: tqdm.write — thermal pose & intrinsics
                if iteration % 50 == 0 and not opt.rgb_only_teacher:
                    extra = []
                    if scene.has_thermal_pose:
                        cam0 = scene.getTrainCameras()[0]
                        if hasattr(cam0, 'thermal_delta_quaternion'):
                            dq = cam0.thermal_delta_quaternion.norm().item()
                            dt = cam0.thermal_delta_translation.norm().item()
                            extra.append(f"dQ={dq:.6f} dT={dt:.6f}")
                    ref_cam = scene.getTrainCameras()[0]
                    if hasattr(ref_cam, 'learnable_tfovx'):
                        effective_fovx, effective_fovy = ref_cam.get_thermal_fovs()
                        extra.append(f"tFovX={effective_fovx.item():.6f} tFovY={effective_fovy.item():.6f}")
                    if extra:
                        tqdm.write("  " + "  |  ".join(extra))
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
 
            if (iteration in saving_iterations and not full_scene_steps):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            timer.start()
            # Densification
            scene_schedule_step = (
                scene_optimizer_step_count + 1
                if strict_step_budget_v2 else iteration)
            if (step_scene_optimizer
                    and scene_schedule_step < opt.densify_until_iter):
                densification_phase_counts[optimizer_phase] += 1
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # In thermal_only mode, skip unified densification stats (modality-aware fills it)
                if not opt.thermal_only:
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter_rgb)

                opacity_threshold = opt.opacity_threshold_fine_init - scene_schedule_step*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - scene_schedule_step*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                grad_rgb_for_densify = viewspace_grad_rgb.detach() if viewspace_grad_rgb is not None else None
                grad_th_for_densify = viewspace_grad_th.detach() if viewspace_grad_th is not None else None
                gaussians.add_modality_densification_stats(
                    grad_rgb_for_densify,
                    grad_th_for_densify,
                    visibility_filter_rgb,
                    visibility_filter_th,
                    thermal_only=opt.thermal_only,
                )

                if (scene_schedule_step > opt.densify_from_iter
                        and scene_schedule_step % opt.densification_interval == 0):
                    size_threshold = 20 if scene_schedule_step > opt.opacity_reset_interval else None
                    gaussians.densify(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        routing_stage=routing_stage,
                        enable_modality_densify=opt.enable_modality_densify or opt.thermal_only,
                        thermal_only=opt.thermal_only,
                    )
                if (scene_schedule_step > opt.pruning_from_iter
                        and scene_schedule_step % opt.pruning_interval == 0):
                    size_threshold = 20 if scene_schedule_step > opt.opacity_reset_interval else None

                    gaussians.prune(
                        densify_threshold,
                        opacity_threshold,
                        scene.cameras_extent,
                        size_threshold,
                        routing_stage=routing_stage,
                        enable_modality_densify=opt.enable_modality_densify,
                    )

                    if (opt.reset_opacity_ratio > 0
                            and scene_schedule_step % opt.pruning_interval == 0):
                        gaussians.reset_opacity(opt.reset_opacity_ratio)

            # Optimizer step
            run_optimizer_block = schedule.should_run_optimizer_block(
                iteration, opt.iterations, full_scene_steps,
                strict_step_budget_v2=strict_step_budget_v2)
            if run_optimizer_block:
                if step_scene_optimizer:
                    gaussians.optimizer.step()
                    scene_optimizer_step_count += 1
                    scene_optimizer_last_step_iteration = iteration
                bad_parameter_groups = []
                bad_optimizer_states = []
                for group in gaussians.optimizer.param_groups:
                    bad_count = sum(
                        int((~torch.isfinite(parameter)).sum().item())
                        for parameter in group["params"]
                    )
                    if bad_count:
                        bad_parameter_groups.append(
                            f"{group.get('name', '?')}:{bad_count}")
                    state_bad_count = 0
                    for parameter in group["params"]:
                        for value in gaussians.optimizer.state.get(parameter, {}).values():
                            if torch.is_tensor(value):
                                state_bad_count += int(
                                    (~torch.isfinite(value)).sum().item())
                    if state_bad_count:
                        bad_optimizer_states.append(
                            f"{group.get('name', '?')}:{state_bad_count}")
                if bad_parameter_groups or bad_optimizer_states:
                    raise FloatingPointError(
                        f"Non-finite Gaussian optimizer result "
                        f"at iteration {iteration}; cameras={sample_names}; "
                        f"parameters={','.join(bad_parameter_groups)}; "
                        f"states={','.join(bad_optimizer_states)}; "
                        f"grad_absmax={gradient_max_groups}"
                    )
                gaussians.optimizer.zero_grad(set_to_none = True)

                # Thermal pose optimizers (Self-Cali-GS style: separate Adam per param type)
                if scene.has_thermal_pose:
                    train_thermal_pose = (
                        iteration >= opt.thermal_pose_start_iter
                        and step_calibration_optimizers
                        and (pose_target_steps < 0
                             or thermal_pose_optimizer_step_count < pose_target_steps)
                        and not (
                            opt.thermal_pose_freeze_after >= 0
                            and iteration > opt.thermal_pose_freeze_after
                        )
                    )
                    if train_thermal_pose:
                        scene.optimizer_thermal_rotation.step()
                        scene.optimizer_thermal_translation.step()
                        thermal_pose_optimizer_step_count += 1
                        if v34_block_calibration:
                            scene.clamp_thermal_pose(
                                max_rotation_degrees=5.0,
                                max_translation_fraction=0.1)
                        scene.scheduler_thermal_rotation.step()
                        scene.scheduler_thermal_translation.step()
                    if iteration == opt.thermal_pose_start_iter:
                        print(f"[ThermalPose] active from iter {opt.thermal_pose_start_iter}")
                    if (opt.thermal_pose_freeze_after >= 0
                            and iteration == opt.thermal_pose_freeze_after + 1):
                        print(f"[ThermalPose] frozen after iter {opt.thermal_pose_freeze_after}")
                    scene.optimizer_thermal_rotation.zero_grad(set_to_none=True)
                    scene.optimizer_thermal_translation.zero_grad(set_to_none=True)

                    intrinsic_freeze_after = (
                        opt.thermal_intrinsic_freeze_after
                        if opt.thermal_intrinsic_freeze_after >= 0
                        else opt.thermal_pose_freeze_after
                    )
                    train_thermal_intrinsic = (
                        opt.thermal_intrinsic_lr > 0
                        and iteration >= opt.thermal_intrinsic_start_iter
                        and step_calibration_optimizers
                        and not (
                            intrinsic_freeze_after >= 0
                            and iteration > intrinsic_freeze_after
                        )
                    )
                    if train_thermal_intrinsic:
                        scene.optimizer_thermal_fovx.step()
                        scene.scheduler_thermal_fovx.step()
                        if scene.optimizer_thermal_fovy is not None:
                            scene.optimizer_thermal_fovy.step()
                            scene.scheduler_thermal_fovy.step()
                        scene.clamp_thermal_intrinsics()
                    if (opt.thermal_intrinsic_lr > 0
                            and iteration == opt.thermal_intrinsic_start_iter):
                        print(f"[ThermalIntrinsic] active from iter {opt.thermal_intrinsic_start_iter}")
                    if (opt.thermal_intrinsic_lr > 0
                            and intrinsic_freeze_after >= 0
                            and iteration == intrinsic_freeze_after + 1):
                        print(f"[ThermalIntrinsic] frozen after iter {intrinsic_freeze_after}")
                    scene.optimizer_thermal_fovx.zero_grad(set_to_none=True)
                    if scene.optimizer_thermal_fovy is not None:
                        scene.optimizer_thermal_fovy.zero_grad(set_to_none=True)

                if scene.temporal_alignment_enabled:
                    train_temporal_offset = (
                        iteration >= opt.temporal_offset_start_iter
                        and (not strict_scene_freeze
                             or step_calibration_optimizers)
                        and (joint_clock_loss_enabled
                             or step_calibration_optimizers)
                        and (joint_clock_loss_enabled
                             or not v34_block_calibration)
                        and (not joint_clock_loss_enabled
                             or clock_loss_iteration >= joint_clock_loss_start_iter)
                and (clock_target_steps < 0
                     or temporal_offset_optimizer_step_count < clock_target_steps)
                        and (strict_step_budget_v2
                             or not (
                                 opt.temporal_offset_freeze_after >= 0
                                 and iteration > opt.temporal_offset_freeze_after))
                    )
                    offset_before = float(
                        scene.temporal_offset_frames().detach().item())
                    drift_before = float(
                        scene.temporal_drift_frames().detach().item())
                    raw_gradient = scene.temporal_offset_raw.grad
                    drift_raw_gradient = (
                        scene.temporal_drift_raw.grad
                        if scene.temporal_affine_clock_enabled else None)
                    if train_temporal_offset:
                        if raw_gradient is None:
                            raise RuntimeError(
                                f"Temporal raw gradient missing at iteration {iteration}")
                        if not bool(torch.isfinite(raw_gradient).all()):
                            raise FloatingPointError(
                                f"Temporal raw gradient is non-finite at iteration {iteration}")
                        if (scene.temporal_affine_clock_enabled
                                and drift_raw_gradient is None):
                            raise RuntimeError(
                                f"Temporal drift gradient missing at iteration {iteration}")
                        if (drift_raw_gradient is not None
                                and not bool(torch.isfinite(
                                    drift_raw_gradient).all())):
                            raise FloatingPointError(
                                f"Temporal drift gradient is non-finite at iteration {iteration}")
                    nuisance_versions_before = None
                    nuisance_parameters = None
                    final_clock_step = (
                        calibration_reconstruction_start_iter
                        + 2 * strict_calibration_steps
                        - calibration_reconstruction_phase_length - 1
                        if strict_step_budget_v2 else min(
                            final_iter,
                            opt.temporal_offset_freeze_after
                            if opt.temporal_offset_freeze_after >= 0
                            else final_iter))
                    if strict_scene_freeze:
                        while (final_clock_step >= 1
                               and (final_clock_step >= final_iter
                                    or schedule.phase_for_iteration(
                                        final_clock_step,
                                        phase_length=(
                                            calibration_reconstruction_phase_length),
                                        start_iteration=(
                                            calibration_reconstruction_start_iter),
                                        calibration_steps=(
                                            strict_calibration_steps
                                            if strict_step_budget_v2 else None),
                                        scene_steps=(
                                            scene_optimizer_target_steps
                                            if strict_step_budget_v2 else None))
                                    != "calibration")):
                            final_clock_step -= 1
                        if final_clock_step < 1:
                            raise RuntimeError(
                                "Strict schedule has no clock optimizer step")
                    audit_clock_step = (
                        train_temporal_offset
                        and iteration in {1, final_clock_step})
                    if audit_clock_step:
                        nuisance_parameters = []
                        for group in gaussians.optimizer.param_groups:
                            nuisance_parameters.extend(group["params"])
                        for camera in scene.getTrainCameras():
                            for name in (
                                    "thermal_delta_quaternion",
                                    "thermal_delta_translation",
                                    "learnable_tfovx", "learnable_tfovy"):
                                parameter = getattr(camera, name, None)
                                if parameter is not None:
                                    nuisance_parameters.append(parameter)
                        clock_ids = {id(scene.temporal_offset_raw)}
                        if scene.temporal_affine_clock_enabled:
                            clock_ids.add(id(scene.temporal_drift_raw))
                        nuisance_parameters = {
                            id(parameter): parameter
                            for parameter in nuisance_parameters
                            if id(parameter) not in clock_ids
                        }
                        nuisance_versions_before = {
                            identity: parameter._version
                            for identity, parameter in nuisance_parameters.items()
                        }
                    if train_temporal_offset:
                        scene.optimizer_temporal_offset.step()
                        temporal_offset_optimizer_step_count += 1
                        if joint_clock_loss_enabled:
                            joint_clock_optimizer_step_count += 1
                            if (joint_clock_alignment_raw_gradient is not None
                                    and float(joint_clock_alignment_raw_gradient.abs().item()) > 0.0):
                                joint_clock_nonzero_alignment_gradient_count += 1
                                joint_clock_nonzero_offset_gradient_count += 1
                            if (joint_clock_alignment_drift_raw_gradient is not None
                                    and float(joint_clock_alignment_drift_raw_gradient.abs().item()) > 0.0):
                                joint_clock_nonzero_drift_gradient_count += 1
                    if audit_clock_step:
                        nuisance_versions_after = {
                            identity: parameter._version
                            for identity, parameter in nuisance_parameters.items()
                        }
                        nuisance_unchanged = (
                            nuisance_versions_before == nuisance_versions_after)
                        print("CLOCK_STEP_NUISANCE_AUDIT " + json.dumps({
                            "iteration": int(iteration),
                            "nuisance_parameter_count": len(nuisance_parameters),
                            "parameter_versions_unchanged": nuisance_unchanged,
                        }, sort_keys=True))
                        if not nuisance_unchanged:
                            raise RuntimeError(
                                "Clock optimizer mutated a nuisance parameter")
                    offset_after = float(
                        scene.temporal_offset_frames().detach().item())
                    drift_after = float(
                        scene.temporal_drift_frames().detach().item())
                    if iteration == opt.temporal_offset_start_iter:
                        print(
                            f"[TemporalAlignment] active from iter "
                            f"{opt.temporal_offset_start_iter}"
                        )
                    if (opt.temporal_offset_freeze_after >= 0
                            and iteration == opt.temporal_offset_freeze_after + 1):
                        print(
                            "[TemporalAlignment] frozen after iter "
                            f"{opt.temporal_offset_freeze_after}; "
                            f"offset_frames={scene.temporal_offset_frames().detach().item():.6f}"
                        )
                    if (strict_step_budget_v2
                            and train_temporal_offset
                            and clock_target_steps > 0
                            and temporal_offset_optimizer_step_count == clock_target_steps):
                        print("STRICT_CLOCK_FREEZE " + json.dumps({
                            "iteration": int(iteration),
                            "clock_optimizer_steps": int(
                                temporal_offset_optimizer_step_count),
                            "offset_frames": float(
                                scene.temporal_offset_frames().detach().item()),
                            "requires_grad_next": False,
                        }, sort_keys=True))
                    log_temporal_step = (
                        iteration == 1
                        or iteration == final_iter
                        or (final_iter <= 500 and iteration % 10 == 0)
                        or (final_iter > 500 and iteration % 500 == 0)
                    )
                    if log_temporal_step:
                        print(
                            "STAGE2_TEMPORAL_STEP " + json.dumps({
                                "iteration": iteration,
                                "clock_step": int(
                                    temporal_offset_optimizer_step_count),
                                "loss": float(loss.detach().item()),
                                "offset_after": offset_after,
                                "offset_before": offset_before,
                                "drift_after": drift_after,
                                "drift_before": drift_before,
                                "endpoint_offsets_after": [
                                    float(value.detach().item())
                                    for value in scene.temporal_endpoint_offsets()
                                ],
                                "raw_gradient": (
                                    None if raw_gradient is None
                                    else float(raw_gradient.detach().item())),
                                "drift_raw_gradient": (
                                    None if drift_raw_gradient is None
                                    else float(drift_raw_gradient.detach().item())),
                                "joint_alignment_gradient": (
                                    None if joint_clock_alignment_raw_gradient is None
                                    else float(joint_clock_alignment_raw_gradient.detach().item())),
                                "joint_alignment_drift_gradient": (
                                    None if joint_clock_alignment_drift_raw_gradient is None
                                    else float(joint_clock_alignment_drift_raw_gradient.detach().item())),
                                "joint_alignment_loss": float(
                                    joint_clock_alignment_loss.detach().item()),
                                "temporal_consensus_loss": (
                                    None if not temporal_consensus_enabled
                                    else float(joint_clock_components[
                                        "temporal_consensus"].detach().item())),
                                "temporal_consensus_left": (
                                    None if not temporal_consensus_enabled
                                    else float(joint_clock_components[
                                        "temporal_consensus_left"].detach().item())),
                                "temporal_consensus_right": (
                                    None if not temporal_consensus_enabled
                                    else float(joint_clock_components[
                                        "temporal_consensus_right"].detach().item())),
                                "joint_clock_weight": joint_clock_weight,
                                "reconstruction_gradient": (
                                    None if joint_clock_reconstruction_raw_gradient is None
                                    else float(joint_clock_reconstruction_raw_gradient.item())),
                                "reconstruction_drift_gradient": (
                                    None if joint_clock_reconstruction_drift_raw_gradient is None
                                    else float(joint_clock_reconstruction_drift_raw_gradient.item())),
                                "trainable": train_temporal_offset,
                            }, sort_keys=True)
                        )
                    scene.optimizer_temporal_offset.zero_grad(set_to_none=True)

                if (calibration_reconstruction_alternation
                        and (iteration == 1
                             or iteration == calibration_reconstruction_start_iter
                             or (iteration >= calibration_reconstruction_start_iter
                                 and (iteration - calibration_reconstruction_start_iter)
                                 % calibration_reconstruction_phase_length == 0))):
                    print("STAGE2_OPTIMIZER_PHASE " + json.dumps({
                        "iteration": iteration,
                        "phase": optimizer_phase,
                        "scene_step": step_scene_optimizer,
                        "thermal_pose_step": bool(
                            scene.has_thermal_pose
                            and iteration >= opt.thermal_pose_start_iter
                            and step_calibration_optimizers),
                        "temporal_step": bool(
                            scene.temporal_alignment_enabled
                            and iteration >= opt.temporal_offset_start_iter
                            and (not strict_scene_freeze
                                 or step_calibration_optimizers)
                            and (joint_clock_loss_enabled
                                 or step_calibration_optimizers)),
                    }, sort_keys=True))

            if iteration in saving_iterations and full_scene_steps:
                print("\n[ITER {}] Saving Gaussians after Scene step {}".format(
                    iteration, scene_optimizer_step_count))
                scene.save(iteration)
                scene_save_step_counts[str(iteration)] = (
                    scene_optimizer_step_count)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                checkpoint_step_counts[str(iteration)] = (
                    scene_optimizer_step_count)

    optimizer_update_counts = {
        "strict_step_budget_v2": bool(strict_step_budget_v2),
        "expected_outer_iterations": int(final_iter),
        "clock_stage2_updates": clock_stage2_update_count,
        "temporal_offset_optimizer_steps": temporal_offset_optimizer_step_count,
        "joint_clock_optimizer_steps": joint_clock_optimizer_step_count,
        "joint_clock_nonzero_alignment_gradients": (
            joint_clock_nonzero_alignment_gradient_count),
        "joint_clock_nonzero_offset_gradients": (
            joint_clock_nonzero_offset_gradient_count),
        "joint_clock_nonzero_drift_gradients": (
            joint_clock_nonzero_drift_gradient_count),
        "gaussian_optimizer_steps": scene_optimizer_step_count,
        "thermal_pose_optimizer_steps": thermal_pose_optimizer_step_count,
        "target_clock_optimizer_steps": clock_target_steps,
        "target_pose_optimizer_steps": pose_target_steps,
        "scene_backward_phase_counts": scene_backward_phase_counts,
        "densification_phase_counts": densification_phase_counts,
        "sh_degree_update_phase_counts": sh_degree_update_phase_counts,
        "last_gaussian_step_iteration": scene_optimizer_last_step_iteration,
        "optimizer_phase_iterations": optimizer_phase_counts,
        "scene_save_step_counts": scene_save_step_counts,
        "checkpoint_step_counts": checkpoint_step_counts,
        "target_gaussian_optimizer_steps": scene_optimizer_target_steps,
        "temporal_consensus_contract": (
            None if temporal_consensus is None
            else temporal_consensus.contract),
        "temporal_consensus_initial_raw_gradients": (
            None if temporal_consensus_initial_gradient is None else {
                "offset_raw": float(
                    temporal_consensus_initial_gradient[0].item()),
                "drift_raw": (
                    None if len(temporal_consensus_initial_gradient) == 1
                    else float(temporal_consensus_initial_gradient[1].item())),
            }),
        "memory_safe_cache_release_count": int(
            memory_safe_cache_release_count),
        "memory_safe_clock_detached": bool(memory_safe_clock_detached),
        "main_clock_optimizer_enabled": bool(joint_clock_loss_enabled),
        "zero_start_clock_phase": zero_start_clock_phase,
    }
    print("FULL_SCENE_OPTIMIZER_COUNTS " + json.dumps(
        optimizer_update_counts, sort_keys=True))
    if ((full_scene_steps or strict_scene_freeze)
            and scene_optimizer_step_count != scene_optimizer_target_steps):
        raise RuntimeError(
            "Gaussian optimizer-step budget mismatch: "
            f"actual={scene_optimizer_step_count}, "
            f"target={scene_optimizer_target_steps}")
    return optimizer_update_counts

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    teacher_files = {}
    teacher_hashes = {}
    temporal_bootstrap = None
    internal_clock_learning = None
    ablation_mode = os.environ.get("ED3DGS_SHIFT20_ABLATION_MODE", "")
    if ablation_mode not in {"", "fixed_clock", "total_gradient", "frozen_pose"}:
        raise RuntimeError(f"Invalid shift20 ablation mode: {ablation_mode!r}")
    fixed_clock_ablation = ablation_mode == "fixed_clock"
    total_gradient_ablation = ablation_mode == "total_gradient"
    frozen_pose_ablation = ablation_mode == "frozen_pose"
    fourarm = os.environ.get("ED3DGS_R25_FOURARM", "")
    valid_fourarms = {"baseline", "time_only", "pose_only", "full"}
    if fourarm and fourarm not in valid_fourarms:
        raise RuntimeError(f"Invalid ED3DGS_R25_FOURARM: {fourarm!r}")
    matched_arm = os.environ.get("ED3DGS_R25_MATCHED_ARM", "")
    matched_run = bool(matched_arm)
    if matched_run and matched_arm not in {
            "shifted_standard", "shifted_v33_clock", "shifted_r25_clock"}:
        raise RuntimeError(f"Invalid matched R25 arm: {matched_arm!r}")
    v34_run = os.environ.get("ED3DGS_INTERNAL_CLOCK_V34") == "1"
    strict_scene_freeze = (
        os.environ.get("ED3DGS_STRICT_SCENE_FREEZE_V1") == "1")
    strict_step_budget_v2 = bool(getattr(
        opt, "strict_step_budget_v2", False)) or (
            os.environ.get("ED3DGS_STRICT_STEP_BUDGET_V2", "0") == "1")
    clock_target_steps = int(getattr(
        opt, "temporal_offset_target_steps", -1))
    pose_target_steps = int(getattr(
        opt, "thermal_pose_target_steps", -1))
    geoflow_run = os.environ.get("ED3DGS_GEOFLOW_SOFTVOLUME_V1") == "1"
    zero_init_routed_run = (
        os.environ.get("ED3DGS_ZERO_INIT_ROUTED_CLOCK_V1") == "1")
    temporal_consensus_enabled = (
        os.environ.get("ED3DGS_TEMPORAL_CONSENSUS_V1") == "1")
    self_calibrating_clock = (
        os.environ.get("ED3DGS_SELF_CALIBRATING_CLOCK_STUDY") == "1")
    temporal_consensus_mode = os.environ.get(
        "ED3DGS_TEMPORAL_CONSENSUS_MODE", "consensus_only")
    if zero_init_routed_run and not geoflow_run:
        raise RuntimeError(
            "zero-init routed clock requires the GeoFlow-compatible loader")
    if v34_run and geoflow_run:
        raise RuntimeError("MIND and GeoFlow clocks are mutually exclusive")
    if fourarm:
        if matched_run:
            raise RuntimeError("Four-arm protocol forbids legacy matched-arm tags")
        v30_arm = fourarm
        expected_clock = fourarm in {"time_only", "full"}
        if (v34_run or geoflow_run) != expected_clock:
            raise RuntimeError("Four-arm clock implementation disagrees with arm")
        if (geoflow_run and fourarm != "full"
                and not (frozen_pose_ablation and fourarm == "time_only")):
            raise RuntimeError("GeoFlow matched release is full-arm only")
    elif matched_run:
        v30_arm = matched_arm
        if v34_run != (matched_arm == "shifted_r25_clock"):
            raise RuntimeError("Matched R25 arm and clock implementation disagree")
    elif v34_run:
        v30_arm = os.environ.get("ED3DGS_V34_RUN_TAG", "")
        if not v30_arm or any(word in v30_arm.lower() for word in (
                "truth", "candidate", "bootstrap")):
            raise RuntimeError("Invalid v34 run tag")
    else:
        if opt.rgb_only_teacher:
            v30_arm = "rgb_only_teacher"
        else:
            v30_arm = os.environ.get("ED3DGS_V33_ARM", "")
            if v30_arm not in {
                    "shifted_standard", "shifted_pose_only", "shifted_full"}:
                raise RuntimeError(f"Invalid ED3DGS_V33_ARM: {v30_arm!r}")
    synthetic_shift = int(os.environ.get("ED3DGS_THERMAL_FRAME_SHIFT", "0"))
    matched_motion_clock_study = (
        os.environ.get("ED3DGS_MATCHED_MOTION_CLOCK_STUDY") == "1")
    synthetic_endpoint_drift = float(os.environ.get(
        "ED3DGS_THERMAL_ENDPOINT_DRIFT_V34", "0"))
    if (fourarm and synthetic_shift != 20
            and not matched_motion_clock_study
            and not self_calibrating_clock):
        raise RuntimeError("R25 four-arm experiment is locked to +20 frames")
    if (matched_run and synthetic_shift != 20):
        raise RuntimeError("Matched R25 experiment is locked to +20 frames")
    if (not opt.rgb_only_teacher
            and not matched_run and not v34_run and synthetic_shift != 20
            and not matched_motion_clock_study
            and not self_calibrating_clock):
        raise RuntimeError(
            "v33 is locked to the Covers +20-frame synthetic stress test")
    if matched_motion_clock_study and (
            fourarm != "full" or synthetic_shift not in {0, 8, 20}):
        raise RuntimeError(
            "Matched motion-clock study requires the full arm and shift 0, 8, or 20")
    valid_self_calibrating_arm = (
        fourarm == "full"
        or (frozen_pose_ablation and fourarm == "time_only"))
    if self_calibrating_clock and (
            not valid_self_calibrating_arm
            or synthetic_shift not in {0, 8, 20}):
        raise RuntimeError(
            "Self-calibrating study requires an approved clock arm and shift")
    if (v34_run and synthetic_shift not in {
            -20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20}):
        raise RuntimeError("v34 shift is outside the locked Gate matrix")
    affine_requested = os.environ.get("ED3DGS_V34_AFFINE_CLOCK", "0") == "1"
    if synthetic_endpoint_drift != 0.0 and not affine_requested:
        raise RuntimeError("Synthetic endpoint drift requires affine clock")
    timer = Timer()
    scene = Scene(
        dataset, gaussians, shuffle=dataset.shuffle, loader=dataset.loader,
        duration=hyper.total_num_frames, opt=opt, load_test_cameras=False)
    pose_perturbation_report = None
    if os.environ.get("ED3DGS_V34_JOINT_POSE_PERTURB", "0") == "1":
        if not v34_run or not scene.has_thermal_pose:
            raise RuntimeError("Joint pose perturbation requires v34 Thermal pose")
        by_side = {}
        for camera in scene.getTrainCameras():
            side, _ = scene._camera_state_side_frame(camera)
            if side in {"left", "right"} and side not in by_side:
                by_side[side] = camera
        if set(by_side) != {"left", "right"}:
            raise RuntimeError("Joint pose perturbation lacks both camera sides")
        pose_perturbation_report = {}
        for side, camera in by_side.items():
            pose_math.set_locked_joint_gate_perturbation_(
                camera.thermal_delta_quaternion,
                camera.thermal_delta_translation, side, scene.cameras_extent)
            rotation, translation_fraction = pose_math.residual_sizes(
                camera.thermal_delta_quaternion,
                camera.thermal_delta_translation, scene.cameras_extent)
            pose_perturbation_report[side] = {
                "rotation_degrees": rotation,
                "translation_fraction_of_extent": translation_fraction,
            }
        print("V34_JOINT_POSE_PERTURBATION " + json.dumps(
            pose_perturbation_report, sort_keys=True))
    if fourarm:
        expected_temporal = fourarm in {"time_only", "full"}
        expected_pose = fourarm in {"pose_only", "full"}
    elif matched_run:
        expected_temporal = matched_arm != "shifted_standard"
        expected_pose = False
    else:
        expected_temporal = True if v34_run else v30_arm == "shifted_full"
        expected_pose = True if v34_run else v30_arm in {
            "shifted_pose_only", "shifted_full"}
    if scene.temporal_alignment_enabled != expected_temporal:
        raise RuntimeError(
            f"v33 arm {v30_arm} temporal configuration is inconsistent")
    if scene.has_thermal_pose != expected_pose:
        raise RuntimeError(
            f"v33 arm {v30_arm} Thermal pose configuration is inconsistent")
    expected_modes = {
        "fixed_clock": ("pose_only", False, True),
        "total_gradient": ("full", True, True),
        "frozen_pose": ("time_only", True, False),
    }
    if ablation_mode:
        expected_arm, expected_time, expected_thermal_pose = expected_modes[
            ablation_mode]
        if (fourarm != expected_arm
                or bool(scene.temporal_alignment_enabled) != expected_time
                or bool(scene.has_thermal_pose) != expected_thermal_pose):
            raise RuntimeError(
                "Shift20 ablation arm, clock, or Thermal pose contract mismatch")
        print("SHIFT20_ABLATION_CONTRACT " + json.dumps({
            "mode": ablation_mode,
            "arm": fourarm,
            "clock_enabled": bool(scene.temporal_alignment_enabled),
            "thermal_pose_enabled": bool(scene.has_thermal_pose),
            "reconstruction_gradient_to_clock": bool(
                total_gradient_ablation),
            "fixed_offset_frames": 0.0 if fixed_clock_ablation else None,
            "shift_truth_used_for_optimization": False,
        }, sort_keys=True))
    if os.environ.get("ED3DGS_STRICT_COMMON_SUPPORT_V33") == "1":
        all_train_cameras = list(scene.getTrainCameras())
        if hasattr(scene, "stage2_all_train_cameras"):
            raise RuntimeError("Stage2 full-camera snapshot was already configured")
        scene.stage2_all_train_cameras = tuple(all_train_cameras)
        frames_by_side = {"left": set(), "right": set()}
        for camera in all_train_cameras:
            side, frame = scene._camera_state_side_frame(camera)
            if side in frames_by_side and frame is not None:
                frames_by_side[side].add(int(frame))
        support_bound = scene.temporal_offset_max_frames
        if getattr(scene, "temporal_affine_clock_enabled", False):
            support_bound += scene.temporal_drift_max_endpoint_frames
        bound = int(support_bound)
        if float(bound) != support_bound:
            raise ValueError("Strict Stage2 support requires an integer offset bound")
        common_train_cameras = []
        for camera in all_train_cameras:
            side, frame = scene._camera_state_side_frame(camera)
            if side not in frames_by_side or frame is None:
                continue
            frame = int(frame)
            side_frames = frames_by_side[side]
            if (frame - bound in side_frames
                    and frame + bound in side_frames
                    and (geoflow_run
                         or int(camera.thermal_frame_shift) == synthetic_shift)):
                common_train_cameras.append(camera)
        scene.stage2_training_cameras = common_train_cameras
        print("STAGE2_COMMON_SUPPORT " + json.dumps({
            "all_trajectory_camera_count": len(all_train_cameras),
            "immutable_camera_snapshot_count": len(
                scene.stage2_all_train_cameras),
            "bound_frames": bound,
            "selected_training_camera_count": len(common_train_cameras),
            "trajectory_knots_left": len(frames_by_side["left"]),
            "trajectory_knots_right": len(frames_by_side["right"]),
        }, sort_keys=True))
        preflight_reconstruction_support_snapshot(scene)
    teacher_root = str(getattr(dataset, "stage2_teacher_model_path", "") or "")
    if teacher_root:
        if checkpoint:
            raise ValueError(
                "Stage2 teacher initialization cannot be combined with start_checkpoint")
        if opt.rgb_only_teacher:
            raise ValueError("Stage2 teacher initialization requires RGB-T training mode")
        teacher_iteration = int(getattr(dataset, "stage2_teacher_iteration", -1))
        if teacher_iteration <= 0:
            raise ValueError("stage2_teacher_iteration must be positive")
        teacher_dir = os.path.join(
            teacher_root, "point_cloud", f"iteration_{teacher_iteration}")
        teacher_files = {
            "point_cloud.ply": os.path.join(teacher_dir, "point_cloud.ply"),
            "deformation.pth": os.path.join(teacher_dir, "deformation.pth"),
        }
        for name, path in teacher_files.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing Stage2 teacher {name}: {path}")
        for marker in ("RUN_COMPLETE", "RGB_TEACHER_COMPLETE"):
            marker_path = os.path.join(teacher_root, marker)
            if not os.path.isfile(marker_path):
                raise FileNotFoundError(f"Missing Stage2 teacher marker: {marker_path}")

        def file_sha256(path):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        teacher_hashes = {
            name: file_sha256(path) for name, path in teacher_files.items()}
        gaussians.load_ply(teacher_files["point_cloud.ply"])
        gaussians.load_model(teacher_dir)
        init_report = {
            "schema": "covers_stage2_teacher_init_v1",
            "teacher_model_path": os.path.abspath(teacher_root),
            "teacher_iteration": teacher_iteration,
            "teacher_hashes": teacher_hashes,
            "thermal_camera_state_loaded": False,
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
        }
        with open(os.path.join(dataset.model_path, "stage2_teacher_init.json"), "w") as handle:
            json.dump(init_report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("STAGE2_TEACHER_INIT " + json.dumps(init_report, sort_keys=True))

        bootstrap_path = str(getattr(
            dataset, "stage2_temporal_bootstrap_report", "") or "")
        if ((os.environ.get("ED3DGS_INTERNAL_CLOCK_V30") == "1"
                or os.environ.get("ED3DGS_INTERNAL_CLOCK_V34") == "1")
                and bootstrap_path):
            raise RuntimeError(
                "v33 forbids every external temporal bootstrap path")
        if bootstrap_path:
            if not os.path.isfile(bootstrap_path):
                raise FileNotFoundError(
                    f"Missing Stage2 temporal bootstrap report: {bootstrap_path}")
            expected_bootstrap_hash = str(getattr(
                dataset, "stage2_temporal_bootstrap_sha256", "") or "")
            actual_bootstrap_hash = file_sha256(bootstrap_path)
            if not expected_bootstrap_hash or actual_bootstrap_hash != expected_bootstrap_hash:
                raise RuntimeError(
                    "Stage2 temporal bootstrap report hash does not match the "
                    "preregistered artifact")
            bootstrap_tag = os.path.splitext(os.path.basename(bootstrap_path))[0]
            bootstrap_root = os.path.dirname(os.path.dirname(bootstrap_path))
            bootstrap_marker = os.path.join(
                bootstrap_root, "outputs", bootstrap_tag, "RUN_COMPLETE")
            if not os.path.isfile(bootstrap_marker):
                raise FileNotFoundError(
                    f"Missing Stage2 temporal bootstrap marker: {bootstrap_marker}")
            with open(bootstrap_path, "r") as handle:
                bootstrap_report = json.load(handle)
            allowed_bootstrap_schemas = {
                "covers_mind_c1_detrended_shared_recovery_v20",
                "covers_mind_c1_detrended_shared_recovery_v24",
            }
            if bootstrap_report.get("schema") not in allowed_bootstrap_schemas:
                raise RuntimeError(
                    f"Invalid Stage2 temporal bootstrap schema: {bootstrap_report.get('schema')!r}")
            required_bootstrap_values = {
                "scene": "Covers",
                "per_scene_optimization": True,
                "shared_scene_clock_across_cameras": True,
                "each_arm_restarts_from_raw_zero": True,
                "candidate_enumeration_in_optimizer": False,
                "external_data_loaded": False,
                "formal_test_cameras_constructed": False,
            }
            for key, expected in required_bootstrap_values.items():
                if bootstrap_report.get(key) != expected:
                    raise RuntimeError(
                        f"Invalid Stage2 temporal bootstrap field {key}: "
                        f"{bootstrap_report.get(key)!r}")
            bootstrap_teacher_hashes = bootstrap_report.get("teacher_hashes", {})
            if any(bootstrap_teacher_hashes.get(name) != digest
                   for name, digest in teacher_hashes.items()):
                raise RuntimeError(
                    "Stage2 temporal bootstrap and RGB teacher hashes differ")
            matching_arms = [
                arm for arm in bootstrap_report.get("recoveries", [])
                if int(arm.get("truth_shift_frames", 10**9)) == synthetic_shift]
            if len(matching_arms) != 1:
                raise RuntimeError(
                    "Stage2 temporal bootstrap must contain exactly one arm for "
                    f"declared shift {synthetic_shift}")
            selected_arm = matching_arms[0]
            bootstrap_offset = float(
                selected_arm.get("recovered_delta_frames", float("nan")))
            bootstrap_raw = float(selected_arm.get("final_raw", float("nan")))
            forced_offset = float(getattr(
                opt, "temporal_offset_force_frames", float("nan")))
            if (float(selected_arm.get("initial_raw", float("nan"))) != 0.0
                    or not math.isfinite(bootstrap_offset)
                    or not math.isfinite(bootstrap_raw)
                    or abs(bootstrap_offset - synthetic_shift) > 2.0
                    or abs(bootstrap_offset - forced_offset) > 1e-6):
                raise RuntimeError(
                    "Stage2 temporal bootstrap recovery is invalid")
            temporal_bootstrap = {
                "arm": selected_arm.get("arm"),
                "initial_raw": 0.0,
                "recovered_offset_frames": bootstrap_offset,
                "recovered_raw": bootstrap_raw,
                "report_path": os.path.abspath(bootstrap_path),
                "report_sha256": actual_bootstrap_hash,
                "strict_gate_pass": bool(bootstrap_report.get("gate_pass", False)),
                "truth_shift_frames": synthetic_shift,
            }
            print("STAGE2_TEMPORAL_BOOTSTRAP " + json.dumps(
                temporal_bootstrap, sort_keys=True))

    if scene.temporal_alignment_enabled:
        if not teacher_root:
            raise RuntimeError(
                "Stage2 temporal joint training requires an external RGB teacher")
        train_cameras = getattr(
            scene, "stage2_training_cameras", scene.getTrainCameras())
        if not train_cameras:
            raise RuntimeError("Stage2 temporal training has no cameras")
        side_counts = {"left": 0, "right": 0}
        for camera in train_cameras:
            side, _ = scene._camera_state_side_frame(camera)
            if side not in side_counts:
                raise RuntimeError(f"Stage2 camera side is invalid: {camera.image_name}")
            side_counts[side] += 1
        if (side_counts["left"] != side_counts["right"]
                or side_counts["left"] < 100):
            raise RuntimeError(
                f"Stage2 framewise common support is insufficient/unbalanced: {side_counts}")
        shifts = {int(camera.thermal_frame_shift) for camera in train_cameras}
        if shifts != {synthetic_shift}:
            raise RuntimeError(
                f"Stage2 observation shift mismatch: declared={synthetic_shift}, observed={shifts}")
        if not all(getattr(camera, "temporal_strict_common_support", False)
                   for camera in train_cameras):
            raise RuntimeError("Stage2 temporal cameras must use strict common support")
        initial_raw = float(scene.temporal_offset_raw.detach().item())
        initial_offset = float(scene.temporal_offset_frames().detach().item())
        initial_drift_raw = float(scene.temporal_drift_raw.detach().item())
        if temporal_bootstrap is None:
            if (initial_raw != 0.0 or initial_offset != 0.0
                    or initial_drift_raw != 0.0):
                raise RuntimeError(
                    "Stage2 temporal raw parameters must start exactly at zero")
        elif abs(initial_offset - temporal_bootstrap["recovered_offset_frames"]) > 1e-5:
            raise RuntimeError(
                "Stage2 temporal state does not match the audited bootstrap")
        integer_pose_error = 0.0
        zero_raw = torch.zeros_like(scene.temporal_offset_raw)
        for camera in train_cameras:
            pose_min = float(camera.temporal_pose_frames[0].detach().item())
            pose_max = float(camera.temporal_pose_frames[-1].detach().item())
            frame = float(camera.frame_no)
            full_clock_bound = scene.temporal_offset_max_frames + (
                scene.temporal_drift_max_endpoint_frames
                if scene.temporal_affine_clock_enabled else 0.0)
            if (frame - full_clock_bound < pose_min
                    or frame + full_clock_bound > pose_max):
                raise RuntimeError(
                    f"Stage2 pose track does not cover full clock domain: {camera.image_name}")
            camera.nctc_temporal_offset_raw_override = zero_raw
            try:
                error = float(torch.max(torch.abs(
                    camera.get_temporal_rgb_world_view_transform()
                    - camera.world_view_transform)).detach().item())
            finally:
                del camera.nctc_temporal_offset_raw_override
            integer_pose_error = max(integer_pose_error, error)
        if integer_pose_error != 0.0:
            raise RuntimeError(
                f"Stage2 C1 integer pose mismatch: {integer_pose_error}")
        print("STAGE2_TEMPORAL_CONTRACT " + json.dumps({
            "candidate_enumeration": False,
            "c1_integer_pose_max_abs_error": integer_pose_error,
            "common_support_camera_count": len(train_cameras),
            "delta_formula": (
                f"{scene.temporal_offset_max_frames:g}*tanh(raw_offset)"
                + (f"+{scene.temporal_drift_max_endpoint_frames:g}*"
                   "tanh(raw_drift)*u(t)"
                   if scene.temporal_affine_clock_enabled else "")),
            "initial_offset_frames": initial_offset,
            "initial_raw": initial_raw,
            "initial_drift_raw": initial_drift_raw,
            "per_scene": True,
            "scene": os.path.basename(os.path.normpath(dataset.source_path)),
            "observed_dataset_frame_shift": synthetic_shift,
            "observed_endpoint_drift_frames": synthetic_endpoint_drift,
            "temporal_bootstrap": temporal_bootstrap,
            "training_cameras_by_side": side_counts,
        }, sort_keys=True))
        clock_v33 = os.environ.get("ED3DGS_INTERNAL_CLOCK_V30") == "1"
        clock_v34 = os.environ.get("ED3DGS_INTERNAL_CLOCK_V34") == "1"
        if sum((clock_v33, clock_v34, geoflow_run)) != 1:
            raise RuntimeError(
                "Exactly one in-process clock implementation is required")
        if temporal_bootstrap is not None:
            raise RuntimeError("Internal clock learning forbids bootstrap")
        if zero_init_routed_run:
            if (float(scene.temporal_offset_raw.detach().item()) != 0.0
                    or float(scene.temporal_offset_frames().detach().item()) != 0.0):
                raise RuntimeError(
                    "zero-init routed clock was changed before training")
            if not scene.temporal_offset_raw.requires_grad:
                raise RuntimeError(
                    "zero-init routed clock must remain trainable")
            internal_clock_learning = {
                "schema": "self_calibrating_global_clock_contract",
                "status": "PASS",
                "initial_raw": 0.0,
                "initial_offset_frames": 0.0,
                "pretraining_offset_assignment": False,
                "selected_camera_count": len(train_cameras),
                "shift_truth_input": False,
                "candidate_enumeration": False,
                "external_bootstrap_or_report_loaded": False,
                "raw_object_id": id(scene.temporal_offset_raw),
                "drift_raw_object_id": None,
                "initial_drift_raw": 0.0,
                "initial_drift_frames": 0.0,
                "optimizer_object_id": id(scene.optimizer_temporal_offset),
                "optimizer_steps": 0,
            }
            selected = list(scene.stage2_training_cameras)
            selected_names = [camera.image_name for camera in selected]
            scene.stage2_reconstruction_cameras = selected
            scene.stage2_reconstruction_support_report = {
                "schema": "covers_zero_init_bounded_support_v1",
                "selection_inputs": ["trajectory_bounds", "clock_domain"],
                "shift_truth_used_for_selection": False,
                "learned_offset_frames": None,
                "calibration_camera_count": len(selected),
                "reconstruction_camera_count": len(selected),
                "reconstruction_cameras_by_side": side_counts,
                "selected_observation_shifts_audit_only": sorted({
                    int(camera.thermal_frame_shift) for camera in selected}),
                "selected_names": selected_names,
                "selected_names_sha256": hashlib.sha256(
                    "\n".join(selected_names).encode("utf-8")).hexdigest(),
            }
            print("ZERO_INIT_ROUTED_CLOCK_BEGIN " + json.dumps({
                "initial_offset_frames": 0.0,
                "initial_raw": 0.0,
                "optimizer_object_id": id(scene.optimizer_temporal_offset),
                "raw_object_id": id(scene.temporal_offset_raw),
                "reconstruction_camera_count": len(selected),
                "stage1_offset_assignment": False,
            }, sort_keys=True))
        else:
            raise RuntimeError(
                "The formal release supports only raw-zero routed clock learning")
        if (id(scene.temporal_offset_raw)
                != internal_clock_learning["raw_object_id"]
                or id(scene.optimizer_temporal_offset)
                != internal_clock_learning["optimizer_object_id"]):
            raise RuntimeError("Clock stage replaced the learned raw or optimizer")
        if (scene.temporal_affine_clock_enabled
                and id(scene.temporal_drift_raw)
                != internal_clock_learning["drift_raw_object_id"]):
            raise RuntimeError("Clock stage replaced the learned drift raw")
        if not zero_init_routed_run:
            raise RuntimeError(
                "The formal release forbids pretraining clock initialization")
        if bool(getattr(opt, "joint_clock_loss_enabled", False)):
            if zero_init_routed_run:
                if not scene.temporal_offset_raw.requires_grad:
                    raise RuntimeError(
                        "zero-init routed clock unexpectedly became frozen")
            else:
                if scene.temporal_offset_raw.requires_grad:
                    raise RuntimeError(
                        "GeoFlow Stage-A must freeze the Scene clock before Stage2")
                scene.temporal_offset_raw.requires_grad_(True)
            scene.optimizer_temporal_offset.zero_grad(set_to_none=True)
            print("JOINT_CLOCK_STAGE2_BEGIN " + json.dumps({
                "initial_offset_frames": float(
                    scene.temporal_offset_frames().detach().item()),
                "optimizer_object_id": id(scene.optimizer_temporal_offset),
                "raw_object_id": id(scene.temporal_offset_raw),
                "drift_raw_object_id": (
                    id(scene.temporal_drift_raw)
                    if scene.temporal_affine_clock_enabled else None),
                "raw_requires_grad": bool(
                    scene.temporal_offset_raw.requires_grad),
                "drift_raw_requires_grad": bool(
                    scene.temporal_drift_raw.requires_grad),
                "stage1": (
                    "none_raw_zero" if zero_init_routed_run
                    else "geoflow_softvolume_v1"),
            }, sort_keys=True))
    if fixed_clock_ablation:
        selected = list(scene.stage2_training_cameras)
        side_counts = {"left": 0, "right": 0}
        for camera in selected:
            side, _ = scene._camera_state_side_frame(camera)
            if side not in side_counts:
                raise RuntimeError(
                    f"Fixed-clock calibration side is invalid: {camera.image_name}")
            side_counts[side] += 1
        if (not selected or len(selected) % 2 != 0
                or side_counts["left"] != side_counts["right"]):
            raise RuntimeError(
                "Fixed-clock calibration support must be balanced by side")
        selected_names = [camera.image_name for camera in selected]
        scene.stage2_reconstruction_cameras = selected
        scene.stage2_reconstruction_support_report = {
            "schema": "covers_fixed_clock_calibration_support",
            "selection_inputs": ["trajectory_bounds", "clock_domain"],
            "shift_truth_used_for_selection": False,
            "fixed_offset_frames": 0.0,
            "calibration_camera_count": len(selected),
            "reconstruction_camera_count": len(selected),
            "reconstruction_cameras_by_side": side_counts,
            "selected_names": selected_names,
            "selected_names_sha256": hashlib.sha256(
                "\n".join(selected_names).encode("utf-8")).hexdigest(),
        }
        print("FIXED_CLOCK_ABLATION_BEGIN " + json.dumps({
            "fixed_offset_frames": 0.0,
            "reconstruction_camera_count": len(selected),
            "shift_truth_used_for_optimization": False,
        }, sort_keys=True))
    timer.start()
    
    start_time = time()
    optimizer_update_counts = scene_reconstruction(
        dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
        checkpoint_iterations, checkpoint, debug_from,
        gaussians, scene, tb_writer, opt.iterations, timer, start_time)
    zero_start_clock_phase = optimizer_update_counts.get(
        "zero_start_clock_phase")
    if zero_start_clock_phase is not None:
        if internal_clock_learning is None:
            raise RuntimeError(
                "Zero-start clock phase is missing the raw-zero contract")
        internal_clock_learning.update({
            "optimizer_steps": int(zero_start_clock_phase.get(
                "optimizer_steps",
                zero_start_clock_phase.get("coarse_optimizer_steps", 0))),
            "nonzero_gradients": int(zero_start_clock_phase.get(
                "nonzero_gradients",
                zero_start_clock_phase.get("coarse_nonzero_gradients", 0))),
            "learned_offset_frames": float(zero_start_clock_phase[
                "final_offset_frames"]),
            "learned_raw": float(zero_start_clock_phase["final_raw"]),
            "zero_start_learning_phase": True,
            "regional_refinement_applied": bool(
                zero_start_clock_phase.get("regional_refinement")),
            "motion_continuation_applied": bool(
                zero_start_clock_phase.get("motion_clock")),
        })
    if self_calibrating_clock:
        internal_clock_learning.update({
            "optimizer_steps": int(optimizer_update_counts.get(
                "temporal_offset_optimizer_steps",
                optimizer_update_counts.get("joint_clock_optimizer_steps", 0))),
            "nonzero_offset_gradients": int(optimizer_update_counts.get(
                "joint_clock_nonzero_offset_gradients", 0)),
            "nonzero_drift_gradients": int(optimizer_update_counts.get(
                "joint_clock_nonzero_drift_gradients", 0)),
            "learned_offset_frames": float(
                scene.temporal_offset_frames().detach().item()),
            "learned_drift_frames": float(
                scene.temporal_drift_frames().detach().item()),
            "learned_raw": float(scene.temporal_offset_raw.detach().item()),
            "learned_drift_raw": float(
                scene.temporal_drift_raw.detach().item()),
            "learned_endpoint_offsets_frames": [
                float(value.detach().item())
                for value in scene.temporal_endpoint_offsets()
            ],
            "unified_main_training": True,
            "pre_reconstruction_clock_phase": False,
        })
    motion_clock_applied = bool(
        zero_start_clock_phase is not None
        and zero_start_clock_phase.get("motion_clock"))
    if opt.rgb_only_teacher:
        result = {
            "schema": "strict_rgb_only_teacher_training_result_v1",
            "scene": os.path.basename(os.path.normpath(dataset.source_path)),
            "iterations": int(opt.iterations),
            "observed_dataset_frame_shift": synthetic_shift,
            "rgb_only_teacher": True,
            "thermal_loss_weight": float(opt.thermal_loss_weight),
            "thermal_pose_enabled": bool(scene.has_thermal_pose),
            "temporal_alignment_enabled": bool(
                scene.temporal_alignment_enabled),
            "gaussian_count": int(gaussians.get_xyz.shape[0]),
            "optimizer_update_counts": optimizer_update_counts,
        }
        with open(os.path.join(
                dataset.model_path, "rgb_teacher_training_result.json"),
                "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("RGB_TEACHER_TRAINING_RESULT " + json.dumps(
            result, sort_keys=True))
    elif scene.temporal_alignment_enabled:
        teacher_hashes_after = {
            name: file_sha256(path) for name, path in teacher_files.items()}
        if teacher_hashes_after != teacher_hashes:
            raise RuntimeError("Stage2 training modified the frozen RGB teacher")
        result_schema = (
            "covers_self_calibrating_global_clock_training_result"
            if self_calibrating_clock else
            "covers_motion_continuation_clock_training_result"
            if motion_clock_applied else
            "covers_zero_start_clock_training_result"
            if zero_start_clock_phase is not None else
            "covers_zero_init_routed_clock_result_v1"
            if zero_init_routed_run else
            "covers_geoflowsync_matched_result_v1" if geoflow_run else
            (("covers_r25_fourarm_dualsupport_result_v1"
                        if os.environ.get("ED3DGS_R25_DUAL_SUPPORT") == "1"
                        else "covers_r25_fourarm_fullscene_result_v1")
                       if fourarm else
                       ("covers_r25_clock_fullscene_result_v1"
                       if bool(getattr(
                           opt, "scene_optimizer_every_iteration", False)) else
                       ("covers_r25_clock_matched_stage2_result_v35"
                       if matched_run else
                       ("covers_observable_spatiotemporal_stage2_result_v34"
                        if v34_run
                        else "covers_shifted_matched_stage2_result_v33")))))
        result = {
            "schema": result_schema,
            "self_calibrating_clock": self_calibrating_clock,
            "arm": v30_arm,
            "ablation_mode": ablation_mode or None,
            "candidate_enumeration": motion_clock_applied,
            "candidate_lag_bank_in_observable": bool(
                motion_clock_applied),
            "final_offset_frames": float(
                scene.temporal_offset_frames().detach().item()),
            "final_raw": float(scene.temporal_offset_raw.detach().item()),
            "affine_clock_enabled": bool(
                scene.temporal_affine_clock_enabled),
            "final_drift_raw": (float(scene.temporal_drift_raw.detach().item())
                                if scene.temporal_affine_clock_enabled
                                else None),
            "final_endpoint_offsets_frames": [float(value.detach().item())
                for value in scene.temporal_endpoint_offsets()],
            "formal_test_cameras_constructed": False,
            "iterations": int(opt.iterations),
            "per_scene": True,
            "scene": os.path.basename(os.path.normpath(dataset.source_path)),
            "observed_dataset_frame_shift": synthetic_shift,
            "observed_endpoint_drift_frames": synthetic_endpoint_drift,
            "temporal_offset_max_frames": float(scene.temporal_offset_max_frames),
            "temporal_alignment_enabled": True,
            "temporal_offset_lr": float(opt.temporal_offset_lr),
            "temporal_offset_start_iter": int(opt.temporal_offset_start_iter),
            "thermal_pose_enabled": bool(scene.has_thermal_pose),
            "thermal_pose_shared_by_side": bool(
                getattr(opt, "thermal_pose_shared_by_side", False)),
            "thermal_pose_lr_rotation": float(opt.thermal_pose_lr_r),
            "thermal_pose_lr_translation": float(opt.thermal_pose_lr_t),
            "thermal_pose_start_iter": int(opt.thermal_pose_start_iter),
            "thermal_intrinsic_lr": float(opt.thermal_intrinsic_lr),
            "pose_renderer_derivative_audit": getattr(
                scene, "v34_pose_derivative_audit", None),
            "calibration_reconstruction_alternation": bool(getattr(
                opt, "calibration_reconstruction_alternation", False)),
            "calibration_reconstruction_phase_length": int(getattr(
                opt, "calibration_reconstruction_phase_length", 8)),
            "scene_extent": float(scene.cameras_extent),
            "clock_drift_from_internal_frames": (
                None if zero_init_routed_run else abs(float(
                    scene.temporal_offset_frames().detach().item())
                    - float(internal_clock_learning["learned_offset_frames"]))),
            "clock_change_from_initial_frames": abs(float(
                scene.temporal_offset_frames().detach().item())
                - float(internal_clock_learning["initial_offset_frames"])),
            "teacher_hashes_after": teacher_hashes_after,
            "teacher_hashes_before": teacher_hashes,
            "full_trajectory_camera_count": len(scene.getTrainCameras()),
            "initial_temporal_bootstrap": temporal_bootstrap,
            "initial_pose_perturbation": pose_perturbation_report,
            "in_process_internal_clock": internal_clock_learning,
            "zero_start_clock_phase": zero_start_clock_phase,
            "clock_module": (
                "continuous_global_motion_cost_volume_clock"
                if self_calibrating_clock else
                "zero_start_motion_correlation_continuation"
                if motion_clock_applied else
                "zero_start_consensus_then_regional_root"
                if zero_start_clock_phase is not None else
                "zero_init_routed_total_v1" if zero_init_routed_run else
                "geoflow_softvolume_v1" if geoflow_run else
                "mind_v34" if v34_run else "mind_v33"),
            "clock_frozen_during_reconstruction": bool(
                strict_scene_freeze
                or zero_start_clock_phase is not None
                or (self_calibrating_clock
                    and opt.temporal_offset_freeze_after >= 0
                    and int(opt.iterations)
                    > opt.temporal_offset_freeze_after)
                or (geoflow_run and not bool(getattr(
                    opt, "joint_clock_loss_enabled", False)))),
            "joint_clock_contract": {
                "enabled": bool(optimizer_update_counts.get(
                    "main_clock_optimizer_enabled", bool(getattr(
                        opt, "joint_clock_loss_enabled", False)))),
                "alignment_gradient_owners": ["scene_clock"],
                "alignment_loss": (
                    "continuous_global_motion_cost_volume"
                    if self_calibrating_clock else
                    "training_sequence_motion_correlation_continuation"
                    if motion_clock_applied
                    else "temporal_activity_consensus_v1"
                    if temporal_consensus_enabled
                       and temporal_consensus_mode == "consensus_only"
                    else "0.5*temporal_activity_consensus_v1+0.5*"
                         "multiscale_observable_polarity_invariant_NGF"
                    if temporal_consensus_enabled
                    else clock_loss.loss_name()),
                "alignment_weight": float(getattr(
                    opt, "joint_clock_loss_weight", 0.0)),
                "clock_freeze_after": int(getattr(
                    opt, "temporal_offset_freeze_after", -1)),
                "clock_target_steps": int(clock_target_steps),
                "strict_step_budget_v2": bool(strict_step_budget_v2),
                "clock_lr": float(opt.temporal_offset_lr),
                "interval": int(getattr(
                    opt, "joint_clock_loss_interval", 1)),
                "ramp_iters": int(getattr(
                    opt, "joint_clock_loss_ramp_iters", 0)),
                "reconstruction_backward_excludes_alignment": True,
                "reconstruction_gradient_to_clock": bool(
                    zero_start_clock_phase is None
                    and (os.environ.get("ED3DGS_CLOCK_TOTAL_GRAD_V1") == "1"
                         or not zero_init_routed_run)),
                "clock_gradient_mode": (
                    "zero_start_motion_continuation_frozen"
                    if motion_clock_applied
                    else "zero_start_coarse_then_regional_root_frozen"
                    if zero_start_clock_phase is not None else
                    "total_loss_routed"
                    if os.environ.get("ED3DGS_CLOCK_TOTAL_GRAD_V1") == "1"
                    else "alignment_only"
                    if zero_init_routed_run else "legacy_total"),
                "stable_modality_routing": bool(
                    os.environ.get("ED3DGS_MODALITY_ROUTING_STABLE_V1") == "1"),
                "ema_modal_loss": bool(
                    os.environ.get("ED3DGS_EMA_MODAL_LOSS_V1") == "1"),
                "memory_safe_backward": bool(
                    os.environ.get("ED3DGS_MEMORY_SAFE_BACKWARD_V1") == "1"),
                "memory_safe_gaussian_threshold": int(os.environ.get(
                    "ED3DGS_MEMORY_SAFE_GAUSSIAN_THRESHOLD_V1", "160000")),
                "shift_truth_input": False,
                "stage1": (
                    "motion_continuation_learned_from_exact_zero"
                    if motion_clock_applied
                    else "learned_from_exact_zero"
                    if zero_start_clock_phase is not None else
                    "none_raw_zero" if zero_init_routed_run
                    else "geoflow_softvolume_v1"),
                "stage2_clock_requires_grad": bool(
                    scene.temporal_offset_raw.requires_grad),
                "stage2_drift_requires_grad": bool(
                    scene.temporal_drift_raw.requires_grad),
                "start_iteration": int(getattr(
                    opt, "joint_clock_loss_start_iter", 0)),
                "temporal_consensus_enabled": temporal_consensus_enabled,
                "temporal_consensus_mode": (
                    temporal_consensus_mode if temporal_consensus_enabled
                    else None),
                "motion_clock_continuation": motion_clock_applied,
                "temporal_consensus_initial_raw_gradients": (
                    optimizer_update_counts.get(
                        "temporal_consensus_initial_raw_gradients")),
                "temporal_consensus_contract": (
                    optimizer_update_counts.get(
                        "temporal_consensus_contract")),
            },
            "optimizer_update_counts": optimizer_update_counts,
            "scene_optimizer_every_iteration": bool(getattr(
                opt, "scene_optimizer_every_iteration", False)),
            "strict_scene_freeze": strict_scene_freeze,
            "temporal_raw_object_id_final": id(scene.temporal_offset_raw),
            "temporal_drift_raw_object_id_final": (
                id(scene.temporal_drift_raw)
                if scene.temporal_affine_clock_enabled else None),
            "temporal_optimizer_object_id_final": id(
                scene.optimizer_temporal_offset),
            "temporal_raw_same_object_end_to_end": bool(
                internal_clock_learning is not None
                and id(scene.temporal_offset_raw)
                == internal_clock_learning["raw_object_id"]
                and id(scene.optimizer_temporal_offset)
                == internal_clock_learning["optimizer_object_id"]
                and (not scene.temporal_affine_clock_enabled
                     or id(scene.temporal_drift_raw)
                     == internal_clock_learning["drift_raw_object_id"])),
            "train_camera_count": len(getattr(
                scene, "stage2_training_cameras", scene.getTrainCameras())),
            "reconstruction_camera_count": len(getattr(
                scene, "stage2_reconstruction_cameras",
                getattr(scene, "stage2_training_cameras", scene.getTrainCameras()))),
            "dual_support": getattr(
                scene, "stage2_reconstruction_support_report", None),
            "support_transition_preflight": getattr(
                scene, "stage2_support_transition_preflight", None),
        }
        pose_by_side = {}
        thermal_cameras = [
            camera for camera in scene.getTrainCameras()
            if getattr(camera, "has_thermal", False)
        ]
        if scene.has_thermal_pose:
            result["thermal_pose_unique_rotation_parameters"] = len({
                id(camera.thermal_delta_quaternion) for camera in thermal_cameras})
            result["thermal_pose_unique_translation_parameters"] = len({
                id(camera.thermal_delta_translation) for camera in thermal_cameras})
            for side in ("left", "right"):
                side_cameras = [
                    camera for camera in thermal_cameras
                    if scene._camera_state_side_frame(camera)[0] == side
                ]
                if not side_cameras:
                    continue
                camera = side_cameras[0]
                quaternion = torch.nn.functional.normalize(
                    camera.thermal_delta_quaternion.detach(), dim=0)
                translation = camera.thermal_delta_translation.detach()
                rotation_degrees, translation_fraction = (
                    pose_math.residual_sizes(
                        quaternion, translation, scene.cameras_extent))
                pose_by_side[side] = {
                    "camera_count": len(side_cameras),
                    "quaternion": [float(value) for value in quaternion.cpu()],
                    "rotation_degrees": rotation_degrees,
                    "translation": [float(value) for value in translation.cpu()],
                    "translation_fraction_of_extent": translation_fraction,
                    "translation_norm": float(translation.norm().cpu()),
                }
        else:
            result["thermal_pose_unique_rotation_parameters"] = 0
            result["thermal_pose_unique_translation_parameters"] = 0
        result["thermal_pose_by_side"] = pose_by_side
        with open(os.path.join(dataset.model_path, "stage2_training_result.json"), "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("STAGE2_TRAINING_RESULT " + json.dumps(result, sort_keys=True))
    else:
        teacher_hashes_after = {
            name: file_sha256(path) for name, path in teacher_files.items()}
        if teacher_hashes_after != teacher_hashes:
            raise RuntimeError("Baseline training modified the frozen RGB teacher")
        result = {
            "schema": ("covers_r25_fourarm_fullscene_result_v1"
                       if fourarm else
                       ("covers_r25_clock_matched_stage2_result_v35"
                       if matched_run
                       else "covers_shifted_matched_stage2_result_v33")),
            "arm": v30_arm,
            "ablation_mode": ablation_mode or None,
            "candidate_enumeration": False,
            "formal_test_cameras_constructed": False,
            "iterations": int(opt.iterations),
            "observed_dataset_frame_shift": synthetic_shift,
            "per_scene": True,
            "scene": os.path.basename(os.path.normpath(dataset.source_path)),
            "teacher_hashes_after": teacher_hashes_after,
            "teacher_hashes_before": teacher_hashes,
            "temporal_alignment_enabled": False,
            "thermal_pose_enabled": bool(scene.has_thermal_pose),
            "thermal_pose_shared_by_side": bool(
                getattr(opt, "thermal_pose_shared_by_side", False)),
            "thermal_pose_lr_rotation": float(opt.thermal_pose_lr_r),
            "thermal_pose_lr_translation": float(opt.thermal_pose_lr_t),
            "thermal_pose_start_iter": int(opt.thermal_pose_start_iter),
            "thermal_intrinsic_lr": float(opt.thermal_intrinsic_lr),
            "calibration_reconstruction_alternation": bool(getattr(
                opt, "calibration_reconstruction_alternation", False)),
            "calibration_reconstruction_phase_length": int(getattr(
                opt, "calibration_reconstruction_phase_length", 8)),
            "scene_extent": float(scene.cameras_extent),
            "optimizer_update_counts": optimizer_update_counts,
            "scene_optimizer_every_iteration": bool(getattr(
                opt, "scene_optimizer_every_iteration", False)),
            "strict_scene_freeze": strict_scene_freeze,
            "train_camera_count": len(getattr(
                scene, "stage2_training_cameras", scene.getTrainCameras())),
            "reconstruction_camera_count": len(getattr(
                scene, "stage2_reconstruction_cameras",
                getattr(scene, "stage2_training_cameras",
                        scene.getTrainCameras()))),
            "dual_support": getattr(
                scene, "stage2_reconstruction_support_report", None),
            "support_transition_preflight": getattr(
                scene, "stage2_support_transition_preflight", None),
        }
        thermal_cameras = [
            camera for camera in scene.getTrainCameras()
            if getattr(camera, "has_thermal", False)
        ]
        result["thermal_pose_unique_rotation_parameters"] = len({
            id(camera.thermal_delta_quaternion) for camera in thermal_cameras
        }) if scene.has_thermal_pose else 0
        result["thermal_pose_unique_translation_parameters"] = len({
            id(camera.thermal_delta_translation) for camera in thermal_cameras
        }) if scene.has_thermal_pose else 0
        pose_by_side = {}
        if scene.has_thermal_pose:
            for side in ("left", "right"):
                side_cameras = [
                    camera for camera in thermal_cameras
                    if scene._camera_state_side_frame(camera)[0] == side
                ]
                if not side_cameras:
                    continue
                camera = side_cameras[0]
                quaternion = torch.nn.functional.normalize(
                    camera.thermal_delta_quaternion.detach(), dim=0)
                translation = camera.thermal_delta_translation.detach()
                scalar = torch.clamp(quaternion[0].abs(), 0.0, 1.0)
                pose_by_side[side] = {
                    "camera_count": len(side_cameras),
                    "quaternion": [float(value) for value in quaternion.cpu()],
                    "rotation_degrees": float(
                        torch.rad2deg(2.0 * torch.acos(scalar)).cpu()),
                    "translation": [float(value) for value in translation.cpu()],
                    "translation_fraction_of_extent": float(
                        translation.norm().cpu()) / float(scene.cameras_extent),
                    "translation_norm": float(translation.norm().cpu()),
                }
        result["thermal_pose_by_side"] = pose_by_side
        with open(os.path.join(dataset.model_path, "stage2_training_result.json"), "w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("STAGE2_TRAINING_RESULT " + json.dumps(result, sort_keys=True))
    end_time = time()
    
    total_time_seconds = end_time - start_time
    hours, remainder = divmod(total_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"training time: {int(hours)}h {int(minutes)}m {seconds}sec")


def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    writer = SummaryWriter(log_dir=os.path.join(args.model_path, "tensorboard"))
    return writer


        
if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 5000, 7000, 14000, 20000, 30000, 45000, 60000, 80000, 100000, 120000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000, 15000, 25000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--seed", type=int, default=6666)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        # import mmcv
        import mmengine
        from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)
    print(f"[Reproducibility] effective_seed={args.seed}")

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
