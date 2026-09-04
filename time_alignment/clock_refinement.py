"""Blind Gate for refining a coarse Covers clock with continuous renders.

The released temporal-consensus loss shifts a frozen scalar activity trace.
This diagnostic instead re-renders the frozen RGB teacher at each continuous
clock value before computing the same MIND activity observable.  The profile
is audit-only; it never initializes or selects a value for an optimizer.
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from time_alignment import feature_runtime as runtime
from time_alignment import pose_calibration
from time_alignment import rgb_fidelity
from time_alignment import temporal_consensus as consensus


PROFILE_RADIUS_FRAMES = 2.0
PROFILE_STEP_FRAMES = 0.25
REGIONAL_GRIDS = (2, 4)
SMOOTHING_KERNELS = (9, 5, 3)
REFINEMENT_BRACKET_STEP_FRAMES = 0.1
REFINEMENT_MAX_BRACKET_STEPS = 16
REFINEMENT_BISECTION_STEPS = 6


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _trainability_snapshot(gaussians, cameras):
    tensors = []
    seen = set()

    def add(name, value):
        if not torch.is_tensor(value) or id(value) in seen:
            return
        seen.add(id(value))
        tensors.append((name, value, bool(value.requires_grad)))

    for name in (
            "_xyz", "_features_dc", "_features_rest", "_thermal_dc",
            "_thermal_rest", "_opacity", "_thermal_opacity", "_scaling",
            "_rotation", "_embedding", "_t_embedding", "_logit_modality",
            "_pose_r", "_pose_t", "_pose_log_s"):
        add(f"gaussian.{name}", getattr(gaussians, name, None))
    for name, value in gaussians._deformation.named_parameters():
        add(f"deformation.{name}", value)
    for head_name in ("pose_head", "thermal_pose_head"):
        head = getattr(gaussians, head_name, None)
        if head is not None:
            for name, value in head.named_parameters():
                add(f"{head_name}.{name}", value)
    for camera_index, camera in enumerate(cameras):
        for name in ("learnable_tfovx", "learnable_tfovy"):
            add(
                f"camera.{camera_index}.{name}",
                getattr(camera, name, None))
    return tensors


def _restore_trainability(snapshot):
    for _, value, requires_grad in snapshot:
        value.requires_grad_(requires_grad)


def _raw_for_offset(offset_frames, max_offset_frames, device):
    _require(abs(offset_frames) < max_offset_frames,
             "Clock-refinement probe reached the tanh boundary")
    return torch.tensor(
        math.atanh(float(offset_frames) / float(max_offset_frames)),
        dtype=torch.float32, device=device)


def _group_cameras(cameras):
    grouped = {"left": {}, "right": {}}
    for camera in cameras:
        side = consensus._side(camera)
        frame = int(camera.frame_no)
        _require(frame not in grouped[side],
                 f"Duplicate refinement camera: {side} {frame}")
        grouped[side][frame] = camera
    for side, by_frame in grouped.items():
        frames = sorted(by_frame)
        steps = {second - first
                 for first, second in zip(frames[:-1], frames[1:])}
        _require(len(frames) >= 100 and len(steps) == 1,
                 f"Invalid refinement camera grid: {side}")
    return grouped


def _camera_manifest(cameras):
    rows = []
    for camera in sorted(
            cameras, key=lambda item: (
                consensus._side(item), int(item.frame_no))):
        row = {
            "side": consensus._side(camera),
            "frame": int(camera.frame_no),
            "frame_no": float(camera.frame_no),
            "temporal_alignment_enabled": bool(
                camera.temporal_alignment_enabled),
            "temporal_observation_correction_enabled": bool(
                camera.temporal_observation_correction_enabled),
            "temporal_strict_common_support": bool(
                camera.temporal_strict_common_support),
            "max_offset_frames": float(camera.temporal_offset_max_frames),
            "duration": float(camera.temporal_duration),
            "thermal_frame_shift": int(camera.thermal_frame_shift),
            "has_raw_override": hasattr(
                camera, "nctc_temporal_offset_raw_override"),
        }
        rows.append(row)
    _require(len(rows) == len(cameras),
             "Camera scalar manifest is incomplete")
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return rows, hashlib.sha256(encoded).hexdigest()


def _render_descriptor(camera, frame, raw, gaussians, pipe, hyper,
                       background, teacher_iteration):
    _require(int(camera.frame_no) == int(frame),
             "Thermal-view clock descriptor frame mismatch")
    rendered = pose_calibration.render_rgb_structure_from_thermal_camera(
        camera, gaussians, pipe, hyper, background, teacher_iteration,
        raw_offset_override=raw)
    return consensus._mind_descriptor(rendered)


def _direct_descriptors(side, grouped, raw, gaussians, pipe, hyper,
                        background, teacher_iteration):
    frames = sorted(grouped[side])
    return [
        _render_descriptor(
            grouped[side][frame], frame, raw, gaussians, pipe, hyper,
            background, teacher_iteration)
        for frame in frames
    ]


def _regional_activity(descriptors):
    transitions = torch.stack([
        (second - first).square().mean(dim=0)
        for first, second in zip(descriptors[:-1], descriptors[1:])
    ])
    symmetric = 0.5 * (transitions[:-1] + transitions[1:])
    result = torch.cat([
        F.adaptive_avg_pool2d(
            symmetric[:, None], (grid, grid)).flatten(1)
        for grid in REGIONAL_GRIDS
    ], dim=1)
    _require(result.ndim == 2 and result.shape[0] >= 32
             and result.shape[1] == sum(grid * grid for grid in REGIONAL_GRIDS),
             "Insufficient regional MIND activity")
    _require(bool(torch.isfinite(result).all()),
             "Non-finite regional MIND activity")
    return result


def _regional_activity_from_three(descriptors):
    backward = (descriptors[1] - descriptors[0]).square().mean(dim=0)
    forward = (descriptors[2] - descriptors[1]).square().mean(dim=0)
    symmetric = 0.5 * (backward + forward)
    result = torch.cat([
        F.adaptive_avg_pool2d(
            symmetric[None, None], (grid, grid)).flatten()
        for grid in REGIONAL_GRIDS
    ])
    _require(bool(torch.isfinite(result).all()),
             "Non-finite regional MIND activity row")
    return result


def _smooth(sequence, kernel):
    if kernel == 1:
        return sequence
    radius = kernel // 2
    return F.avg_pool1d(
        F.pad(sequence.T[None], (radius, radius), mode="replicate"),
        kernel_size=kernel, stride=1)[0].T


def _detrend(sequence):
    coordinate = torch.linspace(
        -1.0, 1.0, sequence.shape[0], device=sequence.device,
        dtype=sequence.dtype)[:, None]
    coordinate = coordinate - coordinate.mean()
    centered = sequence - sequence.mean(dim=0, keepdim=True)
    slope = (centered * coordinate).sum(dim=0, keepdim=True) / (
        coordinate.square().sum().clamp_min(1.0e-12))
    return centered - coordinate * slope


def _multiscale_loss(prediction, target):
    _require(prediction.shape == target.shape and prediction.ndim == 2,
             "Invalid regional activity inputs")
    correlations = []
    for kernel in SMOOTHING_KERNELS:
        first = _detrend(_smooth(prediction, kernel))
        second = _detrend(_smooth(target, kernel))
        denominator = first.norm(dim=0) * second.norm(dim=0)
        _require(bool((denominator > 1.0e-12).all()),
                 "Degenerate regional MIND activity")
        correlations.append((first * second).sum(dim=0) / denominator)
    result = -torch.stack(correlations).mean()
    _require(bool(torch.isfinite(result)),
             "Non-finite regional MIND correlation")
    return result


def _objective(predictions, targets):
    side_losses = {
        side: _multiscale_loss(predictions[side], targets[side])
        for side in ("left", "right")
    }
    return 0.5 * (side_losses["left"] + side_losses["right"]), side_losses


def _profile_row(offset_frames, grouped, targets, centers, max_offset_frames,
                 gaussians, pipe, hyper, background, teacher_iteration):
    raw = _raw_for_offset(offset_frames, max_offset_frames, "cuda")
    with torch.no_grad():
        predictions = {
            side: _regional_activity(_direct_descriptors(
                side, grouped, raw, gaussians, pipe, hyper, background,
                teacher_iteration))[centers[side]]
            for side in ("left", "right")
        }
        total, sides = _objective(predictions, targets)
    row = {
        "offset_frames": float(offset_frames),
        "loss": float(total.item()),
        "left_loss": float(sides["left"].item()),
        "right_loss": float(sides["right"].item()),
    }
    _require(all(math.isfinite(value) for value in row.values()),
             "Non-finite direct-render profile row")
    print("CLOCK_REFINEMENT_PROFILE " + json.dumps(
        row, sort_keys=True), flush=True)
    return row


def _direct_gradient(offset_frames, grouped, targets, centers,
                     max_offset_frames, gaussians, pipe, hyper, background,
                     teacher_iteration):
    with torch.no_grad():
        probe_raw = _raw_for_offset(
            offset_frames, max_offset_frames, "cuda")
        predictions = {
            side: _regional_activity(_direct_descriptors(
                side, grouped, probe_raw, gaussians, pipe, hyper,
                background, teacher_iteration))[centers[side]]
            for side in ("left", "right")
        }
    leaves = {
        side: predictions[side].clone().requires_grad_(True)
        for side in ("left", "right")
    }
    loss, side_losses = _objective(leaves, targets)
    weights = torch.autograd.grad(
        loss, (leaves["left"], leaves["right"]))
    weights = {
        side: weight.detach()
        for side, weight in zip(("left", "right"), weights)
    }
    raw = _raw_for_offset(
        offset_frames, max_offset_frames, "cuda").requires_grad_(True)
    raw_gradients_by_side = {
        side: torch.zeros((), device="cuda")
        for side in ("left", "right")
    }
    for side in ("left", "right"):
        frames = sorted(grouped[side])
        for row_index, center_index in enumerate(centers[side]):
            center_index = int(center_index.item())
            descriptors = [
                _render_descriptor(
                    grouped[side][frames[index]], frames[index], raw,
                    gaussians, pipe, hyper, background, teacher_iteration)
                for index in (
                    center_index, center_index + 1, center_index + 2)
            ]
            activity = _regional_activity_from_three(descriptors)
            weighted = (activity * weights[side][row_index]).sum()
            derivative = torch.autograd.grad(weighted, raw)[0]
            _require(bool(torch.isfinite(derivative)),
                     f"Non-finite regional derivative: {side} {center_index}")
            raw_gradients_by_side[side] += derivative.detach()
            if row_index % 16 == 15:
                torch.cuda.empty_cache()
    raw_gradient = sum(raw_gradients_by_side.values())
    offset_derivative = float(max_offset_frames) * (
        1.0 - torch.tanh(raw.detach()).square())
    gradient = raw_gradient / offset_derivative
    _require(bool(torch.isfinite(raw_gradient))
             and bool(torch.isfinite(gradient)),
             "Non-finite direct-render objective gradient")
    return {
        "offset_frames": float(offset_frames),
        "loss": float(loss.detach().item()),
        "left_loss": float(side_losses["left"].detach().item()),
        "right_loss": float(side_losses["right"].detach().item()),
        "raw_gradient": float(raw_gradient.detach().item()),
        "dloss_doffset": float(gradient.detach().item()),
        "side_raw_gradients": {
            side: float(value.detach().item())
            for side, value in raw_gradients_by_side.items()
        },
        "side_dloss_doffset": {
            side: float((value / offset_derivative).detach().item())
            for side, value in raw_gradients_by_side.items()
        },
    }


def run_refinement(report_path, cameras, coarse_offset_frames,
                   expected_shift_audit_only, max_offset_frames, gaussians,
                   pipe, hyper, background, teacher_iteration=30000):
    """Optimize only the shared clock against the frozen regional objective."""
    report_path = Path(report_path).resolve()
    _require(not report_path.exists(),
             f"Refusing existing refinement report: {report_path}")
    grouped = _group_cameras(cameras)
    with torch.no_grad():
        targets = {}
        for side in ("left", "right"):
            thermal_descriptors = [
                consensus._mind_descriptor(
                    grouped[side][frame].thermal_image.to("cuda"))
                for frame in sorted(grouped[side])
            ]
            targets[side] = _regional_activity(thermal_descriptors).detach()
    centers = {
        side: torch.arange(
            targets[side].shape[0], device=targets[side].device,
            dtype=torch.long)
        for side in ("left", "right")
    }

    trainability_before = _trainability_snapshot(gaussians, cameras)
    _require(all(value.grad is None for _, value, _ in trainability_before),
             "Clock refinement requires clean pre-training gradients")
    frozen = rgb_fidelity.freeze_rgb_state(gaussians, cameras)
    nuisance_before, nuisance_count = runtime.stable_tensor_hash(frozen)
    camera_before, camera_hash_before = _camera_manifest(cameras)
    started = time.time()
    trace = []
    start_gradient = _direct_gradient(
        coarse_offset_frames, grouped, targets, centers, max_offset_frames,
        gaussians, pipe, hyper, background, teacher_iteration)
    trace.append({"phase": "start", "step": 0, **start_gradient})
    print("CLOCK_REFINEMENT_OPTIMIZER_STEP " + json.dumps(
        trace[-1], sort_keys=True), flush=True)
    start_derivative = start_gradient["dloss_doffset"]
    _require(start_derivative != 0.0,
             "Coarse clock is already an exact stationary point")
    direction = -1.0 if start_derivative > 0.0 else 1.0
    previous = (float(coarse_offset_frames), start_derivative)
    bracket = None
    for step in range(1, REFINEMENT_MAX_BRACKET_STEPS + 1):
        offset = (float(coarse_offset_frames)
                  + direction * REFINEMENT_BRACKET_STEP_FRAMES * step)
        _require(abs(offset) < max_offset_frames,
                 "Clock bracket reached the tanh boundary")
        gradient = _direct_gradient(
            offset, grouped, targets, centers, max_offset_frames, gaussians,
            pipe, hyper, background, teacher_iteration)
        trace.append({"phase": "bracket", "step": step, **gradient})
        print("CLOCK_REFINEMENT_OPTIMIZER_STEP " + json.dumps(
            trace[-1], sort_keys=True), flush=True)
        current = (offset, gradient["dloss_doffset"])
        if previous[1] * current[1] <= 0.0:
            bracket = (previous, current)
            break
        previous = current
    _require(bracket is not None,
             "Failed to bracket a regional clock-gradient root")

    first, second = bracket
    for step in range(1, REFINEMENT_BISECTION_STEPS + 1):
        midpoint = 0.5 * (first[0] + second[0])
        gradient = _direct_gradient(
            midpoint, grouped, targets, centers, max_offset_frames,
            gaussians, pipe, hyper, background, teacher_iteration)
        trace.append({"phase": "bisection", "step": step, **gradient})
        print("CLOCK_REFINEMENT_OPTIMIZER_STEP " + json.dumps(
            trace[-1], sort_keys=True), flush=True)
        middle = (midpoint, gradient["dloss_doffset"])
        if first[1] * middle[1] <= 0.0:
            second = middle
        else:
            first = middle
    final_offset = 0.5 * (first[0] + second[0])
    raw_value = math.atanh(final_offset / max_offset_frames)
    final_gradient = _direct_gradient(
        final_offset, grouped, targets, centers, max_offset_frames, gaussians,
        pipe, hyper, background, teacher_iteration)
    nuisance_after, nuisance_after_count = runtime.stable_tensor_hash(
        rgb_fidelity.freeze_rgb_state(gaussians, cameras))
    camera_after, camera_hash_after = _camera_manifest(cameras)
    _restore_trainability(trainability_before)
    trainability_after = _trainability_snapshot(gaussians, cameras)
    trainability_restored = (
        len(trainability_before) == len(trainability_after)
        and all(
            before_name == after_name
            and before_value is after_value
            and before_requires_grad == after_requires_grad
            for (before_name, before_value, before_requires_grad),
            (after_name, after_value, after_requires_grad)
            in zip(trainability_before, trainability_after)))
    checks = {
        "finite_trace": all(
            all(math.isfinite(value) for key, value in row.items()
                if key not in ("phase", "side_raw_gradients",
                               "side_dloss_doffset"))
            for row in trace),
        "gradient_root_bracketed": first[1] * second[1] <= 0.0,
        "final_bracket_width_at_most_configured_resolution": (
            abs(second[0] - first[0])
            <= (REFINEMENT_BRACKET_STEP_FRAMES
                / (2 ** REFINEMENT_BISECTION_STEPS)) + 1.0e-12),
        "objective_decreased": final_gradient["loss"] < trace[0]["loss"],
        "expected_shift_error_at_most_quarter_frame": (
            abs(final_offset - float(expected_shift_audit_only)) <= 0.25),
        "frozen_nuisance_unchanged": (
            nuisance_count == nuisance_after_count
            and nuisance_before == nuisance_after),
        "camera_state_unchanged": (
            camera_before == camera_after
            and camera_hash_before == camera_hash_after),
        "trainability_restored": trainability_restored,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": "covers_regional_mind_gradient_root_refinement",
        "status": status,
        "scene": "Covers",
        "candidate_enumeration": False,
        "shift_truth_input_to_optimizer": False,
        "expected_shift_used_only_after_optimization_for_accuracy_audit": True,
        "expected_shift_audit_only": float(expected_shift_audit_only),
        "expected_shift_error_frames": abs(
            final_offset - float(expected_shift_audit_only)),
        "formal_test_cameras_constructed": False,
        "coarse_offset_frames": float(coarse_offset_frames),
        "refined_offset_frames": float(final_offset),
        "refined_raw": float(raw_value),
        "optimizer": {
            "name": "gradient_sign_bracket_bisection",
            "bracket_step_frames": REFINEMENT_BRACKET_STEP_FRAMES,
            "max_bracket_steps": REFINEMENT_MAX_BRACKET_STEPS,
            "bisection_steps": REFINEMENT_BISECTION_STEPS,
            "executed_gradient_evaluations": len(trace) + 1,
            "final_bracket_offsets_frames": [first[0], second[0]],
            "final_bracket_gradients": [first[1], second[1]],
        },
        "trace": trace,
        "final_gradient": final_gradient,
        "checks": checks,
        "observable": (
            "continuous thermal-view RGB-teacher re-render before fixed "
            "2x2+4x4 regional "
            "MIND symmetric activity; equal-side affine-detrended multiscale "
            "Pearson"),
        "regional_grids": list(REGIONAL_GRIDS),
        "smoothing_kernels": list(SMOOTHING_KERNELS),
        "camera_count_by_side": {
            side: len(grouped[side]) for side in ("left", "right")
        },
        "nuisance_sha256_before": nuisance_before,
        "nuisance_sha256_after": nuisance_after,
        "nuisance_tensor_count": nuisance_count,
        "camera_manifest_sha256_before": camera_hash_before,
        "camera_manifest_sha256_after": camera_hash_after,
        "trainability_tensor_count": len(trainability_before),
        "trainable_tensor_count_before": sum(
            requires_grad for _, _, requires_grad in trainability_before),
        "trainable_tensor_count_after": sum(
            requires_grad for _, _, requires_grad in trainability_after),
        "script_sha256": runtime.sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    marker = report_path.parent / f"CLOCK_REFINEMENT_{status}"
    _require(not marker.exists(), f"Refusing existing marker: {marker}")
    marker.write_text(status + "\n", encoding="utf-8")
    print("CLOCK_REFINEMENT_RESULT " + json.dumps({
        "status": status,
        "coarse_offset_frames": float(coarse_offset_frames),
        "refined_offset_frames": float(final_offset),
        "checks": checks,
        "report": str(report_path),
    }, sort_keys=True), flush=True)
    return report


def run_gate(report_path, cameras, temporal_consensus, coarse_offset_frames,
             expected_shift_audit_only, max_offset_frames, gaussians, pipe,
             hyper, background, teacher_iteration=30000):
    """Write one fail-closed profile/gradient report and return its payload."""
    report_path = Path(report_path).resolve()
    _require(not report_path.exists(),
             f"Refusing existing refinement report: {report_path}")
    grouped = _group_cameras(cameras)
    with torch.no_grad():
        targets = {}
        for side in ("left", "right"):
            thermal_descriptors = [
                consensus._mind_descriptor(
                    grouped[side][frame].thermal_image.to("cuda"))
                for frame in sorted(grouped[side])
            ]
            targets[side] = _regional_activity(thermal_descriptors).detach()
    centers = {
        side: torch.arange(
            targets[side].shape[0],
            device=targets[side].device,
            dtype=torch.long)
        for side in ("left", "right")
    }
    for side in ("left", "right"):
        _require(int(centers[side].max()) + 2 < len(grouped[side]),
                 f"Refinement center exceeds camera support: {side}")

    frozen = rgb_fidelity.freeze_rgb_state(gaussians, cameras)
    nuisance_before, nuisance_count = runtime.stable_tensor_hash(frozen)
    camera_before, camera_hash_before = _camera_manifest(cameras)
    started = time.time()
    offsets = [round(
        float(coarse_offset_frames) + index * PROFILE_STEP_FRAMES, 6)
        for index in range(
            -int(PROFILE_RADIUS_FRAMES / PROFILE_STEP_FRAMES),
            int(PROFILE_RADIUS_FRAMES / PROFILE_STEP_FRAMES) + 1)
    ]
    _require(offsets[0] > -max_offset_frames
             and offsets[-1] < max_offset_frames,
             "Refinement profile exceeds the clock domain")
    rows = [
        _profile_row(
            offset, grouped, targets, centers, max_offset_frames, gaussians,
            pipe, hyper, background, teacher_iteration)
        for offset in offsets
    ]
    gradient = _direct_gradient(
        coarse_offset_frames, grouped, targets, centers, max_offset_frames,
        gaussians, pipe, hyper, background, teacher_iteration)

    best_index = min(range(len(rows)), key=lambda index: rows[index]["loss"])
    best = rows[best_index]
    second = min(
        row["loss"] for index, row in enumerate(rows)
        if index != best_index)
    side_best = {
        side: min(rows, key=lambda row: row[f"{side}_loss"])["offset_frames"]
        for side in ("left", "right")
    }
    lower = next(row for row in rows if math.isclose(
        row["offset_frames"],
        float(coarse_offset_frames) - PROFILE_STEP_FRAMES,
        rel_tol=0.0, abs_tol=1.0e-6))
    upper = next(row for row in rows if math.isclose(
        row["offset_frames"],
        float(coarse_offset_frames) + PROFILE_STEP_FRAMES,
        rel_tol=0.0, abs_tol=1.0e-6))
    central_fd = ((upper["loss"] - lower["loss"])
                  / (2.0 * PROFILE_STEP_FRAMES))
    desired_direction = best["offset_frames"] - float(coarse_offset_frames)
    gradient_points_to_best = (
        (desired_direction > 0.0
         and gradient["dloss_doffset"] < 0.0 and central_fd < 0.0)
        or (desired_direction < 0.0
            and gradient["dloss_doffset"] > 0.0 and central_fd > 0.0)
        or (desired_direction == 0.0
            and abs(gradient["dloss_doffset"]) < 1.0e-6))
    side_gradients_point_to_best = all(
        (desired_direction > 0.0 and value < 0.0)
        or (desired_direction < 0.0 and value > 0.0)
        or (desired_direction == 0.0 and abs(value) < 1.0e-6)
        for value in gradient["side_dloss_doffset"].values())

    nuisance_after, nuisance_after_count = runtime.stable_tensor_hash(
        rgb_fidelity.freeze_rgb_state(gaussians, cameras))
    camera_after, camera_hash_after = _camera_manifest(cameras)
    checks = {
        "finite_profile": all(
            all(math.isfinite(value) for value in row.values())
            for row in rows),
        "unique_interior_minimum": (
            0 < best_index < len(rows) - 1
            and second > best["loss"]),
        "side_gradients_agree_toward_joint_minimum": (
            side_gradients_point_to_best),
        "gradient_points_from_coarse_to_profile_minimum": (
            gradient_points_to_best),
        "expected_shift_error_at_most_quarter_frame": (
            abs(best["offset_frames"]
                - float(expected_shift_audit_only))
            <= PROFILE_STEP_FRAMES),
        "frozen_nuisance_unchanged": (
            nuisance_count == nuisance_after_count
            and nuisance_before == nuisance_after),
        "camera_state_unchanged": (
            camera_before == camera_after
            and camera_hash_before == camera_hash_after),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    coarse_loss, _ = temporal_consensus.loss(torch.tensor(
        float(coarse_offset_frames), device="cuda"))
    report = {
        "schema": "covers_regional_mind_directional_clock_gate",
        "status": status,
        "scene": "Covers",
        "profile_only": True,
        "candidate_enumeration_in_optimizer": False,
        "profile_scan_used_only_for_gate": True,
        "shift_truth_input_to_observable": False,
        "expected_shift_used_only_after_profile_for_accuracy_audit": True,
        "expected_shift_audit_only": float(expected_shift_audit_only),
        "coarse_offset_frames": float(coarse_offset_frames),
        "coarse_offset_source": "previous blind temporal-consensus result",
        "coarse_trace_loss": float(coarse_loss.detach().item()),
        "profile_radius_frames": PROFILE_RADIUS_FRAMES,
        "profile_step_frames": PROFILE_STEP_FRAMES,
        "profile_rows": rows,
        "profile_best_offset_frames": best["offset_frames"],
        "profile_best_loss": best["loss"],
        "profile_winner_margin": second - best["loss"],
        "side_best_offsets_frames": side_best,
        "direct_gradient_at_coarse": gradient,
        "central_fd_dloss_doffset_at_coarse": central_fd,
        "checks": checks,
        "observable": (
            "continuous thermal-view RGB-teacher re-render before fixed "
            "2x2+4x4 regional "
            "MIND symmetric activity; equal-side affine-detrended multiscale "
            "Pearson with per-side gradient agreement"),
        "regional_grids": list(REGIONAL_GRIDS),
        "smoothing_kernels": list(SMOOTHING_KERNELS),
        "support_indices_by_side": {
            side: [int(value) for value in centers[side].tolist()]
            for side in ("left", "right")
        },
        "camera_count_by_side": {
            side: len(grouped[side]) for side in ("left", "right")
        },
        "nuisance_sha256_before": nuisance_before,
        "nuisance_sha256_after": nuisance_after,
        "nuisance_tensor_count": nuisance_count,
        "camera_manifest_sha256_before": camera_hash_before,
        "camera_manifest_sha256_after": camera_hash_after,
        "formal_test_cameras_constructed": False,
        "script_sha256": runtime.sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.time() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    marker = report_path.parent / f"CLOCK_REFINEMENT_GATE_{status}"
    _require(not marker.exists(), f"Refusing existing marker: {marker}")
    marker.write_text(status + "\n", encoding="utf-8")
    print("CLOCK_REFINEMENT_GATE_RESULT " + json.dumps({
        "status": status,
        "profile_best_offset_frames": best["offset_frames"],
        "expected_shift_error_frames": abs(
            best["offset_frames"] - float(expected_shift_audit_only)),
        "checks": checks,
        "report": str(report_path),
    }, sort_keys=True), flush=True)
    return report
