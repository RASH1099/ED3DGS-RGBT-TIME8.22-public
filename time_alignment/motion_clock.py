"""Training-only global motion correlation for one zero-initialized clock."""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from time_alignment import temporal_consensus as consensus


SCHEMA = "covers_zero_start_motion_continuation_clock"
CANDIDATE_OFFSETS = tuple(range(-22, 23, 2))
SMOOTHING_SIGMAS = (12.0, 6.0, 3.0, 1.5, 0.75, 0.375, 0.1875)
STEPS_PER_SIGMA = 100
RAW_LEARNING_RATE = 0.01
SEARCH_RADIUS = 5
MAX_OFFSET_FRAMES = 24.0


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _gray(camera, modality):
    image = (
        camera.original_image if modality == "rgb" else camera.thermal_image)
    _require(image is not None,
             f"Motion clock requires loaded {modality} training images")
    _require(torch.is_tensor(image) and image.ndim == 3,
             f"Missing {modality} tensor for {camera.image_name}")
    array = image.detach().to(device="cpu", dtype=torch.float32).numpy()
    _require(array.shape[0] in (1, 3),
             f"Invalid {modality} channels for {camera.image_name}")
    if array.shape[0] == 3:
        gray = (0.299 * array[0] + 0.587 * array[1] + 0.114 * array[2])
    else:
        gray = array[0]
    gray = cv2.resize(
        gray, (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
    low, high = np.quantile(gray, (0.02, 0.98))
    normalized = np.clip(
        (gray - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)
    _require(bool(np.isfinite(normalized).all()),
             f"Non-finite {modality} image for {camera.image_name}")
    return normalized


def _change(first, second):
    change = cv2.absdiff(second, first)
    change = cv2.GaussianBlur(change, (7, 7), 0)
    change = cv2.resize(change, (40, 30), interpolation=cv2.INTER_AREA)
    change = change - change.mean()
    norm = float(np.linalg.norm(change))
    _require(math.isfinite(norm) and norm > 1.0e-8,
             "Degenerate motion-change map")
    return np.ascontiguousarray(change / norm, dtype=np.float32)


def _group(cameras):
    grouped = {"left": {}, "right": {}}
    for camera in cameras:
        side = consensus._side(camera)
        frame = int(camera.frame_no)
        _require(frame not in grouped[side],
                 f"Duplicate motion-clock camera: {side} {frame}")
        grouped[side][frame] = camera
    for side, by_frame in grouped.items():
        frames = sorted(by_frame)
        steps = {
            second - first for first, second in zip(frames[:-1], frames[1:])}
        _require(len(frames) == 133 and steps == {2},
                 f"Motion clock requires the full train trajectory: {side}")
    return grouped


def _build_changes(grouped):
    changes = {"rgb": {}, "thermal": {}}
    digest = hashlib.sha256()
    for modality in ("rgb", "thermal"):
        for side, by_frame in grouped.items():
            frames = sorted(by_frame)
            images = {
                frame: _gray(by_frame[frame], modality) for frame in frames}
            side_changes = {}
            for first, second in zip(frames[:-1], frames[1:]):
                _require(second - first == 2,
                         f"Invalid motion transition: {side} {first}->{second}")
                value = _change(images[first], images[second])
                side_changes[first] = value
                digest.update(
                    f"{modality}:{side}:{first}\n".encode("utf-8"))
                digest.update(value.tobytes(order="C"))
            changes[modality][side] = side_changes
    return changes, digest.hexdigest()


def _correlation(first, second):
    padded = cv2.copyMakeBorder(
        second, SEARCH_RADIUS, SEARCH_RADIUS, SEARCH_RADIUS, SEARCH_RADIUS,
        cv2.BORDER_REFLECT)
    response = cv2.matchTemplate(
        padded, first, cv2.TM_CCOEFF_NORMED)
    value = float(response.max())
    _require(math.isfinite(value), "Non-finite motion correlation")
    return value


def _bases(changes, side):
    frames = sorted(changes["rgb"][side])
    minimum = frames[0] + int(MAX_OFFSET_FRAMES)
    maximum = frames[-1] - int(MAX_OFFSET_FRAMES)
    result = [frame for frame in frames if minimum <= frame <= maximum]
    _require(len(result) >= 100, f"Insufficient motion support: {side}")
    return result


def _profile(changes, scope):
    rows = []
    for candidate in CANDIDATE_OFFSETS:
        side_scores = {}
        for side in ("left", "right"):
            bases = _bases(changes, side)
            values = [
                _correlation(
                    changes["rgb"][side][base + candidate],
                    changes["thermal"][side][base])
                for base in bases
            ]
            if scope == "early":
                values = values[:len(values) // 2]
            elif scope == "late":
                values = values[len(values) // 2:]
            _require(values, f"Empty {scope} profile for {side}")
            side_scores[side] = float(np.mean(values))
        rows.append({
            "offset_frames": candidate,
            "score": 0.5 * (side_scores["left"] + side_scores["right"]),
            "left_score": side_scores["left"],
            "right_score": side_scores["right"],
        })
    return rows


def _profile_summary(rows):
    ordered = sorted(rows, key=lambda row: row["score"], reverse=True)
    best = ordered[0]
    return {
        "joint_best_offset_frames": best["offset_frames"],
        "left_best_offset_frames": max(
            rows, key=lambda row: row["left_score"])["offset_frames"],
        "right_best_offset_frames": max(
            rows, key=lambda row: row["right_score"])["offset_frames"],
        "winner_margin": best["score"] - ordered[1]["score"],
        "rows": rows,
    }


def _optimizer_parameters(optimizer):
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def _optimize(rows, raw, optimizer):
    _require(isinstance(raw, torch.nn.Parameter) and raw.numel() == 1,
             "Motion clock requires the Scene scalar clock parameter")
    _require(isinstance(optimizer, torch.optim.Adam),
             "Motion clock requires the registered Adam optimizer")
    _require(raw.requires_grad, "Scene clock must be trainable")
    _require(float(raw.detach().item()) == 0.0,
             "Motion clock must start at exact raw zero")
    parameters = _optimizer_parameters(optimizer)
    _require(len(parameters) == 1 and parameters[0] is raw,
             "Motion clock optimizer must own only the Scene clock")
    _require(len(optimizer.state) == 0,
             "Motion clock optimizer must have no prior state")
    for group in optimizer.param_groups:
        group["lr"] = RAW_LEARNING_RATE
    candidates = torch.tensor(
        [row["offset_frames"] for row in rows],
        dtype=raw.dtype, device=raw.device)
    scores = torch.tensor(
        [row["score"] for row in rows],
        dtype=raw.dtype, device=raw.device)
    trace = []
    finite_gradient_count = 0
    nonzero_gradient_count = 0
    for sigma in SMOOTHING_SIGMAS:
        first_loss = None
        last_loss = None
        for step in range(1, STEPS_PER_SIGMA + 1):
            optimizer.zero_grad(set_to_none=True)
            offset = MAX_OFFSET_FRAMES * torch.tanh(raw)
            logits = -(offset - candidates).square() / (2.0 * sigma * sigma)
            weights = torch.softmax(logits, dim=0)
            loss = -(weights * scores).sum()
            loss.backward()
            _require(raw.grad is not None,
                     "Motion-clock loss did not reach the Scene clock")
            gradient = float(raw.grad.detach().item())
            _require(math.isfinite(gradient),
                     "Non-finite motion-clock continuation gradient")
            finite_gradient_count += 1
            nonzero_gradient_count += int(gradient != 0.0)
            optimizer.step()
            last_loss = float(loss.detach().item())
            if first_loss is None:
                first_loss = last_loss
        trace.append({
            "sigma_frames": sigma,
            "steps": STEPS_PER_SIGMA,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "offset_frames": float(
                (MAX_OFFSET_FRAMES * torch.tanh(raw)).detach().item()),
            "raw": float(raw.detach().item()),
        })
    return {
        "final_offset_frames": float(
            (MAX_OFFSET_FRAMES * torch.tanh(raw)).detach().item()),
        "final_raw": float(raw.detach().item()),
        "finite_gradient_count": finite_gradient_count,
        "nonzero_gradient_count": nonzero_gradient_count,
        "optimizer_steps": len(SMOOTHING_SIGMAS) * STEPS_PER_SIGMA,
        "optimizer_state_count": len(optimizer.state),
        "optimizer_owns_scene_clock_only": True,
        "trace": trace,
    }


def run(report_path, cameras, scene_raw_clock, scene_clock_optimizer,
        scene_max_offset_frames, expected_shift_audit_only):
    report_path = Path(report_path).resolve()
    _require(not report_path.exists(),
             f"Refusing existing motion-clock report: {report_path}")
    _require(float(scene_max_offset_frames) == MAX_OFFSET_FRAMES,
             "Motion-clock bound does not match the registered protocol")
    grouped = _group(cameras)
    changes, input_hash = _build_changes(grouped)
    profiles = {
        scope: _profile_summary(_profile(changes, scope))
        for scope in ("all", "early", "late")
    }
    raw_object_id = id(scene_raw_clock)
    optimizer_object_id = id(scene_clock_optimizer)
    optimization = _optimize(
        profiles["all"]["rows"], scene_raw_clock, scene_clock_optimizer)
    final_offset = optimization["final_offset_frames"]
    final_raw = optimization["final_raw"]

    # The registered shift is deliberately first read here, after optimization.
    _require(callable(expected_shift_audit_only),
             "Expected-shift audit must be deferred until after optimization")
    expected_shift = float(expected_shift_audit_only())
    full_peak = float(profiles["all"]["joint_best_offset_frames"])
    checks = {
        "exact_raw_zero_initialization": True,
        "one_raw_clock_scalar": True,
        "scene_parameter_identity_preserved": (
            id(scene_raw_clock) == raw_object_id
            and id(scene_clock_optimizer) == optimizer_object_id),
        "scene_optimizer_owns_clock_only": (
            optimization["optimizer_owns_scene_clock_only"]),
        "fixed_symmetric_candidate_bank": (
            CANDIDATE_OFFSETS == tuple(range(-22, 23, 2))),
        "finite_unique_full_profile": (
            profiles["all"]["winner_margin"] > 0.0),
        "full_left_right_joint_peak_agreement": (
            profiles["all"]["left_best_offset_frames"] == full_peak
            and profiles["all"]["right_best_offset_frames"] == full_peak),
        "early_late_joint_peak_agreement": (
            profiles["early"]["joint_best_offset_frames"] == full_peak
            and profiles["late"]["joint_best_offset_frames"] == full_peak),
        "continuation_matches_blind_profile_within_quarter_frame": (
            abs(final_offset - full_peak) <= 0.25),
        "expected_shift_error_at_most_quarter_frame": (
            abs(final_offset - expected_shift) <= 0.25),
        "finite_optimizer_gradients": (
            optimization["finite_gradient_count"]
            == optimization["optimizer_steps"]),
        "nonzero_optimizer_gradients": (
            optimization["nonzero_gradient_count"] > 0),
        "final_offset_strictly_inside_bound": (
            abs(final_offset) < MAX_OFFSET_FRAMES),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": SCHEMA,
        "status": status,
        "scene": "Covers",
        "initial_raw": 0.0,
        "initial_offset_frames": 0.0,
        "preloaded_offset": False,
        "shift_truth_input_to_observable": False,
        "camera_shift_metadata_read_by_observable": False,
        "expected_shift_used_only_after_optimization_for_accuracy_audit": True,
        "expected_shift_audit_only": expected_shift,
        "expected_shift_error_frames": abs(final_offset - expected_shift),
        "candidate_lag_bank_in_observable": True,
        "trainable_candidate_logits": False,
        "trainable_state": "one_raw_clock_scalar",
        "optimizer": "adam",
        "scene_raw_object_id": raw_object_id,
        "scene_optimizer_object_id": optimizer_object_id,
        "candidate_offsets_frames": list(CANDIDATE_OFFSETS),
        "translation_search_radius_pixels_at_30x40": SEARCH_RADIUS,
        "smoothing_sigmas_frames": list(SMOOTHING_SIGMAS),
        "raw_learning_rate": RAW_LEARNING_RATE,
        "steps_per_sigma": STEPS_PER_SIGMA,
        "training_camera_count": sum(len(value) for value in grouped.values()),
        "training_cameras_by_side": {
            side: len(value) for side, value in grouped.items()},
        "motion_centers_by_side": {
            side: len(_bases(changes, side)) for side in grouped},
        "processed_motion_input_sha256": input_hash,
        "profiles": profiles,
        "optimization": optimization,
        "final_offset_frames": final_offset,
        "final_raw": final_raw,
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    marker = report_path.parent / f"MOTION_CLOCK_{status}"
    _require(not marker.exists(), f"Refusing existing marker: {marker}")
    marker.write_text(status + "\n", encoding="utf-8")
    print("MOTION_CLOCK_RESULT " + json.dumps({
        "status": status,
        "final_offset_frames": final_offset,
        "expected_shift_error_frames": abs(final_offset - expected_shift),
        "full_profile_peak_offset_frames": full_peak,
        "checks": checks,
        "report": str(report_path),
    }, sort_keys=True), flush=True)
    return report
