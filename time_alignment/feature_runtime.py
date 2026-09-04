#!/usr/bin/env python3
"""Sealed Covers NC-TC train-only residual gate.

The runner loads the audited strict RGB-only teacher, fits only static Thermal
SH-DC appearance on a support block, then freezes every nuisance tensor and
updates one scalar clock residual on a disjoint buffered query block through
the real Thermal rasterizer.
"""

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import mmengine
import numpy as np
import torch
import torch.nn.functional as F
import diff_gaussian_rasterization
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from arguments import (
    ModelHiddenParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import ssim
from utils.params_utils import merge_hparams


SCHEMA = "covers_nctc_gate_v1"
FRAME_PATTERN = re.compile(r"covers_(left|right)_rgb_(\d{4})$")
ARM_SHIFTS = {"N": 7.375, "Z": 8.0, "P": 8.625}
EXPECTED_COARSE_OFFSET = 8.0
COARSE_CANDIDATES = tuple(range(-12, 13, 2))
RESIDUAL_BOUND = 1.25
DIRECTION_HALF_WINDOW = 2
PROFILE_RESIDUALS = tuple(float(v) for v in np.arange(-1.125, 1.1251, 0.125))
EXPECTED_RASTERIZER_SHA256 = (
    "b41c1eeb0fbc11b66b6685bc42ade856155bfd1f4ac1e4c402b196f2efefbcfc")
EXPECTED_RASTERIZER_WRAPPER_SHA256 = (
    "27fe693d983140d6f36f443a9a80d33db737b8413cf33c140052830939df4a0a")
EXPECTED_DEFORMATION_SHA256 = (
    "52a3ea1144f9cbc2ad61eaa920d29410c953351628fab2f473f1b0e9400f2f96")
EXPECTED_TEACHER = {
    "point_cloud.ply": "a9627b0a61275b5719989ae21529f31282ac01978b23bcb56effec40f1a45ce0",
    "deformation.pth": "b20e87964f7edf099390251ee9d03cc872b7fca711face6f062b7acebf457bb0",
    "thermal_camera_state.pth": "388d839745e6acb9f8251435977bd2846aeb2cbcbf54d89c6b316b6ad730b5a4",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def side_frame(view):
    match = FRAME_PATTERN.fullmatch(Path(view.image_name).stem)
    require(match is not None, f"Unexpected Covers image name: {view.image_name}")
    return match.group(1), int(match.group(2))


def stable_tensor_hash(named_tensors):
    digest = hashlib.sha256()
    seen = set()
    count = 0
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        require(torch.is_tensor(tensor), f"Non-tensor nuisance entry: {name}")
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        detached = tensor.detach().cpu().contiguous()
        require(bool(torch.isfinite(detached).all()), f"Non-finite nuisance tensor: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(str(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
        count += 1
    return digest.hexdigest(), count


def gaussian_nuisance_tensors(gaussians, views):
    names = (
        "_xyz", "_features_dc", "_features_rest", "_thermal_dc",
        "_thermal_rest", "_opacity", "_thermal_opacity", "_scaling",
        "_rotation", "_embedding", "_t_embedding", "_logit_modality",
        "_pose_r", "_pose_t", "_pose_log_s",
    )
    tensors = []
    for name in names:
        value = getattr(gaussians, name, None)
        require(torch.is_tensor(value), f"Missing Gaussian nuisance tensor: {name}")
        tensors.append((f"gaussian.{name}", value))
    for name, value in gaussians._deformation.named_parameters():
        tensors.append((f"deformation.{name}", value))
    pose_head = getattr(gaussians, "pose_head", None)
    if pose_head is not None:
        for name, value in pose_head.named_parameters():
            tensors.append((f"pose_head.{name}", value))
    thermal_pose_head = getattr(gaussians, "thermal_pose_head", None)
    if thermal_pose_head is not None:
        for name, value in thermal_pose_head.named_parameters():
            tensors.append((f"thermal_pose_head.{name}", value))
    for name in ("init_rotation_rgb", "init_translation_rgb"):
        value = getattr(gaussians, name, None)
        if torch.is_tensor(value):
            tensors.append((f"gaussian_optional.{name}", value))
    camera_leaf_count = 0
    for view in sorted(views, key=lambda item: side_frame(item)):
        side, frame = side_frame(view)
        for name in (
            "thermal_delta_quaternion", "thermal_delta_translation",
            "learnable_tfovx", "learnable_tfovy",
        ):
            value = getattr(view, name, None)
            require(torch.is_tensor(value),
                    f"Missing camera nuisance leaf: {side} {frame} {name}")
            tensors.append((f"camera.{side}.{frame}.{name}", value))
            camera_leaf_count += 1
    require(camera_leaf_count == 266 * 4,
            f"Unexpected camera nuisance leaf count: {camera_leaf_count}")
    return tensors


def full_forward_state_tensors(gaussians, views):
    tensors = list(gaussian_nuisance_tensors(gaussians, views))
    for name in (
        "max_radii2D", "xyz_gradient_accum", "denom",
        "xyz_gradient_accum_rgb", "xyz_gradient_accum_th",
        "denom_rgb", "denom_th",
    ):
        value = getattr(gaussians, name, None)
        require(torch.is_tensor(value), f"Missing Gaussian forward state: {name}")
        tensors.append((f"gaussian_state.{name}", value))
    for name, value in gaussians._deformation.named_buffers():
        tensors.append((f"deformation_buffer.{name}", value))
    pose_head = getattr(gaussians, "pose_head", None)
    if pose_head is not None:
        for name, value in pose_head.named_buffers():
            tensors.append((f"pose_head_buffer.{name}", value))
    thermal_pose_head = getattr(gaussians, "thermal_pose_head", None)
    if thermal_pose_head is not None:
        for name, value in thermal_pose_head.named_buffers():
            tensors.append((f"thermal_pose_head_buffer.{name}", value))
    camera_state_names = (
        "world_view_transform", "full_proj_transform",
        "projection_matrix", "projection_matrix_thermal",
        "full_proj_transform_thermal",
        "thermal_projection_matrix_learnable", "temporal_pose_frames",
        "temporal_pose_track",
    )
    for view in sorted(views, key=lambda item: side_frame(item)):
        side, frame = side_frame(view)
        for name in camera_state_names:
            value = getattr(view, name, None)
            require(torch.is_tensor(value),
                    f"Missing camera forward state: {side} {frame} {name}")
            tensors.append((f"camera_state.{side}.{frame}.{name}", value))
    return tensors


def camera_scalar_manifest(views):
    rows = []
    for view in sorted(views, key=lambda item: side_frame(item)):
        side, frame = side_frame(view)
        row = {
            "side": side,
            "frame": frame,
            "frame_no": float(view.frame_no),
            "temporal_alignment_enabled": bool(view.temporal_alignment_enabled),
            "temporal_observation_correction_enabled": bool(
                view.temporal_observation_correction_enabled),
            "temporal_strict_common_support": bool(
                view.temporal_strict_common_support),
            "coarse_offset": float(view.nctc_coarse_offset),
            "residual_bound": float(view.temporal_offset_max_frames),
            "duration": float(view.temporal_duration),
        }
        for name in ("init_rotation_rgb", "init_translation_rgb"):
            value = getattr(view, name, None)
            if torch.is_tensor(value):
                detached = value.detach().cpu().contiguous()
                require(bool(torch.isfinite(detached).all()),
                        f"Non-finite optional camera state: {side} {frame} {name}")
                row[name] = detached.tolist()
        rows.append(row)
    require(len(rows) == 266, "Camera scalar manifest is incomplete")
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return rows, hashlib.sha256(encoded).hexdigest()


def set_all_nuisance_requires_grad(gaussians, views, enabled_name=None):
    for name, tensor in gaussian_nuisance_tensors(gaussians, views):
        tensor.requires_grad_(name == enabled_name)
        tensor.grad = None


def image_objective(prediction, target, lambda_dssim):
    require(prediction.shape == target.shape, "Thermal image shape mismatch")
    require(prediction.device == target.device, "Thermal image device mismatch")
    require(prediction.dtype == target.dtype, "Thermal image dtype mismatch")
    require(bool(torch.isfinite(prediction).all()), "Non-finite Thermal render")
    require(bool(torch.isfinite(target).all()), "Non-finite Thermal target")
    l1 = torch.abs(prediction - target).mean()
    if lambda_dssim == 0.0:
        return l1, l1, torch.zeros((), device=l1.device)
    ssim_value, _ = ssim(prediction.unsqueeze(0), target.unsqueeze(0))
    dssim = (1.0 - ssim_value) / 2.0
    return l1 + lambda_dssim * dssim, l1, dssim


def target_on_render_device(target, render):
    require(target.dtype == render.dtype,
            f"Observed/render dtype mismatch: {target.dtype} vs {render.dtype}")
    staged = target.to(device=render.device)
    require(staged.device == render.device, "Observed target staging failed")
    return staged


def configure_temporal_views(views_by_side, raw_offset, duration, coarse_offset):
    for side, by_frame in views_by_side.items():
        frames = sorted(by_frame)
        matrices = torch.stack(
            [by_frame[frame].world_view_transform.detach().clone() for frame in frames],
            dim=0,
        )
        frame_tensor = torch.tensor(
            frames, device=matrices.device, dtype=matrices.dtype)
        for frame, view in by_frame.items():
            view.temporal_alignment_enabled = True
            view.temporal_observation_correction_enabled = True
            object.__setattr__(
                view, "nctc_temporal_offset_raw_override", raw_offset)
            view.temporal_offset_max_frames = RESIDUAL_BOUND
            view.temporal_duration = float(duration)
            view.temporal_pose_frames = frame_tensor
            view.temporal_pose_track = matrices
            view.temporal_strict_common_support = True
            view.thermal_frame_shift = 1
            view.nctc_coarse_offset = float(coarse_offset)


def proxy_spec(side, target_frame, shift, frames):
    requested = float(target_frame) + float(shift)
    rounded = int(round(requested))
    if math.isclose(requested, rounded, rel_tol=0.0, abs_tol=1e-12) and rounded in frames:
        return {
            "side": side,
            "target_frame": int(target_frame),
            "shift": float(shift),
            "requested_source_frame": requested,
            "lower": rounded,
            "upper": rounded,
            "upper_weight": 0.0,
        }
    lower_candidates = [frame for frame in frames if frame < requested]
    upper_candidates = [frame for frame in frames if frame > requested]
    require(lower_candidates and upper_candidates,
            f"Missing train-only proxy bracket: {side} target={target_frame} shift={shift}")
    lower = max(lower_candidates)
    upper = min(upper_candidates)
    require(upper - lower == 2,
            f"Proxy bracket crosses a non-train interval: {side} {lower}->{upper}")
    weight = (requested - lower) / (upper - lower)
    require(0.0 < weight < 1.0, "Invalid proxy interpolation weight")
    return {
        "side": side,
        "target_frame": int(target_frame),
        "shift": float(shift),
        "requested_source_frame": requested,
        "lower": int(lower),
        "upper": int(upper),
        "upper_weight": float(weight),
    }


def proxy_image(spec, views_by_side, cache):
    key = (spec["side"], spec["target_frame"], spec["shift"])
    if key in cache:
        return cache[key]
    by_frame = views_by_side[spec["side"]]
    lower_image = by_frame[spec["lower"]].thermal_image.detach()
    require(bool(torch.isfinite(lower_image).all()), "Non-finite lower Thermal source")
    if spec["lower"] == spec["upper"]:
        result = lower_image
    else:
        upper_image = by_frame[spec["upper"]].thermal_image.detach()
        require(bool(torch.isfinite(upper_image).all()), "Non-finite upper Thermal source")
        weight = spec["upper_weight"]
        result = (1.0 - weight) * lower_image + weight * upper_image
    require(bool(torch.isfinite(result).all()), "Non-finite interpolated Thermal proxy")
    cache[key] = result
    return result


def temporal_change_gray(image, size=(120, 160)):
    if image.shape[0] == 3:
        gray = (
            0.299 * image[0:1]
            + 0.587 * image[1:2]
            + 0.114 * image[2:3]
        )
    else:
        gray = image.mean(dim=0, keepdim=True)
    return F.interpolate(
        gray.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    )[0, 0].detach().cpu()


def standardized_temporal_change(first, second):
    change = (second - first).abs()
    border_y = max(1, change.shape[0] // 12)
    border_x = max(1, change.shape[1] // 12)
    change = change[border_y:-border_y, border_x:-border_x]
    std = change.std(unbiased=False)
    require(bool(torch.isfinite(std)), "Non-finite train temporal-change map")
    if float(std.item()) == 0.0:
        return None
    return (change - change.mean()) / std


def build_coarse_target_manifest(views_by_side):
    """Fixed target IDs valid for every registered coarse candidate."""
    targets = {}
    centers = {}
    for side, by_frame in views_by_side.items():
        frames = sorted(by_frame)
        frame_set = set(frames)
        side_targets = [
            frame for frame in frames
            if all(frame + candidate in frame_set
                   for candidate in COARSE_CANDIDATES)
        ]
        side_target_set = set(side_targets)
        side_centers = [
            frame for frame in side_targets
            if frame + DIRECTION_HALF_WINDOW in side_target_set
        ]
        require(len(side_targets) >= 100 and len(side_centers) >= 100,
                f"Insufficient fixed coarse manifest for {side}")
        targets[side] = side_targets
        centers[side] = side_centers
    return targets, centers


def estimate_train_only_coarse(views_by_side, observed_thermal, coarse_centers):
    """Estimate the integer coarse offset without access to registered truth."""
    scan = {candidate: {"left": [], "right": []}
            for candidate in COARSE_CANDIDATES}
    centers_by_side = {}
    with torch.no_grad():
        for side in ("left", "right"):
            by_frame = views_by_side[side]
            frame_set = set(by_frame)
            centers = list(coarse_centers[side])
            require(all(frame in observed_thermal[side]
                        and frame + DIRECTION_HALF_WINDOW in observed_thermal[side]
                        for frame in centers),
                    f"Observed stream misses fixed coarse centers for {side}")
            require(all(
                frame + candidate in frame_set
                and frame + DIRECTION_HALF_WINDOW + candidate in frame_set
                for frame in centers for candidate in COARSE_CANDIDATES),
                f"RGB stream misses fixed coarse candidate source for {side}")
            require(len(centers) >= 60,
                    f"Insufficient coarse-common centers for {side}: {len(centers)}")
            centers_by_side[side] = centers
            rgb_gray = {
                frame: temporal_change_gray(view.original_image)
                for frame, view in by_frame.items()
            }
            thermal_gray = {
                frame: temporal_change_gray(observed_thermal[side][frame])
                for frame in centers
            }
            thermal_gray.update({
                frame + DIRECTION_HALF_WINDOW:
                    temporal_change_gray(observed_thermal[side][frame + DIRECTION_HALF_WINDOW])
                for frame in centers
            })
            for frame in centers:
                thermal_change = standardized_temporal_change(
                    thermal_gray[frame],
                    thermal_gray[frame + DIRECTION_HALF_WINDOW],
                )
                if thermal_change is None:
                    continue
                rgb_changes = {
                    candidate: standardized_temporal_change(
                        rgb_gray[frame + candidate],
                        rgb_gray[frame + DIRECTION_HALF_WINDOW + candidate],
                    )
                    for candidate in COARSE_CANDIDATES
                }
                if any(change is None for change in rgb_changes.values()):
                    continue
                for candidate, rgb_change in rgb_changes.items():
                    denominator = (
                        torch.linalg.vector_norm(thermal_change)
                        * torch.linalg.vector_norm(rgb_change)
                    )
                    require(bool(torch.isfinite(denominator))
                            and float(denominator.item()) != 0.0,
                            "Invalid train temporal-change norm")
                    score = (thermal_change * rgb_change).sum() / denominator
                    require(bool(torch.isfinite(score)),
                            "Non-finite train temporal-change score")
                    scan[candidate][side].append(float(score.item()))

    for side in ("left", "right"):
        candidate_counts = {
            candidate: len(scan[candidate][side])
            for candidate in COARSE_CANDIDATES
        }
        require(len(set(candidate_counts.values())) == 1,
                f"Coarse candidates use unequal {side} counts: {candidate_counts}")

    summary = {}
    for candidate in COARSE_CANDIDATES:
        left = scan[candidate]["left"]
        right = scan[candidate]["right"]
        require(left and right, f"Empty coarse score at candidate {candidate}")
        combined = left + right
        summary[str(candidate)] = {
            "all": float(np.mean(combined)),
            "left": float(np.mean(left)),
            "right": float(np.mean(right)),
            "count_all": len(combined),
            "count_left": len(left),
            "count_right": len(right),
        }
    maxima = {}
    winner_margins = {}
    for key in ("all", "left", "right"):
        ranking = sorted(
            COARSE_CANDIDATES,
            key=lambda value: summary[str(value)][key], reverse=True)
        best, runner_up = ranking[0], ranking[1]
        margin = summary[str(best)][key] - summary[str(runner_up)][key]
        require(margin > 0.0,
                f"Tied train temporal-change coarse maximum for {key}")
        maxima[key] = best
        winner_margins[key] = margin
    require(len(set(maxima.values())) == 1,
            f"Train temporal-change coarse estimate disagrees by side: {maxima}")
    estimate = int(maxima["all"])
    require(estimate not in (COARSE_CANDIDATES[0], COARSE_CANDIDATES[-1]),
            f"Train temporal-change optimum reached search boundary: {estimate}")
    return estimate, {
        "schema": "nctc_train_change_coarse_v1",
        "uses_train_cameras_only": True,
        "registered_truth_visible_to_estimator": False,
        "candidates": list(COARSE_CANDIDATES),
        "score": "mean cosine similarity of standardized absolute temporal-change maps",
        "resolution": [160, 120],
        "centers_by_side": centers_by_side,
        "maxima": maxima,
        "winner_margins": winner_margins,
        "scan": summary,
    }


def build_common_support(views_by_side, coarse_offset, allowed_targets):
    common = {}
    source_specs = {}
    for side, by_frame in views_by_side.items():
        frames = sorted(by_frame)
        frame_set = set(frames)
        target_set = set(allowed_targets[side])
        pose_min, pose_max = frames[0], frames[-1]
        valid = []
        for base in allowed_targets[side]:
            target_frames = (
                base - DIRECTION_HALF_WINDOW,
                base,
                base + DIRECTION_HALF_WINDOW,
            )
            if any(target not in target_set for target in target_frames):
                continue
            query_min = min(target_frames) + coarse_offset - RESIDUAL_BOUND
            query_max = max(target_frames) + coarse_offset + RESIDUAL_BOUND
            if query_min < pose_min or query_max > pose_max:
                continue
            candidate_specs = []
            try:
                for shift in (
                        coarse_offset - RESIDUAL_BOUND,
                        coarse_offset + RESIDUAL_BOUND):
                    for target in target_frames:
                        candidate_specs.append(proxy_spec(side, target, shift, frames))
            except RuntimeError:
                continue
            valid.append(base)
            for spec in candidate_specs:
                key = f"{side}:{spec['target_frame']}:{spec['shift']:.3f}"
                source_specs[key] = spec
        require(len(valid) >= 60, f"Insufficient common support for {side}: {len(valid)}")
        support_end = (4 * len(valid)) // 10
        first_buffer_end = (5 * len(valid)) // 10
        query_end = (7 * len(valid)) // 10
        second_buffer_end = (8 * len(valid)) // 10
        support = valid[:support_end]
        first_buffer = valid[support_end:first_buffer_end]
        query = valid[first_buffer_end:query_end]
        second_buffer = valid[query_end:second_buffer_end]
        probe = valid[second_buffer_end:]
        require(len(support) >= 20 and len(query) >= 20 and len(probe) >= 20,
                f"Insufficient split size for {side}")
        support_query_separation = query[0] - support[-1]
        query_probe_separation = probe[0] - query[-1]
        required_buffer = math.ceil(
            max(abs(value) for value in COARSE_CANDIDATES)
            + RESIDUAL_BOUND + DIRECTION_HALF_WINDOW)
        require(support_query_separation >= required_buffer,
                f"Support/query buffer too small for {side}: "
                f"{support_query_separation} < {required_buffer}")
        require(query_probe_separation >= required_buffer,
                f"Query/probe buffer too small for {side}: "
                f"{query_probe_separation} < {required_buffer}")
        common[side] = {
            "valid": valid,
            "support": support,
            "query": query,
            "probe": probe,
            "first_buffer_frames": first_buffer,
            "second_buffer_frames": second_buffer,
            "support_query_separation": support_query_separation,
            "query_probe_separation": query_probe_separation,
        }
    return common, source_specs


def audit_role_isolation(common, generation_specs, views_by_side,
                         coarse_offset, coarse_selection_centers):
    roles = ("support", "query", "probe")
    audit = {}
    role_sets = {}
    for role in roles:
        target_keys = set()
        observed_source_keys = set()
        trajectory_source_keys = set()
        for side in ("left", "right"):
            frames = sorted(views_by_side[side])
            for base in common[side][role]:
                for target in (
                        base - DIRECTION_HALF_WINDOW,
                        base,
                        base + DIRECTION_HALF_WINDOW):
                    target_keys.add((side, target))
                    observed_spec = generation_specs[f"{side}:{target}"]
                    observed_source_keys.add((side, observed_spec["lower"]))
                    observed_source_keys.add((side, observed_spec["upper"]))
                    for shift in (
                            coarse_offset - RESIDUAL_BOUND,
                            coarse_offset + RESIDUAL_BOUND):
                        trajectory_spec = proxy_spec(side, target, shift, frames)
                        trajectory_source_keys.add((side, trajectory_spec["lower"]))
                        trajectory_source_keys.add((side, trajectory_spec["upper"]))
        role_sets[role] = {
            "target": target_keys,
            "observed_source": observed_source_keys,
            "trajectory_source": trajectory_source_keys,
        }
        audit[role] = {
            name: [f"{side}:{frame}" for side, frame in sorted(values)]
            for name, values in role_sets[role].items()
        }
    for index, first in enumerate(roles):
        for second in roles[index + 1:]:
            for name in ("target", "observed_source", "trajectory_source"):
                require(role_sets[first][name].isdisjoint(role_sets[second][name]),
                        f"Role leakage in {name}: {first} vs {second}")
    coarse_observed_keys = {
        (side, frame)
        for side, centers in coarse_selection_centers.items()
        for center in centers
        for frame in (center, center + DIRECTION_HALF_WINDOW)
    }
    require(coarse_observed_keys.isdisjoint(role_sets["probe"]["target"]),
            "Sealed probe participated in coarse selection")
    audit["coarse_selection_observed_targets"] = [
        f"{side}:{frame}" for side, frame in sorted(coarse_observed_keys)]
    audit["pairwise_disjoint"] = True
    audit["probe_excluded_from_coarse_selection"] = True
    return audit


def render_clock_thermal(view, base_frame, raw_offset, gaussians, pipe, hyper,
                         background_thermal, teacher_iteration,
                         appearance="thermal", pose_raw_offset=None,
                         deformation_raw_offset=None):
    require(appearance in ("rgb", "thermal"), "Invalid render appearance")
    pose_raw = raw_offset if pose_raw_offset is None else pose_raw_offset
    deformation_raw = (
        raw_offset if deformation_raw_offset is None
        else deformation_raw_offset)
    original_frame_no = view.frame_no
    original_raw = view.nctc_temporal_offset_raw_override
    view.frame_no = float(base_frame) + view.nctc_coarse_offset
    object.__setattr__(
        view, "nctc_temporal_offset_raw_override", pose_raw)
    try:
        thermal_viewmatrix = view.get_thermal_world_view_transform()
        view.refresh_thermal_projection()
        thermal_intrinsic = view.thermal_projection_matrix_learnable
        thermal_full_proj = thermal_viewmatrix @ thermal_intrinsic.detach()
        thermal_full_proj_intrinsic = thermal_viewmatrix.detach() @ thermal_intrinsic
        thermal_campos = thermal_viewmatrix.inverse()[3, :3]
        effective_tfovx, effective_tfovy = view.get_thermal_fovs()

        raster_settings = GaussianRasterizationSettings(
            image_height=int(getattr(view, "thermal_height", view.image_height)),
            image_width=int(getattr(view, "thermal_width", view.image_width)),
            tanfovx=torch.tan(effective_tfovx * 0.5),
            tanfovy=torch.tan(effective_tfovy * 0.5),
            bg=background_thermal,
            scale_modifier=1.0,
            viewmatrix=thermal_viewmatrix,
            projmatrix=thermal_full_proj,
            projmatrix_intrinsic=thermal_full_proj_intrinsic,
            intrinsic=thermal_intrinsic,
            sh_degree=gaussians.active_sh_degree,
            campos=thermal_campos,
            prefiltered=False,
            debug=pipe.debug,
            debug_iter=teacher_iteration,
        )

        means3d = gaussians.get_xyz
        object.__setattr__(
            view, "nctc_temporal_offset_raw_override", deformation_raw)
        time_value = view.get_temporal_time().to(
            device=means3d.device, dtype=means3d.dtype)
        time_tensor = time_value.reshape(1, 1).repeat(means3d.shape[0], 1)
        require(not pipe.compute_cov3D_python,
                "NC-TC v1 requires the locked rasterizer covariance path")
        routing = torch.zeros((means3d.shape[0], 3), device=means3d.device)
        routing[:, 0] = 1.0
        deformed = gaussians._deformation(
            means3d,
            gaussians._scaling,
            gaussians._rotation,
            gaussians._opacity,
            gaussians._thermal_opacity,
            time_tensor,
            None,
            gaussians,
            None,
            None,
            gaussians.get_features,
            gaussians.get_thermal_features,
            iter=teacher_iteration,
            num_down_emb_c=hyper.min_embeddings,
            num_down_emb_f=hyper.min_embeddings,
            modality_routing=routing,
            thermal_only=False,
            rgb_only_teacher=True,
        )
        means, scales, rotations = deformed[0], deformed[1], deformed[2]
        scales = gaussians.scaling_activation(scales)
        rotations = gaussians.rotation_activation(rotations)
        render_opacity = gaussians.opacity_activation(
            gaussians._thermal_opacity if appearance == "thermal"
            else gaussians._opacity)
        require(bool(torch.isfinite(means).all()), "Non-finite teacher geometry")
        require(bool(torch.isfinite(scales).all()), "Non-finite teacher scale")
        require(bool(torch.isfinite(rotations).all()), "Non-finite teacher rotation")
        require(bool(torch.isfinite(render_opacity).all()), "Non-finite render opacity")

        screenspace = torch.zeros_like(
            means, dtype=means.dtype, requires_grad=True, device="cuda")
        screenspace_densify = torch.zeros_like(
            means, dtype=means.dtype, requires_grad=True, device="cuda")
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered, radii, _, _, _ = rasterizer(
            means3D=means,
            means2D=screenspace,
            means2D_densify=screenspace_densify,
            shift_factors=torch.zeros(3, device="cuda"),
            shs=(gaussians.get_thermal_features if appearance == "thermal"
                 else gaussians.get_features),
            colors_precomp=None,
            opacities=render_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        require(int((radii > 0).sum().item()) > 0, "No visible NC-TC Gaussians")
        require(bool(torch.isfinite(rendered).all()), "Non-finite NC-TC Thermal render")
        return rendered
    finally:
        view.frame_no = original_frame_no
        object.__setattr__(
            view, "nctc_temporal_offset_raw_override", original_raw)


def dynamic_mask_hash(masks):
    digest = hashlib.sha256()
    for (side, frame), mask in sorted(masks.items()):
        digest.update(f"{side}:{frame}".encode("ascii"))
        digest.update(mask.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_frozen_dynamic_masks(items, views_by_side, gaussians, pipe, hyper,
                               background_thermal, teacher_iteration):
    masks = {}
    raw_zero = torch.zeros((), device="cuda")
    with torch.no_grad():
        for side, base_frame in items:
            minus_frame = base_frame - DIRECTION_HALF_WINDOW
            plus_frame = base_frame + DIRECTION_HALF_WINDOW
            projected_minus = render_clock_thermal(
                views_by_side[side][minus_frame], minus_frame, raw_zero,
                gaussians, pipe, hyper, background_thermal, teacher_iteration,
                appearance="rgb")
            projected_plus = render_clock_thermal(
                views_by_side[side][plus_frame], plus_frame, raw_zero,
                gaussians, pipe, hyper, background_thermal, teacher_iteration,
                appearance="rgb")
            motion = torch.abs(projected_plus - projected_minus).mean(
                dim=0, keepdim=True)
            threshold = torch.quantile(motion.reshape(-1), 0.70)
            mask = (motion >= threshold).detach()
            require(int(mask.sum().item()) > 0,
                    f"Empty frozen dynamic mask: {side} {base_frame}")
            masks[(side, base_frame)] = mask
    return masks, dynamic_mask_hash(masks)


def clock_objective(side, base_frame, raw_offset, views_by_side,
                    observed_thermal, frozen_masks, gaussians, pipe, hyper,
                    background_thermal, teacher_iteration, lambda_dssim,
                    lambda_direction, pose_raw_offset=None,
                    deformation_raw_offset=None):
    by_frame = views_by_side[side]
    center_view = by_frame[base_frame]
    minus_frame = base_frame - DIRECTION_HALF_WINDOW
    plus_frame = base_frame + DIRECTION_HALF_WINDOW
    pred_center = render_clock_thermal(
        center_view, base_frame, raw_offset, gaussians, pipe, hyper,
        background_thermal, teacher_iteration,
        pose_raw_offset=pose_raw_offset,
        deformation_raw_offset=deformation_raw_offset)
    pred_minus = render_clock_thermal(
        by_frame[minus_frame], minus_frame, raw_offset, gaussians, pipe, hyper,
        background_thermal, teacher_iteration,
        pose_raw_offset=pose_raw_offset,
        deformation_raw_offset=deformation_raw_offset)
    pred_plus = render_clock_thermal(
        by_frame[plus_frame], plus_frame, raw_offset, gaussians, pipe, hyper,
        background_thermal, teacher_iteration,
        pose_raw_offset=pose_raw_offset,
        deformation_raw_offset=deformation_raw_offset)
    target_center = target_on_render_device(
        observed_thermal[side][base_frame], pred_center)
    target_minus = target_on_render_device(
        observed_thermal[side][minus_frame], pred_minus)
    target_plus = target_on_render_device(
        observed_thermal[side][plus_frame], pred_plus)
    frame_loss, frame_l1, frame_dssim = image_objective(
        pred_center, target_center, lambda_dssim)
    predicted_direction = pred_plus - pred_minus
    target_direction = target_plus - target_minus
    mask = frozen_masks[(side, base_frame)]
    mask_count = int(mask.sum().item())
    require(mask_count > 0, "Empty dynamic mask")
    direction_error = torch.abs(predicted_direction - target_direction)
    direction_loss = (direction_error * mask).sum() / (mask.sum() * direction_error.shape[0])
    total = frame_loss + lambda_direction * direction_loss
    require(bool(torch.isfinite(total)), "Non-finite clock objective")
    return total, {
        "frame_l1": float(frame_l1.detach().item()),
        "frame_dssim": float(frame_dssim.detach().item()),
        "direction_l1": float(direction_loss.detach().item()),
        "dynamic_pixels": mask_count,
        "render_min": float(pred_center.detach().min().item()),
        "render_max": float(pred_center.detach().max().item()),
    }


def deterministic_schedule(items, steps, seed):
    require(items, "Cannot schedule an empty split")
    rng = random.Random(seed)
    result = []
    while len(result) < steps:
        block = list(items)
        rng.shuffle(block)
        result.extend(block)
    return result[:steps]


def build_profile(common, views_by_side, observed_thermal, frozen_masks,
                  gaussians, pipe, hyper, background_thermal, args):
    profile = {}
    for side in ("left", "right"):
        probe = common[side]["probe"]
        midpoint = len(probe) // 2
        groups = {
            f"{side}_early": probe[:midpoint],
            f"{side}_late": probe[midpoint:],
        }
        for group_name, group_frames in groups.items():
            selected_views = np.linspace(
                0, len(group_frames) - 1,
                args.profile_views_per_block, dtype=int)
            selected_frames = [group_frames[int(index)] for index in selected_views]
            losses = {}
            with torch.no_grad():
                for residual in PROFILE_RESIDUALS:
                    normalized = residual / RESIDUAL_BOUND
                    require(abs(normalized) < 1.0,
                            "Profile residual outside open bound")
                    candidate_raw = torch.tensor(
                        math.atanh(normalized), device="cuda")
                    values = []
                    for frame in selected_frames:
                        value, _ = clock_objective(
                            side, frame, candidate_raw,
                            views_by_side, observed_thermal, frozen_masks,
                            gaussians, pipe, hyper,
                            background_thermal, args.iteration,
                            args.nctc_lambda_dssim, args.lambda_direction)
                        values.append(float(value.item()))
                    losses[f"{residual:+.3f}"] = float(np.mean(values))
            estimate_key = min(losses, key=losses.get)
            estimate = float(estimate_key)
            estimate_index = PROFILE_RESIDUALS.index(estimate)
            curvature = None
            if 0 < estimate_index < len(PROFILE_RESIDUALS) - 1:
                left = losses[f"{PROFILE_RESIDUALS[estimate_index - 1]:+.3f}"]
                center = losses[f"{estimate:+.3f}"]
                right = losses[f"{PROFILE_RESIDUALS[estimate_index + 1]:+.3f}"]
                curvature = left + right - 2.0 * center
            profile[group_name] = {
                "frames": selected_frames,
                "estimate": estimate,
                "curvature": curvature,
                "losses": losses,
            }
            print("PROFILE " + json.dumps(
                {"group": group_name, "estimate": estimate,
                 "curvature": curvature}, sort_keys=True), flush=True)
    return profile


def gradient_contract_for_sample(side, frame, mode, raw_value, raw_step,
                                 views_by_side, observed_thermal, frozen_masks,
                                 gaussians, pipe, hyper, background_thermal,
                                 args):
    mode_activity = {
        "full": (True, True),
        "pose_only": (True, False),
        "deformation_only": (False, True),
    }
    pose_active, deformation_active = mode_activity[mode]
    central_raw = torch.tensor(raw_value, device="cuda")
    analytic_raw = torch.tensor(raw_value, device="cuda", requires_grad=True)
    analytic_loss, _ = clock_objective(
        side, frame, analytic_raw, views_by_side, observed_thermal,
        frozen_masks, gaussians, pipe, hyper, background_thermal,
        args.iteration, args.nctc_lambda_dssim, args.lambda_direction,
        pose_raw_offset=(analytic_raw if pose_active else central_raw),
        deformation_raw_offset=(
            analytic_raw if deformation_active else central_raw))
    (analytic,) = torch.autograd.grad(analytic_loss, (analytic_raw,))
    plus_raw = torch.tensor(raw_value + raw_step, device="cuda")
    minus_raw = torch.tensor(raw_value - raw_step, device="cuda")
    plus_loss, _ = clock_objective(
        side, frame, plus_raw, views_by_side, observed_thermal,
        frozen_masks, gaussians, pipe, hyper, background_thermal,
        args.iteration, args.nctc_lambda_dssim, args.lambda_direction,
        pose_raw_offset=(plus_raw if pose_active else central_raw),
        deformation_raw_offset=(
            plus_raw if deformation_active else central_raw))
    minus_loss, _ = clock_objective(
        side, frame, minus_raw, views_by_side, observed_thermal,
        frozen_masks, gaussians, pipe, hyper, background_thermal,
        args.iteration, args.nctc_lambda_dssim, args.lambda_direction,
        pose_raw_offset=(minus_raw if pose_active else central_raw),
        deformation_raw_offset=(
            minus_raw if deformation_active else central_raw))
    finite_difference = (
        plus_loss.detach() - minus_loss.detach()) / (2.0 * raw_step)
    analytic_value = float(analytic.detach().item())
    finite_value = float(finite_difference.item())
    require(math.isfinite(analytic_value) and math.isfinite(finite_value),
            f"Non-finite {mode} gradient contract: {side} {frame}")
    denominator = max(abs(analytic_value), abs(finite_value))
    if denominator == 0.0:
        relative_error = 0.0
        sign_match = True
    else:
        relative_error = abs(analytic_value - finite_value) / denominator
        sign_match = math.copysign(1.0, analytic_value) == math.copysign(
            1.0, finite_value)
    return {
        "side": side,
        "frame": frame,
        "mode": mode,
        "analytic": analytic_value,
        "finite_difference": finite_value,
        "relative_error": relative_error,
        "sign_match": sign_match,
    }


def evaluate_gradient_contracts(common, views_by_side, observed_thermal,
                                frozen_masks, gaussians, pipe, hyper,
                                background_thermal, args):
    probe_residual = 0.25
    raw_value = math.atanh(probe_residual / RESIDUAL_BOUND)
    raw_step = 0.01
    contracts = []
    pose_track_motion = {}
    for side in ("left", "right"):
        query = common[side]["query"]
        sample_frames = [query[0], query[len(query) // 2]]
        track = next(iter(views_by_side[side].values())).temporal_pose_track
        pose_track_motion[side] = float(
            torch.abs(track[1:] - track[:-1]).max().detach().item())
        for frame in sample_frames:
            for mode in ("full", "pose_only", "deformation_only"):
                record = gradient_contract_for_sample(
                    side, frame, mode, raw_value, raw_step,
                    views_by_side, observed_thermal, frozen_masks, gaussians,
                    pipe, hyper, background_thermal, args)
                contracts.append(record)
                print("GRADIENT_SAMPLE " + json.dumps(
                    record, sort_keys=True), file=sys.stderr, flush=True)
                if mode in ("full", "deformation_only"):
                    require(record["analytic"] != 0.0
                            and record["finite_difference"] != 0.0,
                            f"Zero {mode} gradient: {side} {frame}")
                    require(record["sign_match"],
                            f"{mode} gradient sign mismatch: {side} {frame}")
                    require(record["relative_error"] <= 0.25,
                            f"{mode} gradient relative error too large: "
                            f"{side} {frame} {record['relative_error']}")
                elif pose_track_motion[side] == 0.0:
                    require(record["analytic"] == 0.0
                            and record["finite_difference"] == 0.0,
                            f"Static pose track produced a clock gradient: {side} {frame}")
                else:
                    require(record["analytic"] != 0.0
                            and record["finite_difference"] != 0.0,
                            f"Dynamic pose path has zero gradient: {side} {frame}")
                    require(record["sign_match"]
                            and record["relative_error"] <= 0.25,
                            f"Pose gradient contract failed: {side} {frame}")
    return {
        "probe_residual_frames": probe_residual,
        "probe_raw": raw_value,
        "raw_step": raw_step,
        "pose_track_motion": pose_track_motion,
        "records": contracts,
    }


def main():
    require(sys.flags.optimize == 0, f"Optimized Python forbidden: {sys.flags.optimize}")
    parser = argparse.ArgumentParser()
    model_group = ModelParams(parser, sentinel=True)
    opt_group = OptimizationParams(parser)
    pipe_group = PipelineParams(parser)
    hyper_group = ModelHiddenParams(parser)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--iteration", type=int, default=3000)
    parser.add_argument("--arm", choices=sorted(ARM_SHIFTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--support_steps", type=int, default=500)
    parser.add_argument("--clock_steps", type=int, default=500)
    parser.add_argument("--support_lr", type=float, default=0.01)
    parser.add_argument("--clock_lr", type=float, default=0.05)
    parser.add_argument("--nctc_lambda_dssim", type=float, default=0.2)
    parser.add_argument("--lambda_direction", type=float, default=0.5)
    parser.add_argument("--profile_views_per_block", type=int, default=2)
    parser.add_argument("--mechanical_smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=6666)
    args = get_combined_args(parser)
    require(not args.output.exists(), f"Refusing existing output: {args.output}")
    require(not args.report.exists(), f"Refusing existing report: {args.report}")
    require(args.support_steps >= 1 and args.clock_steps >= 1, "Invalid step count")
    require(args.support_lr > 0.0 and args.clock_lr > 0.0, "Learning rates must be positive")
    require(0.0 <= args.nctc_lambda_dssim <= 1.0, "Invalid DSSIM weight")
    require(args.lambda_direction >= 0.0, "Invalid direction weight")
    require(args.profile_views_per_block >= 1, "Invalid profile view count")
    args = merge_hparams(args, mmengine.Config.fromfile(args.configs))

    # Load Thermal training images, but keep the audited teacher weights frozen.
    args.rgb_only_teacher = False
    args.no_thermal_pose_opt = True
    args.temporal_alignment_enabled = False
    args.change_thermal_geo = False
    args.shuffle = False
    safe_state(True, seed=args.seed)

    dataset = model_group.extract(args)
    hyper = hyper_group.extract(args)
    pipe = pipe_group.extract(args)
    opt = opt_group.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        duration=hyper.total_num_frames,
        loader=dataset.loader,
        opt=opt,
        load_test_cameras=False,
    )
    require(scene.loaded_iter == args.iteration, "Teacher iteration mismatch")
    require(len(scene.getTestCameras()) == 0, "Formal test cameras were constructed")
    require(getattr(gaussians, "optimizer", None) is None,
            "Gaussian optimizer must not exist in NC-TC gate")
    scene_optimizers = {
        name: value for name, value in vars(scene).items()
        if name.startswith("optimizer")
    }
    require(all(value is None for value in scene_optimizers.values()),
            f"Scene optimizer exists in NC-TC gate: {list(scene_optimizers)}")
    rasterizer_path = Path(diff_gaussian_rasterization._C.__file__).resolve()
    rasterizer_sha256 = sha256_file(rasterizer_path)
    require(rasterizer_sha256 == EXPECTED_RASTERIZER_SHA256,
            f"Rasterizer runtime hash mismatch: {rasterizer_path} {rasterizer_sha256}")
    rasterizer_wrapper_path = Path(diff_gaussian_rasterization.__file__).resolve()
    rasterizer_wrapper_sha256 = sha256_file(rasterizer_wrapper_path)
    require(rasterizer_wrapper_sha256 == EXPECTED_RASTERIZER_WRAPPER_SHA256,
            "Rasterizer Python wrapper hash mismatch")
    deformation_path = Path(__file__).resolve().parent / "scene" / "deformation.py"
    deformation_sha256 = sha256_file(deformation_path)
    require(deformation_sha256 == EXPECTED_DEFORMATION_SHA256,
            "NC-TC deformation source hash mismatch")
    teacher_dir = Path(args.model_path) / "point_cloud" / f"iteration_{args.iteration}"
    teacher_hashes = {
        name: sha256_file(teacher_dir / name) for name in EXPECTED_TEACHER
    }
    require(teacher_hashes == EXPECTED_TEACHER,
            f"Teacher artifact mismatch: {teacher_hashes}")

    views = scene.getTrainCameras()
    require(len(views) == 266, f"Unexpected train-camera count: {len(views)}")
    views_by_side = {"left": {}, "right": {}}
    for view in views:
        side, frame = side_frame(view)
        require(frame not in views_by_side[side], f"Duplicate train frame: {side} {frame}")
        require(hasattr(view, "thermal_image") and view.thermal_image is not None,
                f"Thermal image was not loaded: {view.image_name}")
        views_by_side[side][frame] = view

    raw_offset = torch.nn.Parameter(torch.zeros((), device="cuda"), requires_grad=False)
    arm_shift = ARM_SHIFTS[args.arm]
    proxy_cache = {}
    observed_thermal = {"left": {}, "right": {}}
    coarse_targets, coarse_centers = build_coarse_target_manifest(views_by_side)
    coarse_selection_centers = {
        side: centers[:(3 * len(centers)) // 5]
        for side, centers in coarse_centers.items()
    }
    generation_specs = {}
    for side in ("left", "right"):
        frames = sorted(views_by_side[side])
        for frame in coarse_targets[side]:
            spec = proxy_spec(side, frame, arm_shift, frames)
            generation_specs[f"{side}:{frame}"] = spec
            observed_thermal[side][frame] = proxy_image(
                spec, views_by_side, proxy_cache)
    coarse_offset, coarse_report = estimate_train_only_coarse(
        views_by_side, observed_thermal, coarse_selection_centers)
    configure_temporal_views(
        views_by_side, raw_offset, hyper.total_num_frames, coarse_offset)
    common, source_specs = build_common_support(
        views_by_side, coarse_offset, coarse_targets)
    role_isolation_audit = audit_role_isolation(
        common, generation_specs, views_by_side, coarse_offset,
        coarse_selection_centers)
    print("COARSE_ESTIMATE " + json.dumps(coarse_report, sort_keys=True), flush=True)

    with torch.no_grad():
        gaussians._thermal_dc.copy_(gaussians._features_dc)
        gaussians._thermal_rest.zero_()
        gaussians._thermal_opacity.copy_(gaussians._opacity)
    set_all_nuisance_requires_grad(gaussians, views, enabled_name="gaussian._thermal_dc")
    require(gaussians._thermal_dc.requires_grad, "Thermal DC did not enter support phase")
    background_thermal = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    support_items = [
        (side, frame)
        for side in ("left", "right")
        for frame in common[side]["support"]
    ]
    query_items = [
        (side, frame)
        for side in ("left", "right")
        for frame in common[side]["query"]
    ]
    probe_items = [
        (side, frame)
        for side in ("left", "right")
        for frame in common[side]["probe"]
    ]
    support_schedule = deterministic_schedule(
        support_items, args.support_steps, args.seed + 101)
    query_schedule = deterministic_schedule(
        query_items, args.clock_steps, args.seed + 202)

    started = time.time()
    support_records = []
    detached_zero = raw_offset.detach()
    for step, (side, frame) in enumerate(support_schedule, start=1):
        prediction = render_clock_thermal(
            views_by_side[side][frame], frame, detached_zero,
            gaussians, pipe, hyper, background_thermal, args.iteration)
        target = target_on_render_device(
            observed_thermal[side][frame], prediction)
        loss, l1, dssim = image_objective(
            prediction, target, args.nctc_lambda_dssim)
        (gradient,) = torch.autograd.grad(
            loss, (gaussians._thermal_dc,), retain_graph=False,
            create_graph=False, allow_unused=False)
        require(bool(torch.isfinite(gradient).all()), "Non-finite support gradient")
        with torch.no_grad():
            gaussians._thermal_dc.add_(gradient, alpha=-args.support_lr)
        gaussians._thermal_dc.grad = None
        if step == 1 or step == args.support_steps or step % 25 == 0:
            record = {
                "step": step,
                "side": side,
                "frame": frame,
                "loss": float(loss.detach().item()),
                "l1": float(l1.detach().item()),
                "dssim": float(dssim.detach().item()),
                "gradient_absmax": float(gradient.detach().abs().max().item()),
                "render_min": float(prediction.detach().min().item()),
                "render_max": float(prediction.detach().max().item()),
            }
            support_records.append(record)
            print("SUPPORT " + json.dumps(record, sort_keys=True), flush=True)

    set_all_nuisance_requires_grad(gaussians, views, enabled_name=None)
    raw_offset.requires_grad_(True)
    frozen_masks, frozen_mask_hash = build_frozen_dynamic_masks(
        query_items + probe_items, views_by_side, gaussians, pipe, hyper,
        background_thermal, args.iteration)
    for view in views:
        view.refresh_thermal_projection()
    camera_manifest_before, camera_manifest_hash_before = camera_scalar_manifest(views)
    nuisance_before, nuisance_count = stable_tensor_hash(
        full_forward_state_tensors(gaussians, views))
    require(all(tensor.grad is None for _, tensor in gaussian_nuisance_tensors(gaussians, views)),
            "Stale nuisance gradient before clock phase")

    gradient_contract = evaluate_gradient_contracts(
        common, views_by_side, observed_thermal, frozen_masks, gaussians,
        pipe, hyper, background_thermal, args)
    print("GRADIENT_CONTRACT " + json.dumps(gradient_contract, sort_keys=True), flush=True)

    clock_records = []
    for step, (side, frame) in enumerate(query_schedule, start=1):
        loss, diagnostics = clock_objective(
            side, frame, raw_offset, views_by_side, observed_thermal,
            frozen_masks, gaussians, pipe, hyper, background_thermal, args.iteration,
            args.nctc_lambda_dssim, args.lambda_direction)
        (gradient,) = torch.autograd.grad(
            loss, (raw_offset,), retain_graph=False,
            create_graph=False, allow_unused=False)
        require(bool(torch.isfinite(gradient)), "Non-finite clock gradient")
        require(all(tensor.grad is None for _, tensor in gaussian_nuisance_tensors(gaussians, views)),
                "Clock step populated a nuisance gradient")
        with torch.no_grad():
            raw_offset.add_(gradient, alpha=-args.clock_lr)
        residual = RESIDUAL_BOUND * torch.tanh(raw_offset.detach())
        require(bool(torch.isfinite(residual)), "Non-finite learned residual")
        if step == 1 or step == args.clock_steps or step % 25 == 0:
            record = {
                "step": step,
                "side": side,
                "frame": frame,
                "loss": float(loss.detach().item()),
                "gradient": float(gradient.detach().item()),
                "raw": float(raw_offset.detach().item()),
                "residual_frames": float(residual.item()),
                **diagnostics,
            }
            clock_records.append(record)
            print("CLOCK " + json.dumps(record, sort_keys=True), flush=True)

    nuisance_after, nuisance_after_count = stable_tensor_hash(
        full_forward_state_tensors(gaussians, views))
    require(nuisance_count == nuisance_after_count, "Nuisance tensor count changed")
    require(nuisance_before == nuisance_after, "Nuisance state changed during clock phase")
    frozen_mask_hash_after = dynamic_mask_hash(frozen_masks)
    require(frozen_mask_hash == frozen_mask_hash_after,
            "Frozen dynamic masks changed during clock phase")

    learned_residual = float((RESIDUAL_BOUND * torch.tanh(raw_offset.detach())).item())
    expected_residual = arm_shift - coarse_offset
    profile = {} if args.mechanical_smoke else build_profile(
        common, views_by_side, observed_thermal, frozen_masks,
        gaussians, pipe, hyper, background_thermal, args)
    nuisance_final, nuisance_final_count = stable_tensor_hash(
        full_forward_state_tensors(gaussians, views))
    require(nuisance_final_count == nuisance_count,
            "Nuisance tensor count changed during profile")
    require(nuisance_final == nuisance_before,
            "Nuisance state changed during profile")
    camera_manifest_final, camera_manifest_hash_final = camera_scalar_manifest(views)
    require(camera_manifest_hash_final == camera_manifest_hash_before,
            "Camera scalar manifest changed during clock/profile")
    require(camera_manifest_final == camera_manifest_before,
            "Camera scalar values changed during clock/profile")
    profile_estimates = [entry["estimate"] for entry in profile.values()]
    profile_curvatures = [entry["curvature"] for entry in profile.values()]
    tolerance = 0.125 if args.arm == "Z" else 0.25
    learned_error = abs(learned_residual - expected_residual)
    coarse_gate_pass = float(coarse_offset) == EXPECTED_COARSE_OFFSET
    if args.mechanical_smoke:
        arm_pass = None
        run_status = "MECHANICAL_PASS" if coarse_gate_pass else "MECHANICAL_FAIL"
    else:
        arm_pass = (
            coarse_gate_pass
            and learned_error <= tolerance
            and abs(learned_residual) < 1.125
            and all(abs(value - expected_residual) <= 0.25 for value in profile_estimates)
            and all(value is not None and value > 0.0 for value in profile_curvatures)
            and max(profile_estimates) - min(profile_estimates) <= 0.25
        )
        run_status = "PASS" if arm_pass else "FAIL"

    source_hashes = {}
    for side, by_frame in views_by_side.items():
        used_frames = sorted({
            value
            for spec in source_specs.values()
            if spec["side"] == side
            for value in (spec["lower"], spec["upper"])
        })
        for frame in used_frames:
            tensor = by_frame[frame].thermal_image.detach().cpu().contiguous()
            source_hashes[f"{side}:{frame}"] = hashlib.sha256(
                tensor.numpy().tobytes()).hexdigest()
    observed_hashes = {}
    for side in ("left", "right"):
        for frame, tensor in sorted(observed_thermal[side].items()):
            observed_cpu = tensor.detach().cpu().contiguous()
            observed_hashes[f"{side}:{frame}"] = hashlib.sha256(
                observed_cpu.numpy().tobytes()).hexdigest()

    report = {
        "schema": SCHEMA,
        "arm": args.arm,
        "status": run_status,
        "scientific_gate_evaluated": not args.mechanical_smoke,
        "registered_shift": arm_shift,
        "coarse_offset": coarse_offset,
        "expected_coarse_offset": EXPECTED_COARSE_OFFSET,
        "coarse_gate_pass": coarse_gate_pass,
        "coarse_report": coarse_report,
        "expected_residual": expected_residual,
        "residual_bound": RESIDUAL_BOUND,
        "learned_raw": float(raw_offset.detach().item()),
        "learned_residual": learned_residual,
        "learned_effective_offset": coarse_offset + learned_residual,
        "learned_error": learned_error,
        "support_steps": args.support_steps,
        "clock_steps": args.clock_steps,
        "support_lr": args.support_lr,
        "clock_lr": args.clock_lr,
        "lambda_dssim": args.nctc_lambda_dssim,
        "lambda_direction": args.lambda_direction,
        "gradient_contract": gradient_contract,
        "frozen_dynamic_mask_sha256": frozen_mask_hash,
        "frozen_dynamic_mask_sha256_after_clock": frozen_mask_hash_after,
        "frozen_dynamic_mask_count": len(frozen_masks),
        "nuisance_hash_before_clock": nuisance_before,
        "nuisance_hash_after_clock": nuisance_after,
        "nuisance_hash_after_profile": nuisance_final,
        "nuisance_tensor_count": nuisance_count,
        "camera_scalar_manifest": camera_manifest_before,
        "camera_scalar_manifest_sha256": camera_manifest_hash_before,
        "camera_scalar_manifest_sha256_after_profile": camera_manifest_hash_final,
        "formal_test_cameras_constructed": False,
        "optimizers_absent": True,
        "train_camera_count": len(views),
        "coarse_targets": coarse_targets,
        "coarse_centers": coarse_centers,
        "coarse_selection_centers": coarse_selection_centers,
        "common_support": common,
        "role_isolation_audit": role_isolation_audit,
        "generation_specs_audit_only": generation_specs,
        "source_specs": source_specs,
        "source_tensor_hashes": source_hashes,
        "observed_tensor_hashes": observed_hashes,
        "profile": profile,
        "support_log": support_records,
        "clock_log": clock_records,
        "teacher_hashes": teacher_hashes,
        "rasterizer_runtime_path": str(rasterizer_path),
        "rasterizer_runtime_sha256": rasterizer_sha256,
        "rasterizer_wrapper_path": str(rasterizer_wrapper_path),
        "rasterizer_wrapper_sha256": rasterizer_wrapper_sha256,
        "deformation_path": str(deformation_path),
        "deformation_sha256": deformation_sha256,
        "script_sha256": sha256_file(__file__),
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "schema": SCHEMA,
        "arm": args.arm,
        "raw_offset": raw_offset.detach().cpu(),
        "residual_frames": learned_residual,
        "thermal_dc": gaussians._thermal_dc.detach().cpu(),
        "thermal_opacity": gaussians._thermal_opacity.detach().cpu(),
        "teacher_hashes": teacher_hashes,
        "nuisance_hash": nuisance_after,
    }, args.output / "nctc_state.pth")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    require(not args.report.exists(), f"Report appeared during run: {args.report}")
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    if args.mechanical_smoke and run_status == "MECHANICAL_PASS":
        (args.output / "MECHANICAL_SMOKE_PASS").write_text(
            "pass\n", encoding="utf-8")
    elif arm_pass:
        (args.output / "ARM_GATE_PASS").write_text("pass\n", encoding="utf-8")
    print("NCTC_GATE_RESULT " + json.dumps({
        "arm": args.arm,
        "status": report["status"],
        "learned_residual": learned_residual,
        "expected_residual": expected_residual,
        "learned_error": learned_error,
        "profile_estimates": profile_estimates,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
