#!/usr/bin/env python3
"""Covers-only renderer profile of the validated global change-map clock."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mmengine
import torch
import torch.nn.functional as F
import diff_gaussian_rasterization
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from time_alignment import feature_runtime as base


SCHEMA = "covers_rgb_temporal_fidelity_profile_v2"
OFFSET_BOUND = 20.0
ARM_SHIFTS = {"N": -8, "Z": 0, "P": 8}
PROFILE_OFFSETS = (-8, 0, 8)
HALF_WINDOW = 2
FEATURE_SIZE = (120, 160)
EXPECTED_TEACHER = {
    "point_cloud.ply": "cbc0fe6ff116325a6d4e80507d81e35d52e9b1553b5d26be716c62ae5e6077e2",
    "deformation.pth": "20346dd8e4b734142e5af0d37d481377e3ae83e25c16195e0c53400b2158fc7f",
    "thermal_camera_state.pth": "8c0fbf4efdecee170035806b16e58e7ea3981020d26705ccd517f684dfe4fe53",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def configure_views(views_by_side, raw_offset, duration):
    for by_frame in views_by_side.values():
        frames = sorted(by_frame)
        matrices = torch.stack([
            by_frame[frame].world_view_transform.detach().clone()
            for frame in frames
        ], dim=0)
        frame_tensor = torch.tensor(
            frames, device=matrices.device, dtype=matrices.dtype)
        for view in by_frame.values():
            view.temporal_alignment_enabled = True
            view.temporal_observation_correction_enabled = True
            object.__setattr__(
                view, "nctc_temporal_offset_raw_override", raw_offset)
            view.temporal_offset_max_frames = OFFSET_BOUND
            view.temporal_duration = float(duration)
            view.temporal_pose_frames = frame_tensor
            view.temporal_pose_track = matrices
            view.temporal_strict_common_support = True
            view.thermal_frame_shift = 1
            view.nctc_coarse_offset = 0.0


def raw_for_offset(offset, device):
    require(abs(offset) < OFFSET_BOUND, "Offset reached tanh boundary")
    return torch.tensor(
        math.atanh(float(offset) / OFFSET_BOUND),
        dtype=torch.float32, device=device)


def gray(image):
    require(image.ndim == 3 and bool(torch.isfinite(image).all()),
            "Invalid CHW image")
    if image.shape[0] == 3:
        value = (0.299 * image[0:1] + 0.587 * image[1:2]
                 + 0.114 * image[2:3])
    else:
        value = image.mean(dim=0, keepdim=True)
    return F.interpolate(
        value.unsqueeze(0), size=FEATURE_SIZE,
        mode="bilinear", align_corners=False)[0, 0]


def standardized_absolute_change(first, second):
    change = torch.abs(gray(second) - gray(first))
    border_y = max(1, change.shape[0] // 12)
    border_x = max(1, change.shape[1] // 12)
    cropped = change[border_y:-border_y, border_x:-border_x]
    std = cropped.std(unbiased=False)
    require(bool(torch.isfinite(std)) and float(std.detach().item()) > 0.0,
            "Degenerate temporal change map")
    result = (cropped - cropped.mean()) / std
    require(bool(torch.isfinite(result).all()),
            "Non-finite standardized temporal change")
    return result


def cosine(first, second):
    require(first.shape == second.shape, "Change-map shape mismatch")
    denominator = (torch.linalg.vector_norm(first)
                   * torch.linalg.vector_norm(second))
    require(bool(torch.isfinite(denominator))
            and float(denominator.detach().item()) > 0.0,
            "Degenerate change-map cosine")
    value = torch.sum(first * second) / denominator
    require(bool(torch.isfinite(value)), "Non-finite change-map cosine")
    return value


def build_centers(views_by_side):
    margin = max(abs(value) for value in ARM_SHIFTS.values())
    centers = {}
    for side, by_frame in views_by_side.items():
        frame_set = set(by_frame)
        values = [
            frame for frame in sorted(by_frame)
            if all(frame + shift in frame_set
                   and frame + HALF_WINDOW + shift in frame_set
                   for shift in ARM_SHIFTS.values())
        ]
        require(len(values) >= 100,
                f"Insufficient full-scene common centers: {side}")
        require(all(frame - margin in frame_set
                    and frame + HALF_WINDOW + margin in frame_set
                    for frame in values), "Common-support construction failed")
        centers[side] = values
    return centers


def render_clock_rgb(view, base_frame, raw_offset, gaussians, pipe, hyper,
                     background, iteration):
    original_frame_no = view.frame_no
    original_raw = view.nctc_temporal_offset_raw_override
    view.frame_no = float(base_frame)
    object.__setattr__(
        view, "nctc_temporal_offset_raw_override", raw_offset)
    try:
        viewmatrix = view.get_temporal_rgb_world_view_transform()
        intrinsic = view.projection_matrix
        full_projection = viewmatrix @ intrinsic
        campos = viewmatrix.inverse()[3, :3]
        raster_settings = GaussianRasterizationSettings(
            image_height=int(view.image_height),
            image_width=int(view.image_width),
            tanfovx=math.tan(float(view.FoVx) * 0.5),
            tanfovy=math.tan(float(view.FoVy) * 0.5),
            bg=background,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=full_projection,
            projmatrix_intrinsic=full_projection.detach(),
            intrinsic=intrinsic,
            sh_degree=gaussians.active_sh_degree,
            campos=campos,
            prefiltered=False,
            debug=pipe.debug,
            debug_iter=iteration,
        )

        means3d = gaussians.get_xyz
        time_value = view.get_temporal_time().to(
            device=means3d.device, dtype=means3d.dtype)
        time_tensor = time_value.reshape(1, 1).repeat(means3d.shape[0], 1)
        require(not pipe.compute_cov3D_python,
                "RGB temporal Gate requires the locked covariance path")
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
            iter=iteration,
            num_down_emb_c=hyper.min_embeddings,
            num_down_emb_f=hyper.min_embeddings,
            modality_routing=routing,
            thermal_only=False,
            rgb_only_teacher=True,
        )
        means = deformed[0]
        scales = gaussians.scaling_activation(deformed[1])
        rotations = gaussians.rotation_activation(deformed[2])
        opacity = gaussians.opacity_activation(gaussians._opacity)
        for name, value in (
            ("means", means), ("scales", scales),
            ("rotations", rotations), ("opacity", opacity),
        ):
            require(bool(torch.isfinite(value).all()),
                    f"Non-finite RGB teacher {name}")

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
            shs=gaussians.get_features,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        require(int((radii > 0).sum().item()) > 0,
                "No visible RGB teacher Gaussians")
        require(bool(torch.isfinite(rendered).all()),
                "Non-finite RGB teacher render")
        return rendered
    finally:
        view.frame_no = original_frame_no
        object.__setattr__(
            view, "nctc_temporal_offset_raw_override", original_raw)


def freeze_rgb_state(gaussians, views):
    tensors = []
    gaussian_names = (
        "_xyz", "_features_dc", "_features_rest", "_thermal_dc",
        "_thermal_rest", "_opacity", "_thermal_opacity", "_scaling",
        "_rotation", "_embedding", "_t_embedding", "_logit_modality",
        "_pose_r", "_pose_t", "_pose_log_s",
    )
    for name in gaussian_names:
        value = getattr(gaussians, name, None)
        require(torch.is_tensor(value), f"Missing Gaussian tensor: {name}")
        value.requires_grad_(False)
        value.grad = None
        tensors.append((f"gaussian.{name}", value))
    for name, value in gaussians._deformation.named_parameters():
        value.requires_grad_(False)
        value.grad = None
        tensors.append((f"deformation.{name}", value))
    for head_name in ("pose_head", "thermal_pose_head"):
        head = getattr(gaussians, head_name, None)
        if head is not None:
            for name, value in head.named_parameters():
                value.requires_grad_(False)
                value.grad = None
                tensors.append((f"{head_name}.{name}", value))
    for name, value in gaussians._deformation.named_buffers():
        tensors.append((f"deformation_buffer.{name}", value))
    for head_name in ("pose_head", "thermal_pose_head"):
        head = getattr(gaussians, head_name, None)
        if head is not None:
            for name, value in head.named_buffers():
                tensors.append((f"{head_name}_buffer.{name}", value))
    for name in (
        "max_radii2D", "xyz_gradient_accum", "denom",
        "xyz_gradient_accum_rgb", "xyz_gradient_accum_th",
        "denom_rgb", "denom_th",
    ):
        value = getattr(gaussians, name, None)
        require(torch.is_tensor(value), f"Missing Gaussian state: {name}")
        tensors.append((f"gaussian_state.{name}", value))

    camera_tensor_names = (
        "world_view_transform", "full_proj_transform", "projection_matrix",
        "projection_matrix_thermal", "full_proj_transform_thermal",
        "thermal_projection_matrix_learnable", "temporal_pose_frames",
        "temporal_pose_track",
    )
    for view in sorted(views, key=base.side_frame):
        side, frame = base.side_frame(view)
        for name in ("learnable_tfovx", "learnable_tfovy"):
            value = getattr(view, name, None)
            require(torch.is_tensor(value),
                    f"Missing camera parameter: {side} {frame} {name}")
            value.requires_grad_(False)
            value.grad = None
            tensors.append((f"camera_parameter.{side}.{frame}.{name}", value))
        for name in camera_tensor_names:
            value = getattr(view, name, None)
            require(torch.is_tensor(value),
                    f"Missing camera state: {side} {frame} {name}")
            tensors.append((f"camera_state.{side}.{frame}.{name}", value))
    return tensors


def render_at(side, frame, offset, views_by_side, gaussians, pipe, hyper,
              background, iteration, cache):
    key = (side, int(frame), int(offset))
    if key not in cache:
        cache[key] = render_clock_rgb(
            views_by_side[side][frame], frame,
            raw_for_offset(offset, "cuda"), gaussians, pipe, hyper,
            background, iteration).detach()
    return cache[key]


def main():
    require(sys.flags.optimize == 0, "Optimized Python is forbidden")
    parser = argparse.ArgumentParser()
    model_group = base.ModelParams(parser, sentinel=True)
    opt_group = base.OptimizationParams(parser)
    pipe_group = base.PipelineParams(parser)
    hyper_group = base.ModelHiddenParams(parser)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6666)
    args = base.get_combined_args(parser)
    require(not args.output.exists(), f"Refusing existing output: {args.output}")
    require(not args.report.exists(), f"Refusing existing report: {args.report}")
    args = base.merge_hparams(args, mmengine.Config.fromfile(args.configs))

    args.rgb_only_teacher = True
    args.no_thermal_pose_opt = True
    args.temporal_alignment_enabled = False
    args.change_thermal_geo = False
    args.shuffle = False
    base.safe_state(False, seed=args.seed)

    dataset = model_group.extract(args)
    hyper = hyper_group.extract(args)
    pipe = pipe_group.extract(args)
    opt = opt_group.extract(args)
    gaussians = base.GaussianModel(dataset.sh_degree, hyper)
    scene = base.Scene(
        dataset, gaussians, load_iteration=args.iteration, shuffle=False,
        duration=hyper.total_num_frames, loader=dataset.loader, opt=opt,
        load_test_cameras=False)
    require(scene.loaded_iter == args.iteration, "Teacher iteration mismatch")
    require(len(scene.getTestCameras()) == 0,
            "Formal test cameras were constructed")
    require(getattr(gaussians, "optimizer", None) is None,
            "Gaussian optimizer exists")

    rasterizer_path = Path(diff_gaussian_rasterization._C.__file__).resolve()
    wrapper_path = Path(diff_gaussian_rasterization.__file__).resolve()
    deformation_path = Path(__file__).resolve().parent / "scene" / "deformation.py"
    require(base.sha256_file(rasterizer_path)
            == base.EXPECTED_RASTERIZER_SHA256,
            "Rasterizer runtime hash mismatch")
    require(base.sha256_file(wrapper_path)
            == base.EXPECTED_RASTERIZER_WRAPPER_SHA256,
            "Rasterizer wrapper hash mismatch")
    require(base.sha256_file(deformation_path)
            == base.EXPECTED_DEFORMATION_SHA256,
            "Deformation source hash mismatch")
    teacher_dir = (Path(args.model_path) / "point_cloud"
                   / f"iteration_{args.iteration}")
    teacher_hashes = {
        name: base.sha256_file(teacher_dir / name)
        for name in EXPECTED_TEACHER
    }
    require(teacher_hashes == EXPECTED_TEACHER,
            "Teacher artifact mismatch")
    camera_state = torch.load(
        teacher_dir / "thermal_camera_state.pth", map_location="cpu")
    require(camera_state.get("num_cameras") == 0
            and camera_state.get("cameras") == [],
            "RGB-only teacher contains fitted Thermal cameras")

    views = scene.getTrainCameras()
    require(len(views) == 266, "Unexpected train-camera count")
    views_by_side = {"left": {}, "right": {}}
    for view in views:
        side, frame = base.side_frame(view)
        require(frame not in views_by_side[side], "Duplicate train frame")
        require(view.original_image is not None, "RGB image missing")
        require(view.thermal_image is None, "Thermal image was opened")
        require(not view.has_thermal, "Thermal camera path is active")
        expected_time = float(frame) / float(hyper.total_num_frames)
        require(math.isclose(float(view.time), expected_time,
                             rel_tol=0.0, abs_tol=1e-7),
                f"RGB timestamp/frame mismatch: {side} {frame}")
        views_by_side[side][frame] = view
    raw_offset = torch.nn.Parameter(
        torch.zeros((), device="cuda"), requires_grad=False)
    configure_views(views_by_side, raw_offset, hyper.total_num_frames)
    frozen_tensors = freeze_rgb_state(gaussians, views)
    centers = build_centers(views_by_side)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background
        else [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    nuisance_before, nuisance_count = base.stable_tensor_hash(frozen_tensors)
    camera_before, camera_hash_before = base.camera_scalar_manifest(views)
    started = time.time()
    render_cache = {}
    observed_rgb = {
        side: {frame: view.original_image.detach().to("cuda")
               for frame, view in by_frame.items()}
        for side, by_frame in views_by_side.items()
    }
    scan = {
        arm: {offset: {"left": [], "right": []}
              for offset in PROFILE_OFFSETS}
        for arm in ARM_SHIFTS
    }
    with torch.no_grad():
        for side in ("left", "right"):
            for frame in centers[side]:
                rendered_changes = {}
                for offset in PROFILE_OFFSETS:
                    first = render_at(
                        side, frame, offset, views_by_side, gaussians, pipe,
                        hyper, background, args.iteration, render_cache)
                    second = render_at(
                        side, frame + HALF_WINDOW, offset, views_by_side,
                        gaussians, pipe, hyper, background, args.iteration,
                        render_cache)
                    rendered_changes[offset] = standardized_absolute_change(
                        first, second)
                for arm, shift in ARM_SHIFTS.items():
                    target_change = standardized_absolute_change(
                        observed_rgb[side][frame + shift],
                        observed_rgb[side][frame + HALF_WINDOW + shift])
                    for offset in PROFILE_OFFSETS:
                        scan[arm][offset][side].append(float(cosine(
                            rendered_changes[offset], target_change).item()))

    rows = []
    for arm, truth in ARM_SHIFTS.items():
        summary = {}
        for offset in PROFILE_OFFSETS:
            left = scan[arm][offset]["left"]
            right = scan[arm][offset]["right"]
            require(left and right, "Empty global score row")
            summary[str(offset)] = {
                "all": float(sum(left + right) / len(left + right)),
                "left": float(sum(left) / len(left)),
                "right": float(sum(right) / len(right)),
                "count_all": len(left + right),
            }
        winners = {}
        margins = {}
        for scope in ("all", "left", "right"):
            ranking = sorted(
                PROFILE_OFFSETS,
                key=lambda value: summary[str(value)][scope], reverse=True)
            winners[scope] = ranking[0]
            margins[scope] = (summary[str(ranking[0])][scope]
                              - summary[str(ranking[1])][scope])
        row = {
            "arm": arm,
            "truth_frames": truth,
            "scores": summary,
            "winners": winners,
            "winner_margins": margins,
            "pass": all(value == truth for value in winners.values())
            and all(value > 0.0 for value in margins.values()),
        }
        rows.append(row)
        print("RGB_TEMPORAL_FIDELITY_PROFILE " + json.dumps(
            row, sort_keys=True), flush=True)

    nuisance_after, nuisance_after_count = base.stable_tensor_hash(
        freeze_rgb_state(gaussians, views))
    camera_after, camera_hash_after = base.camera_scalar_manifest(views)
    require(nuisance_count == nuisance_after_count
            and nuisance_before == nuisance_after,
            "Frozen scene state changed")
    require(camera_before == camera_after
            and camera_hash_before == camera_hash_after,
            "Camera scalar state changed")
    profile_pass = all(row["pass"] for row in rows)
    status = ("RGB_TEMPORAL_FIDELITY_PASS" if profile_pass
              else "RGB_TEMPORAL_FIDELITY_FAIL")
    report = {
        "schema": SCHEMA,
        "status": status,
        "scene": "Covers",
        "per_scene_optimization": True,
        "profile_only": True,
        "rgb_image_count": len(views),
        "thermal_images_opened": False,
        "observable": "mean cosine of standardized absolute temporal-change maps",
        "comparison": "continuous RGB-only 4DGS renders versus observed RGB frames",
        "centers_by_side": centers,
        "rows": rows,
        "all_rows_pass": profile_pass,
        "render_cache_entries": len(render_cache),
        "nuisance_sha256_before": nuisance_before,
        "nuisance_sha256_after": nuisance_after,
        "nuisance_tensor_count": nuisance_count,
        "camera_manifest_sha256_before": camera_hash_before,
        "camera_manifest_sha256_after": camera_hash_after,
        "camera_path": "time-aligned RGB trajectory with RGB intrinsics",
        "formal_test_cameras_constructed": False,
        "future_trainable_parameter": "delta=20*tanh(raw_offset)",
        "teacher_hashes": teacher_hashes,
        "script_sha256": base.sha256_file(__file__),
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (args.output / "RUN_COMPLETE").write_text(
        "complete\n", encoding="utf-8")
    (args.output / status).write_text(status + "\n", encoding="utf-8")
    print("RGB_TEMPORAL_FIDELITY_RESULT " + json.dumps({
        "status": status,
        "pass_count": sum(row["pass"] for row in rows),
        "row_count": len(rows),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
