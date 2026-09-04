#!/usr/bin/env python3
"""Cross-modal structural Thermal-pose calibration for Covers v34."""

import math

import torch
import torch.nn.functional as F
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from time_alignment import mind_descriptor as mind


MIND_WEIGHT = 0.5
NGF_WEIGHT = 0.5
ROTATION_RAW_FD_STEP = 1e-3
TRANSLATION_EXTENT_FD_STEP = 1e-3


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _gray(image):
    require(image.ndim == 3 and image.shape[0] == 3,
            "Structural pose image must be [3,H,W]")
    weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
    return (image * weights).sum(dim=0)


def _normalized_gradient(image):
    gray = _gray(image).unsqueeze(0).unsqueeze(0)
    kernel_x = gray.new_tensor(((-1.0, 0.0, 1.0),
                                (-2.0, 0.0, 2.0),
                                (-1.0, 0.0, 1.0))).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel_x)[0, 0]
    gy = F.conv2d(padded, kernel_y)[0, 0]
    norm = torch.sqrt(gx.square() + gy.square() + 1e-6)
    return gx / norm, gy / norm


def structural_pose_loss(rendered_rgb, observed_thermal):
    require(rendered_rgb.shape == observed_thermal.shape,
            "Pose render/observation shape mismatch")
    rendered_mind = mind.mind_descriptor(rendered_rgb)
    observed_mind = mind.mind_descriptor(observed_thermal).detach()
    mind_loss = torch.abs(rendered_mind - observed_mind).mean()
    render_gx, render_gy = _normalized_gradient(rendered_rgb)
    target_gx, target_gy = _normalized_gradient(observed_thermal)
    dot = render_gx * target_gx.detach() + render_gy * target_gy.detach()
    ngf_loss = (1.0 - dot.square()).mean()
    total = MIND_WEIGHT * mind_loss + NGF_WEIGHT * ngf_loss
    require(bool(torch.isfinite(total)), "Non-finite structural pose loss")
    return total, {"mind": mind_loss, "ngf": ngf_loss}


def render_rgb_structure_from_thermal_camera(view, gaussians, pipe, hyper,
                                             background, iteration,
                                             raw_offset_override=None):
    """Render frozen RGB structure through the current Thermal camera pose."""
    require(getattr(view, "has_thermal", False),
            f"Missing Thermal camera: {view.image_name}")
    sentinel = object()
    old_offset = getattr(view, "nctc_temporal_offset_raw_override", sentinel)
    old_drift = getattr(view, "nctc_temporal_drift_raw_override", sentinel)
    has_clock = view.temporal_offset_raw is not None
    if has_clock:
        applied_raw = (
            view.temporal_offset_raw.detach()
            if raw_offset_override is None else raw_offset_override)
        object.__setattr__(
            view, "nctc_temporal_offset_raw_override",
            applied_raw)
    if view.temporal_drift_raw is not None:
        object.__setattr__(
            view, "nctc_temporal_drift_raw_override",
            view.temporal_drift_raw.detach())
    try:
        viewmatrix = view.get_thermal_world_view_transform()
        view.refresh_thermal_projection()
        intrinsic = view.thermal_projection_matrix_learnable.detach()
        full_projection = viewmatrix @ intrinsic
        campos = viewmatrix.inverse()[3, :3]
        fovx, fovy = view.get_thermal_fovs()
        raster_settings = GaussianRasterizationSettings(
            image_height=int(view.thermal_height or view.image_height),
            image_width=int(view.thermal_width or view.image_width),
            tanfovx=torch.tan(fovx.detach() * 0.5),
            tanfovy=torch.tan(fovy.detach() * 0.5),
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
        routing = torch.zeros((means3d.shape[0], 3), device=means3d.device)
        routing[:, 0] = 1.0
        clock_grad = bool(
            torch.is_tensor(raw_offset_override)
            and raw_offset_override.requires_grad)
        deformation_context = (
            torch.enable_grad() if clock_grad else torch.no_grad())
        with deformation_context:
            deformed = gaussians._deformation(
                means3d, gaussians._scaling, gaussians._rotation,
                gaussians._opacity, gaussians._thermal_opacity, time_tensor,
                None, gaussians, None, None, gaussians.get_features,
                gaussians.get_thermal_features, iter=iteration,
                num_down_emb_c=hyper.min_embeddings,
                num_down_emb_f=hyper.min_embeddings,
                modality_routing=routing, thermal_only=False,
                rgb_only_teacher=True)
            means = deformed[0]
            scales = gaussians.scaling_activation(deformed[1])
            rotations = gaussians.rotation_activation(deformed[2])
            if not clock_grad:
                means = means.detach()
                scales = scales.detach()
                rotations = rotations.detach()
            opacity = gaussians.opacity_activation(
                gaussians._opacity).detach()
            rgb_features = gaussians.get_features.detach()
        screenspace = torch.zeros_like(
            means, requires_grad=True, device="cuda")
        screenspace_densify = torch.zeros_like(
            means, requires_grad=True, device="cuda")
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered, radii, _, _, _ = rasterizer(
            means3D=means,
            means2D=screenspace,
            means2D_densify=screenspace_densify,
            shift_factors=torch.zeros(3, device="cuda"),
            shs=rgb_features,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        require(int((radii > 0).sum().item()) > 0,
                "No visible RGB Gaussians in Thermal pose view")
        require(bool(torch.isfinite(rendered).all()),
                "Non-finite pose structural render")
        return rendered
    finally:
        if has_clock:
            if old_offset is sentinel:
                delattr(view, "nctc_temporal_offset_raw_override")
            else:
                object.__setattr__(
                    view, "nctc_temporal_offset_raw_override", old_offset)
        if view.temporal_drift_raw is not None:
            if old_drift is sentinel:
                delattr(view, "nctc_temporal_drift_raw_override")
            else:
                object.__setattr__(
                    view, "nctc_temporal_drift_raw_override", old_drift)


def batch_pose_loss(views, gaussians, pipe, hyper, background, iteration):
    losses, mind_losses, ngf_losses = [], [], []
    for view in views:
        rendered = render_rgb_structure_from_thermal_camera(
            view, gaussians, pipe, hyper, background, iteration)
        loss, components = structural_pose_loss(
            rendered, view.thermal_image.cuda())
        losses.append(loss)
        mind_losses.append(components["mind"])
        ngf_losses.append(components["ngf"])
    require(losses, "Pose calibration batch is empty")
    return torch.stack(losses).mean(), {
        "mind": torch.stack(mind_losses).mean(),
        "ngf": torch.stack(ngf_losses).mean(),
    }


def directional_derivative_audit(view, gaussians, pipe, hyper, background,
                                 iteration, scene_extent,
                                 relative_error_max=0.05):
    """Compare pose autograd with centered FD along fixed directions."""
    quaternion = view.thermal_delta_quaternion
    translation = view.thermal_delta_translation
    rendered = render_rgb_structure_from_thermal_camera(
        view, gaussians, pipe, hyper, background, iteration)
    loss, _ = structural_pose_loss(rendered, view.thermal_image.cuda())
    gradients = torch.autograd.grad(loss, (quaternion, translation))
    directions = (
        ("rotation", quaternion,
         quaternion.new_tensor((0.0, 1.0, 0.0, 0.0)),
         ROTATION_RAW_FD_STEP,
         gradients[0]),
        ("translation", translation,
         translation.new_tensor((1.0, 1.0, 1.0)) / math.sqrt(3.0),
         TRANSLATION_EXTENT_FD_STEP * float(scene_extent), gradients[1]),
    )
    rows = []
    for name, parameter, direction, step, gradient in directions:
        original = parameter.detach().clone()
        losses = []
        for sign in (-1.0, 1.0):
            with torch.no_grad():
                parameter.copy_(original + sign * step * direction)
            probe = render_rgb_structure_from_thermal_camera(
                view, gaussians, pipe, hyper, background, iteration)
            probe_loss, _ = structural_pose_loss(
                probe, view.thermal_image.cuda())
            losses.append(float(probe_loss.detach().item()))
        with torch.no_grad():
            parameter.copy_(original)
        finite = (losses[1] - losses[0]) / (2.0 * step)
        autograd_value = float((gradient * direction).sum().detach().item())
        relative = abs(autograd_value - finite) / max(
            abs(autograd_value), abs(finite), 1e-8)
        rows.append({
            "parameter": name,
            "step": step,
            "loss_minus": losses[0],
            "loss_plus": losses[1],
            "autograd_directional_derivative": autograd_value,
            "centered_fd_directional_derivative": finite,
            "relative_error": relative,
            "pass": relative <= relative_error_max,
        })
    report = {
        "relative_error_max": relative_error_max,
        "rows": rows,
        "pass": all(row["pass"] for row in rows),
    }
    require(report["pass"], "Pose renderer derivative audit failed")
    return report
