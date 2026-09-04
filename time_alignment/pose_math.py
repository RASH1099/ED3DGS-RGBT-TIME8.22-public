#!/usr/bin/env python3
"""Bounded side-shared Thermal pose projection for Covers v34."""

import math

import torch


MAX_ROTATION_DEGREES = 5.0
MAX_TRANSLATION_FRACTION = 0.1


def project_pose_(quaternion, translation, scene_extent,
                  max_rotation_degrees=MAX_ROTATION_DEGREES,
                  max_translation_fraction=MAX_TRANSLATION_FRACTION):
    """Project one [w,x,y,z] residual and translation in place."""
    if float(scene_extent) <= 0.0:
        raise RuntimeError("Scene extent must be positive")
    with torch.no_grad():
        norm = torch.linalg.vector_norm(quaternion)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
            raise RuntimeError("Invalid Thermal pose quaternion")
        normalized = quaternion / norm
        if float(normalized[0]) < 0.0:
            normalized = -normalized
        vector_norm = torch.linalg.vector_norm(normalized[1:])
        angle = 2.0 * torch.atan2(vector_norm, normalized[0].clamp_min(0.0))
        maximum = math.radians(float(max_rotation_degrees))
        if float(angle) > maximum:
            if float(vector_norm) <= 0.0:
                raise RuntimeError("Degenerate quaternion axis")
            axis = normalized[1:] / vector_norm
            half = 0.5 * maximum
            normalized = torch.cat((
                torch.cos(torch.as_tensor(half, device=quaternion.device,
                                          dtype=quaternion.dtype)).reshape(1),
                axis * math.sin(half),
            ))
        quaternion.copy_(normalized)

        translation_limit = (float(scene_extent)
                             * float(max_translation_fraction))
        translation_norm = torch.linalg.vector_norm(translation)
        if not bool(torch.isfinite(translation_norm)):
            raise RuntimeError("Invalid Thermal pose translation")
        if float(translation_norm) > translation_limit:
            translation.mul_(translation_limit / translation_norm)


def residual_sizes(quaternion, translation, scene_extent):
    normalized = quaternion.detach() / torch.linalg.vector_norm(
        quaternion.detach())
    scalar = abs(float(normalized[0]))
    vector_norm = float(torch.linalg.vector_norm(normalized[1:]))
    rotation_degrees = math.degrees(
        2.0 * math.atan2(vector_norm, scalar))
    translation_fraction = (
        float(torch.linalg.vector_norm(translation.detach()))
        / float(scene_extent))
    return rotation_degrees, translation_fraction


def set_locked_joint_gate_perturbation_(quaternion, translation, side,
                                        scene_extent):
    """Inject the preregistered 2-degree / 1%-extent Gate perturbation."""
    if side not in {"left", "right"}:
        raise RuntimeError(f"Invalid pose-perturbation side: {side}")
    half_angle = math.radians(2.0) / 2.0
    axis = (quaternion.new_tensor((1.0, 0.0, 0.0)) if side == "left"
            else quaternion.new_tensor((0.0, 1.0, 0.0)))
    direction = (translation.new_tensor((1.0, 1.0, 0.0)) if side == "left"
                 else translation.new_tensor((-1.0, 1.0, 0.0)))
    direction = direction / torch.linalg.vector_norm(direction)
    with torch.no_grad():
        quaternion.copy_(torch.cat((
            quaternion.new_tensor((math.cos(half_angle),)),
            axis * math.sin(half_angle))))
        translation.copy_(direction * (0.01 * float(scene_extent)))
