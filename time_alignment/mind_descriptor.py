#!/usr/bin/env python3
"""Covers-only one-raw shift-equivariant recovery Gate."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import diff_gaussian_rasterization
import mmengine
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scene.deformation as deformation_module

from time_alignment import feature_runtime as base
from time_alignment import rgb_fidelity as rgbgate


SCHEMA = "covers_mind_shift_equivariant_recovery_v6"
ARM_SHIFTS = {"N": -8, "Z": 0, "P": 8}
PROFILE_OFFSETS = (-8, 0, 8)
HALF_WINDOW = 2
FD_STEPS = (1.0, 0.5)
EVAL_DELTAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
RECOVERY_STEPS = 60
RAW_LR = 0.05
EQUIVARIANCE_TOLERANCE = 1.0
LANDSCAPE_ONLY = False
LANDSCAPE_DELTAS = tuple(float(value) for value in range(-12, 13))
DIRECTIONAL_GATE_ONLY = False
DIRECTIONAL_PROFILE_DELTAS = LANDSCAPE_DELTAS
DIRECTIONAL_EVAL_DELTAS = tuple(sorted(set(
    DIRECTIONAL_PROFILE_DELTAS + (-0.5, 0.5))))
TRANSPORT_GATE_ONLY = False
TRANSPORT_PROFILE_DELTAS = LANDSCAPE_DELTAS
TRANSPORT_EVAL_DELTAS = DIRECTIONAL_EVAL_DELTAS
MULTISCALE_GATE_ONLY = False
MULTISCALE_KERNELS = (9, 5, 3, 1)
COARSE_MULTISCALE_GATE_ONLY = False
COARSE_DERIVATIVE_AUDIT_ONLY = False
C1_DERIVATIVE_AUDIT_ONLY = False
DETREND_C1_MULTISCALE_GATE_ONLY = False
C1_PHASE_GATE_ONLY = False
C1_TEMPORAL_DERIVATIVE_GATE_ONLY = False
C1_DETRENDED_SHARED_RECOVERY_ONLY = False
DERIVATIVE_AUDIT_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))
EXPECTED_TEACHER = rgbgate.EXPECTED_TEACHER
EXPECTED_SOURCE = "/data/linlifeng/home_storage/project/covers/covers"
THERMAL_IMAGE_ROOT = Path(EXPECTED_SOURCE) / "thermal" / "images" / "4x"
ENTRY_SCRIPT_PATH = Path(__file__).resolve()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def mind_descriptor(image):
    gray = rgbgate.gray(image)
    height, width = gray.shape
    source = gray[None, None]
    padded = F.pad(source, (1, 1, 1, 1), mode="replicate")
    distances = []
    for dy, dx in OFFSETS:
        shifted = padded[:, :, 1 + dy:1 + dy + height,
                         1 + dx:1 + dx + width]
        squared = (source - shifted).square()
        smoothed = F.avg_pool2d(
            F.pad(squared, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3, stride=1)[0, 0]
        distances.append(smoothed)
    distance = torch.stack(distances, dim=0)
    variance = distance.mean(dim=0, keepdim=True)
    denominator = variance + torch.finfo(distance.dtype).eps
    descriptor = torch.exp(-distance / denominator)
    require(descriptor.shape == (len(OFFSETS), 120, 160),
            "Unexpected MIND shape")
    require(bool(torch.isfinite(descriptor).all()),
            "Non-finite MIND descriptor")
    return descriptor


def cosine(first, second):
    require(first.shape == second.shape, "MIND change shape mismatch")
    denominator = (torch.linalg.vector_norm(first)
                   * torch.linalg.vector_norm(second))
    require(bool(torch.isfinite(denominator))
            and float(denominator.detach().item()) > 0.0,
            "Degenerate MIND change cosine")
    value = torch.sum(first * second) / denominator
    require(bool(torch.isfinite(value)), "Non-finite MIND cosine")
    return value


def pearson(first, second):
    require(len(first) == len(second) and len(first) >= 100,
            "Invalid activity sequence lengths")
    first_tensor = torch.tensor(first, dtype=torch.float64)
    second_tensor = torch.tensor(second, dtype=torch.float64)
    first_centered = first_tensor - first_tensor.mean()
    second_centered = second_tensor - second_tensor.mean()
    denominator = (torch.linalg.vector_norm(first_centered)
                   * torch.linalg.vector_norm(second_centered))
    require(bool(torch.isfinite(denominator))
            and float(denominator.item()) > 0.0,
            "Degenerate MIND activity correlation")
    value = torch.sum(first_centered * second_centered) / denominator
    require(bool(torch.isfinite(value)), "Non-finite activity correlation")
    return float(value.item())


def pearson_tensor(first, second):
    require(first.shape == second.shape and first.ndim == 1
            and first.shape[0] >= 100, "Invalid activity tensors")
    first_centered = first - first.mean()
    second_centered = second - second.mean()
    denominator = (torch.linalg.vector_norm(first_centered)
                   * torch.linalg.vector_norm(second_centered))
    require(bool(torch.isfinite(denominator))
            and float(denominator.detach().item()) > 0.0,
            "Degenerate differentiable activity correlation")
    value = torch.sum(first_centered * second_centered) / denominator
    require(bool(torch.isfinite(value)), "Non-finite activity correlation")
    return value


def losses_from_activities(rendered, targets):
    losses = {}
    for arm in ARM_SHIFTS:
        left = -pearson_tensor(rendered["left"], targets[arm]["left"])
        right = -pearson_tensor(rendered["right"], targets[arm]["right"])
        all_loss = -pearson_tensor(
            torch.cat((rendered["left"], rendered["right"])),
            torch.cat((targets[arm]["left"], targets[arm]["right"])))
        losses[arm] = {"all": all_loss, "left": left, "right": right}
    return losses


def render_activity_sequence(side, centers, delta, views_by_side, gaussians,
                             pipe, hyper, background, iteration):
    raw = rgbgate.raw_for_offset(delta, "cuda")
    values = []
    for frame in centers[side]:
        before = rgbgate.render_clock_rgb(
            views_by_side[side][frame - HALF_WINDOW],
            frame - HALF_WINDOW, raw, gaussians, pipe, hyper,
            background, iteration)
        center = rgbgate.render_clock_rgb(
            views_by_side[side][frame], frame, raw, gaussians, pipe, hyper,
            background, iteration)
        after = rgbgate.render_clock_rgb(
            views_by_side[side][frame + HALF_WINDOW],
            frame + HALF_WINDOW, raw, gaussians, pipe, hyper, background,
            iteration)
        before_desc = mind_descriptor(before)
        center_desc = mind_descriptor(center)
        after_desc = mind_descriptor(after)
        backward = (center_desc - before_desc).square().mean()
        forward = (after_desc - center_desc).square().mean()
        values.append(0.5 * (backward + forward))
    result = torch.stack(values)
    require(bool(torch.isfinite(result).all()), "Non-finite render activities")
    return result


def placeholder_weights(render_zero, targets):
    weights = {arm: {scope: {} for scope in ("all", "left", "right")}
               for arm in ARM_SHIFTS}
    for arm in ARM_SHIFTS:
        left = render_zero["left"].detach().clone().requires_grad_(True)
        right = render_zero["right"].detach().clone().requires_grad_(True)
        all_loss = -pearson_tensor(
            torch.cat((left, right)),
            torch.cat((targets[arm]["left"], targets[arm]["right"])))
        left_loss = -pearson_tensor(left, targets[arm]["left"])
        right_loss = -pearson_tensor(right, targets[arm]["right"])
        all_left, all_right = torch.autograd.grad(
            all_loss, (left, right), retain_graph=True)
        left_gradient = torch.autograd.grad(left_loss, left, retain_graph=True)[0]
        right_gradient = torch.autograd.grad(right_loss, right)[0]
        weights[arm]["all"] = {
            "left": all_left.detach(), "right": all_right.detach()}
        weights[arm]["left"] = {
            "left": left_gradient.detach(),
            "right": torch.zeros_like(right).detach()}
        weights[arm]["right"] = {
            "left": torch.zeros_like(left).detach(),
            "right": right_gradient.detach()}
    return weights


def accumulated_delta_gradients(centers, weights, views_by_side, gaussians,
                                pipe, hyper, background, iteration):
    raw = torch.nn.Parameter(torch.zeros((), device="cuda"))
    totals = {arm: {scope: torch.zeros((), device="cuda")
                    for scope in ("all", "left", "right")}
              for arm in ARM_SHIFTS}
    for side in ("left", "right"):
        for index, frame in enumerate(centers[side]):
            before = rgbgate.render_clock_rgb(
                views_by_side[side][frame - HALF_WINDOW],
                frame - HALF_WINDOW, raw, gaussians, pipe,
                hyper, background, iteration)
            center = rgbgate.render_clock_rgb(
                views_by_side[side][frame], frame, raw, gaussians, pipe,
                hyper, background, iteration)
            after = rgbgate.render_clock_rgb(
                views_by_side[side][frame + HALF_WINDOW],
                frame + HALF_WINDOW, raw, gaussians, pipe, hyper, background,
                iteration)
            before_desc = mind_descriptor(before)
            center_desc = mind_descriptor(center)
            after_desc = mind_descriptor(after)
            activity = 0.5 * (
                (center_desc - before_desc).square().mean()
                + (after_desc - center_desc).square().mean())
            derivative = torch.autograd.grad(activity, raw)[0].detach()
            require(bool(torch.isfinite(derivative)),
                    f"Non-finite activity derivative: {side} {frame}")
            for arm in ARM_SHIFTS:
                for scope in ("all", "left", "right"):
                    totals[arm][scope] += (
                        weights[arm][scope][side][index] * derivative)
    return {
        arm: {scope: float((value / rgbgate.OFFSET_BOUND).item())
              for scope, value in scopes.items()}
        for arm, scopes in totals.items()
    }


def finite_difference_rows(loss_values, autograd_values):
    rows = []
    for arm, truth in ARM_SHIFTS.items():
        scopes = {}
        for scope in ("all", "left", "right"):
            values = {delta: loss_values[delta][arm][scope]
                      for delta in EVAL_DELTAS}
            central = {
                str(step): (values[step] - values[-step]) / (2.0 * step)
                for step in FD_STEPS}
            auto = autograd_values[arm][scope]
            if truth < 0:
                passed = auto > 0.0 and all(
                    value > 0.0 for value in central.values())
            elif truth > 0:
                passed = auto < 0.0 and all(
                    value < 0.0 for value in central.values())
            else:
                left_near = (values[0.0] - values[-0.5]) / 0.5
                right_near = (values[0.5] - values[0.0]) / 0.5
                passed = (
                    values[-1.0] > values[-0.5] > values[0.0]
                    and values[1.0] > values[0.5] > values[0.0]
                    and left_near < 0.0 < right_near
                    and abs(auto) < min(abs(left_near), abs(right_near)))
            scopes[scope] = {
                "losses": {str(delta): values[delta]
                           for delta in EVAL_DELTAS},
                "autograd_dloss_ddelta": auto,
                "central_fd": central, "pass": passed}
        row = {"arm": arm, "truth_frames": truth, "scopes": scopes,
               "pass": all(value["pass"] for value in scopes.values())}
        rows.append(row)
        print("MIND_SYMMETRIC_ACTIVITY_GRADIENT_ROW " + json.dumps(
            row, sort_keys=True), flush=True)
    return rows


def activity_and_raw_derivative(raw, side, centers, views_by_side, gaussians,
                                pipe, hyper, background, iteration):
    activities, derivatives = [], []
    for frame in centers[side]:
        before = rgbgate.render_clock_rgb(
            views_by_side[side][frame - HALF_WINDOW],
            frame - HALF_WINDOW, raw, gaussians, pipe, hyper, background,
            iteration)
        center = rgbgate.render_clock_rgb(
            views_by_side[side][frame], frame, raw, gaussians, pipe, hyper,
            background, iteration)
        after = rgbgate.render_clock_rgb(
            views_by_side[side][frame + HALF_WINDOW],
            frame + HALF_WINDOW, raw, gaussians, pipe, hyper, background,
            iteration)
        before_desc = mind_descriptor(before)
        center_desc = mind_descriptor(center)
        after_desc = mind_descriptor(after)
        activity = 0.5 * (
            (center_desc - before_desc).square().mean()
            + (after_desc - center_desc).square().mean())
        derivative = torch.autograd.grad(activity, raw)[0]
        require(bool(torch.isfinite(activity))
                and bool(torch.isfinite(derivative)),
                f"Non-finite recovery activity: {side} {frame}")
        activities.append(activity.detach())
        derivatives.append(derivative.detach())
    return torch.stack(activities), torch.stack(derivatives)


def directional_sequences(side, centers, delta, views_by_side, gaussians,
                          pipe, hyper, background, iteration):
    raw = rgbgate.raw_for_offset(delta, "cuda")
    symmetric, directional = [], []
    for frame in centers[side]:
        before = rgbgate.render_clock_rgb(
            views_by_side[side][frame - HALF_WINDOW],
            frame - HALF_WINDOW, raw, gaussians, pipe, hyper, background,
            iteration)
        center = rgbgate.render_clock_rgb(
            views_by_side[side][frame], frame, raw, gaussians, pipe, hyper,
            background, iteration)
        after = rgbgate.render_clock_rgb(
            views_by_side[side][frame + HALF_WINDOW],
            frame + HALF_WINDOW, raw, gaussians, pipe, hyper, background,
            iteration)
        before_desc = mind_descriptor(before)
        center_desc = mind_descriptor(center)
        after_desc = mind_descriptor(after)
        backward = (center_desc - before_desc).square().mean()
        forward = (after_desc - center_desc).square().mean()
        symmetric.append(0.5 * (backward + forward))
        directional.append(forward - backward)
    result = {
        "symmetric": torch.stack(symmetric),
        "directional": torch.stack(directional),
    }
    require(all(bool(torch.isfinite(value).all())
                for value in result.values()),
            "Non-finite directional sequences")
    return result


def directional_sequences_and_raw_derivatives(
        raw, side, centers, views_by_side, gaussians, pipe, hyper, background,
        iteration):
    values = {"symmetric": [], "directional": []}
    derivatives = {"symmetric": [], "directional": []}
    for frame in centers[side]:
        before = rgbgate.render_clock_rgb(
            views_by_side[side][frame - HALF_WINDOW],
            frame - HALF_WINDOW, raw, gaussians, pipe, hyper, background,
            iteration)
        center = rgbgate.render_clock_rgb(
            views_by_side[side][frame], frame, raw, gaussians, pipe, hyper,
            background, iteration)
        after = rgbgate.render_clock_rgb(
            views_by_side[side][frame + HALF_WINDOW],
            frame + HALF_WINDOW, raw, gaussians, pipe, hyper, background,
            iteration)
        before_desc = mind_descriptor(before)
        center_desc = mind_descriptor(center)
        after_desc = mind_descriptor(after)
        backward = (center_desc - before_desc).square().mean()
        forward = (after_desc - center_desc).square().mean()
        current = {
            "symmetric": 0.5 * (backward + forward),
            "directional": forward - backward,
        }
        derivatives["symmetric"].append(torch.autograd.grad(
            current["symmetric"], raw, retain_graph=True)[0].detach())
        derivatives["directional"].append(torch.autograd.grad(
            current["directional"], raw)[0].detach())
        for component in current:
            require(bool(torch.isfinite(current[component]))
                    and bool(torch.isfinite(derivatives[component][-1])),
                    f"Non-finite directional derivative: {side} {frame}")
            values[component].append(current[component].detach())
    return (
        {key: torch.stack(items) for key, items in values.items()},
        {key: torch.stack(items) for key, items in derivatives.items()},
    )


def combined_directional_loss(rendered, targets, arm, scope):
    sides = ("left", "right") if scope == "all" else (scope,)
    losses = []
    for component in ("symmetric", "directional"):
        prediction = torch.cat([
            rendered[side][component] for side in sides])
        target = torch.cat([
            targets[arm][side][component] for side in sides])
        losses.append(-pearson_tensor(prediction, target))
    return 0.5 * (losses[0] + losses[1])


def temporal_transport_loss(prediction, target):
    require(prediction.shape == target.shape and prediction.ndim == 1
            and prediction.shape[0] >= 100,
            "Invalid temporal transport tensors")
    require(bool((prediction >= 0.0).all())
            and bool((target >= 0.0).all()),
            "Temporal transport requires nonnegative activity")
    prediction_sum = prediction.sum()
    target_sum = target.sum()
    require(bool(torch.isfinite(prediction_sum))
            and bool(torch.isfinite(target_sum))
            and float(prediction_sum.detach().item()) > 0.0
            and float(target_sum.detach().item()) > 0.0,
            "Degenerate temporal activity mass")
    prediction_cdf = torch.cumsum(prediction / prediction_sum, dim=0)
    target_cdf = torch.cumsum(target / target_sum, dim=0)
    loss = (prediction_cdf - target_cdf).square().mean()
    require(bool(torch.isfinite(loss)), "Non-finite temporal transport loss")
    return loss


def combined_transport_loss(rendered, targets, arm, scope):
    sides = ("left", "right") if scope == "all" else (scope,)
    return sum(
        temporal_transport_loss(rendered[side], targets[arm][side])
        for side in sides) / float(len(sides))


def smooth_temporal_sequence(sequence, kernel):
    require(kernel >= 1 and kernel % 2 == 1,
            "Temporal smoothing kernel must be positive and odd")
    if kernel == 1:
        return sequence
    radius = kernel // 2
    source = sequence.reshape(1, 1, -1)
    return F.avg_pool1d(
        F.pad(source, (radius, radius), mode="replicate"),
        kernel_size=kernel, stride=1)[0, 0]


def combined_multiscale_loss(rendered, targets, arm, scope):
    sides = ("left", "right") if scope == "all" else (scope,)
    scale_losses = []
    for kernel in MULTISCALE_KERNELS:
        prediction = torch.cat([
            smooth_temporal_sequence(rendered[side], kernel)
            for side in sides])
        target = torch.cat([
            smooth_temporal_sequence(targets[arm][side], kernel)
            for side in sides])
        scale_losses.append(-pearson_tensor(prediction, target))
    return sum(scale_losses) / float(len(scale_losses))


def detrend_temporal_sequence(sequence):
    require(sequence.ndim == 1 and sequence.shape[0] >= 100,
            "Invalid temporal sequence for affine detrending")
    coordinate = torch.linspace(
        -1.0, 1.0, sequence.shape[0], device=sequence.device,
        dtype=sequence.dtype)
    centered = sequence - sequence.mean()
    slope = torch.sum(centered * coordinate) / torch.sum(coordinate.square())
    detrended = centered - slope * coordinate
    require(bool(torch.isfinite(detrended).all()),
            "Non-finite detrended temporal sequence")
    return detrended


def combined_detrended_multiscale_loss(rendered, targets, arm, scope):
    sides = ("left", "right") if scope == "all" else (scope,)
    scale_losses = []
    for kernel in MULTISCALE_KERNELS:
        prediction = torch.cat([
            detrend_temporal_sequence(smooth_temporal_sequence(
                rendered[side], kernel)) for side in sides])
        target = torch.cat([
            detrend_temporal_sequence(smooth_temporal_sequence(
                targets[arm][side], kernel)) for side in sides])
        scale_losses.append(-pearson_tensor(prediction, target))
    return sum(scale_losses) / float(len(scale_losses))


def gcc_phat_zero_lag_loss(prediction, target):
    require(prediction.shape == target.shape and prediction.ndim == 1
            and prediction.shape[0] >= 100,
            "Invalid temporal sequence for GCC-PHAT")
    prediction_spectrum = torch.fft.rfft(
        prediction - prediction.mean(), norm="ortho")
    target_spectrum = torch.fft.rfft(
        target - target.mean(), norm="ortho")
    cross_spectrum = prediction_spectrum * target_spectrum.conj()
    non_dc = cross_spectrum[1:]
    normalized = non_dc / (
        non_dc.abs() + torch.finfo(non_dc.real.dtype).eps)
    loss = -normalized.real.mean()
    require(bool(torch.isfinite(loss)), "Non-finite GCC-PHAT loss")
    return loss


def combined_gcc_phat_loss(rendered, targets, arm, scope):
    sides = ("left", "right") if scope == "all" else (scope,)
    return sum(
        gcc_phat_zero_lag_loss(rendered[side], targets[arm][side])
        for side in sides) / float(len(sides))


def centered_temporal_derivative(sequence):
    require(sequence.ndim == 1 and sequence.shape[0] >= 100,
            "Invalid temporal sequence for centered derivative")
    derivative = 0.5 * (sequence[2:] - sequence[:-2])
    require(bool(torch.isfinite(derivative).all()),
            "Non-finite centered temporal derivative")
    return derivative


def combined_temporal_derivative_multiscale_loss(
        rendered, targets, arm, scope):
    sides = ("left", "right") if scope == "all" else (scope,)
    scale_losses = []
    for kernel in MULTISCALE_KERNELS:
        prediction = torch.cat([
            centered_temporal_derivative(smooth_temporal_sequence(
                rendered[side], kernel)) for side in sides])
        target = torch.cat([
            centered_temporal_derivative(smooth_temporal_sequence(
                targets[arm][side], kernel)) for side in sides])
        scale_losses.append(-pearson_tensor(prediction, target))
    return sum(scale_losses) / float(len(scale_losses))


def optimize_arm(arm, targets, centers, views_by_side, gaussians, pipe, hyper,
                 background, iteration):
    raw = torch.nn.Parameter(torch.zeros((), device="cuda"))
    optimizer = torch.optim.Adam([raw], lr=RAW_LR)
    history = []
    for step in range(1, RECOVERY_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered, jacobian = {}, {}
        for side in ("left", "right"):
            rendered[side], jacobian[side] = activity_and_raw_derivative(
                raw, side, centers, views_by_side, gaussians, pipe, hyper,
                background, iteration)
        left = rendered["left"].clone().requires_grad_(True)
        right = rendered["right"].clone().requires_grad_(True)
        loss = -pearson_tensor(
            torch.cat((left, right)),
            torch.cat((targets[arm]["left"], targets[arm]["right"])))
        left_weight, right_weight = torch.autograd.grad(loss, (left, right))
        raw_gradient = (torch.sum(left_weight * jacobian["left"])
                        + torch.sum(right_weight * jacobian["right"]))
        require(bool(torch.isfinite(raw_gradient)),
                f"Non-finite recovery gradient: {arm} {step}")
        raw.grad = raw_gradient.detach().clone()
        delta_before = float((rgbgate.OFFSET_BOUND * torch.tanh(raw)).item())
        optimizer.step()
        delta_after = float((rgbgate.OFFSET_BOUND * torch.tanh(raw)).item())
        if step == 1 or step % 10 == 0 or step == RECOVERY_STEPS:
            row = {"arm": arm, "step": step, "loss": float(loss.item()),
                   "raw_gradient": float(raw_gradient.item()),
                   "delta_before": delta_before, "delta_after": delta_after}
            history.append(row)
            print("MIND_EQUIVARIANT_RECOVERY_STEP " + json.dumps(
                row, sort_keys=True), flush=True)

    final_delta = float((rgbgate.OFFSET_BOUND * torch.tanh(raw)).item())
    with torch.no_grad():
        final_rendered = {
            side: render_activity_sequence(
                side, centers, final_delta, views_by_side, gaussians, pipe,
                hyper, background, iteration).detach()
            for side in ("left", "right")}
        left_loss = -pearson_tensor(
            final_rendered["left"], targets[arm]["left"])
        right_loss = -pearson_tensor(
            final_rendered["right"], targets[arm]["right"])
        all_loss = -pearson_tensor(
            torch.cat((final_rendered["left"], final_rendered["right"])),
            torch.cat((targets[arm]["left"], targets[arm]["right"])))
        final_losses = {"all": float(all_loss.item()),
                        "left": float(left_loss.item()),
                        "right": float(right_loss.item())}
    return {"arm": arm, "truth_shift_frames": ARM_SHIFTS[arm],
            "final_raw": float(raw.detach().item()),
            "recovered_delta_frames": final_delta,
            "final_losses": final_losses, "history": history}


def optimize_detrended_shared_arm(
        arm, targets, centers, views_by_side, gaussians, pipe, hyper,
        background, iteration):
    raw = torch.nn.Parameter(torch.zeros((), device="cuda"))
    optimizer = torch.optim.Adam([raw], lr=RAW_LR)
    history = []
    for step in range(1, RECOVERY_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        rendered, jacobian = {}, {}
        for side in ("left", "right"):
            rendered[side], jacobian[side] = activity_and_raw_derivative(
                raw, side, centers, views_by_side, gaussians, pipe, hyper,
                background, iteration)
        leaves = {
            side: rendered[side].clone().requires_grad_(True)
            for side in ("left", "right")}
        loss = combined_detrended_multiscale_loss(
            leaves, targets, arm, "all")
        weights = torch.autograd.grad(
            loss, (leaves["left"], leaves["right"]))
        raw_gradient = sum(
            weight.mul(jacobian[side]).sum()
            for weight, side in zip(weights, ("left", "right")))
        require(bool(torch.isfinite(loss))
                and bool(torch.isfinite(raw_gradient)),
                f"Non-finite shared recovery state: {arm} {step}")
        raw.grad = raw_gradient.detach().clone()
        delta_before = float(
            (rgbgate.OFFSET_BOUND * torch.tanh(raw)).item())
        optimizer.step()
        delta_after = float(
            (rgbgate.OFFSET_BOUND * torch.tanh(raw)).item())
        if step == 1 or step % 10 == 0 or step == RECOVERY_STEPS:
            row = {
                "arm": arm, "step": step, "loss": float(loss.item()),
                "raw_gradient": float(raw_gradient.item()),
                "delta_before": delta_before, "delta_after": delta_after,
            }
            history.append(row)
            print("MIND_C1_DETRENDED_SHARED_RECOVERY_STEP " + json.dumps(
                row, sort_keys=True), flush=True)

    final_delta = float((rgbgate.OFFSET_BOUND * torch.tanh(raw)).item())
    with torch.no_grad():
        final_rendered = {
            side: render_activity_sequence(
                side, centers, final_delta, views_by_side, gaussians, pipe,
                hyper, background, iteration).detach()
            for side in ("left", "right")}
        final_losses = {
            scope: float(combined_detrended_multiscale_loss(
                final_rendered, targets, arm, scope).item())
            for scope in ("all", "left", "right")}
    return {
        "arm": arm, "truth_shift_frames": ARM_SHIFTS[arm],
        "initial_raw": 0.0, "final_raw": float(raw.detach().item()),
        "recovered_delta_frames": final_delta,
        "final_losses": final_losses, "history": history,
    }


def load_thermal_observation(side, frame):
    path = THERMAL_IMAGE_ROOT / f"covers_{side}_thermal_{frame:04d}.png"
    require(path.is_file(), f"Missing locked Thermal observation: {path}")
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)) / 255.0
    require(tensor.ndim == 3 and tensor.shape[0] == 3
            and bool(torch.isfinite(tensor).all()),
            f"Invalid Thermal observation: {path}")
    return tensor


def build_symmetric_centers(views_by_side):
    centers = {}
    for side, by_frame in views_by_side.items():
        frame_set = set(by_frame)
        values = [
            frame for frame in sorted(by_frame)
            if all(frame + shift + endpoint in frame_set
                   for shift in ARM_SHIFTS.values()
                   for endpoint in (-HALF_WINDOW, 0, HALF_WINDOW))]
        require(len(values) >= 100,
                f"Insufficient symmetric common centers: {side}")
        centers[side] = values
    return centers


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
    parser.add_argument("--target-modality", choices=("rgb", "thermal"),
                        required=True)
    parser.add_argument("--rgb-profile-report", type=Path, required=True)
    parser.add_argument("--thermal-profile-report", type=Path, required=True)
    parser.add_argument("--gradient-report", type=Path, required=True)
    parser.add_argument("--landscape-report", type=Path)
    parser.add_argument("--directional-report", type=Path)
    parser.add_argument("--transport-report", type=Path)
    parser.add_argument("--multiscale-report", type=Path)
    parser.add_argument("--coarse-multiscale-report", type=Path)
    parser.add_argument("--derivative-audit-report", type=Path)
    parser.add_argument("--c1-derivative-audit-report", type=Path)
    parser.add_argument("--detrended-report", type=Path)
    parser.add_argument("--gcc-phat-report", type=Path)
    parser.add_argument("--temporal-derivative-report", type=Path)
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
    require(str(dataset.source_path) == EXPECTED_SOURCE,
            "Dataset scope is not the locked Covers scene")
    require(args.target_modality == "thermal",
            "Equivariant recovery is locked to Thermal observations")
    for path, modality in ((args.rgb_profile_report, "rgb"),
                           (args.thermal_profile_report, "thermal")):
        require(path.is_file(), f"Missing v4 profile report: {path}")
        profile = json.loads(path.read_text(encoding="utf-8"))
        require(profile.get("schema") == "covers_mind_symmetric_activity_profile_v4"
                and profile.get("status") == "MIND_SYMMETRIC_ACTIVITY_PROFILE_PASS"
                and profile.get("target_modality") == modality
                and profile.get("teacher_hashes") == EXPECTED_TEACHER,
                f"Invalid v4 profile prerequisite: {modality}")
    require(args.gradient_report.is_file(),
            f"Missing v5 gradient report: {args.gradient_report}")
    gradient_report = json.loads(
        args.gradient_report.read_text(encoding="utf-8"))
    gradient_rows = {
        row.get("arm"): row for row in gradient_report.get("rows", [])}
    require(
        gradient_report.get("schema")
        == "covers_mind_symmetric_activity_gradient_v5"
        and gradient_report.get("status")
        == "MIND_SYMMETRIC_ACTIVITY_GRADIENT_FAIL"
        and gradient_report.get("target_modality") == "thermal"
        and gradient_report.get("teacher_hashes") == EXPECTED_TEACHER
        and set(gradient_rows) == set(ARM_SHIFTS)
        and gradient_rows["N"].get("pass") is True
        and gradient_rows["Z"].get("pass") is False
        and gradient_rows["P"].get("pass") is True,
        "Invalid v5 gradient prerequisite")
    if DIRECTIONAL_GATE_ONLY:
        require(args.landscape_report is not None
                and args.landscape_report.is_file(),
                "Missing v8 landscape report")
        landscape_prerequisite = json.loads(
            args.landscape_report.read_text(encoding="utf-8"))
        require(
            landscape_prerequisite.get("schema")
            == "covers_mind_p_landscape_v8"
            and landscape_prerequisite.get("status")
            == "MIND_P_LANDSCAPE_COMPLETE"
            and landscape_prerequisite.get("grid_winners", {}).get("all")
            == 9.0
            and landscape_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid v8 landscape prerequisite")
    if TRANSPORT_GATE_ONLY:
        require(args.directional_report is not None
                and args.directional_report.is_file(),
                "Missing v9 directional report")
        directional_prerequisite = json.loads(
            args.directional_report.read_text(encoding="utf-8"))
        require(
            directional_prerequisite.get("schema")
            == "covers_mind_directional_gate_v9"
            and directional_prerequisite.get("status")
            == "MIND_DIRECTIONAL_GATE_FAIL"
            and directional_prerequisite.get("profile_pass") is True
            and directional_prerequisite.get("gradient_pass") is False
            and directional_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid v9 directional prerequisite")
    if MULTISCALE_GATE_ONLY:
        require(args.transport_report is not None
                and args.transport_report.is_file(),
                "Missing v10 transport report")
        transport_prerequisite = json.loads(
            args.transport_report.read_text(encoding="utf-8"))
        require(
            transport_prerequisite.get("schema")
            == "covers_mind_transport_gate_v10"
            and transport_prerequisite.get("status")
            == "MIND_TRANSPORT_GATE_FAIL"
            and transport_prerequisite.get("profile_pass") is False
            and transport_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid v10 transport prerequisite")
    if COARSE_MULTISCALE_GATE_ONLY:
        require(args.multiscale_report is not None
                and args.multiscale_report.is_file(),
                "Missing v11 multiscale report")
        multiscale_prerequisite = json.loads(
            args.multiscale_report.read_text(encoding="utf-8"))
        require(
            multiscale_prerequisite.get("schema")
            == "covers_mind_multiscale_gate_v11"
            and multiscale_prerequisite.get("status")
            == "MIND_MULTISCALE_GATE_FAIL"
            and multiscale_prerequisite.get("gradient_pass") is False
            and multiscale_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid v11 multiscale prerequisite")
    if COARSE_DERIVATIVE_AUDIT_ONLY:
        require(args.coarse_multiscale_report is not None
                and args.coarse_multiscale_report.is_file(),
                "Missing v12 coarse multiscale report")
        coarse_prerequisite = json.loads(
            args.coarse_multiscale_report.read_text(encoding="utf-8"))
        require(
            coarse_prerequisite.get("schema")
            == "covers_mind_coarse_multiscale_gate_v12"
            and coarse_prerequisite.get("status")
            == "MIND_COARSE_MULTISCALE_GATE_FAIL"
            and coarse_prerequisite.get("profile_pass") is False
            and coarse_prerequisite.get("gradient_pass") is False
            and coarse_prerequisite.get(
                "multiscale_kernels_center_samples") == [9, 5, 3]
            and coarse_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid v12 coarse multiscale prerequisite")
    if C1_DERIVATIVE_AUDIT_ONLY:
        require(args.derivative_audit_report is not None
                and args.derivative_audit_report.is_file(),
                "Missing v13 derivative audit report")
        derivative_prerequisite = json.loads(
            args.derivative_audit_report.read_text(encoding="utf-8"))
        require(
            derivative_prerequisite.get("schema")
            == "covers_mind_coarse_derivative_audit_v13"
            and derivative_prerequisite.get("status")
            == "MIND_COARSE_DERIVATIVE_AUDIT_MISMATCH"
            and derivative_prerequisite.get("derivative_converged") is False
            and derivative_prerequisite.get(
                "multiscale_kernels_center_samples") == [9, 5, 3]
            and derivative_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid v13 derivative audit prerequisite")
    if DETREND_C1_MULTISCALE_GATE_ONLY:
        require(args.c1_derivative_audit_report is not None
                and args.c1_derivative_audit_report.is_file(),
                "Missing v15 C1 derivative audit report")
        c1_prerequisite = json.loads(
            args.c1_derivative_audit_report.read_text(encoding="utf-8"))
        require(
            c1_prerequisite.get("schema")
            == "covers_mind_c1_derivative_audit_v15"
            and c1_prerequisite.get("status")
            == "MIND_C1_DERIVATIVE_AUDIT_MISMATCH"
            and c1_prerequisite.get("derivative_converged") is False
            and c1_prerequisite.get("integer_pose_max_abs_error") == 0.0
            and c1_prerequisite.get("teacher_hashes") == EXPECTED_TEACHER,
            "Invalid v15 C1 derivative audit prerequisite")
    if C1_PHASE_GATE_ONLY:
        require(args.detrended_report is not None
                and args.detrended_report.is_file(),
                "Missing v16 detrended Gate report")
        detrended_prerequisite = json.loads(
            args.detrended_report.read_text(encoding="utf-8"))
        require(
            detrended_prerequisite.get("schema")
            == "covers_mind_c1_detrended_multiscale_gate_v16"
            and detrended_prerequisite.get("status")
            == "MIND_C1_DETRENDED_MULTISCALE_GATE_FAIL"
            and detrended_prerequisite.get("profile_pass") is False
            and detrended_prerequisite.get("gradient_pass") is False
            and detrended_prerequisite.get("integer_pose_max_abs_error") == 0.0
            and detrended_prerequisite.get("scene") == "Covers"
            and detrended_prerequisite.get("scene_loader_rgb_only") is True
            and detrended_prerequisite.get(
                "formal_test_cameras_constructed") is False
            and detrended_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid v16 detrended Gate prerequisite")
    if C1_TEMPORAL_DERIVATIVE_GATE_ONLY:
        require(args.gcc_phat_report is not None
                and args.gcc_phat_report.is_file(),
                "Missing v17 GCC-PHAT Gate report")
        phase_prerequisite = json.loads(
            args.gcc_phat_report.read_text(encoding="utf-8"))
        require(
            phase_prerequisite.get("schema")
            == "covers_mind_c1_gcc_phat_gate_v17"
            and phase_prerequisite.get("status")
            == "MIND_C1_GCC_PHAT_GATE_FAIL"
            and phase_prerequisite.get("profile_pass") is False
            and phase_prerequisite.get("gradient_pass") is False
            and phase_prerequisite.get("integer_pose_max_abs_error") == 0.0
            and phase_prerequisite.get("scene") == "Covers"
            and phase_prerequisite.get("scene_loader_rgb_only") is True
            and phase_prerequisite.get(
                "formal_test_cameras_constructed") is False
            and phase_prerequisite.get("teacher_hashes") == EXPECTED_TEACHER,
            "Invalid v17 GCC-PHAT Gate prerequisite")
    if C1_DETRENDED_SHARED_RECOVERY_ONLY:
        require(args.detrended_report is not None
                and args.detrended_report.is_file(),
                "Missing v16 detrended Gate report")
        require(args.temporal_derivative_report is not None
                and args.temporal_derivative_report.is_file(),
                "Missing v18 temporal-derivative Gate report")
        detrended_recovery_prerequisite = json.loads(
            args.detrended_report.read_text(encoding="utf-8"))
        temporal_derivative_prerequisite = json.loads(
            args.temporal_derivative_report.read_text(encoding="utf-8"))
        require(
            detrended_recovery_prerequisite.get("schema")
            == "covers_mind_c1_detrended_multiscale_gate_v16"
            and detrended_recovery_prerequisite.get("status")
            == "MIND_C1_DETRENDED_MULTISCALE_GATE_FAIL"
            and detrended_recovery_prerequisite.get(
                "profile_scopes", {}).get("all", {}).get("pass") is True
            and detrended_recovery_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER
            and temporal_derivative_prerequisite.get("schema")
            == "covers_mind_c1_temporal_derivative_gate_v18"
            and temporal_derivative_prerequisite.get("status")
            == "MIND_C1_TEMPORAL_DERIVATIVE_GATE_FAIL"
            and temporal_derivative_prerequisite.get(
                "profile_scopes", {}).get("all", {}).get("pass") is True
            and temporal_derivative_prerequisite.get("teacher_hashes")
            == EXPECTED_TEACHER,
            "Invalid shared recovery prerequisites")
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
            "Formal cameras were constructed")
    require(getattr(gaussians, "optimizer", None) is None,
            "Gaussian optimizer exists")

    rasterizer_path = Path(diff_gaussian_rasterization._C.__file__).resolve()
    wrapper_path = Path(diff_gaussian_rasterization.__file__).resolve()
    deformation_path = Path(deformation_module.__file__).resolve()
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
    require(teacher_hashes == EXPECTED_TEACHER, "Teacher artifact mismatch")
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
        require(view.thermal_image is None, "Thermal image entered Scene")
        require(not view.has_thermal, "Thermal camera path entered Scene")
        expected_time = float(frame) / float(hyper.total_num_frames)
        require(math.isclose(float(view.time), expected_time,
                             rel_tol=0.0, abs_tol=1e-7),
                f"Timestamp/frame mismatch: {side} {frame}")
        views_by_side[side][frame] = view

    configured_raw = torch.nn.Parameter(
        torch.zeros((), device="cuda"), requires_grad=False)
    rgbgate.configure_views(
        views_by_side, configured_raw, hyper.total_num_frames)
    c1_integer_pose_max_abs_error = None
    if (C1_DERIVATIVE_AUDIT_ONLY
            or DETREND_C1_MULTISCALE_GATE_ONLY
            or C1_PHASE_GATE_ONLY
            or C1_TEMPORAL_DERIVATIVE_GATE_ONLY
            or C1_DETRENDED_SHARED_RECOVERY_ONLY):
        pose_errors = []
        for side in ("left", "right"):
            for frame, view in views_by_side[side].items():
                rendered_pose = view.get_temporal_rgb_world_view_transform()
                frame_tensor = torch.as_tensor(
                    float(frame), device=view.temporal_pose_frames.device,
                    dtype=view.temporal_pose_frames.dtype)
                index = int(torch.searchsorted(
                    view.temporal_pose_frames, frame_tensor).item())
                require(index < view.temporal_pose_frames.shape[0]
                        and bool((view.temporal_pose_frames[index]
                                  == frame_tensor).item()),
                        f"Missing exact pose knot: {side} {frame}")
                pose_errors.append(float((
                    rendered_pose - view.temporal_pose_track[index]
                ).abs().max().item()))
        c1_integer_pose_max_abs_error = max(pose_errors)
        require(c1_integer_pose_max_abs_error == 0.0,
                "C1 interpolation changed an integer camera pose")
        print("MIND_C1_INTEGER_POSE_AUDIT " + json.dumps({
            "camera_count": len(pose_errors),
            "max_abs_error": c1_integer_pose_max_abs_error,
        }, sort_keys=True), flush=True)
    frozen_tensors = rgbgate.freeze_rgb_state(gaussians, views)
    centers = build_symmetric_centers(views_by_side)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background
        else [0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    nuisance_before, nuisance_count = base.stable_tensor_hash(
        frozen_tensors)
    camera_before, camera_hash_before = base.camera_scalar_manifest(views)

    observed = {"left": {}, "right": {}}
    needed_frames = {
        side: sorted({frame + shift + endpoint
                      for frame in centers[side]
                      for shift in ARM_SHIFTS.values()
                      for endpoint in (-HALF_WINDOW, 0, HALF_WINDOW)})
        for side in ("left", "right")
    }
    with torch.no_grad():
        for side in ("left", "right"):
            for frame in needed_frames[side]:
                view = views_by_side[side][frame]
                image = (view.original_image if args.target_modality == "rgb"
                         else load_thermal_observation(side, frame))
                observed[side][frame] = mind_descriptor(
                    image.to("cuda")).detach()

    targets = {arm: {} for arm in ARM_SHIFTS}
    directional_targets = {arm: {} for arm in ARM_SHIFTS}
    for arm, shift in ARM_SHIFTS.items():
        for side in ("left", "right"):
            values = []
            for frame in centers[side]:
                backward = (
                    observed[side][frame + shift]
                    - observed[side][frame - HALF_WINDOW + shift]
                ).square().mean()
                forward = (
                    observed[side][frame + HALF_WINDOW + shift]
                    - observed[side][frame + shift]
                ).square().mean()
                values.append(float((0.5 * (backward + forward)).item()))
            targets[arm][side] = torch.tensor(
                values, device="cuda", dtype=torch.float32)
            symmetric_values, directional_values = [], []
            for frame in centers[side]:
                backward = (
                    observed[side][frame + shift]
                    - observed[side][frame - HALF_WINDOW + shift]
                ).square().mean()
                forward = (
                    observed[side][frame + HALF_WINDOW + shift]
                    - observed[side][frame + shift]
                ).square().mean()
                symmetric_values.append(float(
                    (0.5 * (backward + forward)).item()))
                directional_values.append(float((forward - backward).item()))
            directional_targets[arm][side] = {
                "symmetric": torch.tensor(
                    symmetric_values, device="cuda", dtype=torch.float32),
                "directional": torch.tensor(
                    directional_values, device="cuda", dtype=torch.float32),
            }

    if C1_DETRENDED_SHARED_RECOVERY_ONLY:
        started = time.time()
        recoveries = [
            optimize_detrended_shared_arm(
                arm, targets, centers, views_by_side, gaussians, pipe, hyper,
                background, args.iteration)
            for arm in ARM_SHIFTS]
        recovered = {
            row["arm"]: row["recovered_delta_frames"]
            for row in recoveries}
        negative_relative = recovered["N"] - recovered["Z"]
        positive_relative = recovered["P"] - recovered["Z"]
        negative_residual = negative_relative - ARM_SHIFTS["N"]
        positive_residual = positive_relative - ARM_SHIFTS["P"]
        records_finite = all(
            row["initial_raw"] == 0.0
            and len(row["history"]) == 7
            and math.isfinite(row["final_raw"])
            and math.isfinite(row["recovered_delta_frames"])
            and all(math.isfinite(value)
                    for value in row["final_losses"].values())
            for row in recoveries)
        nuisance_after, nuisance_after_count = base.stable_tensor_hash(
            rgbgate.freeze_rgb_state(gaussians, views))
        camera_after, camera_hash_after = base.camera_scalar_manifest(views)
        require(nuisance_count == nuisance_after_count
                and nuisance_before == nuisance_after,
                "Frozen Scene state changed")
        require(camera_before == camera_after
                and camera_hash_before == camera_hash_after,
                "Camera scalar state changed")
        gate_pass = (
            records_finite
            and recovered["N"] < recovered["Z"] < recovered["P"]
            and abs(negative_residual) <= EQUIVARIANCE_TOLERANCE
            and abs(positive_residual) <= EQUIVARIANCE_TOLERANCE)
        status = (
            "MIND_C1_DETRENDED_SHARED_RECOVERY_PASS" if gate_pass
            else "MIND_C1_DETRENDED_SHARED_RECOVERY_FAIL")
        report = {
            "schema": SCHEMA, "status": status, "scene": "Covers",
            "per_scene_optimization": True,
            "shared_scene_clock_across_cameras": True,
            "single_camera_profiles_used_as_diagnostics_only": True,
            "target_modality": args.target_modality,
            "source_path": str(dataset.source_path),
            "external_data_loaded": False,
            "scene_loader_rgb_only": True,
            "thermal_observation_loader": str(THERMAL_IMAGE_ROOT),
            "trainable_parameter_count": 1,
            "trainable_parameter": "one raw scalar per independent arm; delta=20*tanh(raw)",
            "each_arm_restarts_from_raw_zero": True,
            "candidate_enumeration_in_optimizer": False,
            "optimizer": "Adam", "raw_learning_rate": RAW_LR,
            "steps_per_arm": RECOVERY_STEPS,
            "camera_pose_interpolation": (
                "C1 cubic Hermite translation and hemisphere-aligned normalized quaternion components"),
            "integer_pose_max_abs_error": c1_integer_pose_max_abs_error,
            "observable": (
                "closed-form affine-detrended equal-weight Pearson correlation over fixed temporal smoothing scales"),
            "multiscale_kernels_center_samples": list(MULTISCALE_KERNELS),
            "recoveries": recoveries,
            "recovered_delta_frames": recovered,
            "negative_relative_frames": negative_relative,
            "positive_relative_frames": positive_relative,
            "negative_residual_frames": negative_residual,
            "positive_residual_frames": positive_residual,
            "tolerance_frames": EQUIVARIANCE_TOLERANCE,
            "gate_pass": gate_pass,
            "centers_by_side": centers,
            "nuisance_sha256_before": nuisance_before,
            "nuisance_sha256_after": nuisance_after,
            "nuisance_tensor_count": nuisance_count,
            "camera_manifest_sha256_before": camera_hash_before,
            "camera_manifest_sha256_after": camera_hash_after,
            "formal_test_cameras_constructed": False,
            "teacher_hashes": teacher_hashes,
            "v16_detrended_report": str(args.detrended_report),
            "v18_temporal_derivative_report": str(
                args.temporal_derivative_report),
            "entry_script_path": str(ENTRY_SCRIPT_PATH),
            "script_sha256": base.sha256_file(ENTRY_SCRIPT_PATH),
            "implementation_sha256": base.sha256_file(__file__),
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
        print("MIND_C1_DETRENDED_SHARED_RECOVERY_RESULT " + json.dumps({
            "status": status,
            "negative_relative_frames": negative_relative,
            "positive_relative_frames": positive_relative,
        }, sort_keys=True), flush=True)
        return

    if COARSE_DERIVATIVE_AUDIT_ONLY:
        started = time.time()
        derivative_log_prefix = (
            "MIND_C1_DERIVATIVE" if C1_DERIVATIVE_AUDIT_ONLY
            else "MIND_COARSE_DERIVATIVE")
        audit_deltas = tuple(sorted({0.0} | {
            sign * step
            for step in DERIVATIVE_AUDIT_STEPS
            for sign in (-1.0, 1.0)}))
        loss_values = {}
        with torch.no_grad():
            for delta in audit_deltas:
                rendered = {
                    side: render_activity_sequence(
                        side, centers, delta, views_by_side, gaussians, pipe,
                        hyper, background, args.iteration).detach()
                    for side in ("left", "right")}
                loss_values[delta] = {
                    arm: {
                        scope: float(combined_multiscale_loss(
                            rendered, targets, arm, scope).item())
                        for scope in ("all", "left", "right")}
                    for arm in ARM_SHIFTS}
                print(derivative_log_prefix + "_LOSS_POINT " + json.dumps({
                    "delta_frames": delta, "losses": loss_values[delta],
                }, sort_keys=True), flush=True)

        raw = torch.nn.Parameter(torch.zeros((), device="cuda"))
        zero_values, zero_jacobians = {}, {}
        for side in ("left", "right"):
            zero_values[side], zero_jacobians[side] = (
                activity_and_raw_derivative(
                    raw, side, centers, views_by_side, gaussians, pipe, hyper,
                    background, args.iteration))
        rows = []
        smallest_steps = DERIVATIVE_AUDIT_STEPS[-2:]
        for arm in ARM_SHIFTS:
            scopes = {}
            for scope in ("all", "left", "right"):
                sides = ("left", "right") if scope == "all" else (scope,)
                leaves = {
                    side: zero_values[side].detach().clone().requires_grad_(True)
                    for side in sides}
                loss = combined_multiscale_loss(leaves, targets, arm, scope)
                weights = torch.autograd.grad(loss, tuple(leaves.values()))
                raw_gradient = sum(
                    weight.mul(zero_jacobians[side]).sum()
                    for weight, side in zip(weights, sides))
                auto = float((raw_gradient / rgbgate.OFFSET_BOUND).item())
                finite_differences = {
                    str(step): (
                        loss_values[step][arm][scope]
                        - loss_values[-step][arm][scope]) / (2.0 * step)
                    for step in DERIVATIVE_AUDIT_STEPS}
                sign_matches = {
                    str(step): auto * finite_differences[str(step)] > 0.0
                    for step in DERIVATIVE_AUDIT_STEPS}
                smallest_sign_match = all(
                    sign_matches[str(step)] for step in smallest_steps)
                scopes[scope] = {
                    "autograd_dloss_ddelta": auto,
                    "central_fd": finite_differences,
                    "sign_match": sign_matches,
                    "smallest_two_sign_match": smallest_sign_match,
                }
            row = {
                "arm": arm, "scopes": scopes,
                "smallest_two_sign_match": all(
                    value["smallest_two_sign_match"]
                    for value in scopes.values()),
            }
            rows.append(row)
            print(derivative_log_prefix + "_ROW " + json.dumps(
                row, sort_keys=True), flush=True)

        nuisance_after, nuisance_after_count = base.stable_tensor_hash(
            rgbgate.freeze_rgb_state(gaussians, views))
        camera_after, camera_hash_after = base.camera_scalar_manifest(views)
        require(nuisance_count == nuisance_after_count
                and nuisance_before == nuisance_after,
                "Frozen Scene state changed")
        require(camera_before == camera_after
                and camera_hash_before == camera_hash_after,
                "Camera scalar state changed")
        derivative_converged = all(
            row["smallest_two_sign_match"] for row in rows)
        if C1_DERIVATIVE_AUDIT_ONLY:
            status = (
                "MIND_C1_DERIVATIVE_AUDIT_CONVERGED"
                if derivative_converged
                else "MIND_C1_DERIVATIVE_AUDIT_MISMATCH")
        else:
            status = (
                "MIND_COARSE_DERIVATIVE_AUDIT_CONVERGED"
                if derivative_converged
                else "MIND_COARSE_DERIVATIVE_AUDIT_MISMATCH")
        report = {
            "schema": SCHEMA, "status": status, "scene": "Covers",
            "per_scene_optimization": True, "audit_only": True,
            "target_modality": args.target_modality,
            "source_path": str(dataset.source_path),
            "external_data_loaded": False,
            "scene_loader_rgb_only": True,
            "thermal_observation_loader": str(THERMAL_IMAGE_ROOT),
            "trainable_parameter_count": 1,
            "trainable_parameter": "one raw scalar; delta=20*tanh(raw)",
            "optimizer_constructed": False,
            "candidate_enumeration_in_optimizer": False,
            "camera_pose_interpolation": (
                "C1 cubic Hermite translation and hemisphere-aligned normalized quaternion components"
                if C1_DERIVATIVE_AUDIT_ONLY
                else "original piecewise linear/nlerp"),
            "integer_pose_max_abs_error": c1_integer_pose_max_abs_error,
            "multiscale_kernels_center_samples": list(MULTISCALE_KERNELS),
            "finite_difference_steps_frames": list(DERIVATIVE_AUDIT_STEPS),
            "smallest_two_steps_frames": list(smallest_steps),
            "derivative_converged": derivative_converged,
            "rows": rows,
            "loss_values": {str(delta): value
                            for delta, value in loss_values.items()},
            "centers_by_side": centers,
            "nuisance_sha256_before": nuisance_before,
            "nuisance_sha256_after": nuisance_after,
            "nuisance_tensor_count": nuisance_count,
            "camera_manifest_sha256_before": camera_hash_before,
            "camera_manifest_sha256_after": camera_hash_after,
            "formal_test_cameras_constructed": False,
            "teacher_hashes": teacher_hashes,
            "v12_coarse_multiscale_report": str(
                args.coarse_multiscale_report),
            "v13_derivative_audit_report": (
                str(args.derivative_audit_report)
                if C1_DERIVATIVE_AUDIT_ONLY else None),
            "entry_script_path": str(ENTRY_SCRIPT_PATH),
            "script_sha256": base.sha256_file(ENTRY_SCRIPT_PATH),
            "implementation_sha256": base.sha256_file(__file__),
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
        print(derivative_log_prefix + "_AUDIT_RESULT " + json.dumps({
            "status": status, "derivative_converged": derivative_converged,
        }, sort_keys=True), flush=True)
        return

    if DIRECTIONAL_GATE_ONLY:
        started = time.time()
        loss_values = {}
        with torch.no_grad():
            for delta in DIRECTIONAL_EVAL_DELTAS:
                rendered = {
                    side: directional_sequences(
                        side, centers, delta, views_by_side, gaussians, pipe,
                        hyper, background, args.iteration)
                    for side in ("left", "right")}
                loss_values[delta] = {
                    arm: {
                        scope: float(combined_directional_loss(
                            rendered, directional_targets, arm, scope).item())
                        for scope in ("all", "left", "right")}
                    for arm in ARM_SHIFTS}
                print("MIND_DIRECTIONAL_LOSS_POINT " + json.dumps({
                    "delta_frames": delta, "losses": loss_values[delta],
                }, sort_keys=True), flush=True)

        profile_winners, profile_margins, profile_rows = {}, {}, []
        for arm in ARM_SHIFTS:
            profile_winners[arm] = {}
            profile_margins[arm] = {}
            for scope in ("all", "left", "right"):
                ranking = sorted(
                    DIRECTIONAL_PROFILE_DELTAS,
                    key=lambda delta: loss_values[delta][arm][scope])
                profile_winners[arm][scope] = ranking[0]
                profile_margins[arm][scope] = (
                    loss_values[ranking[1]][arm][scope]
                    - loss_values[ranking[0]][arm][scope])
            profile_rows.append({
                "arm": arm, "truth_shift_frames": ARM_SHIFTS[arm],
                "winners": profile_winners[arm],
                "winner_margins": profile_margins[arm],
            })
        profile_scopes = {}
        for scope in ("all", "left", "right"):
            d_n = profile_winners["N"][scope]
            d_z = profile_winners["Z"][scope]
            d_p = profile_winners["P"][scope]
            negative_relative = d_n - d_z
            positive_relative = d_p - d_z
            passed = (
                d_n < d_z < d_p
                and all(profile_margins[arm][scope] > 0.0
                        for arm in ARM_SHIFTS)
                and abs(negative_relative + 8.0)
                <= EQUIVARIANCE_TOLERANCE
                and abs(positive_relative - 8.0)
                <= EQUIVARIANCE_TOLERANCE)
            profile_scopes[scope] = {
                "negative_relative_frames": negative_relative,
                "positive_relative_frames": positive_relative,
                "pass": passed,
            }
        profile_pass = all(
            row["pass"] for row in profile_scopes.values())

        raw = torch.nn.Parameter(torch.zeros((), device="cuda"))
        zero_values, zero_jacobians = {}, {}
        for side in ("left", "right"):
            zero_values[side], zero_jacobians[side] = (
                directional_sequences_and_raw_derivatives(
                    raw, side, centers, views_by_side, gaussians, pipe, hyper,
                    background, args.iteration))
        gradient_rows = []
        for arm in ARM_SHIFTS:
            scopes = {}
            for scope in ("all", "left", "right"):
                sides = ("left", "right") if scope == "all" else (scope,)
                leaves, jacobians, target_components = {}, {}, {}
                for component in ("symmetric", "directional"):
                    leaves[component] = torch.cat([
                        zero_values[side][component] for side in sides
                    ]).detach().clone().requires_grad_(True)
                    jacobians[component] = torch.cat([
                        zero_jacobians[side][component] for side in sides])
                    target_components[component] = torch.cat([
                        directional_targets[arm][side][component]
                        for side in sides])
                placeholder_loss = 0.5 * sum(
                    -pearson_tensor(
                        leaves[component], target_components[component])
                    for component in ("symmetric", "directional"))
                weights = torch.autograd.grad(
                    placeholder_loss,
                    (leaves["symmetric"], leaves["directional"]))
                raw_gradient = sum(
                    torch.sum(weight * jacobians[component])
                    for weight, component in zip(
                        weights, ("symmetric", "directional")))
                dloss_ddelta = float(
                    (raw_gradient / rgbgate.OFFSET_BOUND).item())
                central_fd = {
                    "1.0": (
                        loss_values[1.0][arm][scope]
                        - loss_values[-1.0][arm][scope]) / 2.0,
                    "0.5": (
                        loss_values[0.5][arm][scope]
                        - loss_values[-0.5][arm][scope]),
                }
                winner = profile_winners[arm][scope]
                if winner < 0.0:
                    passed = dloss_ddelta > 0.0 and all(
                        value > 0.0 for value in central_fd.values())
                elif winner > 0.0:
                    passed = dloss_ddelta < 0.0 and all(
                        value < 0.0 for value in central_fd.values())
                else:
                    left_near = (
                        loss_values[0.0][arm][scope]
                        - loss_values[-0.5][arm][scope]) / 0.5
                    right_near = (
                        loss_values[0.5][arm][scope]
                        - loss_values[0.0][arm][scope]) / 0.5
                    passed = (
                        loss_values[-1.0][arm][scope]
                        > loss_values[-0.5][arm][scope]
                        > loss_values[0.0][arm][scope]
                        and loss_values[1.0][arm][scope]
                        > loss_values[0.5][arm][scope]
                        > loss_values[0.0][arm][scope]
                        and left_near < 0.0 < right_near
                        and abs(dloss_ddelta)
                        < min(abs(left_near), abs(right_near)))
                scopes[scope] = {
                    "profile_winner_frames": winner,
                    "autograd_dloss_ddelta": dloss_ddelta,
                    "central_fd": central_fd, "pass": passed,
                }
            row = {"arm": arm, "scopes": scopes,
                   "pass": all(value["pass"]
                               for value in scopes.values())}
            gradient_rows.append(row)
            print("MIND_DIRECTIONAL_GRADIENT_ROW " + json.dumps(
                row, sort_keys=True), flush=True)

        nuisance_after, nuisance_after_count = base.stable_tensor_hash(
            rgbgate.freeze_rgb_state(gaussians, views))
        camera_after, camera_hash_after = base.camera_scalar_manifest(views)
        require(nuisance_count == nuisance_after_count
                and nuisance_before == nuisance_after,
                "Frozen Scene state changed")
        require(camera_before == camera_after
                and camera_hash_before == camera_hash_after,
                "Camera scalar state changed")
        gradient_pass = all(row["pass"] for row in gradient_rows)
        gate_pass = profile_pass and gradient_pass
        status = ("MIND_DIRECTIONAL_GATE_PASS" if gate_pass
                  else "MIND_DIRECTIONAL_GATE_FAIL")
        report = {
            "schema": SCHEMA, "status": status, "scene": "Covers",
            "per_scene_optimization": True, "gate_only": True,
            "target_modality": args.target_modality,
            "source_path": str(dataset.source_path),
            "external_data_loaded": False,
            "scene_loader_rgb_only": True,
            "thermal_observation_loader": str(THERMAL_IMAGE_ROOT),
            "trainable_parameter_count": 1,
            "trainable_parameter": "one raw scalar; delta=20*tanh(raw)",
            "candidate_enumeration_in_optimizer": False,
            "profile_scan_used_only_for_gate": True,
            "observable": "equal-weight Pearson loss of symmetric and time-antisymmetric MIND activity",
            "directional_component": "forward_energy-backward_energy",
            "directional_component_squared_or_abs": False,
            "profile_deltas_frames": list(DIRECTIONAL_PROFILE_DELTAS),
            "profile_winners": profile_winners,
            "profile_winner_margins": profile_margins,
            "profile_rows": profile_rows,
            "profile_scopes": profile_scopes,
            "profile_pass": profile_pass,
            "gradient_rows": gradient_rows,
            "gradient_pass": gradient_pass,
            "gate_pass": gate_pass,
            "centers_by_side": centers,
            "nuisance_sha256_before": nuisance_before,
            "nuisance_sha256_after": nuisance_after,
            "nuisance_tensor_count": nuisance_count,
            "camera_manifest_sha256_before": camera_hash_before,
            "camera_manifest_sha256_after": camera_hash_after,
            "formal_test_cameras_constructed": False,
            "teacher_hashes": teacher_hashes,
            "v8_landscape_report": str(args.landscape_report),
            "entry_script_path": str(ENTRY_SCRIPT_PATH),
            "script_sha256": base.sha256_file(ENTRY_SCRIPT_PATH),
            "implementation_sha256": base.sha256_file(__file__),
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
        print("MIND_DIRECTIONAL_GATE_RESULT " + json.dumps({
            "status": status, "profile_pass": profile_pass,
            "gradient_pass": gradient_pass,
        }, sort_keys=True), flush=True)
        return

    if (TRANSPORT_GATE_ONLY or MULTISCALE_GATE_ONLY
            or COARSE_MULTISCALE_GATE_ONLY
            or DETREND_C1_MULTISCALE_GATE_ONLY
            or C1_PHASE_GATE_ONLY
            or C1_TEMPORAL_DERIVATIVE_GATE_ONLY):
        started = time.time()
        loss_values = {}
        if C1_TEMPORAL_DERIVATIVE_GATE_ONLY:
            gate_loss = combined_temporal_derivative_multiscale_loss
        elif C1_PHASE_GATE_ONLY:
            gate_loss = combined_gcc_phat_loss
        elif DETREND_C1_MULTISCALE_GATE_ONLY:
            gate_loss = combined_detrended_multiscale_loss
        elif MULTISCALE_GATE_ONLY or COARSE_MULTISCALE_GATE_ONLY:
            gate_loss = combined_multiscale_loss
        else:
            gate_loss = combined_transport_loss
        if C1_TEMPORAL_DERIVATIVE_GATE_ONLY:
            log_prefix = "MIND_C1_TEMPORAL_DERIVATIVE"
        elif C1_PHASE_GATE_ONLY:
            log_prefix = "MIND_C1_GCC_PHAT"
        elif DETREND_C1_MULTISCALE_GATE_ONLY:
            log_prefix = "MIND_C1_DETRENDED_MULTISCALE"
        elif COARSE_MULTISCALE_GATE_ONLY:
            log_prefix = "MIND_COARSE_MULTISCALE"
        elif MULTISCALE_GATE_ONLY:
            log_prefix = "MIND_MULTISCALE"
        else:
            log_prefix = "MIND_TRANSPORT"
        with torch.no_grad():
            for delta in TRANSPORT_EVAL_DELTAS:
                rendered = {
                    side: render_activity_sequence(
                        side, centers, delta, views_by_side, gaussians, pipe,
                        hyper, background, args.iteration).detach()
                    for side in ("left", "right")}
                loss_values[delta] = {
                    arm: {
                        scope: float(gate_loss(
                            rendered, targets, arm, scope).item())
                        for scope in ("all", "left", "right")}
                    for arm in ARM_SHIFTS}
                print(log_prefix + "_LOSS_POINT " + json.dumps({
                    "delta_frames": delta, "losses": loss_values[delta],
                }, sort_keys=True), flush=True)

        profile_winners, profile_margins, profile_rows = {}, {}, []
        for arm in ARM_SHIFTS:
            profile_winners[arm] = {}
            profile_margins[arm] = {}
            for scope in ("all", "left", "right"):
                ranking = sorted(
                    TRANSPORT_PROFILE_DELTAS,
                    key=lambda delta: loss_values[delta][arm][scope])
                profile_winners[arm][scope] = ranking[0]
                profile_margins[arm][scope] = (
                    loss_values[ranking[1]][arm][scope]
                    - loss_values[ranking[0]][arm][scope])
            profile_rows.append({
                "arm": arm, "truth_shift_frames": ARM_SHIFTS[arm],
                "winners": profile_winners[arm],
                "winner_margins": profile_margins[arm],
            })
        profile_scopes = {}
        for scope in ("all", "left", "right"):
            d_n = profile_winners["N"][scope]
            d_z = profile_winners["Z"][scope]
            d_p = profile_winners["P"][scope]
            negative_relative = d_n - d_z
            positive_relative = d_p - d_z
            passed = (
                d_n < d_z < d_p
                and all(profile_margins[arm][scope] > 0.0
                        for arm in ARM_SHIFTS)
                and abs(negative_relative + 8.0)
                <= EQUIVARIANCE_TOLERANCE
                and abs(positive_relative - 8.0)
                <= EQUIVARIANCE_TOLERANCE)
            profile_scopes[scope] = {
                "negative_relative_frames": negative_relative,
                "positive_relative_frames": positive_relative,
                "pass": passed,
            }
        profile_pass = all(
            row["pass"] for row in profile_scopes.values())

        raw = torch.nn.Parameter(torch.zeros((), device="cuda"))
        zero_values, zero_jacobians = {}, {}
        for side in ("left", "right"):
            zero_values[side], zero_jacobians[side] = (
                activity_and_raw_derivative(
                    raw, side, centers, views_by_side, gaussians, pipe, hyper,
                    background, args.iteration))
        gradient_rows = []
        for arm in ARM_SHIFTS:
            scopes = {}
            for scope in ("all", "left", "right"):
                sides = ("left", "right") if scope == "all" else (scope,)
                leaves = {
                    side: zero_values[side].detach().clone().requires_grad_(True)
                    for side in sides}
                placeholder_loss = gate_loss(leaves, targets, arm, scope)
                weights = torch.autograd.grad(
                    placeholder_loss, tuple(leaves.values()))
                raw_gradient = sum(
                    weight.mul(zero_jacobians[side]).sum()
                    for weight, side in zip(weights, sides))
                dloss_ddelta = float(
                    (raw_gradient / rgbgate.OFFSET_BOUND).item())
                central_fd = {
                    "1.0": (
                        loss_values[1.0][arm][scope]
                        - loss_values[-1.0][arm][scope]) / 2.0,
                    "0.5": (
                        loss_values[0.5][arm][scope]
                        - loss_values[-0.5][arm][scope]),
                }
                winner = profile_winners[arm][scope]
                if winner < 0.0:
                    passed = dloss_ddelta > 0.0 and all(
                        value > 0.0 for value in central_fd.values())
                elif winner > 0.0:
                    passed = dloss_ddelta < 0.0 and all(
                        value < 0.0 for value in central_fd.values())
                else:
                    left_near = (
                        loss_values[0.0][arm][scope]
                        - loss_values[-0.5][arm][scope]) / 0.5
                    right_near = (
                        loss_values[0.5][arm][scope]
                        - loss_values[0.0][arm][scope]) / 0.5
                    passed = (
                        loss_values[-1.0][arm][scope]
                        > loss_values[-0.5][arm][scope]
                        > loss_values[0.0][arm][scope]
                        and loss_values[1.0][arm][scope]
                        > loss_values[0.5][arm][scope]
                        > loss_values[0.0][arm][scope]
                        and left_near < 0.0 < right_near
                        and abs(dloss_ddelta)
                        < min(abs(left_near), abs(right_near)))
                scopes[scope] = {
                    "profile_winner_frames": winner,
                    "autograd_dloss_ddelta": dloss_ddelta,
                    "central_fd": central_fd, "pass": passed,
                }
            row = {"arm": arm, "scopes": scopes,
                   "pass": all(value["pass"]
                               for value in scopes.values())}
            gradient_rows.append(row)
            print(log_prefix + "_GRADIENT_ROW " + json.dumps(
                row, sort_keys=True), flush=True)

        nuisance_after, nuisance_after_count = base.stable_tensor_hash(
            rgbgate.freeze_rgb_state(gaussians, views))
        camera_after, camera_hash_after = base.camera_scalar_manifest(views)
        require(nuisance_count == nuisance_after_count
                and nuisance_before == nuisance_after,
                "Frozen Scene state changed")
        require(camera_before == camera_after
                and camera_hash_before == camera_hash_after,
                "Camera scalar state changed")
        gradient_pass = all(row["pass"] for row in gradient_rows)
        gate_pass = profile_pass and gradient_pass
        if C1_TEMPORAL_DERIVATIVE_GATE_ONLY:
            status = (
                "MIND_C1_TEMPORAL_DERIVATIVE_GATE_PASS" if gate_pass
                else "MIND_C1_TEMPORAL_DERIVATIVE_GATE_FAIL")
        elif C1_PHASE_GATE_ONLY:
            status = (
                "MIND_C1_GCC_PHAT_GATE_PASS" if gate_pass
                else "MIND_C1_GCC_PHAT_GATE_FAIL")
        elif DETREND_C1_MULTISCALE_GATE_ONLY:
            status = (
                "MIND_C1_DETRENDED_MULTISCALE_GATE_PASS" if gate_pass
                else "MIND_C1_DETRENDED_MULTISCALE_GATE_FAIL")
        elif COARSE_MULTISCALE_GATE_ONLY:
            status = ("MIND_COARSE_MULTISCALE_GATE_PASS" if gate_pass
                      else "MIND_COARSE_MULTISCALE_GATE_FAIL")
        elif MULTISCALE_GATE_ONLY:
            status = ("MIND_MULTISCALE_GATE_PASS" if gate_pass
                      else "MIND_MULTISCALE_GATE_FAIL")
        else:
            status = ("MIND_TRANSPORT_GATE_PASS" if gate_pass
                      else "MIND_TRANSPORT_GATE_FAIL")
        report = {
            "schema": SCHEMA, "status": status, "scene": "Covers",
            "per_scene_optimization": True, "gate_only": True,
            "target_modality": args.target_modality,
            "source_path": str(dataset.source_path),
            "external_data_loaded": False,
            "scene_loader_rgb_only": True,
            "thermal_observation_loader": str(THERMAL_IMAGE_ROOT),
            "trainable_parameter_count": 1,
            "trainable_parameter": "one raw scalar; delta=20*tanh(raw)",
            "candidate_enumeration_in_optimizer": False,
            "profile_scan_used_only_for_gate": True,
            "observable": (
                "equal-weight Pearson of centered temporal derivatives after fixed smoothing scales"
                if C1_TEMPORAL_DERIVATIVE_GATE_ONLY else
                "per-side zero-lag GCC-PHAT of non-DC symmetric MIND activity"
                if C1_PHASE_GATE_ONLY else
                "closed-form affine-detrended equal-weight Pearson correlation over fixed temporal smoothing scales"
                if DETREND_C1_MULTISCALE_GATE_ONLY else
                "equal-weight Pearson correlation over fixed temporal smoothing scales"
                if (MULTISCALE_GATE_ONLY or COARSE_MULTISCALE_GATE_ONLY) else
                "per-side squared CDF distance of normalized nonnegative symmetric MIND activity"),
            "transport_hyperparameters": (
                None if TRANSPORT_GATE_ONLY else "not applicable"),
            "multiscale_kernels_center_samples": (
                list(MULTISCALE_KERNELS)
                if (MULTISCALE_GATE_ONLY or COARSE_MULTISCALE_GATE_ONLY
                    or DETREND_C1_MULTISCALE_GATE_ONLY
                    or C1_TEMPORAL_DERIVATIVE_GATE_ONLY)
                else None),
            "all_scope_definition": (
                "concatenated per-side centered temporal derivatives per smoothing scale"
                if C1_TEMPORAL_DERIVATIVE_GATE_ONLY else
                "equal mean of per-side GCC-PHAT losses"
                if C1_PHASE_GATE_ONLY else
                "concatenated left/right sequence per smoothing scale"
                if (MULTISCALE_GATE_ONLY or COARSE_MULTISCALE_GATE_ONLY
                    or DETREND_C1_MULTISCALE_GATE_ONLY) else
                "equal mean of left and right transport losses"),
            "affine_temporal_detrending": DETREND_C1_MULTISCALE_GATE_ONLY,
            "gcc_phat_non_dc_only": C1_PHASE_GATE_ONLY,
            "centered_temporal_derivative": C1_TEMPORAL_DERIVATIVE_GATE_ONLY,
            "gcc_phat_spectral_weighting": (
                "unit cross-spectrum magnitude with machine-epsilon denominator"
                if C1_PHASE_GATE_ONLY else None),
            "camera_pose_interpolation": (
                "C1 cubic Hermite translation and hemisphere-aligned normalized quaternion components"
                if (DETREND_C1_MULTISCALE_GATE_ONLY or C1_PHASE_GATE_ONLY
                    or C1_TEMPORAL_DERIVATIVE_GATE_ONLY)
                else None),
            "integer_pose_max_abs_error": c1_integer_pose_max_abs_error,
            "profile_deltas_frames": list(TRANSPORT_PROFILE_DELTAS),
            "profile_winners": profile_winners,
            "profile_winner_margins": profile_margins,
            "profile_rows": profile_rows,
            "profile_scopes": profile_scopes,
            "profile_pass": profile_pass,
            "gradient_rows": gradient_rows,
            "gradient_pass": gradient_pass,
            "gate_pass": gate_pass,
            "centers_by_side": centers,
            "nuisance_sha256_before": nuisance_before,
            "nuisance_sha256_after": nuisance_after,
            "nuisance_tensor_count": nuisance_count,
            "camera_manifest_sha256_before": camera_hash_before,
            "camera_manifest_sha256_after": camera_hash_after,
            "formal_test_cameras_constructed": False,
            "teacher_hashes": teacher_hashes,
            "v9_directional_report": (
                str(args.directional_report) if TRANSPORT_GATE_ONLY else None),
            "v10_transport_report": (
                str(args.transport_report) if MULTISCALE_GATE_ONLY else None),
            "v11_multiscale_report": (
                str(args.multiscale_report)
                if COARSE_MULTISCALE_GATE_ONLY else None),
            "v15_c1_derivative_audit_report": (
                str(args.c1_derivative_audit_report)
                if DETREND_C1_MULTISCALE_GATE_ONLY else None),
            "v16_detrended_report": (
                str(args.detrended_report) if C1_PHASE_GATE_ONLY else None),
            "v17_gcc_phat_report": (
                str(args.gcc_phat_report)
                if C1_TEMPORAL_DERIVATIVE_GATE_ONLY else None),
            "entry_script_path": str(ENTRY_SCRIPT_PATH),
            "script_sha256": base.sha256_file(ENTRY_SCRIPT_PATH),
            "implementation_sha256": base.sha256_file(__file__),
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
        print(log_prefix + "_GATE_RESULT " + json.dumps({
            "status": status, "profile_pass": profile_pass,
            "gradient_pass": gradient_pass,
        }, sort_keys=True), flush=True)
        return

    if LANDSCAPE_ONLY:
        started = time.time()
        landscape = []
        with torch.no_grad():
            for delta in LANDSCAPE_DELTAS:
                rendered = {
                    side: render_activity_sequence(
                        side, centers, delta, views_by_side, gaussians, pipe,
                        hyper, background, args.iteration).detach()
                    for side in ("left", "right")}
                left_loss = -pearson_tensor(
                    rendered["left"], targets["P"]["left"])
                right_loss = -pearson_tensor(
                    rendered["right"], targets["P"]["right"])
                all_loss = -pearson_tensor(
                    torch.cat((rendered["left"], rendered["right"])),
                    torch.cat((targets["P"]["left"],
                               targets["P"]["right"])))
                row = {
                    "delta_frames": delta,
                    "losses": {
                        "all": float(all_loss.item()),
                        "left": float(left_loss.item()),
                        "right": float(right_loss.item()),
                    },
                }
                landscape.append(row)
                print("MIND_P_LANDSCAPE_POINT " + json.dumps(
                    row, sort_keys=True), flush=True)
        by_delta = {row["delta_frames"]: row for row in landscape}
        central_fd = [
            {
                "delta_frames": delta,
                "all_dloss_ddelta": (
                    by_delta[delta + 1.0]["losses"]["all"]
                    - by_delta[delta - 1.0]["losses"]["all"]) / 2.0,
            }
            for delta in LANDSCAPE_DELTAS[1:-1]
        ]
        winners = {
            scope: min(
                LANDSCAPE_DELTAS,
                key=lambda delta: by_delta[delta]["losses"][scope])
            for scope in ("all", "left", "right")
        }
        nuisance_after, nuisance_after_count = base.stable_tensor_hash(
            rgbgate.freeze_rgb_state(gaussians, views))
        camera_after, camera_hash_after = base.camera_scalar_manifest(views)
        require(nuisance_count == nuisance_after_count
                and nuisance_before == nuisance_after,
                "Frozen Scene state changed")
        require(camera_before == camera_after
                and camera_hash_before == camera_hash_after,
                "Camera scalar state changed")
        status = "MIND_P_LANDSCAPE_COMPLETE"
        report = {
            "schema": SCHEMA, "status": status, "scene": "Covers",
            "per_scene_optimization": True, "diagnostic_only": True,
            "target_modality": args.target_modality,
            "source_path": str(dataset.source_path),
            "external_data_loaded": False,
            "scene_loader_rgb_only": True,
            "thermal_observation_loader": str(THERMAL_IMAGE_ROOT),
            "trainable_parameter_count": 0,
            "candidate_enumeration_in_optimizer": False,
            "diagnostic_uniform_delta_scan": True,
            "landscape_arm": "P",
            "landscape_deltas_frames": list(LANDSCAPE_DELTAS),
            "landscape": landscape,
            "central_fd": central_fd,
            "grid_winners": winners,
            "centers_by_side": centers,
            "nuisance_sha256_before": nuisance_before,
            "nuisance_sha256_after": nuisance_after,
            "nuisance_tensor_count": nuisance_count,
            "camera_manifest_sha256_before": camera_hash_before,
            "camera_manifest_sha256_after": camera_hash_after,
            "formal_test_cameras_constructed": False,
            "teacher_hashes": teacher_hashes,
            "entry_script_path": str(ENTRY_SCRIPT_PATH),
            "script_sha256": base.sha256_file(ENTRY_SCRIPT_PATH),
            "implementation_sha256": base.sha256_file(__file__),
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
        print("MIND_P_LANDSCAPE_RESULT " + json.dumps({
            "status": status, "grid_winners": winners,
        }, sort_keys=True), flush=True)
        return

    started = time.time()
    recoveries = [
        optimize_arm(
            arm, targets, centers, views_by_side, gaussians, pipe, hyper,
            background, args.iteration)
        for arm in ARM_SHIFTS]
    recovery_by_arm = {row["arm"]: row for row in recoveries}
    recovered = {
        arm: recovery_by_arm[arm]["recovered_delta_frames"]
        for arm in ARM_SHIFTS}
    negative_relative = recovered["N"] - recovered["Z"]
    positive_relative = recovered["P"] - recovered["Z"]
    negative_residual = negative_relative - ARM_SHIFTS["N"]
    positive_residual = positive_relative - ARM_SHIFTS["P"]
    recovery_records_finite = all(
        len(row["history"]) == 7
        and math.isfinite(row["final_raw"])
        and all(math.isfinite(value)
                for value in row["final_losses"].values())
        for row in recoveries)

    nuisance_after, nuisance_after_count = base.stable_tensor_hash(
        rgbgate.freeze_rgb_state(gaussians, views))
    camera_after, camera_hash_after = base.camera_scalar_manifest(views)
    require(nuisance_count == nuisance_after_count
            and nuisance_before == nuisance_after,
            "Frozen Scene state changed")
    require(camera_before == camera_after
            and camera_hash_before == camera_hash_after,
            "Camera scalar state changed")
    gate_pass = (
        all(math.isfinite(value) for value in recovered.values())
        and recovery_records_finite
        and recovered["N"] < recovered["Z"] < recovered["P"]
        and abs(negative_residual) <= EQUIVARIANCE_TOLERANCE
        and abs(positive_residual) <= EQUIVARIANCE_TOLERANCE)
    status = ("MIND_SHIFT_EQUIVARIANT_RECOVERY_PASS" if gate_pass
              else "MIND_SHIFT_EQUIVARIANT_RECOVERY_FAIL")
    report = {
        "schema": SCHEMA, "status": status, "scene": "Covers",
        "per_scene_optimization": True, "gate_only": True,
        "target_modality": args.target_modality,
        "source_path": str(dataset.source_path),
        "external_data_loaded": False,
        "scene_loader_rgb_only": True,
        "thermal_observation_loader": (
            None if args.target_modality == "rgb"
            else str(THERMAL_IMAGE_ROOT)),
        "trainable_parameter_count": 1,
        "trainable_parameter": "one raw scalar; delta=20*tanh(raw)",
        "candidate_enumeration_in_optimizer": False,
        "independent_arms_from_raw_zero": True,
        "optimizer": {
            "name": "Adam", "raw_learning_rate": RAW_LR,
            "steps_per_arm": RECOVERY_STEPS,
            "target_scope": "all common-support centers",
        },
        "feature": "fixed eight-channel 120x160 local self-similarity descriptor",
        "observable": "Pearson correlation of time-symmetric backward/forward spatial-mean squared MIND activity curves",
        "time_reversal_symmetric": True,
        "spatial_correspondence_used": False,
        "absolute_zero_assumed": False,
        "unknown_per_scene_baseline": "independently recovered Z arm",
        "equivariance_gate": {
            "required_negative_relative_frames": ARM_SHIFTS["N"],
            "required_positive_relative_frames": ARM_SHIFTS["P"],
            "tolerance_frames": EQUIVARIANCE_TOLERANCE,
            "recovered_negative_relative_frames": negative_relative,
            "recovered_positive_relative_frames": positive_relative,
            "negative_residual_frames": negative_residual,
            "positive_residual_frames": positive_residual,
            "ordered_N_Z_P": recovered["N"] < recovered["Z"] < recovered["P"],
            "pass": gate_pass,
        },
        "centers_by_side": centers,
        "recoveries": recoveries,
        "equivariance_gate_pass": gate_pass,
        "prerequisites": {
            "rgb_profile_report": str(args.rgb_profile_report),
            "thermal_profile_report": str(args.thermal_profile_report),
            "v5_gradient_report": str(args.gradient_report),
        },
        "nuisance_sha256_before": nuisance_before,
        "nuisance_sha256_after": nuisance_after,
        "nuisance_tensor_count": nuisance_count,
        "camera_manifest_sha256_before": camera_hash_before,
        "camera_manifest_sha256_after": camera_hash_after,
        "formal_test_cameras_constructed": False,
        "teacher_hashes": teacher_hashes,
        "entry_script_path": str(ENTRY_SCRIPT_PATH),
        "script_sha256": base.sha256_file(ENTRY_SCRIPT_PATH),
        "implementation_sha256": base.sha256_file(__file__),
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (args.output / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    (args.output / status).write_text(status + "\n", encoding="utf-8")
    print("MIND_SHIFT_EQUIVARIANT_RECOVERY_RESULT " + json.dumps({
        "status": status,
        "recovered_delta_frames": recovered,
        "negative_relative_frames": negative_relative,
        "positive_relative_frames": positive_relative,
        "tolerance_frames": EQUIVARIANCE_TOLERANCE,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
