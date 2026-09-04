"""Differentiable blind affine Scene-clock loss from frozen dynamic traces."""

import json
import math

import torch
import torch.nn.functional as F

from time_alignment import rgb_fidelity as rgbgate


MIND_SIZE = (120, 160)
MIND_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)
SMOOTHING_KERNELS = (9, 5, 3)
PROFILE_OFFSETS = tuple(range(-24, 25))
REGIONS = (
    ("full", 0, 120, 0, 160),
    ("top_left", 0, 60, 0, 80),
    ("top_right", 0, 60, 80, 160),
    ("bottom_left", 60, 120, 0, 80),
    ("bottom_right", 60, 120, 80, 160),
    ("left", 0, 120, 0, 80),
    ("right", 0, 120, 80, 160),
    ("top", 0, 60, 0, 160),
    ("bottom", 60, 120, 0, 160),
)
ACTIVE_REGION_COUNT = 5


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _side(camera):
    name = str(camera.image_name).lower()
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    raise ValueError(f"Camera side missing from name: {camera.image_name}")


def _gray(image):
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("expected a CHW grayscale or RGB image")
    if image.shape[0] == 1:
        value = image
    else:
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
        value = (image * weights).sum(dim=0, keepdim=True)
    return F.interpolate(
        value.unsqueeze(0), size=MIND_SIZE, mode="bilinear",
        align_corners=False)[0, 0]


def _mind_descriptor(image):
    gray = _gray(image)
    height, width = gray.shape
    source = gray[None, None]
    padded = F.pad(source, (1, 1, 1, 1), mode="replicate")
    distances = []
    for dy, dx in MIND_OFFSETS:
        shifted = padded[:, :, 1 + dy:1 + dy + height,
                         1 + dx:1 + dx + width]
        squared = (source - shifted).square()
        distances.append(F.avg_pool2d(
            F.pad(squared, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3, stride=1)[0, 0])
    distance = torch.stack(distances)
    variance = distance.mean(dim=0, keepdim=True)
    descriptor = torch.exp(-distance / variance.clamp_min(1.0e-8))
    _require(bool(torch.isfinite(descriptor).all()),
             "non-finite MIND descriptor")
    return descriptor


def _symmetric_activity(descriptors):
    """Return temporal MIND activity for fixed spatial regions."""
    transitions = []
    for first, second in zip(descriptors[:-1], descriptors[1:]):
        squared = (second - first).square()
        transitions.append(torch.stack([
            squared[:, y0:y1, x0:x1].mean()
            for _, y0, y1, x0, x1 in REGIONS
        ]))
    transitions = torch.stack(transitions)
    result = 0.5 * (transitions[:-1] + transitions[1:])
    _require(result.ndim == 2 and result.shape[0] >= 32,
             "insufficient temporal activity trace")
    _require(bool(torch.isfinite(result).all()),
             "non-finite temporal activity trace")
    return result.detach()


def _catmull_rom(sequence, position):
    lower = torch.floor(position).to(torch.long)
    weight = position - lower.to(position.dtype)
    if sequence.ndim > 1:
        weight = weight.unsqueeze(-1)
    _require(int(lower.min()) >= 1, "query below fixed temporal support")
    _require(int(lower.max()) + 2 < sequence.shape[0],
             "query above fixed temporal support")
    p0, p1 = sequence[lower - 1], sequence[lower]
    p2, p3 = sequence[lower + 1], sequence[lower + 2]
    return 0.5 * (
        2.0 * p1 + (-p0 + p2) * weight
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * weight.square()
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * weight.pow(3))


def _smooth(sequence, kernel):
    if kernel == 1:
        return sequence
    radius = kernel // 2
    return F.avg_pool1d(
        F.pad(sequence[None, None], (radius, radius), mode="replicate"),
        kernel_size=kernel, stride=1)[0, 0]


def _detrend(sequence):
    coordinate = torch.linspace(
        -1.0, 1.0, sequence.numel(), device=sequence.device,
        dtype=sequence.dtype)
    centered = sequence - sequence.mean()
    slope = (centered * coordinate).sum() / coordinate.square().sum()
    result = centered - slope * coordinate
    _require(bool(torch.isfinite(result).all()),
             "non-finite detrended temporal trace")
    return result


def _pearson_loss(first, second):
    _require(first.shape == second.shape and first.ndim == 1,
             "invalid temporal correlation inputs")
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    _require(float(denominator.detach().item()) > 1.0e-12,
             "degenerate temporal correlation")
    result = -(first * second).sum() / denominator
    _require(bool(torch.isfinite(result)), "non-finite temporal correlation")
    return result


class TemporalTraceLoss:
    """Cheap differentiable loss over fixed teacher/observation traces."""

    def __init__(self, traces, max_offset_frames, max_drift_frames,
                 duration):
        self.data = {}
        for side in ("left", "right"):
            teacher = traces[side]["teacher"]
            thermal = traces[side]["thermal"]
            step = float(traces[side]["frame_step"])
            activity_frames = traces[side]["activity_frames"]
            _require(teacher.shape == thermal.shape and teacher.ndim == 2,
                     f"trace shape mismatch for {side}")
            _require(activity_frames.shape == teacher.shape[:1],
                     f"activity-frame shape mismatch for {side}")
            region_score = teacher.var(dim=0) * thermal.var(dim=0)
            _require(bool(torch.isfinite(region_score).all()),
                     f"non-finite dynamic-region score for {side}")
            selected = torch.topk(
                region_score, k=min(ACTIVE_REGION_COUNT, region_score.numel()),
                largest=True).indices.tolist()
            if 0 not in selected:
                selected[-1] = 0
            selected = sorted(set(selected))
            teacher = teacher[:, selected]
            thermal = thermal[:, selected]
            margin = int(math.ceil(
                (float(max_offset_frames) + float(max_drift_frames)) / step)) + 2
            centers = torch.arange(
                margin, teacher.shape[0] - margin,
                dtype=torch.long, device=teacher.device)
            _require(centers.numel() >= 32,
                     f"insufficient fixed temporal support for {side}")
            self.data[side] = {
                "teacher": teacher.detach(),
                "thermal": thermal.detach(),
                "frame_step": step,
                "centers": centers,
                "activity_frames": activity_frames.detach(),
                "active_region_indices": selected,
                "active_region_names": [REGIONS[index][0] for index in selected],
            }
        self.duration = float(duration)
        _require(self.duration > 1.0, "invalid affine-clock duration")

    @staticmethod
    def _window_centers(centers, window):
        if window == "full":
            return centers
        midpoint = centers.numel() // 2
        if window == "early":
            return centers[:midpoint]
        if window == "late":
            return centers[midpoint:]
        raise ValueError(f"invalid temporal window: {window}")

    def loss(self, offset_frames, drift_frames=None, sides=("left", "right"),
             window="full"):
        if offset_frames.ndim != 0:
            raise ValueError("Scene clock offset must be scalar")
        if drift_frames is None:
            drift_frames = torch.zeros_like(offset_frames)
        if drift_frames.ndim != 0:
            raise ValueError("Scene clock drift must be scalar")
        side_losses = {}
        for side in sides:
            state = self.data[side]
            centers = self._window_centers(state["centers"], window)
            frame = state["activity_frames"][centers].to(offset_frames.dtype)
            coordinate = 2.0 * frame / (self.duration - 1.0) - 1.0
            effective_offset = offset_frames + drift_frames * coordinate
            position = centers.to(offset_frames.dtype) + (
                effective_offset / state["frame_step"])
            prediction = _catmull_rom(state["teacher"], position)
            target = state["thermal"][centers]
            per_scale = []
            for region in range(prediction.shape[1]):
                for kernel in SMOOTHING_KERNELS:
                    per_scale.append(_pearson_loss(
                        _detrend(_smooth(prediction[:, region], kernel)),
                        _detrend(_smooth(target[:, region], kernel))))
            side_losses[side] = torch.stack(per_scale).mean()
        total = torch.stack(list(side_losses.values())).mean()
        _require(bool(torch.isfinite(total)),
                 "non-finite teacher temporal-consensus loss")
        return total, side_losses

    def audit_profiles(self, device):
        groups = {
            "full": (("left", "right"), "full"),
            "left": (("left",), "full"),
            "right": (("right",), "full"),
            "early": (("left", "right"), "early"),
            "late": (("left", "right"), "late"),
        }
        result = {}
        zero_drift = torch.zeros((), device=device)
        for name, (sides, window) in groups.items():
            losses = {}
            for offset in PROFILE_OFFSETS:
                value, _ = self.loss(
                    torch.tensor(float(offset), device=device), zero_drift,
                    sides=sides, window=window)
                losses[str(offset)] = float(value.detach().item())
            best = min(losses, key=losses.get)
            result[name] = {
                "best_integer_offset_frames": int(best),
                "losses": losses,
            }
        return result


class TemporalConsensusLoss:
    """Build one frozen teacher trace, then optimize only the Scene clock."""

    def __init__(self, cameras, max_offset_frames, max_drift_frames,
                 duration, gaussians, pipe, hyper, background,
                 teacher_iteration=30000, device="cuda"):
        grouped = {"left": {}, "right": {}}
        for camera in cameras:
            if camera.frame_no is None:
                raise ValueError(
                    f"Frame number missing from camera: {camera.image_name}")
            if camera.original_image is None or camera.thermal_image is None:
                raise RuntimeError(
                    f"Temporal consensus requires loaded train images: "
                    f"{camera.image_name}")
            side, frame = _side(camera), int(camera.frame_no)
            if frame in grouped[side]:
                raise ValueError(f"Duplicate temporal camera: {side} {frame}")
            grouped[side][frame] = camera

        device = torch.device(device)
        zero_raw = torch.zeros((), device=device)
        traces, side_contract = {}, {}
        with torch.no_grad():
            for side, by_frame in grouped.items():
                frames = sorted(by_frame)
                steps = {second - first
                         for first, second in zip(frames[:-1], frames[1:])}
                _require(len(steps) == 1, f"nonuniform frame grid for {side}")
                frame_step = steps.pop()
                _require(frame_step > 0, f"invalid frame step for {side}")
                teacher_descriptors, thermal_descriptors = [], []
                for frame in frames:
                    camera = by_frame[frame]
                    had_override = hasattr(
                        camera, "nctc_temporal_offset_raw_override")
                    if not had_override:
                        object.__setattr__(
                            camera, "nctc_temporal_offset_raw_override",
                            zero_raw)
                    try:
                        rendered = rgbgate.render_clock_rgb(
                            camera, frame, zero_raw, gaussians, pipe, hyper,
                            background, teacher_iteration)
                    finally:
                        if not had_override:
                            delattr(camera,
                                    "nctc_temporal_offset_raw_override")
                    teacher_descriptors.append(_mind_descriptor(rendered))
                    thermal_descriptors.append(
                        _mind_descriptor(camera.thermal_image.to(device)))
                traces[side] = {
                    "teacher": _symmetric_activity(teacher_descriptors),
                    "thermal": _symmetric_activity(thermal_descriptors),
                    "frame_step": frame_step,
                    "activity_frames": torch.tensor(
                        frames[1:-1], dtype=torch.float32, device=device),
                }
                side_contract[side] = {
                    "camera_count": len(frames),
                    "activity_count": int(traces[side]["teacher"].numel()),
                    "first_frame": frames[0],
                    "last_frame": frames[-1],
                    "frame_step": frame_step,
                }

        self.trace_loss = TemporalTraceLoss(
            traces, max_offset_frames, max_drift_frames, duration)
        profiles = self.trace_loss.audit_profiles(device)
        self.contract = {
            "schema": "covers_teacher_regional_affine_clock_loss",
            "candidate_enumeration_in_optimizer": False,
            "profile_scan_used_only_for_audit": True,
            "profiles": profiles,
            "teacher_trace_built_at_raw_zero": True,
            "teacher_trace_detached": True,
            "shift_truth_input": False,
            "spatial_correspondence_used": True,
            "affine_clock": True,
            "offset_bound_frames": float(max_offset_frames),
            "drift_endpoint_bound_frames": float(max_drift_frames),
            "duration": float(duration),
            "observable": (
                "frozen-teacher regional MIND dynamics versus Thermal dynamics, "
                "affine-detrended multiscale Pearson"),
            "smoothing_kernels": list(SMOOTHING_KERNELS),
            "active_regions": {
                side: self.trace_loss.data[side]["active_region_names"]
                for side in ("left", "right")
            },
            "sides": side_contract,
        }
        print("TEMPORAL_CONSENSUS_CONTRACT " + json.dumps(
            self.contract, sort_keys=True), flush=True)

    def loss(self, offset_frames, drift_frames):
        return self.trace_loss.loss(offset_frames, drift_frames)
