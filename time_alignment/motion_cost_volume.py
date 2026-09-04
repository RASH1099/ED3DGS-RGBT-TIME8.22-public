"""Continuous global clock loss over a frozen RGB-T motion cost volume."""

import hashlib
import json
import math

import cv2
import numpy as np
import torch


IMAGE_SIZE = (160, 120)
MOTION_SIZE = (40, 30)
SEARCH_RADIUS = 5
PROFILE_OFFSETS = tuple(range(-22, 23, 2))
SMOOTHING_SIGMAS_FRAMES = (12.0, 6.0, 3.0)
STEPS_PER_SIGMA = 100
LINEAR_START_ITERATION = 301
TEMPORAL_BLOCKS_PER_SIDE = 4


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


def _gray(camera, modality):
    image = camera.original_image if modality == "rgb" else camera.thermal_image
    _require(image is not None and torch.is_tensor(image) and image.ndim == 3,
             f"Missing {modality} image for {camera.image_name}")
    array = image.detach().to(device="cpu", dtype=torch.float32).numpy()
    _require(array.shape[0] in (1, 3),
             f"Invalid {modality} channels for {camera.image_name}")
    if array.shape[0] == 3:
        gray = 0.299 * array[0] + 0.587 * array[1] + 0.114 * array[2]
    else:
        gray = array[0]
    gray = cv2.resize(
        gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    low, high = np.quantile(gray, (0.02, 0.98))
    normalized = np.clip(
        (gray - low) / max(float(high - low), 1.0e-6), 0.0, 1.0)
    _require(bool(np.isfinite(normalized).all()),
             f"Non-finite {modality} image for {camera.image_name}")
    return normalized


def _change(first, second):
    change = cv2.absdiff(second, first)
    change = cv2.GaussianBlur(change, (7, 7), 0)
    change = cv2.resize(change, MOTION_SIZE, interpolation=cv2.INTER_AREA)
    change = change - change.mean()
    norm = float(np.linalg.norm(change))
    _require(math.isfinite(norm) and norm > 1.0e-8,
             "Degenerate motion-change map")
    return np.ascontiguousarray(change / norm, dtype=np.float32)


def _correlation(first, second):
    padded = cv2.copyMakeBorder(
        second, SEARCH_RADIUS, SEARCH_RADIUS, SEARCH_RADIUS, SEARCH_RADIUS,
        cv2.BORDER_REFLECT)
    response = cv2.matchTemplate(
        padded, first, cv2.TM_CCOEFF_NORMED)
    value = float(response.max())
    _require(math.isfinite(value), "Non-finite motion correlation")
    return value


def _group(cameras):
    grouped = {"left": {}, "right": {}}
    for camera in cameras:
        side = _side(camera)
        frame = int(camera.frame_no)
        _require(frame not in grouped[side],
                 f"Duplicate motion camera: {side} {frame}")
        grouped[side][frame] = camera
    for side, by_frame in grouped.items():
        frames = sorted(by_frame)
        steps = {second - first
                 for first, second in zip(frames[:-1], frames[1:])}
        _require(len(frames) >= 2 and steps == {2},
                 f"Motion cost volume requires the full trajectory: {side}")
    left_frames = sorted(grouped["left"])
    right_frames = sorted(grouped["right"])
    _require(
        len(left_frames) == len(right_frames),
        "Motion cost volume requires balanced left/right trajectories")
    combined = sorted(left_frames + right_frames)
    _require(
        len(combined) == len(set(combined))
        and combined == list(range(combined[0], combined[-1] + 1)),
        "Motion cost volume requires one complete interleaved trajectory")
    return grouped


def _build_changes(grouped):
    changes = {"rgb": {}, "thermal": {}}
    digest = hashlib.sha256()
    for modality in ("rgb", "thermal"):
        for side, by_frame in grouped.items():
            frames = sorted(by_frame)
            images = {frame: _gray(by_frame[frame], modality)
                      for frame in frames}
            maps = []
            for first, second in zip(frames[:-1], frames[1:]):
                value = _change(images[first], images[second])
                maps.append(value)
                digest.update(f"{modality}:{side}:{first}\n".encode("utf-8"))
                digest.update(value.tobytes(order="C"))
            changes[modality][side] = np.stack(maps)
    return changes, digest.hexdigest()


def _build_volume(rgb, thermal):
    _require(rgb.shape == thermal.shape and rgb.ndim == 3,
             "Invalid motion-map sequences")
    volume = np.empty((thermal.shape[0], rgb.shape[0]), dtype=np.float32)
    for target_index, target in enumerate(thermal):
        for teacher_index, teacher in enumerate(rgb):
            volume[target_index, teacher_index] = _correlation(teacher, target)
    _require(bool(np.isfinite(volume).all()), "Non-finite motion cost volume")
    return volume


class MotionCostVolumeLoss:
    """Query a fixed pairwise motion volume with a continuous affine clock."""

    def __init__(self, cameras, max_offset_frames, max_drift_frames,
                 duration, device="cuda"):
        grouped = _group(cameras)
        changes, input_hash = _build_changes(grouped)
        device = torch.device(device)
        support_bound = float(max_offset_frames) + float(max_drift_frames)
        self.data = {}
        for side in ("left", "right"):
            camera_frames = sorted(grouped[side])
            transition_frames = torch.tensor(
                camera_frames[:-1], dtype=torch.float32, device=device)
            frame_step = float(camera_frames[1] - camera_frames[0])
            centers = torch.nonzero(
                (transition_frames >= transition_frames[0] + support_bound)
                & (transition_frames <= transition_frames[-1] - support_bound),
                as_tuple=False).flatten()
            _require(centers.numel() >= 100,
                     f"Insufficient affine motion support for {side}")
            volume = _build_volume(
                changes["rgb"][side], changes["thermal"][side])
            volume_tensor = torch.tensor(volume, device=device)
            row_mean = volume_tensor.mean(dim=1, keepdim=True)
            row_scale = volume_tensor.std(
                dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
            volume_tensor = (volume_tensor - row_mean) / row_scale
            self.data[side] = {
                "frames": transition_frames,
                "centers": centers,
                "volume": volume_tensor,
                "frame_step": frame_step,
                "camera_first_frame": camera_frames[0],
                "camera_last_frame": camera_frames[-1],
            }
        self.duration = float(duration)
        self.max_offset_frames = float(max_offset_frames)
        self.max_drift_frames = float(max_drift_frames)
        profiles = self.audit_profiles()
        self.contract = {
            "schema": "covers_continuous_global_motion_cost_volume_clock",
            "affine_clock": False,
            "global_offset_only": True,
            "candidate_lag_bank_in_optimizer": False,
            "clock_initialized_from_profile": False,
            "profile_scan_used_only_for_audit": True,
            "shift_truth_input": False,
            "camera_shift_metadata_read": False,
            "cost_volume_detached": True,
            "cost_volume_built_before_clock_updates": True,
            "motion_source": "observed_rgb_and_thermal_training_images",
            "motion_resolution": list(MOTION_SIZE[::-1]),
            "translation_search_radius_pixels": SEARCH_RADIUS,
            "smoothing_sigmas_frames": list(SMOOTHING_SIGMAS_FRAMES),
            "steps_per_sigma": STEPS_PER_SIGMA,
            "continuous_query": "gaussian_continuation_then_linear",
            "linear_start_iteration": LINEAR_START_ITERATION,
            "row_standardization": "per_thermal_transition_zscore",
            "aggregation": {
                "schema": "global_mean_all_supported_transitions",
                "blocks_per_side": TEMPORAL_BLOCKS_PER_SIDE,
                "total_blocks": 2 * TEMPORAL_BLOCKS_PER_SIDE,
                "block_profiles_used_only_for_audit": True,
            },
            "processed_motion_input_sha256": input_hash,
            "profiles": profiles,
            "block_profiles": self.audit_block_profiles(),
            "sides": {
                side: {
                    "camera_count": len(grouped[side]),
                    "motion_map_count": int(self.data[side]["frames"].numel()),
                    "optimization_center_count": int(
                        self.data[side]["centers"].numel()),
                    "cost_volume_shape": list(
                        self.data[side]["volume"].shape),
                    "first_frame": self.data[side]["camera_first_frame"],
                    "last_frame": self.data[side]["camera_last_frame"],
                    "frame_step": self.data[side]["frame_step"],
                }
                for side in ("left", "right")
            },
        }
        print("MOTION_COST_VOLUME_CONTRACT " + json.dumps(
            self.contract, sort_keys=True), flush=True)

    @staticmethod
    def sigma_for_iteration(iteration):
        index = min(
            max(int(iteration) - 1, 0) // STEPS_PER_SIGMA,
            len(SMOOTHING_SIGMAS_FRAMES) - 1)
        return SMOOTHING_SIGMAS_FRAMES[index]

    @staticmethod
    def _window_centers(centers, window):
        if window == "full":
            return centers
        midpoint = centers.numel() // 2
        if window == "early":
            return centers[:midpoint]
        if window == "late":
            return centers[midpoint:]
        raise ValueError(f"Invalid motion-cost window: {window}")

    def _scores_for_centers(self, side, centers, offset_frames, drift_frames,
                            sigma_frames):
        state = self.data[side]
        base_frames = state["frames"][centers]
        coordinate = 2.0 * base_frames / (self.duration - 1.0) - 1.0
        query_frames = base_frames + offset_frames + drift_frames * coordinate
        distance = state["frames"].unsqueeze(0) - query_frames.unsqueeze(1)
        logits = -distance.square() / (2.0 * float(sigma_frames) ** 2)
        weights = torch.softmax(logits, dim=1)
        score = (weights * state["volume"][centers]).sum(dim=1)
        _require(bool(torch.isfinite(score).all()),
                 f"Non-finite motion cost-volume scores for {side}")
        return score

    def _linear_scores_for_centers(self, side, centers, offset_frames,
                                   drift_frames):
        state = self.data[side]
        base_frames = state["frames"][centers]
        coordinate = 2.0 * base_frames / (self.duration - 1.0) - 1.0
        query_frames = base_frames + offset_frames + drift_frames * coordinate
        position = ((query_frames - state["frames"][0])
                    / state["frame_step"])
        index = torch.floor(position).to(torch.long)
        _require(bool((index >= 0).all() and
                      (index <= state["frames"].numel() - 2).all()),
                 f"Linear motion query exceeds support for {side}")
        fraction = position - index.to(position.dtype)
        volume = state["volume"]
        lower = volume[centers, index]
        upper = volume[centers, index + 1]
        score = lower + (upper - lower) * fraction
        _require(bool(torch.isfinite(score).all()),
                 f"Non-finite linear motion scores for {side}")
        return score

    def _side_loss(self, side, offset_frames, drift_frames, sigma_frames,
                   window="full", linear=False):
        state = self.data[side]
        centers = self._window_centers(state["centers"], window)
        score = (self._linear_scores_for_centers(
            side, centers, offset_frames, drift_frames)
                 if linear else self._scores_for_centers(
                     side, centers, offset_frames, drift_frames,
                     sigma_frames))
        loss = -score.mean()
        _require(bool(torch.isfinite(loss)),
                 f"Non-finite motion cost-volume loss for {side}")
        return loss

    def loss(self, offset_frames, drift_frames, iteration):
        if offset_frames.ndim != 0 or drift_frames.ndim != 0:
            raise ValueError("Scene clock parameters must be scalar")
        sigma = self.sigma_for_iteration(iteration)
        linear = int(iteration) >= LINEAR_START_ITERATION
        side_losses = {
            side: self._side_loss(
                side, offset_frames, drift_frames, sigma, linear=linear)
            for side in ("left", "right")
        }
        loss = torch.stack(list(side_losses.values())).mean()
        _require(bool(torch.isfinite(loss)), "Non-finite global motion loss")
        return loss, side_losses

    def _profile_loss(self, sides, window, offset):
        values = []
        for side in sides:
            state = self.data[side]
            centers = self._window_centers(state["centers"], window)
            query = state["frames"][centers] + float(offset)
            indices = torch.round(
                (query - state["frames"][0]) / state["frame_step"]
            ).to(torch.long)
            values.append(-state["volume"][centers, indices].mean())
        return torch.stack(values).mean()

    def audit_profiles(self):
        groups = {
            "full": (("left", "right"), "full"),
            "left": (("left",), "full"),
            "right": (("right",), "full"),
            "early": (("left", "right"), "early"),
            "late": (("left", "right"), "late"),
        }
        profiles = {}
        for name, (sides, window) in groups.items():
            losses = {
                str(offset): float(self._profile_loss(
                    sides, window, offset).detach().item())
                for offset in PROFILE_OFFSETS
            }
            ordered = sorted(losses, key=losses.get)
            profiles[name] = {
                "best_integer_offset_frames": int(ordered[0]),
                "winner_margin": losses[ordered[1]] - losses[ordered[0]],
                "losses": losses,
            }
        return profiles

    def audit_block_profiles(self):
        profiles = {}
        for side in ("left", "right"):
            chunks = torch.tensor_split(
                self.data[side]["centers"], TEMPORAL_BLOCKS_PER_SIDE)
            for block_index, centers in enumerate(chunks):
                losses = {}
                state = self.data[side]
                for offset in PROFILE_OFFSETS:
                    query = state["frames"][centers] + float(offset)
                    indices = torch.round(
                        (query - state["frames"][0])
                        / state["frame_step"]).to(torch.long)
                    losses[str(offset)] = float(
                        -state["volume"][centers, indices].mean().detach())
                ordered = sorted(losses, key=losses.get)
                profiles[f"{side}_{block_index}"] = {
                    "best_integer_offset_frames": int(ordered[0]),
                    "winner_margin": losses[ordered[1]] - losses[ordered[0]],
                    "losses": losses,
                }
        return profiles
