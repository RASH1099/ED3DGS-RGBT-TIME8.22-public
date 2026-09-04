#!/usr/bin/env python3
"""CPU-only audit probe for blind RGB/Thermal temporal activity alignment."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MIND_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1))
PROFILE_OFFSETS = tuple(np.arange(4.0, 12.0001, 0.25).tolist())
GRIDS = (2, 4)
SMOOTHING_KERNELS = (9, 5, 3)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def image_descriptor(path):
    image = Image.open(path).convert("RGB").resize((160, 120), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    source = gray[None, None]
    padded = F.pad(source, (1, 1, 1, 1), mode="replicate")
    distances = []
    for dy, dx in MIND_OFFSETS:
        shifted = padded[:, :, 1 + dy:121 + dy, 1 + dx:161 + dx]
        squared = (source - shifted).square()
        distances.append(F.avg_pool2d(
            F.pad(squared, (1, 1, 1, 1), mode="replicate"),
            kernel_size=3, stride=1)[0, 0])
    distance = torch.stack(distances)
    variance = distance.mean(dim=0, keepdim=True).clamp_min(1.0e-8)
    result = torch.exp(-distance / variance)
    require(result.shape == (8, 120, 160), "Unexpected MIND descriptor shape")
    require(bool(torch.isfinite(result).all()), "Non-finite MIND descriptor")
    return result


def regional_transition(first, second):
    change = (second - first).square().mean(dim=0, keepdim=True)
    return torch.cat([
        F.adaptive_avg_pool2d(change[None], (grid, grid)).flatten()
        for grid in GRIDS
    ])


def activity_sequence(paths):
    previous = image_descriptor(paths[0])
    transitions = []
    for path in paths[1:]:
        current = image_descriptor(path)
        transitions.append(regional_transition(previous, current))
        previous = current
    transitions = torch.stack(transitions)
    activity = 0.5 * (transitions[:-1] + transitions[1:])
    result = activity.numpy().astype(np.float64)
    require(result.shape == (len(paths) - 2, 20), "Invalid regional activity")
    require(np.isfinite(result).all(), "Non-finite regional activity")
    return result


def smooth(sequence, kernel):
    if kernel == 1:
        return sequence
    radius = kernel // 2
    padded = np.pad(sequence, ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    cumulative = np.vstack((np.zeros((1, sequence.shape[1])), cumulative))
    return (cumulative[kernel:] - cumulative[:-kernel]) / kernel


def detrend(sequence):
    coordinate = np.linspace(-1.0, 1.0, sequence.shape[0])[:, None]
    design = np.concatenate((np.ones_like(coordinate), coordinate), axis=1)
    coefficients = np.linalg.lstsq(design, sequence, rcond=None)[0]
    return sequence - design @ coefficients


def interpolate(sequence, positions):
    source = np.arange(sequence.shape[0], dtype=np.float64)
    return np.stack([
        np.interp(positions, source, sequence[:, column])
        for column in range(sequence.shape[1])
    ], axis=1)


def regional_correlations(rgb, thermal, offset_frames, base_indices):
    prediction = interpolate(rgb, base_indices + offset_frames / 2.0)
    target = thermal[base_indices]
    correlations = []
    for kernel in SMOOTHING_KERNELS:
        first = detrend(smooth(prediction, kernel))
        second = detrend(smooth(target, kernel))
        numerator = (first * second).sum(axis=0)
        denominator = np.linalg.norm(first, axis=0) * np.linalg.norm(second, axis=0)
        require(np.all(denominator > 1.0e-12), "Degenerate regional trace")
        correlations.append(numerator / denominator)
    return np.stack(correlations).mean(axis=0)


def load_paths(root, observation_shift):
    datasets = {}
    for modality in ("rgb", "thermal"):
        datasets[modality] = json.loads(
            (root / modality / "dataset.json").read_text(encoding="utf-8"))
    thermal_train = set(datasets["thermal"]["train_ids"])
    result = {side: {"rgb": [], "thermal": []}
              for side in ("left", "right")}
    for rgb_id in datasets["rgb"]["train_ids"]:
        side = "left" if "_left_" in rgb_id else "right"
        frame = int(rgb_id.rsplit("_", 1)[1])
        thermal_id = rgb_id.replace("_rgb_", "_thermal_")
        shifted = thermal_id.rsplit("_", 1)[0] + f"_{frame + observation_shift:04d}"
        observed_id = shifted if shifted in thermal_train else thermal_id
        result[side]["rgb"].append(root / "rgb" / "images" / "1x" / f"{rgb_id}.png")
        result[side]["thermal"].append(
            root / "thermal" / "images" / "1x" / f"{observed_id}.png")
    for side in result:
        require(len(result[side]["rgb"]) == 133, f"Invalid {side} support")
        require(all(path.is_file() for paths in result[side].values() for path in paths),
                f"Missing {side} image")
    return result


def profile(root, observation_shift, expected_shift):
    paths = load_paths(root, observation_shift)
    activities = {
        side: {modality: activity_sequence(paths[side][modality])
               for modality in ("rgb", "thermal")}
        for side in ("left", "right")
    }
    base = np.arange(4, 123, dtype=np.int64)
    rows = []
    for offset in PROFILE_OFFSETS:
        side_regions = {
            side: regional_correlations(
                activities[side]["rgb"], activities[side]["thermal"],
                offset, base)
            for side in ("left", "right")
        }
        side_scores = {side: float(values.mean())
                       for side, values in side_regions.items()}
        joined = np.concatenate(tuple(side_regions.values()))
        rows.append({
            "offset_frames": float(offset),
            "loss": float(-joined.mean()),
            "left_loss": float(-side_scores["left"]),
            "right_loss": float(-side_scores["right"]),
            "median_loss": float(-np.median(joined)),
        })
    best = min(rows, key=lambda row: row["loss"])
    side_best = {
        side: min(rows, key=lambda row: row[f"{side}_loss"])["offset_frames"]
        for side in ("left", "right")
    }
    median_best = min(rows, key=lambda row: row["median_loss"])
    report = {
        "schema": "covers_raw_regional_mind_clock_probe",
        "status": "PASS" if (
            abs(best["offset_frames"] - expected_shift) <= 0.25
            and abs(side_best["left"] - side_best["right"]) <= 0.5
        ) else "FAIL",
        "observation_generation": {
            "shift_frames": observation_shift,
            "separate_from_estimator": True,
        },
        "estimator": {
            "reads_shift_metadata": False,
            "profile_offsets_frames": list(PROFILE_OFFSETS),
            "fixed_support_indices": base.tolist(),
            "grids": list(GRIDS),
            "smoothing_kernels": list(SMOOTHING_KERNELS),
        },
        "profile_rows": rows,
        "profile_best_offset_frames": best["offset_frames"],
        "median_profile_best_offset_frames": median_best["offset_frames"],
        "side_best_offsets_frames": side_best,
        "expected_shift_audit_only": expected_shift,
        "expected_shift_error_frames": abs(best["offset_frames"] - expected_shift),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observation-shift", type=int, required=True)
    parser.add_argument("--expected-shift", type=float, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(not args.report.exists(), f"Refusing existing report: {args.report}")
    report = profile(args.dataset.resolve(), args.observation_shift,
                     args.expected_shift)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "profile_best_offset_frames",
        "median_profile_best_offset_frames", "side_best_offsets_frames",
        "expected_shift_error_frames")}, sort_keys=True))


if __name__ == "__main__":
    main()
