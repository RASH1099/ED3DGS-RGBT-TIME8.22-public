#!/usr/bin/env python3
"""Bounded monotonic constant/affine per-scene clock parameterization."""

import torch


OFFSET_BOUND_FRAMES = 24.0
DRIFT_ENDPOINT_BOUND_FRAMES = 4.0


def normalized_time_coordinate(frame, duration):
    if float(duration) <= 1.0:
        raise RuntimeError("Temporal clock requires duration > 1")
    return 2.0 * float(frame) / (float(duration) - 1.0) - 1.0


def affine_offset_frames(raw_offset, raw_drift, frame, duration,
                         offset_bound=OFFSET_BOUND_FRAMES,
                         drift_bound=DRIFT_ENDPOINT_BOUND_FRAMES):
    offset = float(offset_bound) * torch.tanh(raw_offset)
    if raw_drift is None:
        return offset
    coordinate = normalized_time_coordinate(frame, duration)
    drift = float(drift_bound) * torch.tanh(raw_drift)
    return offset + drift * coordinate


def endpoint_offsets(raw_offset, raw_drift,
                     offset_bound=OFFSET_BOUND_FRAMES,
                     drift_bound=DRIFT_ENDPOINT_BOUND_FRAMES):
    offset = float(offset_bound) * torch.tanh(raw_offset)
    drift = (torch.zeros_like(offset) if raw_drift is None else
             float(drift_bound) * torch.tanh(raw_drift))
    return offset - drift, offset + drift


def minimum_clock_slope(raw_drift, duration,
                        drift_bound=DRIFT_ENDPOINT_BOUND_FRAMES):
    if raw_drift is None:
        return 1.0
    drift = float(drift_bound) * torch.tanh(raw_drift)
    return 1.0 - 2.0 * abs(float(drift.detach())) / (float(duration) - 1.0)
