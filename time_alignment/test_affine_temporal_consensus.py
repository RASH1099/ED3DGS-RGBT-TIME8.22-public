#!/usr/bin/env python3
"""Deterministic CPU contract tests for the affine temporal trace loss."""

import torch

from time_alignment.temporal_consensus import TemporalTraceLoss, _catmull_rom


def make_trace(offset, drift):
    length = 192
    duration = float(length)
    frame = torch.arange(length, dtype=torch.float32)
    teacher = torch.stack([
        torch.sin(frame * 0.071 + phase)
        + 0.37 * torch.sin(frame * 0.137 + 0.5 * phase)
        + 0.19 * torch.cos(frame * 0.043 - phase)
        for phase in (0.0, 0.7, 1.4, 2.1, 2.8)
    ], dim=1)
    thermal = teacher.clone()
    centers = torch.arange(30, length - 30, dtype=torch.long)
    coordinate = 2.0 * frame[centers] / (duration - 1.0) - 1.0
    position = centers.to(torch.float32) + offset + drift * coordinate
    thermal[centers] = _catmull_rom(teacher, position)
    traces = {
        side: {
            "teacher": teacher + side_index * 0.03,
            "thermal": thermal + side_index * 0.03,
            "frame_step": 1.0,
            "activity_frames": frame,
        }
        for side_index, side in enumerate(("left", "right"))
    }
    return traces, duration


def recover(offset_truth, drift_truth):
    traces, duration = make_trace(offset_truth, drift_truth)
    objective = TemporalTraceLoss(traces, 24.0, 4.0, duration)
    raw_offset = torch.nn.Parameter(torch.zeros(()))
    raw_drift = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam((raw_offset, raw_drift), lr=0.03)
    for _ in range(800):
        optimizer.zero_grad(set_to_none=True)
        offset = 24.0 * torch.tanh(raw_offset)
        drift = 4.0 * torch.tanh(raw_drift)
        loss, _ = objective.loss(offset, drift)
        loss.backward()
        optimizer.step()
    learned_offset = float((24.0 * torch.tanh(raw_offset)).detach())
    learned_drift = float((4.0 * torch.tanh(raw_drift)).detach())
    assert abs(learned_offset - offset_truth) <= 0.10, (
        offset_truth, learned_offset)
    assert abs(learned_drift - drift_truth) <= 0.10, (
        drift_truth, learned_drift)


def main():
    for offset in (0.0, 8.0, 20.0):
        recover(offset, 0.0)
    recover(8.0, 1.5)
    print("AFFINE_TEMPORAL_CONSENSUS_TEST_PASS")


if __name__ == "__main__":
    main()
