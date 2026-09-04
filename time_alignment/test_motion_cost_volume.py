#!/usr/bin/env python3
"""CPU contract tests for continuous global motion-volume queries."""

from types import SimpleNamespace

import torch

from time_alignment.motion_cost_volume import MotionCostVolumeLoss, _group


def make_cameras(frames_per_side):
    cameras = []
    for frame in range(2 * frames_per_side):
        side = "left" if frame % 2 == 0 else "right"
        cameras.append(SimpleNamespace(
            image_name=f"scene_{side}_rgb_{frame:04d}",
            frame_no=frame))
    return cameras


def check_trajectory_contract():
    for frames_per_side in (133, 404):
        grouped = _group(make_cameras(frames_per_side))
        assert len(grouped["left"]) == frames_per_side
        assert len(grouped["right"]) == frames_per_side

    incomplete = make_cameras(404)
    incomplete.pop(200)
    try:
        _group(incomplete)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Incomplete trajectory was accepted")


def make_objective(offset_truth):
    objective = object.__new__(MotionCostVolumeLoss)
    objective.duration = 266.0
    objective.max_offset_frames = 24.0
    objective.max_drift_frames = 0.0
    frames = torch.arange(132, dtype=torch.float32) * 2.0
    centers = torch.nonzero(
        (frames >= 28.0) & (frames <= frames[-1] - 28.0),
        as_tuple=False).flatten()
    query = frames + offset_truth
    distance = frames.unsqueeze(0) - query.unsqueeze(1)
    volume = torch.exp(-distance.square() / 2.0)
    volume = (
        (volume - volume.mean(dim=1, keepdim=True))
        / volume.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6))
    objective.data = {
        side: {
            "frames": frames,
            "centers": centers,
            "volume": volume,
            "frame_step": 2.0,
        }
        for side in ("left", "right")
    }
    return objective


def recover(offset_truth):
    objective = make_objective(offset_truth)
    raw_offset = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam((raw_offset,), lr=0.005)
    for iteration in range(1, 1001):
        optimizer.zero_grad(set_to_none=True)
        offset = 24.0 * torch.tanh(raw_offset)
        loss, _ = objective.loss(offset, torch.zeros(()), iteration)
        loss.backward()
        optimizer.step()
    learned_offset = float((24.0 * torch.tanh(raw_offset)).detach())
    assert abs(learned_offset - offset_truth) <= 0.25, (
        offset_truth, learned_offset)


def main():
    check_trajectory_contract()
    for offset in (0.0, 8.0, 20.0):
        recover(offset)
    print("MOTION_COST_VOLUME_TEST_PASS")


if __name__ == "__main__":
    main()
