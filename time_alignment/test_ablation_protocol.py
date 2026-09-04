#!/usr/bin/env python3
"""CPU checks for the locked shift20 ablation configuration and support."""

import ast
import hashlib
import json
import math
import os
import runpy
from dataclasses import dataclass
from pathlib import Path

from time_alignment import schedule


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "arguments" / "fixed_support_shift20.json"
SHIFT0_SUPPORT = ROOT / "arguments" / "fixed_support_shift0.json"


@dataclass
class FakeCamera:
    image_name: str
    side: str
    frame: int
    thermal_frame_shift: int = 20

    def effective_temporal_offset_frames(self):
        return FakeValue(getattr(self, "effective_offset", 0.0))

    @property
    def temporal_pose_frames(self):
        return [FakeValue(0.0), FakeValue(265.0)]


class FakeValue:
    def __init__(self, value):
        self.value = float(value)

    def detach(self):
        return self

    def item(self):
        return self.value


class FakeScene:
    temporal_alignment_enabled = False
    maxtime = 266

    def __init__(self, cameras):
        self._cameras = cameras
        self.stage2_training_cameras = cameras[24:242]
        self.stage2_all_train_cameras = tuple(cameras)

    def getTrainCameras(self):
        # Reproduce the runtime failure: the active Stage-1 pool has only 218
        # cameras when the Stage-2 support switch is requested.
        return self.stage2_training_cameras

    @staticmethod
    def _camera_state_side_frame(camera):
        return camera.side, camera.frame


def load_support_function(name):
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
        "Path": Path,
    }
    exec(compile(module, str(ROOT / "train.py"), "exec"), namespace)
    return namespace[node.name]


def check_configuration_matrix():
    cases = {
        "fixed_clock": ("pose_only", False, True, False),
        "total_gradient": ("full", True, True, True),
        "frozen_pose": ("time_only", True, False, True),
    }
    original = dict(os.environ)
    try:
        for mode, (arm, time_enabled, pose_enabled, joint_clock) in cases.items():
            os.environ.update({
                "ED3DGS_SHIFT20_ABLATION_MODE": mode,
                "ED3DGS_R25_FOURARM": arm,
                "ED3DGS_ITERATIONS": "1000",
                "ED3DGS_CAPACITY_ARM": "modality_densify",
                "ED3DGS_STAGE2_TEACHER_MODEL_PATH": "cpu-test-teacher",
                "ED3DGS_JOINT_CLOCK_LOSS": "1" if joint_clock else "0",
                "ED3DGS_TEMPORAL_CONSENSUS_V1": "1" if joint_clock else "0",
                "ED3DGS_SELF_CALIBRATING_CLOCK_STUDY": (
                    "1" if joint_clock else "0"),
                "ED3DGS_JOINT_CLOCK_LOSS_VARIANT": "routed_ngf",
            })
            values = runpy.run_path(str(ROOT / "arguments" / "covers.py"))
            assert values["TIME_ENABLED"] is time_enabled
            assert values["POSE_ENABLED"] is pose_enabled
            assert values["JOINT_CLOCK"] is joint_clock
    finally:
        os.environ.clear()
        os.environ.update(original)


def check_strict_scene_freeze_schedule():
    assert schedule.strict_block_step_counts(64) == {
        "scene": 31, "calibration": 32}
    assert schedule.strict_block_step_counts(1000) == {
        "scene": 496, "calibration": 503}
    assert schedule.strict_block_step_counts(2560) == {
        "scene": 1279, "calibration": 1280}

    original = dict(os.environ)
    try:
        os.environ.update({
            "ED3DGS_R25_FOURARM": "full",
            "ED3DGS_ITERATIONS": "2560",
            "ED3DGS_CAPACITY_ARM": "modality_densify",
            "ED3DGS_STAGE2_TEACHER_MODEL_PATH": "cpu-test-teacher",
            "ED3DGS_JOINT_CLOCK_LOSS": "1",
            "ED3DGS_TEMPORAL_CONSENSUS_V1": "1",
            "ED3DGS_SELF_CALIBRATING_CLOCK_STUDY": "1",
            "ED3DGS_JOINT_CLOCK_LOSS_VARIANT": "routed_ngf",
            "ED3DGS_STRICT_SCENE_FREEZE_V1": "1",
        })
        values = runpy.run_path(str(ROOT / "arguments" / "covers.py"))
        optimization = values["OptimizationParams"]
        assert optimization["scene_optimizer_every_iteration"] is False
        assert optimization["scene_optimizer_target_steps"] == 1279
    finally:
        os.environ.clear()
        os.environ.update(original)


def check_fixed_support():
    contract = json.loads(SUPPORT.read_text(encoding="utf-8"))
    cameras = []
    for frame in range(266):
        side = "left" if frame % 2 == 0 else "right"
        cameras.append(FakeCamera(
            image_name=f"covers_{side}_rgb_{frame:04d}",
            side=side,
            frame=frame,
        ))
    scene = FakeScene(cameras)
    original = dict(os.environ)
    try:
        os.environ.update({
            "ED3DGS_R25_DUAL_SUPPORT": "1",
            "ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT": str(SUPPORT),
            "ED3DGS_THERMAL_FRAME_SHIFT": "20",
        })
        report = load_support_function(
            "configure_fixed_clock_reconstruction_support")(scene)
    finally:
        os.environ.clear()
        os.environ.update(original)
    assert report["schema"] == "covers_fixed_clock_dual_support"
    assert report["reconstruction_camera_count"] == 244
    assert report["reconstruction_cameras_by_side"] == {
        "left": 122, "right": 122}
    assert report["selected_names"] == contract["training_support_names"]
    assert report["shift_truth_used_for_selection"] is False
    assert report["learned_clock_used_for_selection"] is False
    assert [Path(camera.image_name).stem + ".png"
            for camera in scene.stage2_reconstruction_cameras] \
        == contract["training_support_names"]


def check_transition_preflight():
    cameras = []
    for frame in range(266):
        side = "left" if frame % 2 == 0 else "right"
        cameras.append(FakeCamera(
            image_name=f"covers_{side}_rgb_{frame:04d}",
            side=side,
            frame=frame,
        ))
    preflight = load_support_function(
        "preflight_reconstruction_support_snapshot")
    original = dict(os.environ)
    try:
        for path, expected_count in ((SUPPORT, 244), (SHIFT0_SUPPORT, 256)):
            scene = FakeScene(cameras)
            os.environ.update({
                "ED3DGS_R25_DUAL_SUPPORT": "1",
                "ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT": str(path),
            })
            report = preflight(scene)
            contract = json.loads(path.read_text(encoding="utf-8"))
            assert report["immutable_camera_snapshot"] is True
            assert report["all_trajectory_camera_count"] == 266
            assert report["calibration_camera_count"] == 218
            assert report["reconstruction_camera_count"] == expected_count
            assert report["selected_names_sha256"] \
                == contract["training_support_names_sha256"]
            assert report["evaluation_frame_start"] \
                == contract["evaluation_frame_start"]
            assert report["evaluation_frame_end_exclusive"] \
                == contract["evaluation_frame_end_exclusive"]
            assert not hasattr(scene, "stage2_reconstruction_cameras")
    finally:
        os.environ.clear()
        os.environ.update(original)


class FakeClockScene(FakeScene):
    temporal_alignment_enabled = True

    def __init__(self, cameras, offset):
        super().__init__(cameras)
        self._offset = float(offset)
        for camera in cameras:
            camera.effective_offset = float(offset)

    def temporal_offset_frames(self):
        return FakeValue(self._offset)

    def temporal_drift_frames(self):
        return FakeValue(0.0)

    def temporal_endpoint_offsets(self):
        return [FakeValue(self._offset), FakeValue(self._offset)]


def check_shift0_boundary_safe_support():
    contract = json.loads(SHIFT0_SUPPORT.read_text(encoding="utf-8"))
    names = list(contract["training_support_names"])
    digest = hashlib.sha256(
        ("\n".join(names) + "\n").encode("utf-8")).hexdigest()
    assert contract["support_count"] == 256
    assert contract["training_frame_start"] == 1
    assert contract["training_frame_end_exclusive"] == 257
    assert contract["evaluation_frame_start"] == 1
    assert contract["evaluation_frame_end_exclusive"] == 257
    assert digest == contract["training_support_names_sha256"]
    evaluation_digest = hashlib.sha256(
        ("\n".join(contract["support_names"]) + "\n").encode(
            "utf-8")).hexdigest()
    assert evaluation_digest == contract["support_names_sha256"]

    cameras = []
    for frame in range(266):
        side = "left" if frame % 2 == 0 else "right"
        cameras.append(FakeCamera(
            image_name=f"covers_{side}_rgb_{frame:04d}",
            side=side,
            frame=frame,
            thermal_frame_shift=0,
        ))
    scene = FakeClockScene(cameras, offset=-0.019163241609930992)
    original = dict(os.environ)
    try:
        os.environ.update({
            "ED3DGS_R25_DUAL_SUPPORT": "1",
            "ED3DGS_RECONSTRUCTION_SUPPORT_CONTRACT": str(SHIFT0_SUPPORT),
            "ED3DGS_THERMAL_FRAME_SHIFT": "0",
        })
        report = load_support_function(
            "configure_blind_reconstruction_support")(
                scene, {"learned_offset_frames": scene._offset})
    finally:
        os.environ.clear()
        os.environ.update(original)
    assert report["reconstruction_camera_count"] == 256
    assert report["reconstruction_cameras_by_side"] == {
        "left": 128, "right": 128}
    assert report["selection_policy"] == "pre_registered_train_frames_1_256"
    assert report["selected_names"] == names


if __name__ == "__main__":
    check_configuration_matrix()
    check_strict_scene_freeze_schedule()
    check_fixed_support()
    check_transition_preflight()
    check_shift0_boundary_safe_support()
    print("SHIFT20_ABLATION_CPU_TEST_PASS")
