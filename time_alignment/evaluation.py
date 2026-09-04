import hashlib
import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import (
    ModelHiddenParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from gaussian_renderer import GaussianModel
from render import render_set
from scene import Scene
from utils.general_utils import safe_state


ARMS = {
    "baseline": (False, False),
    "time_only": (True, False),
    "pose_only": (False, True),
    "full": (True, True),
}


def camera_side(camera):
    name = camera.image_name.lower()
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    raise ValueError(f"Camera side missing from {camera.image_name}")


def attach_clock_to_test_cameras(scene):
    templates = {}
    for camera in scene.getTrainCameras():
        side = camera_side(camera)
        if side not in templates:
            templates[side] = camera
    if set(templates) != {"left", "right"}:
        raise RuntimeError(f"Missing train trajectory templates: {sorted(templates)}")

    for camera in scene.getTestCameras():
        template = templates[camera_side(camera)]
        camera.temporal_alignment_enabled = True
        camera.temporal_observation_correction_enabled = True
        camera.temporal_offset_raw = scene.temporal_offset_raw
        camera.temporal_offset_max_frames = scene.temporal_offset_max_frames
        camera.temporal_drift_raw = None
        camera.temporal_drift_max_endpoint_frames = 0.0
        camera.temporal_duration = template.temporal_duration
        camera.temporal_pose_frames = template.temporal_pose_frames
        camera.temporal_pose_track = template.temporal_pose_track
        camera.temporal_strict_common_support = True
    return templates


def use_learned_clock(scene):
    loaded_offset = float(scene.temporal_offset_frames().detach().cpu())
    if scene.temporal_affine_clock_enabled:
        raise RuntimeError("Four-arm evaluation requires a scalar clock")
    if not math.isfinite(loaded_offset):
        raise RuntimeError("Loaded clock is non-finite")
    return loaded_offset


def fixed_support(scene, support_contract):
    expected_names = support_contract["support_names"]
    expected_count = int(support_contract.get("support_count", 0))
    if expected_count <= 0 or len(expected_names) != expected_count:
        raise RuntimeError("Frozen matched support count is invalid")
    by_name = {camera.image_name: camera for camera in scene.getTestCameras()}
    if len(by_name) != len(scene.getTestCameras()):
        raise RuntimeError("Duplicate test camera names")
    missing = [name for name in expected_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Frozen support names missing: {missing[:3]}")
    return [by_name[name] for name in expected_names]


def clear_truth_metadata(cameras):
    for camera in cameras:
        camera.thermal_frame_shift = 0
        camera.thermal_source_frame = None
        camera.thermal_source_name = None


def validate_queries(cameras, applied_offset, duration):
    observed = []
    for camera in cameras:
        if not camera.temporal_observation_correction_enabled:
            raise RuntimeError("Temporal correction disabled on a test camera")
        query_frame = float(
            (camera.get_temporal_time().detach().cpu() * duration).item())
        delta = query_frame - float(camera.frame_no)
        if not math.isfinite(delta) or abs(delta - applied_offset) > 5e-4:
            raise RuntimeError(
                f"Test query mismatch for {camera.image_name}: {delta} vs {applied_offset}")
        pose = camera.get_temporal_rgb_world_view_transform()
        if not torch.isfinite(pose).all():
            raise RuntimeError(f"Non-finite temporal pose for {camera.image_name}")
        observed.append(delta)
    return min(observed), max(observed)


def validate_renderer_trace(trace, cameras, applied_offset, duration, time_enabled):
    if len(trace) != len(cameras):
        raise RuntimeError(
            f"Renderer trace count mismatch: {len(trace)} vs {len(cameras)}")
    deltas = []
    for row, camera in zip(trace, cameras):
        if row["image_name"] != camera.image_name:
            raise RuntimeError(
                f"Renderer trace order mismatch: {row['image_name']} vs {camera.image_name}")
        branch_applied = bool(row["temporal_branch_applied"])
        if branch_applied != time_enabled:
            raise RuntimeError(
                f"Renderer temporal branch mismatch for {camera.image_name}: "
                f"{branch_applied} vs {time_enabled}")
        delta = (float(row["thermal_time"]) - float(row["rgb_time"])) * duration
        if not math.isfinite(delta) or abs(delta - applied_offset) > 5e-4:
            raise RuntimeError(
                f"Renderer-consumed time mismatch for {camera.image_name}: "
                f"{delta} vs {applied_offset}")
        row["actual_delta_frames"] = delta
        deltas.append(delta)
    return min(deltas), max(deltas)


def main():
    parser = ArgumentParser(description="Clock-applied pure-forward test renderer")
    model = ModelParams(parser, sentinel=True)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hidden = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--configs", required=True, type=str)
    parser.add_argument("--output-root", required=True, type=str)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--support-contract", required=True, type=str)
    parser.add_argument("--expected-test-shift", required=True, type=int)
    parser.add_argument("--trace-gate-views", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    import mmengine
    from utils.params_utils import merge_hparams

    args = merge_hparams(args, mmengine.Config.fromfile(args.configs))
    safe_state(args.quiet)
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing existing output root: {output_root}")
    output_root.mkdir(parents=True)

    eval_shift = int(os.environ.get("ED3DGS_EVAL_THERMAL_FRAME_SHIFT", "0"))
    train_shift = int(os.environ.get("ED3DGS_THERMAL_FRAME_SHIFT", "0"))
    if eval_shift != args.expected_test_shift or train_shift != 0:
        raise RuntimeError(
            f"Shift environment mismatch: train={train_shift}, eval={eval_shift}")

    dataset = model.extract(args)
    opt = optimization.extract(args)
    pipe = pipeline.extract(args)
    hyper = hidden.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        duration=hyper.total_num_frames,
        loader=dataset.loader,
        opt=opt,
    )
    time_enabled, pose_enabled = ARMS[args.arm]
    if bool(scene.temporal_alignment_enabled) != time_enabled:
        raise RuntimeError("Loaded temporal flag disagrees with arm")
    if bool(scene.has_thermal_pose) != pose_enabled:
        raise RuntimeError("Loaded pose flag disagrees with arm")
    support_contract = json.loads(Path(args.support_contract).read_text())
    views = fixed_support(scene, support_contract)
    full_support_names = [camera.image_name for camera in views]
    full_support_digest = hashlib.sha256(
        ("\n".join(full_support_names) + "\n").encode("utf-8")).hexdigest()
    if full_support_digest != support_contract.get("support_names_sha256"):
        raise RuntimeError(
            "Fixed-support names do not match their declared SHA-256: "
            f"computed={full_support_digest} "
            f"declared={support_contract.get('support_names_sha256')}")
    if args.trace_gate_views < 0 or args.trace_gate_views > len(views):
        raise RuntimeError("Invalid trace Gate view count")
    gate_only = args.trace_gate_views > 0
    if gate_only:
        views = views[:args.trace_gate_views]
    support_names = [camera.image_name for camera in views]
    full_support_count = len(views)
    if time_enabled:
        attach_clock_to_test_cameras(scene)
        loaded_offset = use_learned_clock(scene)
        applied_offset = loaded_offset
    else:
        loaded_offset = None
        applied_offset = 0.0

    # The evaluation harness used the injected shift only to construct GT and
    # choose common support. Remove all truth metadata before model inference.
    clear_truth_metadata(views)
    if time_enabled:
        query_min, query_max = validate_queries(
            views, applied_offset, float(scene.maxtime))
    else:
        query_min = query_max = 0.0

    background_value = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(background_value, dtype=torch.float32, device="cuda")
    background_thermal = torch.tensor(
        background_value, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        renderer_time_trace = render_set(
            str(output_root), "test", scene.loaded_iter, views, gaussians, pipe,
            background, background_thermal, hyperparam=hyper)

    actual_query_min, actual_query_max = validate_renderer_trace(
        renderer_time_trace, views, applied_offset, float(scene.maxtime),
        time_enabled)
    (output_root / "renderer_time_trace.json").write_text(
        json.dumps(renderer_time_trace, indent=2, sort_keys=True) + "\n")

    method = f"ours_{scene.loaded_iter}"
    support_digest = hashlib.sha256(
        ("\n".join(support_names) + "\n").encode("utf-8")).hexdigest()
    contract = {
        "schema": ("covers_r25_renderer_clock_gate_r7" if gate_only
                   else "covers_r25_fourarm_eval_v2"),
        "status": "PASS",
        "arm": args.arm,
        "loaded_offset_frames": loaded_offset,
        "applied_offset_frames": applied_offset,
        "synthetic_test_shift_frames": int(args.expected_test_shift),
        "train_shift_frames": train_shift,
        "full_common_support_count": full_support_count,
        "rendered_support_count": len(views),
        "support_names_sha256": support_digest,
        "query_delta_min_frames": query_min,
        "query_delta_max_frames": query_max,
        "renderer_actual_delta_min_frames": actual_query_min,
        "renderer_actual_delta_max_frames": actual_query_max,
        "renderer_temporal_branch_applied_all": all(
            row["temporal_branch_applied"] for row in renderer_time_trace),
        "renderer_time_trace_file": "renderer_time_trace.json",
        "test_cameras_temporal_enabled": time_enabled,
        "thermal_pose_enabled": pose_enabled,
        "truth_metadata_cleared_before_forward": True,
        "renderer_reads_shift_truth": False,
        "test_time_pose_or_fov_optimization": False,
        "method": method,
        "support_names": support_names,
    }
    (output_root / "render_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n")
    if gate_only:
        (output_root / "RENDER_CLOCK_GATE_PASS").touch()
        print("RENDER_CLOCK_GATE_PASS " + json.dumps(contract, sort_keys=True))
        return
    print("CLOCK_APPLIED_RENDER_COMPLETE " + json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
