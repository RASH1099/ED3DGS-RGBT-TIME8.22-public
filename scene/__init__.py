#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import math
import json
import re
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from PIL import Image 
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfosv2, cameraList_from_camInfosHyper
from utils.graphics_utils import recordpointshelper
import torch
from time_alignment.pose_math import project_pose_
import torch.nn.functional as F


def _temporal_change_gray(image, size=(120, 160)):
    if image.shape[0] == 3:
        gray = (
            0.299 * image[0:1]
            + 0.587 * image[1:2]
            + 0.114 * image[2:3]
        )
    else:
        gray = image.mean(dim=0, keepdim=True)
    return F.interpolate(
        gray.unsqueeze(0), size=size, mode="bilinear", align_corners=False
    )[0, 0].detach().cpu()


def _standardized_temporal_change(first, second):
    change = (second - first).abs()
    border_y = max(1, change.shape[0] // 12)
    border_x = max(1, change.shape[1] // 12)
    change = change[border_y:-border_y, border_x:-border_x]
    std = change.std(unbiased=False)
    if not torch.isfinite(std):
        raise ValueError("Non-finite train temporal-change map")
    if std.item() == 0.0:
        return None
    return (change - change.mean()) / std


def _train_change_offset_estimate(train_cameras, max_offset_frames):
    cameras_by_side = {"left": [], "right": []}
    for camera in train_cameras:
        name = camera.image_name.lower()
        if "left" in name:
            side = "left"
        elif "right" in name:
            side = "right"
        else:
            raise ValueError(f"Camera side missing from train id: {camera.image_name}")
        if camera.frame_no is None:
            raise ValueError(f"Frame number missing from train camera: {camera.image_name}")
        cameras_by_side[side].append(camera)

    rgb_changes = {}
    thermal_changes = {}
    with torch.no_grad():
        for side, cameras in cameras_by_side.items():
            cameras.sort(key=lambda camera: int(camera.frame_no))
            rgb_frames = {
                int(camera.frame_no): _temporal_change_gray(camera.original_image)
                for camera in cameras
            }
            thermal_frames = {
                int(camera.frame_no): _temporal_change_gray(camera.thermal_image)
                for camera in cameras
            }
            frames = sorted(rgb_frames)
            for first, second in zip(frames[:-1], frames[1:]):
                gap = second - first
                key = (side, first, gap)
                rgb_change = _standardized_temporal_change(
                    rgb_frames[first], rgb_frames[second])
                thermal_change = _standardized_temporal_change(
                    thermal_frames[first], thermal_frames[second])
                if rgb_change is None or thermal_change is None:
                    continue
                rgb_changes[key] = rgb_change
                thermal_changes[key] = thermal_change

    max_offset = int(max_offset_frames)
    scan = {}
    for offset in range(-max_offset, max_offset + 1):
        side_scores = {"left": [], "right": []}
        for (side, frame, gap), thermal_change in thermal_changes.items():
            rgb_change = rgb_changes.get((side, frame + offset, gap))
            if rgb_change is None:
                continue
            denominator = torch.linalg.vector_norm(thermal_change) * torch.linalg.vector_norm(rgb_change)
            if not torch.isfinite(denominator) or denominator.item() == 0.0:
                raise ValueError("Invalid train temporal-change norm")
            score = (thermal_change * rgb_change).sum() / denominator
            side_scores[side].append(score.item())
        if min(len(scores) for scores in side_scores.values()) == 0:
            continue
        combined = side_scores["left"] + side_scores["right"]
        scan[offset] = {
            "all": sum(combined) / len(combined),
            "left": sum(side_scores["left"]) / len(side_scores["left"]),
            "right": sum(side_scores["right"]) / len(side_scores["right"]),
            "count_all": len(combined),
            "count_left": len(side_scores["left"]),
            "count_right": len(side_scores["right"]),
        }

    maxima = {
        key: max(scan, key=lambda offset: scan[offset][key])
        for key in ("all", "left", "right")
    }
    if len(set(maxima.values())) != 1:
        raise RuntimeError(f"Train temporal-change offset disagrees by side: {maxima}")
    estimate = maxima["all"]
    if abs(estimate) >= max_offset_frames:
        raise RuntimeError(
            f"Train temporal-change optimum reached search boundary: {estimate}"
        )
    return estimate, {
        "schema": "train_change_temporal_offset_v1",
        "uses_train_cameras_only": True,
        "score": "mean cosine similarity of standardized absolute temporal-change maps",
        "resolution": [160, 120],
        "maxima": maxima,
        "scan": {str(offset): values for offset, values in scan.items()},
    }


class Scene:
    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True,
                 duration=None, loader=None, testonly=None, opt=None,
                 load_test_cameras=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.refmodelpath = None
        self.maxtime = duration
        self.rgb_only_teacher = bool(getattr(opt, "rgb_only_teacher", False))
        args.rgb_only_teacher = self.rgb_only_teacher

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        resolution_scales= [1.0]
        raydict = {}
        print("loader", loader)

        if loader == "dynerf":
            scene_info = sceneLoadTypeCallbacks["Dynerf"](args.source_path, args.source_path, args.eval, duration=300)
        elif loader == "technicolor" or loader == "technicolorvalid" :
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args.source_path, args.images, args.eval, duration=50, testonly=testonly)
        elif loader == "nerfies":
            scene_info = sceneLoadTypeCallbacks["Nerfies"](
                args.source_path, False, args.eval,
                load_test_cameras=load_test_cameras,
                rgb_only_teacher=self.rgb_only_teacher)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # for id, cam in enumerate(camlist):
            #     json_cams.append(camera_to_JSON(id, cam))
            # with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            #     json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling


        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")  
            if loader in ["technicolorvalid"]:         
                self.train_cameras[resolution_scale] = [] # no training data
            elif loader in ["nerfies"]:
                self.train_cameras[resolution_scale] = cameraList_from_camInfosHyper(scene_info.train_cameras, resolution_scale, args)
            else: 
                self.train_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.train_cameras, resolution_scale, args)
            
            
            if load_test_cameras:
                print("Loading Test Cameras")
                if loader  in ["technicolorvalid", "technicolor", "dynerf", "dynerfvalid"]: # we need gt for metrics
                    self.test_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.test_cameras, resolution_scale, args)
                elif loader in ["nerfies"]:
                    self.test_cameras[resolution_scale] = cameraList_from_camInfosHyper(scene_info.test_cameras, resolution_scale, args)
            else:
                self.test_cameras[resolution_scale] = []
                print("[Protocol] Formal test cameras NOT CONSTRUCTED")

            self._setup_temporal_alignment(opt, duration, resolution_scale)

            # print("Loading Video Cameras")
            # if loader  in ["technicolorvalid", "technicolor", "dynerf", "dynerfvalid"]: # we need gt for metrics
            #     self.video_cameras[resolution_scale] = cameraList_from_camInfosv2(scene_info.video_cameras, resolution_scale, args)
            # elif loader in ["nerfies"]:
            #     self.video_cameras[resolution_scale] = cameraList_from_camInfosHyper(scene_info.video_cameras, resolution_scale, args)


        if loader not in ["nerfies", "dynerf"]:
            for cam in self.test_cameras[resolution_scale]:
                if cam.image_name[:4] not in raydict and cam.rayo is not None:
                    raydict[cam.image_name[:4]] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W

            for cam in self.test_cameras[resolution_scale]:
                cam.rays = raydict[cam.image_name[:4]] # should be direct ?

            if not testonly:
                for cam in self.train_cameras[resolution_scale]:
                    if cam.image_name[:4] not in raydict and cam.rayo is not None:
                        raydict[cam.image_name[:4]] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda() # 1 x 6 x H x W

                for cam in self.train_cameras[resolution_scale]:
                    cam.rays = raydict[cam.image_name[:4]] # should be direct ?


        if self.loaded_iter :
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
            thermal_state_path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "thermal_camera_state.pth",
            )
            self.load_thermal_camera_state(thermal_state_path, scale=resolution_scales[0], apply_to_test=True)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime,self.train_cameras)

        # ===== Thermal pose optimizers (Self-Cali-GS style: per-camera, separate Adam) =====
        self.has_thermal_pose = False
        self.optimizer_thermal_rotation = None
        self.optimizer_thermal_translation = None
        self.optimizer_thermal_fovx = None
        self.optimizer_thermal_fovy = None
        self.scheduler_thermal_fovx = None
        self.scheduler_thermal_fovy = None
        self.thermal_intrinsic_max_rel_change = float(
            getattr(opt, 'thermal_intrinsic_max_rel_change', 0.0))
        if not 0.0 <= self.thermal_intrinsic_max_rel_change < 1.0:
            raise ValueError("thermal_intrinsic_max_rel_change must be in [0, 1)")

        thermal_cams = [c for c in self.train_cameras[resolution_scale]
                        if hasattr(c, 'has_thermal') and c.has_thermal]
        all_thermal_cams = thermal_cams + [
            c for c in self.test_cameras[resolution_scale]
            if hasattr(c, 'has_thermal') and c.has_thermal
        ]
        tied_aspect = bool(getattr(opt, 'thermal_intrinsic_tied_aspect', False))
        for cam in all_thermal_cams:
            cam.thermal_intrinsic_tied_aspect = tied_aspect
        if getattr(opt, 'no_thermal_pose_opt', False):
            print("[ThermalPose] SKIPPED — no_thermal_pose_opt=True, thermal delta stays identity")
        elif thermal_cams:
            self.has_thermal_pose = True
            if getattr(opt, 'thermal_pose_shared_by_side', False):
                shared_pose = {}
                for cam in thermal_cams:
                    side, _ = self._camera_state_side_frame(cam)
                    if side is None:
                        raise ValueError(
                            "thermal_pose_shared_by_side requires left/right in every camera name: "
                            f"{cam.image_name}"
                        )
                    if side not in shared_pose:
                        shared_pose[side] = (
                            cam.thermal_delta_quaternion,
                            cam.thermal_delta_translation,
                        )
                    else:
                        cam.thermal_delta_quaternion = shared_pose[side][0]
                        cam.thermal_delta_translation = shared_pose[side][1]
                print(
                    "[ThermalPose] shared rigid deltas by side: "
                    + ", ".join(sorted(shared_pose))
                )

            if getattr(opt, 'thermal_intrinsic_shared_by_side', False):
                shared_intrinsics = {}
                for cam in thermal_cams:
                    side, _ = self._camera_state_side_frame(cam)
                    if side is None:
                        raise ValueError(
                            "thermal_intrinsic_shared_by_side requires left/right in every camera name: "
                            f"{cam.image_name}"
                        )
                    if side not in shared_intrinsics:
                        shared_intrinsics[side] = (
                            cam.learnable_tfovx,
                            cam.learnable_tfovy,
                        )
                    else:
                        cam.learnable_tfovx = shared_intrinsics[side][0]
                        cam.learnable_tfovy = shared_intrinsics[side][1]
                print(
                    "[ThermalIntrinsic] shared FoV parameters by side: "
                    + ", ".join(sorted(shared_intrinsics))
                )

            def unique_param_groups(cameras, name, lr):
                groups = []
                seen = set()
                for camera in cameras:
                    param = getattr(camera, name)
                    if id(param) in seen:
                        continue
                    seen.add(id(param))
                    groups.append({'params': [param], 'lr': lr})
                return groups

            # Rotation: delta_quaternion, lr from config, milestones [7000, 30000], gamma=0.5
            l_rot = unique_param_groups(
                thermal_cams, 'thermal_delta_quaternion', opt.thermal_pose_lr_r)
            self.optimizer_thermal_rotation = torch.optim.Adam(l_rot, eps=1e-15)
            self.scheduler_thermal_rotation = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_thermal_rotation, milestones=[7000, 30000], gamma=0.5)
            # Translation: delta_translation
            l_tr = unique_param_groups(
                thermal_cams, 'thermal_delta_translation', opt.thermal_pose_lr_t)
            self.optimizer_thermal_translation = torch.optim.Adam(l_tr, eps=1e-15)
            self.scheduler_thermal_translation = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_thermal_translation, milestones=[7000, 30000], gamma=0.5)
            # Thermal FoV
            l_fx = unique_param_groups(
                thermal_cams, 'learnable_tfovx', opt.thermal_intrinsic_lr)
            self.optimizer_thermal_fovx = torch.optim.Adam(l_fx, eps=1e-15)
            self.scheduler_thermal_fovx = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_thermal_fovx, milestones=[10000], gamma=0.5)
            if not tied_aspect:
                l_fy = unique_param_groups(
                    thermal_cams, 'learnable_tfovy', opt.thermal_intrinsic_lr)
                self.optimizer_thermal_fovy = torch.optim.Adam(l_fy, eps=1e-15)
                self.scheduler_thermal_fovy = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer_thermal_fovy, milestones=[10000], gamma=0.5)

    def _setup_temporal_alignment(self, opt, duration, scale):
        self.temporal_alignment_enabled = bool(
            getattr(opt, 'temporal_alignment_enabled', False))
        self.temporal_offset_max_frames = float(
            getattr(opt, 'temporal_offset_max_frames', 12.0))
        if self.temporal_offset_max_frames <= 0:
            raise ValueError("temporal_offset_max_frames must be positive")

        self.temporal_offset_raw = torch.nn.Parameter(
            torch.zeros((), device="cuda"),
            requires_grad=self.temporal_alignment_enabled,
        )
        self.temporal_affine_clock_enabled = bool(
            getattr(opt, 'temporal_affine_clock_enabled', False))
        self.temporal_drift_max_endpoint_frames = float(
            getattr(opt, 'temporal_drift_max_endpoint_frames', 4.0))
        if (self.temporal_affine_clock_enabled
                and self.temporal_drift_max_endpoint_frames <= 0.0):
            raise ValueError(
                "temporal_drift_max_endpoint_frames must be positive")
        self.temporal_drift_raw = torch.nn.Parameter(
            torch.zeros((), device="cuda"),
            requires_grad=(self.temporal_alignment_enabled
                           and self.temporal_affine_clock_enabled),
        )
        self.optimizer_temporal_offset = None
        if not self.temporal_alignment_enabled:
            return

        duration = int(duration)
        if duration <= 1:
            raise ValueError("Temporal alignment requires duration > 1")
        train_cameras = self.train_cameras[scale]
        forced_offset = getattr(opt, 'temporal_offset_force_frames', None)
        internal_v30 = os.environ.get("ED3DGS_INTERNAL_CLOCK_V30") == "1"
        internal_v34 = os.environ.get("ED3DGS_INTERNAL_CLOCK_V34") == "1"
        internal_geoflow = (
            os.environ.get("ED3DGS_GEOFLOW_SOFTVOLUME_V1") == "1")
        if sum((internal_v30, internal_v34, internal_geoflow)) > 1:
            raise ValueError("Multiple internal clock implementations enabled")
        internal_clock = internal_v30 or internal_v34 or internal_geoflow
        if internal_clock:
            if forced_offset is not None:
                raise ValueError("v30 forbids forced temporal initialization")
            estimated_offset = None
            selected_offset = 0.0
            initialization = "same_process_raw_zero_learning"
            change_report = {
                "schema": ("in_process_temporal_offset_geoflow_v1"
                           if internal_geoflow else
                           "in_process_temporal_offset_v34"
                           if internal_v34
                           else "in_process_temporal_offset_v30"),
                "uses_train_cameras_only": True,
                "candidate_enumeration": False,
                "forced_offset_frames": None,
                "selected_offset_frames": 0.0,
                "external_bootstrap_or_report_loaded": False,
            }
        elif forced_offset is None:
            estimated_offset, change_report = _train_change_offset_estimate(
                train_cameras, self.temporal_offset_max_frames)
            selected_offset = float(estimated_offset)
            initialization = "train_change_scan"
        else:
            estimated_offset = None
            selected_offset = float(forced_offset)
            initialization = "forced_without_candidate_scan"
            change_report = {
                "schema": "forced_temporal_offset_v1",
                "uses_train_cameras_only": True,
                "candidate_enumeration": False,
                "forced_offset_frames": selected_offset,
            }
        if (not math.isfinite(selected_offset)
                or abs(selected_offset) >= self.temporal_offset_max_frames):
            raise ValueError(
                "temporal_offset_force_frames must be finite and strictly within "
                "temporal_offset_max_frames")
        normalized_offset = torch.as_tensor(
            selected_offset / self.temporal_offset_max_frames,
            device=self.temporal_offset_raw.device,
            dtype=self.temporal_offset_raw.dtype,
        )
        with torch.no_grad():
            self.temporal_offset_raw.copy_(torch.atanh(normalized_offset))
        change_report["estimated_offset_frames"] = estimated_offset
        change_report["forced_offset_frames"] = (
            None if forced_offset is None else selected_offset)
        change_report["selected_offset_frames"] = selected_offset
        report_path = os.path.join(
            self.model_path, "train_change_temporal_offset.json")
        if os.path.exists(report_path):
            with open(report_path, "r") as report_file:
                existing_report = json.load(report_file)
            if existing_report != change_report:
                raise RuntimeError(
                    f"Train temporal-change report mismatch: {report_path}")
        else:
            with open(report_path, "w") as report_file:
                json.dump(change_report, report_file, indent=2, sort_keys=True)
                report_file.write("\n")
        print(
            "[TemporalAlignment] train-change initialization: "
            f"mode={initialization} "
            f"estimated_offset_frames={estimated_offset} "
            f"selected_offset_frames={selected_offset:+.6f} "
            f"forced={forced_offset is not None} train_only=true "
            f"candidate_enumeration={change_report.get('candidate_enumeration', True)}"
        )
        tracks = {}
        track_frames = {}
        track_frame_sets = {}
        for side in ("left", "right"):
            by_frame = {}
            for cam in train_cameras:
                cam_side, frame = self._camera_state_side_frame(cam)
                if cam_side == side and frame is not None:
                    if frame in by_frame:
                        raise AssertionError(f"Duplicate {side} trajectory frame {frame}")
                    by_frame[frame] = cam.world_view_transform.detach().clone()
            frames = sorted(by_frame)
            if len(frames) < 2:
                raise AssertionError(f"Insufficient train-only {side} trajectory knots")
            tracks[side] = torch.stack([by_frame[frame] for frame in frames], dim=0)
            track_frames[side] = torch.tensor(
                frames, device=tracks[side].device, dtype=tracks[side].dtype)
            track_frame_sets[side] = set(frames)

        valid_shifted = 0
        for cam in train_cameras:
            side, _ = self._camera_state_side_frame(cam)
            if side not in tracks:
                raise AssertionError(f"Temporal trajectory side missing for {cam.image_name}")
            if cam.thermal_frame_shift != 0:
                _, frame = self._camera_state_side_frame(cam)
                source_frame = frame + cam.thermal_frame_shift
                if source_frame not in track_frame_sets[side]:
                    raise AssertionError(
                        f"Shifted source pose is not train-only: {cam.image_name} -> {source_frame}")
                valid_shifted += 1
            cam.temporal_alignment_enabled = True
            cam.temporal_observation_correction_enabled = True
            cam.temporal_offset_raw = self.temporal_offset_raw
            cam.temporal_offset_max_frames = self.temporal_offset_max_frames
            cam.temporal_drift_raw = (
                self.temporal_drift_raw
                if self.temporal_affine_clock_enabled else None)
            cam.temporal_drift_max_endpoint_frames = (
                self.temporal_drift_max_endpoint_frames)
            cam.temporal_duration = float(duration)
            cam.temporal_pose_frames = track_frames[side]
            cam.temporal_pose_track = tracks[side]
            cam.temporal_strict_common_support = bool(
                getattr(opt, 'temporal_strict_common_support', False))

        lr = float(getattr(opt, 'temporal_offset_lr', 0.0))
        if lr <= 0:
            raise ValueError("temporal_offset_lr must be positive when temporal alignment is enabled")
        clock_parameters = [self.temporal_offset_raw]
        if self.temporal_affine_clock_enabled:
            clock_parameters.append(self.temporal_drift_raw)
        self.optimizer_temporal_offset = torch.optim.Adam(
            clock_parameters, lr=lr, eps=1e-15)
        print(
            "[TemporalAlignment] enabled: shared_offset=true "
            f"max_frames={self.temporal_offset_max_frames} lr={lr} duration={duration} "
            f"train_only=true valid_shifted={valid_shifted} "
            f"knots_left={len(tracks['left'])} knots_right={len(tracks['right'])}"
        )

    def temporal_offset_frames(self):
        return self.temporal_offset_max_frames * torch.tanh(self.temporal_offset_raw)

    def temporal_drift_frames(self):
        if not self.temporal_affine_clock_enabled:
            return torch.zeros_like(self.temporal_offset_raw)
        return (self.temporal_drift_max_endpoint_frames
                * torch.tanh(self.temporal_drift_raw))

    def temporal_endpoint_offsets(self):
        offset = self.temporal_offset_frames()
        drift = self.temporal_drift_frames()
        return offset - drift, offset + drift

    def clamp_thermal_pose(self, max_rotation_degrees=5.0,
                           max_translation_fraction=0.1, scale=1.0):
        seen = set()
        for camera in self.train_cameras[scale]:
            if not getattr(camera, "has_thermal", False):
                continue
            key = (id(camera.thermal_delta_quaternion),
                   id(camera.thermal_delta_translation))
            if key in seen:
                continue
            seen.add(key)
            project_pose_(
                camera.thermal_delta_quaternion,
                camera.thermal_delta_translation,
                self.cameras_extent,
                max_rotation_degrees=max_rotation_degrees,
                max_translation_fraction=max_translation_fraction)

    def clamp_thermal_intrinsics(self, scale=1.0):
        """Project learned FoVs to a physically plausible focal-length interval."""
        max_rel = self.thermal_intrinsic_max_rel_change
        if max_rel <= 0:
            return

        seen_x = set()
        seen_y = set()
        with torch.no_grad():
            for cam in self.train_cameras.get(scale, []):
                if not (hasattr(cam, 'has_thermal') and cam.has_thermal):
                    continue
                if id(cam.learnable_tfovx) not in seen_x:
                    seen_x.add(id(cam.learnable_tfovx))
                    tan_half = torch.tan(torch.as_tensor(
                        0.5 * cam.TFoVx,
                        device=cam.learnable_tfovx.device,
                        dtype=cam.learnable_tfovx.dtype,
                    ))
                    lower = 2.0 * torch.atan(tan_half / (1.0 + max_rel))
                    upper = 2.0 * torch.atan(tan_half / (1.0 - max_rel))
                    cam.learnable_tfovx.clamp_(lower.item(), upper.item())
                if (not cam.thermal_intrinsic_tied_aspect
                        and id(cam.learnable_tfovy) not in seen_y):
                    seen_y.add(id(cam.learnable_tfovy))
                    tan_half = torch.tan(torch.as_tensor(
                        0.5 * cam.TFoVy,
                        device=cam.learnable_tfovy.device,
                        dtype=cam.learnable_tfovy.dtype,
                    ))
                    lower = 2.0 * torch.atan(tan_half / (1.0 + max_rel))
                    upper = 2.0 * torch.atan(tan_half / (1.0 - max_rel))
                    cam.learnable_tfovy.clamp_(lower.item(), upper.item())

    @staticmethod
    def _camera_state_name(cam_or_name):
        name = getattr(cam_or_name, "image_name", cam_or_name)
        return os.path.splitext(os.path.basename(str(name)))[0]

    @staticmethod
    def _camera_state_side_frame(cam_or_name):
        base = Scene._camera_state_name(cam_or_name).lower()
        side = None
        if "left" in base:
            side = "left"
        elif "right" in base:
            side = "right"
        match = re.search(r"(\d+)$", base)
        frame = int(match.group(1)) if match else None
        return side, frame

    def capture_thermal_camera_state(self, iteration=None, scale=1.0, include_test=False):
        entries = []
        split_lists = [("train", self.train_cameras.get(scale, []))]
        if include_test:
            split_lists.append(("test", self.test_cameras.get(scale, [])))

        for split, cameras in split_lists:
            for cam in cameras:
                if not (hasattr(cam, "has_thermal") and cam.has_thermal):
                    continue
                side, frame = self._camera_state_side_frame(cam)
                effective_fovx, effective_fovy = cam.get_thermal_fovs()
                entries.append({
                    "split": split,
                    "image_name": cam.image_name,
                    "name_key": self._camera_state_name(cam),
                    "uid": int(cam.uid),
                    "colmap_id": int(cam.colmap_id),
                    "cam_no": None if cam.cam_no is None else int(cam.cam_no),
                    "frame_no": None if cam.frame_no is None else int(cam.frame_no),
                    "side": side,
                    "parsed_frame": frame,
                    "thermal_delta_quaternion": cam.thermal_delta_quaternion.detach().cpu(),
                    "thermal_delta_translation": cam.thermal_delta_translation.detach().cpu(),
                    "learnable_tfovx": float(effective_fovx.detach().cpu()),
                    "learnable_tfovy": float(effective_fovy.detach().cpu()),
                })

        return {
            "format": "thermal_camera_state_v1",
            "iteration": iteration,
            "num_cameras": len(entries),
            "temporal_alignment": {
                "enabled": self.temporal_alignment_enabled,
                "offset_raw": float(self.temporal_offset_raw.detach().cpu()),
                "offset_frames": float(self.temporal_offset_frames().detach().cpu()),
                "max_frames": self.temporal_offset_max_frames,
                "affine_enabled": self.temporal_affine_clock_enabled,
                "drift_raw": float(self.temporal_drift_raw.detach().cpu()),
                "drift_endpoint_frames": float(
                    self.temporal_drift_frames().detach().cpu()),
                "drift_max_endpoint_frames": (
                    self.temporal_drift_max_endpoint_frames),
            },
            "cameras": entries,
        }

    def save_thermal_camera_state(self, path, iteration=None, scale=1.0):
        payload = self.capture_thermal_camera_state(iteration=iteration, scale=scale, include_test=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)
        print(f"[ThermalPose] saved {payload['num_cameras']} camera states: {path}")

    @staticmethod
    def _blend_thermal_camera_entries(entry_a, entry_b, weight):
        if entry_a is entry_b:
            blended = dict(entry_a)
            blended["interpolated_from"] = [entry_a.get("name_key", entry_a.get("image_name"))]
            return blended

        w = float(max(0.0, min(1.0, weight)))
        qa = torch.as_tensor(entry_a["thermal_delta_quaternion"]).float()
        qb = torch.as_tensor(entry_b["thermal_delta_quaternion"]).float()
        if torch.dot(qa, qb) < 0:
            qb = -qb
        q = qa * (1.0 - w) + qb * w
        q = q / (q.norm() + 1e-8)

        ta = torch.as_tensor(entry_a["thermal_delta_translation"]).float()
        tb = torch.as_tensor(entry_b["thermal_delta_translation"]).float()
        blended = dict(entry_a)
        blended["thermal_delta_quaternion"] = q
        blended["thermal_delta_translation"] = ta * (1.0 - w) + tb * w
        blended["learnable_tfovx"] = (
            float(entry_a["learnable_tfovx"]) * (1.0 - w)
            + float(entry_b["learnable_tfovx"]) * w
        )
        blended["learnable_tfovy"] = (
            float(entry_a["learnable_tfovy"]) * (1.0 - w)
            + float(entry_b["learnable_tfovy"]) * w
        )
        blended["interpolated_from"] = [
            entry_a.get("name_key", entry_a.get("image_name")),
            entry_b.get("name_key", entry_b.get("image_name")),
        ]
        return blended

    def _interpolate_thermal_camera_state(self, cam, train_entries):
        side, frame = self._camera_state_side_frame(cam)
        if frame is None:
            return None

        candidates = [
            entry for entry in train_entries
            if entry.get("parsed_frame") is not None and (side is None or entry.get("side") == side)
        ]
        if not candidates:
            candidates = [entry for entry in train_entries if entry.get("parsed_frame") is not None]
        if not candidates:
            return None

        before = [entry for entry in candidates if int(entry["parsed_frame"]) <= frame]
        after = [entry for entry in candidates if int(entry["parsed_frame"]) >= frame]
        prev_entry = max(before, key=lambda e: int(e["parsed_frame"])) if before else None
        next_entry = min(after, key=lambda e: int(e["parsed_frame"])) if after else None

        if prev_entry is not None and next_entry is not None:
            prev_frame = int(prev_entry["parsed_frame"])
            next_frame = int(next_entry["parsed_frame"])
            if prev_frame != next_frame:
                weight = (frame - prev_frame) / float(next_frame - prev_frame)
                return self._blend_thermal_camera_entries(prev_entry, next_entry, weight)
            return self._blend_thermal_camera_entries(prev_entry, prev_entry, 0.0)

        nearest = min(candidates, key=lambda e: abs(int(e["parsed_frame"]) - frame))
        return self._blend_thermal_camera_entries(nearest, nearest, 0.0)

    @staticmethod
    def _apply_thermal_camera_entry(cam, entry):
        if not (hasattr(cam, "has_thermal") and cam.has_thermal):
            return False
        with torch.no_grad():
            q = torch.as_tensor(
                entry["thermal_delta_quaternion"],
                device=cam.thermal_delta_quaternion.device,
                dtype=cam.thermal_delta_quaternion.dtype,
            )
            t = torch.as_tensor(
                entry["thermal_delta_translation"],
                device=cam.thermal_delta_translation.device,
                dtype=cam.thermal_delta_translation.dtype,
            )
            cam.thermal_delta_quaternion.copy_(q)
            cam.thermal_delta_translation.copy_(t)
            cam.learnable_tfovx.copy_(
                torch.as_tensor(entry["learnable_tfovx"], device=cam.learnable_tfovx.device,
                                dtype=cam.learnable_tfovx.dtype)
            )
            cam.learnable_tfovy.copy_(
                torch.as_tensor(entry["learnable_tfovy"], device=cam.learnable_tfovy.device,
                                dtype=cam.learnable_tfovy.dtype)
            )
            cam.refresh_thermal_projection()
        return True

    def load_thermal_camera_state(self, source, scale=1.0, apply_to_test=True):
        if isinstance(source, str):
            if not os.path.exists(source):
                print(f"[ThermalPose] no saved camera state found: {source}")
                return None
            payload = torch.load(source, map_location="cpu")
            source_name = source
        else:
            payload = source
            source_name = "<checkpoint>"

        temporal_state = payload.get("temporal_alignment", {})
        if self.temporal_alignment_enabled:
            if not temporal_state.get("enabled", False):
                raise RuntimeError(
                    f"Temporal alignment state missing or disabled in {source_name}")
            saved_max = float(temporal_state.get("max_frames", -1.0))
            if abs(saved_max - self.temporal_offset_max_frames) > 1e-9:
                raise RuntimeError(
                    f"Temporal max mismatch: saved={saved_max}, config={self.temporal_offset_max_frames}")
            with torch.no_grad():
                self.temporal_offset_raw.copy_(torch.as_tensor(
                    temporal_state["offset_raw"],
                    device=self.temporal_offset_raw.device,
                    dtype=self.temporal_offset_raw.dtype,
                ))
                saved_affine = bool(temporal_state.get(
                    "affine_enabled", False))
                if saved_affine != self.temporal_affine_clock_enabled:
                    raise RuntimeError(
                        "Temporal affine-clock mode mismatch")
                if self.temporal_affine_clock_enabled:
                    saved_drift_max = float(temporal_state.get(
                        "drift_max_endpoint_frames", -1.0))
                    if abs(saved_drift_max
                           - self.temporal_drift_max_endpoint_frames) > 1e-9:
                        raise RuntimeError(
                            "Temporal drift endpoint bound mismatch")
                    self.temporal_drift_raw.copy_(torch.as_tensor(
                        temporal_state["drift_raw"],
                        device=self.temporal_drift_raw.device,
                        dtype=self.temporal_drift_raw.dtype,
                    ))
            print(
                "[TemporalAlignment] loaded offset: "
                f"frames={float(self.temporal_offset_frames().detach().cpu()):.6f}"
            )

        entries = payload.get("cameras", payload if isinstance(payload, list) else [])
        train_entries = [entry for entry in entries if entry.get("split", "train") == "train"]
        by_name = {
            str(entry.get("name_key", self._camera_state_name(entry.get("image_name", "")))): entry
            for entry in train_entries
        }

        train_exact = 0
        test_exact = 0
        test_interp = 0
        missed = 0

        for cam in self.train_cameras.get(scale, []):
            key = self._camera_state_name(cam)
            entry = by_name.get(key)
            if entry is not None and self._apply_thermal_camera_entry(cam, entry):
                train_exact += 1
            elif hasattr(cam, "has_thermal") and cam.has_thermal:
                missed += 1

        if apply_to_test:
            for cam in self.test_cameras.get(scale, []):
                key = self._camera_state_name(cam)
                entry = by_name.get(key)
                if entry is not None:
                    if self._apply_thermal_camera_entry(cam, entry):
                        test_exact += 1
                    continue
                entry = self._interpolate_thermal_camera_state(cam, train_entries)
                if entry is not None and self._apply_thermal_camera_entry(cam, entry):
                    test_interp += 1
                elif hasattr(cam, "has_thermal") and cam.has_thermal:
                    missed += 1

        print(
            "[ThermalPose] loaded camera state from "
            f"{source_name}: train_exact={train_exact}, "
            f"test_exact={test_exact}, test_interp={test_interp}, missed={missed}"
        )
        return {
            "train_exact": train_exact,
            "test_exact": test_exact,
            "test_interp": test_interp,
            "missed": missed,
        }

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
        self.save_thermal_camera_state(
            os.path.join(point_cloud_path, "thermal_camera_state.pth"),
            iteration=iteration,
        )
    
    def recordpoints(self, iteration, string):
        txtpath = os.path.join(self.model_path, "exp_log.txt")
        numpoints = self.gaussians._xyz.shape[0]
        recordpointshelper(self.model_path, numpoints, iteration, string)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    def getVideoCameras(self, scale=1.0):
        return self.video_cameras[scale]
