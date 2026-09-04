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

import math

import torch
from time_alignment.affine_clock import affine_offset_frames
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV
from utils.graphics_utils import fov2focal, pix2ndc
from kornia import create_meshgrid
from kornia.geometry.conversions import (
    quaternion_to_rotation_matrix as kornia_quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
import random 
from torchvision import transforms
from PIL import Image


def _aligned_temporal_quaternion(reference, value):
    return torch.where(torch.dot(reference, value) < 0.0, -value, value)


def _temporal_hermite(value0, value1, tangent0, tangent1, weight):
    weight2 = weight * weight
    weight3 = weight2 * weight
    return (
        (2.0 * weight3 - 3.0 * weight2 + 1.0) * value0
        + (weight3 - 2.0 * weight2 + weight) * tangent0
        + (-2.0 * weight3 + 3.0 * weight2) * value1
        + (weight3 - weight2) * tangent1)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, TFoVx, TFoVy,image, thermal, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0, timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0,
                 cam_no=None, frame_no=None, image_path=None, img_wh=None,
                 thermal_source_frame=None, thermal_source_name=None, thermal_frame_shift=0):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.TFoVx = TFoVx
        self.TFoVy = TFoVy
        self.image_name = image_name
        self.time = timestamp
        self.cam_no = cam_no
        self.frame_no = frame_no
        self.thermal_source_frame = thermal_source_frame
        self.thermal_source_name = thermal_source_name
        self.thermal_frame_shift = int(thermal_frame_shift)
        self.temporal_alignment_enabled = False
        self.temporal_observation_correction_enabled = False
        self.temporal_offset_raw = None
        self.temporal_offset_max_frames = 0.0
        self.temporal_drift_raw = None
        self.temporal_drift_max_endpoint_frames = 0.0
        self.temporal_duration = 1.0
        self.temporal_pose_frames = None
        self.temporal_pose_track = None

        self.transform = transforms.ToTensor()
        self.gt_alpha_mask = gt_alpha_mask
        self.img_wh = img_wh
        self.image_path = image_path
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # image is real image 
        if not isinstance(image, tuple) and image is not None:
            if "camera_" not in image_name:
                self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.original_image = image.clamp(0.0, 1.0).half().to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        elif isinstance(image, tuple) and image is not None:
            self.image_width = image[0]
            self.image_height = image[1]
            self.original_image = None
        
        else: # image: None
            self.image_width = None
            self.image_height = None
            self.original_image = None

        # thermal is real thermal image
        if not isinstance(thermal, tuple) and thermal is not None:
            if "camera_" not in image_name:
                self.thermal_image = thermal.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.thermal_image = thermal.clamp(0.0, 1.0).half().to(self.data_device)
            self.thermal_width = self.thermal_image.shape[2]
            self.thermal_height = self.thermal_image.shape[1]
            if gt_alpha_mask is not None:
                self.thermal_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.thermal_image *= torch.ones((1, self.thermal_height, self.thermal_width), device=self.data_device)

        elif isinstance(thermal, tuple) and thermal is not None:
            self.thermal_width = thermal[0]
            self.thermal_height = thermal[1]
            self.thermal_image = None
        
        else: # image: None
            self.thermal_width = None
            self.thermal_height = None
            self.thermal_image = None


        self.zfar = 100.0
        self.znear = 0.01  

        self.trans = trans
        self.scale = scale
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.projection_matrix_thermal = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.TFoVx, fovY=self.TFoVy).transpose(0,1).cuda()
        self.full_proj_transform_thermal = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix_thermal.unsqueeze(0))).squeeze(0)

        # ---- Learnable thermal FoV + projection matrix (for thermal pose optimization) ----
        self.has_thermal = (thermal is not None and not isinstance(thermal, tuple))
        if self.has_thermal:
            self.learnable_tfovx = nn.Parameter(
                torch.tensor(self.TFoVx, device="cuda").requires_grad_(True))
            self.learnable_tfovy = nn.Parameter(
                torch.tensor(self.TFoVy, device="cuda").requires_grad_(True))
        else:
            self.learnable_tfovx = nn.Parameter(
                torch.tensor(self.FoVx, device="cuda").requires_grad_(False))
            self.learnable_tfovy = nn.Parameter(
                torch.tensor(self.FoVy, device="cuda").requires_grad_(False))
        self.thermal_intrinsic_tied_aspect = False
        self.thermal_projection_matrix_learnable = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar,
            fovX=self.learnable_tfovx, fovY=self.learnable_tfovy).transpose(0, 1).cuda()

        # ---- Thermal extrinsics delta (Self-Cali-GS style: per-camera, separate optimizer) ----
        # Store init quaternion and translation from RGB w2c for thermal pose learning
        if self.has_thermal:
            w2c_row = self.world_view_transform.T.contiguous()  # row-major
            self.init_rotation_rgb = w2c_row[:3, :3].detach().clone()
            self.init_translation_rgb = w2c_row[:3, 3].detach().clone()
            # Identity quat [w=1,x=0,y=0,z=0]: gradients non-zero unlike zeros(4)
            self.thermal_delta_quaternion = nn.Parameter(
                torch.tensor([1., 0., 0., 0.], device="cuda").requires_grad_(True))
            self.thermal_delta_translation = nn.Parameter(
                torch.zeros(3, device="cuda").requires_grad_(True))

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
      
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] #v 


            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)    
            

        else :
            self.rayo = None
            self.rayd = None
            
    def effective_temporal_offset_frames(self):
        offset_raw = getattr(
            self, "nctc_temporal_offset_raw_override", self.temporal_offset_raw)
        if (not self.temporal_alignment_enabled
                or not self.temporal_observation_correction_enabled
                or offset_raw is None):
            return torch.zeros((), device=self.world_view_transform.device,
                               dtype=self.world_view_transform.dtype)
        drift_raw = getattr(
            self, "nctc_temporal_drift_raw_override", self.temporal_drift_raw)
        return affine_offset_frames(
            offset_raw, drift_raw, float(self.frame_no), self.temporal_duration,
            self.temporal_offset_max_frames,
            self.temporal_drift_max_endpoint_frames)

    def get_temporal_time(self):
        if not self.temporal_observation_correction_enabled:
            return torch.as_tensor(self.time, device=self.world_view_transform.device,
                                   dtype=self.world_view_transform.dtype)
        query_frame = torch.as_tensor(
            float(self.frame_no), device=self.world_view_transform.device,
            dtype=self.world_view_transform.dtype) + self.effective_temporal_offset_frames()
        if getattr(self, "temporal_strict_common_support", False):
            query_value = float(query_frame.detach().item())
            if (not math.isfinite(query_value)
                    or query_value < 0.0
                    or query_value > self.temporal_duration - 1.0):
                raise RuntimeError(
                    f"Strict temporal query outside common support: {query_value}")
        else:
            query_frame = query_frame.clamp(0.0, self.temporal_duration - 1.0)
        return query_frame / self.temporal_duration

    def get_temporal_rgb_world_view_transform(self):
        if not self.temporal_observation_correction_enabled:
            return self.world_view_transform
        if self.temporal_pose_track is None or self.temporal_pose_frames is None:
            raise RuntimeError(f"Temporal pose track missing for {self.image_name}")

        query_frame = torch.as_tensor(
            float(self.frame_no), device=self.world_view_transform.device,
            dtype=self.world_view_transform.dtype) + self.effective_temporal_offset_frames()
        if getattr(self, "temporal_strict_common_support", False):
            query_value = float(query_frame.detach().item())
            pose_min = float(self.temporal_pose_frames[0].detach().item())
            pose_max = float(self.temporal_pose_frames[-1].detach().item())
            if (not math.isfinite(query_value)
                    or query_value < pose_min
                    or query_value > pose_max):
                raise RuntimeError(
                    "Strict temporal pose query outside common support: "
                    f"{query_value} not in [{pose_min}, {pose_max}]")
        else:
            query_frame = query_frame.clamp(
                self.temporal_pose_frames[0], self.temporal_pose_frames[-1])
        upper = int(torch.searchsorted(
            self.temporal_pose_frames, query_frame.detach(), right=True).item())
        upper = min(max(upper, 1), self.temporal_pose_track.shape[0] - 1)
        lower = upper - 1
        before = max(lower - 1, 0)
        after = min(upper + 1, self.temporal_pose_track.shape[0] - 1)
        frame_lower = self.temporal_pose_frames[lower]
        frame_upper = self.temporal_pose_frames[upper]
        interval = frame_upper - frame_lower
        interval_value = float(interval.detach().item())
        if not math.isfinite(interval_value) or interval_value <= 0.0:
            raise RuntimeError(f"Invalid C1 temporal pose interval: {interval_value}")
        weight = (query_frame - frame_lower) / interval

        c2ws = [
            torch.linalg.inv(self.temporal_pose_track[index].T.contiguous())
            for index in (before, lower, upper, after)]
        positions = [value[:3, 3] for value in c2ws]
        quaternions = [
            rotation_matrix_to_quaternion(
                value[:3, :3].contiguous().unsqueeze(0))[0]
            for value in c2ws]
        quaternions[0] = _aligned_temporal_quaternion(
            quaternions[1], quaternions[0])
        quaternions[2] = _aligned_temporal_quaternion(
            quaternions[1], quaternions[2])
        quaternions[3] = _aligned_temporal_quaternion(
            quaternions[2], quaternions[3])

        position_tangent0 = (
            positions[2] - positions[1] if before == lower
            else 0.5 * (positions[2] - positions[0]))
        position_tangent1 = (
            positions[2] - positions[1] if after == upper
            else 0.5 * (positions[3] - positions[1]))
        quaternion_tangent0 = (
            quaternions[2] - quaternions[1] if before == lower
            else 0.5 * (quaternions[2] - quaternions[0]))
        quaternion_tangent1 = (
            quaternions[2] - quaternions[1] if after == upper
            else 0.5 * (quaternions[3] - quaternions[1]))

        center = _temporal_hermite(
            positions[1], positions[2], position_tangent0,
            position_tangent1, weight)
        quaternion = torch.nn.functional.normalize(_temporal_hermite(
            quaternions[1], quaternions[2], quaternion_tangent0,
            quaternion_tangent1, weight), dim=0)
        rotation = kornia_quaternion_to_rotation_matrix(
            quaternion.unsqueeze(0))[0]

        c2w = torch.eye(4, device=rotation.device, dtype=rotation.dtype)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = center
        interpolated = torch.linalg.inv(c2w).T.contiguous()
        if bool((query_frame.detach() == frame_lower.detach()).item()):
            exact = self.temporal_pose_track[lower]
            return exact + (interpolated - interpolated.detach())
        if bool((query_frame.detach() == frame_upper.detach()).item()):
            exact = self.temporal_pose_track[upper]
            return exact + (interpolated - interpolated.detach())
        return interpolated

    def get_thermal_world_view_transform(self):
        """Self-Cali-GS style: RGB init + thermal delta → w2c column-major."""
        if not self.has_thermal:
            return self.world_view_transform
        # Quaternion delta: normalize and convert to rotation matrix
        quaternion_norm = self.thermal_delta_quaternion.norm()
        if getattr(self, "temporal_strict_common_support", False):
            norm_value = float(quaternion_norm.detach().item())
            if not math.isfinite(norm_value) or norm_value <= 0.0:
                raise RuntimeError(f"Invalid strict Thermal quaternion norm: {norm_value}")
            q = self.thermal_delta_quaternion / quaternion_norm
        else:
            q = self.thermal_delta_quaternion / (quaternion_norm + 1e-8)
        delta_R = quaternion_to_rotation_matrix(
            q, already_normalized=getattr(
                self, "temporal_strict_common_support", False))  # (3,3)
        delta_t = self.thermal_delta_translation     # (3,)
        # Apply the spatial delta on top of the time-aligned RGB trajectory pose.
        base_w2c = self.get_temporal_rgb_world_view_transform().T.contiguous()
        base_R = base_w2c[:3, :3]
        base_t = base_w2c[:3, 3]
        R_th = delta_R @ base_R
        t_th = delta_R @ base_t + delta_t
        # Build row-major then transpose to column-major
        Rt = torch.cat([R_th, t_th.unsqueeze(1)], dim=1)  # (3,4)
        last_row = torch.tensor([[0., 0., 0., 1.]], device=R_th.device, dtype=R_th.dtype)
        w2c = torch.cat([Rt, last_row], dim=0)             # (4,4) row-major
        return w2c.T.contiguous()                          # column-major

    def get_thermal_fovs(self):
        """Return effective thermal FoVs, optionally preserving one focal length."""
        fovX = self.learnable_tfovx
        if not self.thermal_intrinsic_tied_aspect:
            return fovX, self.learnable_tfovy

        width = float(self.thermal_width or self.image_width)
        height = float(self.thermal_height or self.image_height)
        focal = 0.5 * width / torch.tan(0.5 * fovX)
        fovY = 2.0 * torch.atan(0.5 * height / focal)
        return fovX, fovY

    def refresh_thermal_projection(self):
        """Rebuild thermal projection matrix from learnable FoV using torch ops (preserves gradient)."""
        fovX, fovY = self.get_thermal_fovs()
        tanHalfFovY = torch.tan(fovY / 2.0)
        tanHalfFovX = torch.tan(fovX / 2.0)
        top = tanHalfFovY * self.znear
        bottom = -top
        right = tanHalfFovX * self.znear
        left = -right
        P = torch.zeros(4, 4, device=fovX.device, dtype=fovX.dtype)
        P[0, 0] = 2.0 * self.znear / (right - left)
        P[1, 1] = 2.0 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = 1.0
        P[2, 2] = (self.zfar + self.znear) / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        self.thermal_projection_matrix_learnable = P.T.contiguous()

    # def load_image(self):
    #     original_image = Image.open(self.image_path)
    #     original_image = original_image.resize(self.img_wh, Image.LANCZOS)
    #     self.original_image = self.transform(original_image)
    #     self.image_width = self.original_image.shape[2]
    #     self.image_height = self.original_image.shape[1]
    #     if self.gt_alpha_mask is not None:
    #         self.original_image *= self.gt_alpha_mask.to(self.data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class Camerass(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, thermal, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", near=0.01, far=100.0, timestamp=0.0, rayo=None, rayd=None, rays=None, cxr=0.0,cyr=0.0,
                 ):
        super(Camerass, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.fisheyemapper = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # image is real image 
        if not isinstance(image, tuple):
            if "camera_" not in image_name:
                self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.original_image = image.clamp(0.0, 1.0).half().to(self.data_device)
            print("read one")# lazy loader?
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

        else:
            self.image_width = image[0] 
            self.image_height = image[1] 
            self.original_image = None

        # thermal is real thermal image
        if not isinstance(thermal, tuple) and thermal is not None:
            if "camera_" not in image_name:
                self.thermal_image = thermal.clamp(0.0, 1.0).to(self.data_device)
            else:
                self.thermal_image = thermal.clamp(0.0, 1.0).half().to(self.data_device)
            print("read one")# lazy loader?
            self.thermal_width = self.thermal_image.shape[2]
            self.thermal_height = self.thermal_image.shape[1]
        
        else: # image: None
            self.thermal_width = thermal[0]
            self.thermal_height = thermal[1]
            self.thermal_image = None
        
        self.image_width = 2 * self.image_width
        self.image_height = 2 * self.image_height # 

        self.thermal_width = 2 * self.thermal_width
        self.thermal_height = 2 * self.thermal_height

        self.zfar = 100.0
        self.znear = 0.01  
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
      
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None

def quaternion_to_rotation_matrix(quaternion, already_normalized=False):
    """Convert (w,x,y,z) quaternion to 3x3 rotation matrix."""
    if not already_normalized:
        quaternion = quaternion / (quaternion.norm() + 1e-8)
    w, x, y, z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    R = torch.zeros((3, 3), device=quaternion.device, dtype=quaternion.dtype)
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R
