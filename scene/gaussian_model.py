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

import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from random import randint
from utils.sh_utils import RGB2SH
from .poses import LearnPose 
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.deformation import multiemb_deform_network, multiemb_thermal_deform_network, multiemb_MLP_thermal_deform_network

class GaussianModel:

    def setup_functions(self): 
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._deformation = multiemb_thermal_deform_network(W=args.net_width, D=args.defor_depth, 
                                min_embeddings=args.min_embeddings, max_embeddings=args.max_embeddings, 
                                num_frames=args.total_num_frames,
                                args=args)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._thermal_dc = torch.empty(0)
        self._thermal_rest = torch.empty(0)        
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._thermal_opacity = torch.empty(0)
        self._embedding = torch.empty(0)

        self._t_embedding = torch.empty(0)
        # Modality identity logits per Gaussian: [shared, rgb-only, thermal-only].
        # This is a structural identity variable, not a continuous gate.
        self._logit_modality = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.xyz_gradient_accum_rgb = torch.zeros_like(self._xyz)
        self.xyz_gradient_accum_th = torch.zeros_like(self._xyz)
        self.denom_rgb = torch.zeros((self._xyz.shape[0], 1), device=self._xyz.device)
        self.denom_th = torch.zeros((self._xyz.shape[0], 1), device=self._xyz.device)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        try:
            self.max_gaussians = int(os.environ.get(
                "ED3DGS_MAX_GAUSSIANS", "0"))
        except ValueError as error:
            raise ValueError(
                "ED3DGS_MAX_GAUSSIANS must be a non-negative integer") from error
        if self.max_gaussians < 0:
            raise ValueError("ED3DGS_MAX_GAUSSIANS must be a non-negative integer")
        self.capacity_limited_events = 0
        print("GAUSSIAN_CAPACITY_CONFIG " + json.dumps({
            "max_gaussians": self.max_gaussians,
            "enabled": self.max_gaussians > 0,
        }, sort_keys=True))
        self._pose_r = torch.empty(0)
        self._pose_t = torch.empty(0)
        self._pose_log_s = torch.empty(0)
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._features_dc,
            self._features_rest,
            self._thermal_dc,
            self._thermal_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._thermal_opacity,
            self._pose_r,
            self._pose_t,
            self._pose_log_s,
            self._logit_modality,
            self._embedding,
            self._t_embedding,
            self.thermal_pose_head.state_dict(),
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        thermal_pose_state = None  # for backward compat
        deformation_state = None
        if len(model_args) == 23:
            # Current format: deformation and thermal pose are state dicts.
            (self.active_sh_degree,
             self._xyz,
             deformation_state,
             self._features_dc,
             self._features_rest,
             self._thermal_dc,
             self._thermal_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self._thermal_opacity,
             self._pose_r,
             self._pose_t,
             self._pose_log_s,
             self._logit_modality,
             self._embedding,
             self._t_embedding,
             thermal_pose_state,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
        elif len(model_args) == 20:
            # New format: includes thermal_pose_head state dict, no pose tensors.
            (self.active_sh_degree,
             self._xyz,
             self._deformation,
             self._features_dc,
             self._features_rest,
             self._thermal_dc,
             self._thermal_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self._thermal_opacity,
             self._logit_modality,
             self._embedding,
             self._t_embedding,
             thermal_pose_state,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
        elif len(model_args) == 18:
            # Old format with logit_modality (no thermal_pose_head)
            (self.active_sh_degree,
             self._xyz,
             self._deformation,
             self._features_dc,
             self._features_rest,
             self._thermal_dc,
             self._thermal_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self._thermal_opacity,
             self._logit_modality,
             self._embedding,
             self._t_embedding,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
        elif len(model_args) == 17:
            # Older format: no _pose_r/_pose_t/_pose_log_s (loaded separately)
            (self.active_sh_degree,
             self._xyz,
             self._deformation,
             self._features_dc,
             self._features_rest,
             self._thermal_dc,
             self._thermal_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self._thermal_opacity,
             self._pose_r,
             self._pose_t,
             self._pose_log_s,
             self._embedding,
             self._t_embedding,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
            init_logits = torch.tensor([2.0, 0.0, 0.0], device=self._xyz.device).repeat(self._xyz.shape[0], 1)
            self._logit_modality = nn.Parameter(init_logits.requires_grad_(True))
        else:
            raise ValueError(f"Unknown checkpoint format with {len(model_args)} items")
        if deformation_state is not None:
            self._deformation.load_state_dict(deformation_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        # Restore thermal pose if available
        if thermal_pose_state is not None:
            self.thermal_pose_head.load_state_dict(thermal_pose_state)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_Tcw12(self, cam_uid: int) -> torch.Tensor:
        """列优先 Tcw(12)，直接喂 CUDA。"""
        return self.pose_head(cam_uid, out="Tcw12")  
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_thermal_features(self):
        thermal_dc = self._thermal_dc
        thermal_rest = self._thermal_rest
        return torch.cat((thermal_dc, thermal_rest), dim=1)

    @property
    def get_deformed_features(self, dc):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_deformed_thermal_features(self, dc):
        thermal_dc = self._thermal_dc
        thermal_rest = self._thermal_rest
        return torch.cat((thermal_dc, thermal_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_thermal_opacity(self):
        return self.opacity_activation(self._thermal_opacity)
    
    @property
    def get_embedding(self):
        return self._embedding
    
    @property
    def get_thermal_embedding(self):
        return self._t_embedding
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, time_line: int,camera_list):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        thermal_features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        thermal_features[:, :, :] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        scales = torch.clamp(scales, max=1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1 

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        thermal_opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        embedding = torch.zeros((fused_color.shape[0], self._deformation.gaussian_embedding_dim)).float().cuda()  # [jm]
        thermal_embedding = torch.zeros((fused_color.shape[0], self._deformation.thermal_embedding_dim)).float().cuda()  # [jm]
        modality_logits = torch.tensor([2.0, 0.0, 0.0], device="cuda").repeat(fused_point_cloud.shape[0], 1)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda") 
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_dc = nn.Parameter(thermal_features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_rest = nn.Parameter(thermal_features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._thermal_opacity = nn.Parameter(thermal_opacities.requires_grad_(True))
        self._logit_modality = nn.Parameter(modality_logits.requires_grad_(True))
        self._embedding = nn.Parameter(embedding.requires_grad_(True))  # [jm]
        self._t_embedding = nn.Parameter(thermal_embedding.requires_grad_(True))  # [jm]
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        def _flatten(seq):
            """递归展开 list/tuple/set/字典的 values；其他类型原样 yield。"""
            if isinstance(seq, dict):
                for v in seq.values():
                    yield from _flatten(v)
            elif isinstance(seq, (list, tuple, set)):
                for x in seq:
                    yield from _flatten(x)
            else:
                yield seq

        def _extract_cameras(obj):
            """
            支持几种常见形态：
            - {scale: [cams, ...]}  (你的情况)
            - [cam, cam, ...] / (cam, ...) / 嵌套
            - dataset.cameras 这种带属性的容器
            只返回真正带 world_view_transform 的 Camera 对象。
            """
            base = obj.cameras if hasattr(obj, "cameras") else obj
            cams = [x for x in _flatten(base) if hasattr(x, "world_view_transform")]
            return cams

        cams = _extract_cameras(camera_list)
        if len(cams) == 0:
            # 打印点信息帮调
            flat = list(_flatten(camera_list))
            raise ValueError(f"camera_list 里没有可用相机，前几个元素类型: {[type(x).__name__ for x in flat[:8]]}")

        # 构建 (N,4,4) init_c2w
        init_c2w_list = []
        for cam in cams:
            M = torch.as_tensor(cam.world_view_transform, dtype=torch.float32, device="cuda")
            if M.shape == (3, 4):
                M44 = torch.eye(4, dtype=torch.float32, device="cuda")
                M44[:3, :4] = M
            elif M.shape == (4, 4):
                M44 = M
            else:
                raise ValueError(f"Unexpected pose shape: {tuple(M.shape)}")
            init_c2w_list.append(M44)

        init_c2w = torch.stack(init_c2w_list, dim=0)  # (N,4,4)

        # 推荐：右乘增量，学 R/t/s
        self.pose_head = LearnPose(
            num_cams=init_c2w.shape[0],
            learn_R=True, learn_t=True, learn_s=True,
            init_c2w=init_c2w,
            compose="right",
            device="cuda", dtype=torch.float32
        )

        # Thermal pose head: same init_c2w, learns independent delta for thermal extrinsics
        self.thermal_pose_head = LearnPose(
            num_cams=init_c2w.shape[0],
            learn_R=True, learn_t=True, learn_s=False,
            init_c2w=init_c2w,
            compose="right",
            device="cuda", dtype=torch.float32
        )

        def reset_bad_views_poses(self, bad_views):
            """
            重置坏视角的位姿增量
            """
            for cam in bad_views:
                cam_id = cam.uid  # 获取坏视角的相机ID
                self.pose_head.reset_delta(cam_id) 

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_rgb = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_th = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_rgb = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_th = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters()), 'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': [self._deformation.offsets], 'lr': training_args.offsets_lr, "name": "offsets"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / training_args.feature_lr_div_factor, "name": "f_rest"},
            {'params': [self._thermal_dc], 'lr': training_args.thermal_feature_lr, "name": "thermal_dc"},
            {'params': [self._thermal_rest], 'lr': training_args.thermal_feature_lr / training_args.thermal_feature_lr_div_factor, "name": "thermal_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._thermal_opacity], 'lr': training_args.thermal_opacity_lr, "name": "thermal_opacity"},
            {'params': [self._logit_modality], 'lr': training_args.modality_lr, "name": "logit_modality"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._embedding], 'lr': training_args.feature_lr, "name": "embedding"},
            {'params': [self._t_embedding], 'lr': training_args.thermal_feature_lr, "name": "thermal_embedding"}
        ]

        l += self.pose_head.pose_param_groups(
                    lr_r=0,   # RGB pose: freeze (不学位姿)
                    lr_t=0,   # RGB pose: freeze
                    lr_s=0    # 不加这组 = 冻结 S
                )

        # Thermal pose: now handled by Camera-level params + Scene optimizers
        # thermal_pose_head exists but is NOT trained here (Scene handles it)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.deformation_lr_max_steps)    
        self.pose_r_sched = get_expon_lr_func(
            lr_init=training_args.c2w_lr ,
            lr_final=training_args.c2w_lr_final,
            max_steps=training_args.position_lr_max_steps
        )
        self.pose_t_sched = get_expon_lr_func(
            lr_init=training_args.c2w_lr,
            lr_final=training_args.c2w_lr_final,
            max_steps=training_args.position_lr_max_steps)

        # Thermal pose schedulers: now handled by Scene MultiStepLR

    def register_thermal_intrinsics(self, cameras, lr):
        """Add learnable thermal FoV params from cameras to optimizer."""
        for cam in cameras:
            if hasattr(cam, 'learnable_tfovx') and cam.learnable_tfovx.requires_grad:
                self.optimizer.add_param_group({
                    'params': [cam.learnable_tfovx], 'lr': lr, 'name': 'thermal_fovx'
                })
            if hasattr(cam, 'learnable_tfovy') and cam.learnable_tfovy.requires_grad:
                self.optimizer.add_param_group({
                    'params': [cam.learnable_tfovy], 'lr': lr, 'name': 'thermal_fovy'
                })
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "deformation":
                start_iter = max(
                    int(getattr(self._deformation.args, "deformation_start_iter", 0)),
                    0,
                )
                lr = self.deformation_scheduler_args(max(iteration - start_iter, 0))
                param_group['lr'] = lr
    def update_pose(self, iteration):
        """
        分阶段更新 Pose 学习率：
        - 每 5000 轮重新从 lr_init 开始衰减（阶段性重启）
        - 不影响 xyz 或其他参数组
        """
        # ====== 参数 ======
        stage_size = 30000         # 每个阶段长度
        #################为了和高斯球强拟合度  这里可能要调整###########
        local_iter = iteration % stage_size   # 当前阶段内步数
        stage_idx  = iteration // stage_size  # 第几个阶段（用于日志）

        # ====== 计算当前学习率 ======
        lr_r = self.pose_r_sched(local_iter)
        lr_t = self.pose_t_sched(local_iter)
        lr_s = None
        if hasattr(self, "pose_s_sched"):
            lr_s = self.pose_s_sched(local_iter)

        # ====== 应用到优化器 (RGB 位姿冻结，只更新 thermal) ======
        for group in self.optimizer.param_groups:
            name = group.get("name", "")
            if name in getattr(self, "frozen_groups", []):
                group["lr"] = 0.0
                continue

            # RGB pose: frozen (lr stays at 0). Thermal pose handled by Scene MultiStepLR.
            if name in ("pose_r", "pose_t", "pose_log_s"):
                group["lr"] = 0.0

        # ====== 可选日志输出 ======
        if local_iter == 0:
            print(f"[Pose-LR Restart] Stage {stage_idx} @ iter {iteration} → "
                f"lr_r={lr_r:.2e}, lr_t={lr_t:.2e} (thermal: Scene MultiStepLR)")

        return lr_r, lr_t, lr_s
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self._thermal_dc.shape[1]*self._thermal_dc.shape[2]):
            l.append('t_dc_{}'.format(i))
        for i in range(self._thermal_rest.shape[1]*self._thermal_rest.shape[2]):
            l.append('t_rest_{}'.format(i))
        for i in range(self._opacity.shape[1]):
            l.append('opacity_{}'.format(i))
        for i in range(self._thermal_opacity.shape[1]):
            l.append('thermal_opacity_{}'.format(i))
        for i in range(self._logit_modality.shape[1]):
            l.append('logit_modality_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._embedding.shape[1]):
            l.append('embedding_{}'.format(i))
        for i in range(self._t_embedding.shape[1]):
            l.append('thermal_embedding_{}'.format(i))
        return l

    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path,"deformation.pth"),map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(),os.path.join(path, "deformation.pth"))
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        thermal_dc = self._thermal_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        thermal_rest = self._thermal_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        thermal_opacities = self._thermal_opacity.detach().cpu().numpy()
        modality_logits = self._logit_modality.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        embedding = self._embedding.detach().cpu().numpy()
        thermal_embedding = self._t_embedding.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, thermal_dc, thermal_rest, opacities, thermal_opacities, modality_logits, scale, rotation, embedding, thermal_embedding), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
    def reset_opacity(self, ratio=0):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        thermal_opacities_new = inverse_sigmoid(torch.min(self.get_thermal_opacity, torch.ones_like(self.get_thermal_opacity)*0.01))
        if ratio is not None:
            mask = torch.rand(self.get_opacity.shape[0], device="cuda") < ratio
            opacities_new[~mask] = self.get_opacity[~mask]
            thermal_opacities_new[~mask] = self.get_thermal_opacity[~mask]
            print(f"reset opacity: {mask.sum()} points")
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        optimizable_tensors = self.replace_tensor_to_optimizer(thermal_opacities_new, "thermal_opacity")
        self._opacity = optimizable_tensors["opacity"]
        self._thermal_opacity = optimizable_tensors["thermal_opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        # opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        # thermal_opacities = np.asarray(plydata.elements[0]["thermal_opacity"])[..., np.newaxis]
        opacities_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_")]
        thermal_opacities_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("thermal_opacity_")]
        opacities_name = sorted(opacities_name, key = lambda x: int(x.split('_')[-1]))
        thermal_opacities_name = sorted(thermal_opacities_name, key = lambda x: int(x.split('_')[-1]))
        opacities = np.zeros((xyz.shape[0], len(opacities_name)))
        thermal_opacities = np.zeros((xyz.shape[0], len(thermal_opacities_name)))
        for idx, attr_name in enumerate(opacities_name):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
        for idx, attr_name in enumerate(thermal_opacities_name):
            thermal_opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
        modality_logits_name = [p.name for p in plydata.elements[0].properties if p.name.startswith("logit_modality_")]
        modality_logits_name = sorted(modality_logits_name, key = lambda x: int(x.split('_')[-1]))
        modality_logits = np.zeros((xyz.shape[0], len(modality_logits_name)))
        for idx, attr_name in enumerate(modality_logits_name):
            modality_logits[:, idx] = np.asarray(plydata.elements[0][attr_name])      
            
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        thermal_dc = np.zeros((xyz.shape[0], 3, 1))
        thermal_dc[:, 0, 0] = np.asarray(plydata.elements[0]["t_dc_0"])
        thermal_dc[:, 1, 0] = np.asarray(plydata.elements[0]["t_dc_1"])
        thermal_dc[:, 2, 0] = np.asarray(plydata.elements[0]["t_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        extra_t_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("t_rest_")]
        extra_t_names = sorted(extra_t_names, key = lambda x: int(x.split('_')[-1]))

        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

        assert len(extra_t_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        thermal_extra = np.zeros((xyz.shape[0], len(extra_t_names)))

        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        for idx, attr_name in enumerate(extra_t_names):
            thermal_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        thermal_extra = thermal_extra.reshape((thermal_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        embedding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("embedding")]
        embedding_names = sorted(embedding_names, key = lambda x: int(x.split('_')[-1]))
        embeddings = np.zeros((xyz.shape[0], len(embedding_names)))
        for idx, attr_name in enumerate(embedding_names):
            embeddings[:, idx] = np.asarray(plydata.elements[0][attr_name])

        thermal_embedding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("thermal_embedding")]
        thermal_embedding_names = sorted(thermal_embedding_names, key = lambda x: int(x.split('_')[-1]))
        thermal_embeddings = np.zeros((xyz.shape[0], len(thermal_embedding_names)))
        for idx, attr_name in enumerate(thermal_embedding_names):
            thermal_embeddings[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_dc = nn.Parameter(torch.tensor(thermal_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._thermal_rest = nn.Parameter(torch.tensor(thermal_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._thermal_opacity = nn.Parameter(torch.tensor(thermal_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._logit_modality = nn.Parameter(torch.tensor(modality_logits, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._embedding = nn.Parameter(torch.tensor(embeddings, dtype=torch.float, device="cuda").requires_grad_(True))
        self._t_embedding = nn.Parameter(torch.tensor(thermal_embeddings, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1 or group["name"] == "offsets":
                continue
            name = group.get("name", "")
            # 跳过与点数无关的参数（例如相机位姿/尺度向量/thermal FoV）
            if name in ("pose_r", "pose_t", "pose_log_s",
                        "thermal_pose_r", "thermal_pose_t",
                        "thermal_fovx", "thermal_fovy"):
                optimizable_tensors[name] = group["params"][0]
                continue

            param = group["params"][0]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["thermal_dc"]
        self._thermal_rest = optimizable_tensors["thermal_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._thermal_opacity = optimizable_tensors["thermal_opacity"]
        self._logit_modality = optimizable_tensors["logit_modality"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._embedding = optimizable_tensors["embedding"]
        self._t_embedding = optimizable_tensors["thermal_embedding"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.xyz_gradient_accum_rgb = self.xyz_gradient_accum_rgb[valid_points_mask]
        self.xyz_gradient_accum_th = self.xyz_gradient_accum_th[valid_points_mask]
        self.denom_rgb = self.denom_rgb[valid_points_mask]
        self.denom_th = self.denom_th[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1 or group["name"] == "offsets":continue
            assert len(group["params"]) == 1
            # 对于 c2w/s，直接保持不变
            name = group.get("name", "")
            param = group["params"][0]
            if name in ("pose_r", "pose_t", "pose_log_s",
                        "thermal_pose_r", "thermal_pose_t",
                        "thermal_fovx", "thermal_fovy"):
                optimizable_tensors[name] = param
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)

            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacities, new_thermal_opacities, 
                              new_logit_modality, new_scaling, new_rotation, new_embedding, new_t_embedding,new_pose_r,
                          new_pose_t,
                          new_pose_los_s):
        d = {"xyz": new_xyz, 
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "thermal_dc": new_thermal_dc,
        "thermal_rest": new_thermal_rest,
        "opacity": new_opacities,
        "thermal_opacity": new_thermal_opacities,
        "logit_modality": new_logit_modality,    
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "embedding" : new_embedding,
        "thermal_embedding" : new_t_embedding,
        "pose_r":            new_pose_r,
        "pose_t":            new_pose_t,
        "pose_log_s":        new_pose_los_s,

       }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._thermal_dc = optimizable_tensors["thermal_dc"]
        self._thermal_rest = optimizable_tensors["thermal_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._thermal_opacity = optimizable_tensors["thermal_opacity"]
        self._logit_modality = optimizable_tensors["logit_modality"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._embedding = optimizable_tensors["embedding"]
        self._t_embedding = optimizable_tensors["thermal_embedding"]
        
        self._pose_r           = optimizable_tensors["pose_r"]
        self._pose_t           = optimizable_tensors["pose_t"]
        self._pose_log_s       = optimizable_tensors["pose_log_s"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_rgb = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_th = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_rgb = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_th = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _limit_densification_mask(self, selected_pts_mask, scores,
                                  copies_per_selected, operation):
        if self.max_gaussians <= 0:
            return selected_pts_mask

        selected_count = int(selected_pts_mask.sum().item())
        current_count = int(self.get_xyz.shape[0])
        available = max(0, self.max_gaussians - current_count)
        allowed_count = available // copies_per_selected
        if selected_count <= allowed_count:
            return selected_pts_mask

        limited_mask = torch.zeros_like(selected_pts_mask)
        if allowed_count > 0:
            candidate_indices = torch.nonzero(
                selected_pts_mask, as_tuple=False).flatten()
            candidate_scores = scores.reshape(-1)[candidate_indices]
            selected_order = torch.argsort(
                candidate_scores, descending=True, stable=True)[:allowed_count]
            limited_mask[candidate_indices[selected_order]] = True

        self.capacity_limited_events += 1
        print("GAUSSIAN_CAPACITY_LIMIT " + json.dumps({
            "operation": operation,
            "current_count": current_count,
            "max_gaussians": self.max_gaussians,
            "available_slots": available,
            "copies_per_selected": copies_per_selected,
            "requested_candidates": selected_count,
            "accepted_candidates": int(limited_mask.sum().item()),
            "event": self.capacity_limited_events,
        }, sort_keys=True))
        return limited_mask

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        selected_pts_mask = self._limit_densification_mask(
            selected_pts_mask, padded_grad, N, "split")
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_thermal_dc = self._thermal_dc[selected_pts_mask].repeat(N,1,1)
        new_thermal_rest = self._thermal_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_thermal_opacity = self._thermal_opacity[selected_pts_mask].repeat(N,1)
        new_logit_modality = self._logit_modality[selected_pts_mask].repeat(N,1)
        new_embedding = self._embedding[selected_pts_mask].repeat(N,1)
        new_t_embedding = self._t_embedding[selected_pts_mask].repeat(N,1)

        new_pose_r = self._pose_r           # 相机参数保持不变
        new_pose_t = self._pose_t
        new_pose_log_s = self._pose_log_s
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacity, new_thermal_opacity, 
                                   new_logit_modality, new_scaling, new_rotation, new_embedding, new_t_embedding, new_pose_r, new_pose_t,new_pose_log_s)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        selected_pts_mask = self._limit_densification_mask(
            selected_pts_mask, torch.norm(grads, dim=-1), 1, "clone")
        if not selected_pts_mask.any():
            return
        
        new_xyz = self._xyz[selected_pts_mask] 
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_thermal_dc = self._thermal_dc[selected_pts_mask]
        new_thermal_rest = self._thermal_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_thermal_opacities = self._thermal_opacity[selected_pts_mask]
        new_logit_modality = self._logit_modality[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_embedding = self._embedding[selected_pts_mask]
        new_t_embedding = self._t_embedding[selected_pts_mask]

        new_pose_r = self._pose_r           # 相机参数保持不变
        new_pose_t = self._pose_t
        new_pose_log_s = self._pose_log_s

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacities, new_thermal_opacities, new_logit_modality,
                                    new_scaling, new_rotation, new_embedding, new_t_embedding, new_pose_r, new_pose_t,new_pose_log_s)

    def decompose_shared_to_modalities(self, shared_mask):
        """Split shared Gaussians into rgb-only and thermal-only structure identities."""
        if shared_mask is None or not shared_mask.any():
            return
        shared_mask = shared_mask.to(self._xyz.device)
        old_count = self.get_xyz.shape[0]
        shared_count = shared_mask.sum()

        new_xyz = self._xyz[shared_mask].repeat(2, 1)
        new_features_dc = self._features_dc[shared_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[shared_mask].repeat(2, 1, 1)
        new_thermal_dc = self._thermal_dc[shared_mask].repeat(2, 1, 1)
        new_thermal_rest = self._thermal_rest[shared_mask].repeat(2, 1, 1)
        new_opacities = self._opacity[shared_mask].repeat(2, 1)
        new_thermal_opacities = self._thermal_opacity[shared_mask].repeat(2, 1)
        new_scaling = self._scaling[shared_mask].repeat(2, 1)
        new_rotation = self._rotation[shared_mask].repeat(2, 1)
        new_embedding = self._embedding[shared_mask].repeat(2, 1)
        new_t_embedding = self._t_embedding[shared_mask].repeat(2, 1)

        # Structural modality split: shared -> rgb-only & thermal-only.
        rgb_logits = torch.tensor([-1e4, 1e4, -1e4], device=self._xyz.device).repeat(shared_count, 1)
        thermal_logits = torch.tensor([-1e4, -1e4, 1e4], device=self._xyz.device).repeat(shared_count, 1)
        new_logit_modality = torch.cat([rgb_logits, thermal_logits], dim=0)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_thermal_dc, new_thermal_rest, new_opacities, new_thermal_opacities, new_logit_modality, new_scaling, new_rotation, new_embedding, new_t_embedding)

        prune_mask = torch.zeros(old_count + 2 * shared_count, device=self._xyz.device, dtype=torch.bool)
        prune_mask[:old_count] = shared_mask
        self.prune_points(prune_mask)

    def prune(self, max_grad, min_opacity, extent, max_screen_size, use_mean=False, routing_stage=None, enable_modality_densify=True):
        opacity_for_prune = self.get_opacity
        use_modality_prune = (
            enable_modality_densify
            and routing_stage in {"C", "D"}
        )
        if use_modality_prune:
            rgb_opacity = self.get_opacity
            thermal_opacity = self.get_thermal_opacity
            routing_idx = torch.argmax(self._logit_modality, dim=-1)
            mask_shared = routing_idx == 0
            mask_rgb = routing_idx == 1
            mask_thermal = routing_idx == 2
            opacity_for_prune = torch.zeros_like(rgb_opacity)
            opacity_for_prune[mask_shared] = torch.maximum(
                rgb_opacity[mask_shared], thermal_opacity[mask_shared]
            )
            opacity_for_prune[mask_rgb] = rgb_opacity[mask_rgb]
            opacity_for_prune[mask_thermal] = thermal_opacity[mask_thermal]

        if use_mean:
            prune_mask = (
                opacity_for_prune
                < (
                    opacity_for_prune.max()
                    - (opacity_for_prune.max() - opacity_for_prune.min()) * 0.5
                )
            ).squeeze()
        else:
            prune_mask = (opacity_for_prune < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size, routing_stage=None, enable_modality_densify=True, thermal_only=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if thermal_only:
            # Thermal-only: all Gaussians use thermal gradient for densification
            grad_th = self.xyz_gradient_accum_th / (self.denom_th + 1e-8)
            grad_th[grad_th.isnan()] = 0.0
            if enable_modality_densify:
                grads = grad_th

        # Stage A/B: keep original 3DGS densification behavior.
        if not thermal_only and routing_stage in {"C", "D"}:
            use_modality_densify = enable_modality_densify
            if use_modality_densify:
                # === Modality-aware densification (Stage C/D) ===
                routing_idx = torch.argmax(self._logit_modality, dim=-1)
                mask_shared = routing_idx == 0
                mask_rgb = routing_idx == 1
                mask_thermal = routing_idx == 2
                grad_rgb = self.xyz_gradient_accum_rgb / (self.denom_rgb + 1e-8)
                grad_th = self.xyz_gradient_accum_th / (self.denom_th + 1e-8)
                grad_for_densify = torch.zeros_like(grad_rgb)
                grad_for_densify[mask_shared] = torch.maximum(
                    grad_rgb[mask_shared], grad_th[mask_shared]
                )
                grad_for_densify[mask_rgb] = grad_rgb[mask_rgb]
                grad_for_densify[mask_thermal] = grad_th[mask_thermal]
                grad_for_densify[grad_for_densify.isnan()] = 0.0
                grads = grad_for_densify

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        self.xyz_gradient_accum_rgb.zero_()
        self.xyz_gradient_accum_th.zero_()
        self.denom_rgb.zero_()
        self.denom_th.zero_()

    def _visibility_column(self, visibility_filter):
        return visibility_filter.to(self._xyz.device).float().unsqueeze(-1)

    def add_modality_densification_stats(self, grad_rgb, grad_th,
                                         visibility_filter_rgb,
                                         visibility_filter_th=None,
                                         thermal_only=False):
        if visibility_filter_th is None:
            visibility_filter_th = visibility_filter_rgb

        if thermal_only:
            if grad_th is None:
                return
            grad_th_mag = torch.norm(grad_th[:, :2], dim=-1, keepdim=True).detach().to(self._xyz.device)
            visible = self._visibility_column(visibility_filter_th)
            grad_th_mag *= visible
            self.xyz_gradient_accum_th += grad_th_mag
            self.denom_th += visible
            # Also fill unified accum for backward compat (used by add_densification_stats fallback)
            self.xyz_gradient_accum += grad_th_mag
            self.denom += visible
            return

        if grad_rgb is None or grad_th is None:
            return
        grad_rgb_mag = torch.norm(grad_rgb[:, :2], dim=-1, keepdim=True).detach().to(self._xyz.device)
        grad_th_mag = torch.norm(grad_th[:, :2], dim=-1, keepdim=True).detach().to(self._xyz.device)

        visible_rgb = self._visibility_column(visibility_filter_rgb)
        visible_th = self._visibility_column(visibility_filter_th)

        grad_rgb_mag *= visible_rgb
        grad_th_mag *= visible_th

        self.xyz_gradient_accum_rgb += grad_rgb_mag
        self.xyz_gradient_accum_th += grad_th_mag
        self.denom_rgb += visible_rgb
        self.denom_th += visible_th

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def print_deformation_weight_grad(self):
        for name, weight in self._deformation.named_parameters():
            if weight.requires_grad:
                if weight.grad is None:
                    
                    print(name," :",weight.grad)
                else:
                    if weight.grad.mean() != 0:
                        print(name," :",weight.grad.mean(), weight.grad.min(), weight.grad.max())
        print("-"*50)
