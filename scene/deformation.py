import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# 用于自定义的C++拓展
from torch.utils.cpp_extension import load
import torch.nn.init as init


def _run_deformation_head(module, value):
    enabled = os.environ.get("ED3DGS_DEFORMATION_CHECKPOINT", "0") == "1"
    if not enabled or not module.training or not torch.is_grad_enabled():
        return module(value)
    return checkpoint(module, value, use_reentrant=False)


def _mean_active_offset(offsets):
    if not bool(torch.isfinite(offsets).all()):
        raise RuntimeError("Deformation offsets are non-finite")
    nonzero_offsets = torch.masked_select(offsets, offsets.ne(0))
    offset = offsets.sum() if nonzero_offsets.numel() == 0 else nonzero_offsets.mean()
    if not bool(torch.isfinite(offset)):
        raise RuntimeError("Deformation offset mean is non-finite")
    return offset

def apply_modality_residual(dx_shared, dx_rgb_residual, dx_th_residual, s_hard):
    """Combine shared displacement with modality-specific residuals.

    Shared displacement models geometry common to all modalities, while residuals
    capture rgb/thermal-specific deviations that are gated by routing identity.
    """
    if s_hard is None:
        return dx_shared + dx_rgb_residual + dx_th_residual
    return dx_shared + s_hard[:, 1:2] * dx_rgb_residual + s_hard[:, 2:3] * dx_th_residual

# 定义 deform_network 类，继承自 PyTorch 的 nn.Module
class deform_network(nn.Module):
    ''' 
    构造函数:__init__
    初始化形变网络，接受多个参数：
    D: 网络的层数（深度）。
    W: 每层的宽度（神经元数量）。
    min_embeddings, max_embeddings: 用于时间和高斯特征嵌入的最小和最大值。
    num_frames: 输入数据的帧数。
    args: 存储自定义的参数，例如 temporal_embedding_dim（时间嵌入维度）和 gaussian_embedding_dim（高斯嵌入维度）。
    '''
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(deform_network, self).__init__()
        self.D = D
        self.W = W

        self.args = args
        self.min_embeddings = min_embeddings
        self.max_embeddings = max_embeddings
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        self.c2f_temporal_iter = args.c2f_temporal_iter

        self.feature_out_c, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c = self.create_net()
        self.feature_out_f, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f = self.create_net()

        if args.zero_temporal:
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))

    def create_net(self):
        self.feature_out = [nn.Linear(self.temporal_embedding_dim + self.gaussian_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        feature_out = nn.Sequential(*self.feature_out)
        return  \
            feature_out,\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]

        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2

        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        emb = emb.repeat(1,1,N,1).squeeze()

        return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if type(pc) == type(None):
            h = torch.cat([h, embeddings], dim=-1)
        else:        
            h = torch.cat([h, pc.get_embedding], dim=-1)

        h = _run_deformation_head(feature_out, h)
        return h

    def deform(self, hidden, pts, scales, rotations, opacity, sh_coefs, pos_deform, scales_deform, rotations_deform, opacity_deform, rgb_deform, scale=1., scale_c=1., scale_o=1., coef_s=1.):
        dx, ds, dr, do = pos_deform(hidden), None, None, None
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = scales_deform(hidden)
            scales = scales + ds * scale * coef_s
        if not self.args.no_dr:
            dr = rotations_deform(hidden)
            rotations = rotations + dr * scale
        if not self.args.no_do:
            do = opacity_deform(hidden) 
            opacity = opacity + do * scale * scale_o
        if not self.args.no_dc:
            dc = rgb_deform(hidden) 
            sh_coefs = sh_coefs + dc.view(-1,16,3) * scale_c
        return pts, scales, rotations, opacity, sh_coefs
    
    def forward(self, point, scales=None, rotations=None, opacity=None, time_emb=None, cam_no=None, pc=None, embeddings=None, sh_coefs=None, iter=None, num_down_emb_c=30, num_down_emb_f=30):
        pts, scales, rotations, opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig = pts, scales, rotations, opacity, sh_coefs
        
        if type(cam_no) == type(None):
            offset = _mean_active_offset(self.offsets)
        else:
            offset = self.offsets[cam_no]
        time_emb += offset

        use_anneal = self.args.use_anneal
        coef = 1 if not use_anneal else np.clip(iter/1000,0,1) 
        coef_c = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_o = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_s = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)

        if self.args.no_coarse_deform:
            pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig
        else:
            hidden = self.query_time(pts, scales, rotations, time_emb, pc, embeddings, sh_coefs, iter, self.feature_out_c, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()        
            pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = self.deform(hidden, pts, scales, rotations, opacity, sh_coefs,\
                self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c, coef, coef_c, coef_o, coef_s)

        if self.args.no_fine_deform:
            pts, scales, rotations, opacity, sh_coefs = pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub
        else:
            hidden = self.query_time(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings, sh_coefs_sub, iter, self.feature_out_f, num_down_emb=num_down_emb_f).float()
            pts, scales, rotations, opacity, sh_coefs = self.deform(hidden, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub,\
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f, coef, coef_c, coef_o, coef_s)
                        
        return pts, scales, rotations, opacity, sh_coefs, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, sh_coefs_orig))
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list


class new_deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(new_deform_network, self).__init__()
        self.D = D
        self.W = W

        self.args = args
        self.min_embeddings = min_embeddings
        self.max_embeddings = max_embeddings
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        
        # new thermal_embedding_dim
        self.thermal_embedding_dim = args.thermal_embedding_dim

        self.c2f_temporal_iter = args.c2f_temporal_iter

        self.feature_out_c, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.thermal_opacity_deform_c, self.rgb_deform_c, self.thermal_deform_c = self.create_net()
        self.feature_out_f, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.thermal_opacity_deform_f, self.rgb_deform_f, self.thermal_deform_f = self.create_net()

        if args.zero_temporal:
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))

    # add thermal_embedding_dim
    def create_net(self):
        self.feature_out = [nn.Linear(self.temporal_embedding_dim + self.gaussian_embedding_dim + self.thermal_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        feature_out = nn.Sequential(*self.feature_out)
        return  \
            feature_out,\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]

        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2

        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        emb = emb.repeat(1,1,N,1).squeeze()

        return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, thermal_sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if embeddings is not None:
            D_rgb = self.gaussian_embedding_dim
            rgb_emb = embeddings[:, :D_rgb]
            thermal_emb = embeddings[:, D_rgb:]
        else:
            rgb_emb, thermal_emb = None, None

        if type(pc) == type(None):
            h_rgb = torch.cat([h, rgb_emb], dim=-1)
            h_thermal = torch.cat([h, thermal_emb], dim=-1)
        else:        
            h_rgb = torch.cat([h, pc.get_embedding], dim=-1)
            h_thermal = torch.cat([h, pc.get_embedding], dim=-1)

        h_rgb = feature_out(h_rgb)
        h_thermal = feature_out(h_thermal)
        return h_rgb, h_thermal

    def deform(self, hidden_pair, pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, 
               pos_deform, scales_deform, rotations_deform, opacity_deform, thermal_opacity_deform, rgb_deform, thermal_deform, 
               scale=1., scale_c=1., scale_o=1., coef_s=1.):
        hidden_rgb, hidden_thermal = hidden_pair
        dx= pos_deform(hidden_rgb)
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = scales_deform(hidden_rgb)
            scales = scales + ds * scale * coef_s
        if not self.args.no_dr:
            dr = rotations_deform(hidden_rgb)
            rotations = rotations + dr * scale
        if not self.args.no_do:
            do = opacity_deform(hidden_rgb) 
            opacity = opacity + do * scale * scale_o
            dto = thermal_opacity_deform(hidden_thermal)
            thermal_opacity = thermal_opacity + dto * scale * scale_o

        if not self.args.no_dc:
            dc = rgb_deform(hidden_rgb) 
            sh_coefs = sh_coefs + dc.view(-1,16,3) * scale_c
            dt = thermal_deform(hidden_thermal)
            thermal_sh_coefs = thermal_sh_coefs +dt.view(-1,16,3) * scale_c
        return pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs
    
    def forward(self, point, scales=None, rotations=None, opacity=None, thermal_opacity=None, 
                time_emb=None, cam_no=None, pc=None, embeddings=None, 
                sh_coefs=None, thermal_sh_coefs=None, iter=None, num_down_emb_c=30, num_down_emb_f=30):
        pts, scales, rotations, opacity, thermal_opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1], thermal_opacity[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig = pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs
        
        if type(cam_no) == type(None):
            offset = _mean_active_offset(self.offsets)
        else:
            offset = self.offsets[cam_no]
        time_emb += offset

        use_anneal = self.args.use_anneal
        coef = 1 if not use_anneal else np.clip(iter/1000,0,1) 
        coef_c = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_o = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)
        coef_s = 1 if not use_anneal else np.clip((iter-self.args.deform_from_iter)/1000,0,1)

        if self.args.no_coarse_deform:
            pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig
        else:
            hidden_pair = self.query_time(pts, scales, rotations, time_emb, pc, embeddings, 
                                     sh_coefs, thermal_sh_coefs, iter, self.feature_out_c, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c)        
            pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub = self.deform(hidden_pair, pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, \
                self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.thermal_opacity_deform_c, self.rgb_deform_c, self.thermal_deform_c,
                  coef, coef_c, coef_o, coef_s)

        if self.args.no_fine_deform:
            pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs = pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub
        else:
            hidden_pair = self.query_time(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings,
                                      sh_coefs_sub, thermal_sh_coefs_sub, iter, self.feature_out_f, num_down_emb=num_down_emb_f)
            pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs = self.deform(hidden_pair, pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub,\
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.thermal_opacity_deform_f, self.rgb_deform_f, self.thermal_deform_f,
                  coef, coef_c, coef_o, coef_s)
                        
        return pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig))
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list    


class multiemb_deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(multiemb_deform_network, self).__init__()
        self.D = D
        self.W = W

        self.args = args
        self.min_embeddings = min_embeddings
        self.max_embeddings = max_embeddings
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        
        # new thermal_embedding_dim
        self.thermal_embedding_dim = args.thermal_embedding_dim

        self.c2f_temporal_iter = args.c2f_temporal_iter

        self.feature_out_c_1, self.feature_out_c_2, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.thermal_opacity_deform_c, self.rgb_deform_c, self.thermal_deform_c = self.create_net()
        self.feature_out_f_1, self.feature_out_f_2, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.thermal_opacity_deform_f, self.rgb_deform_f, self.thermal_deform_f = self.create_net()

        if args.zero_temporal:
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))

    # add thermal_embedding_dim
    def create_net(self):
        self.feature_out_1 = [nn.Linear(self.temporal_embedding_dim + self.gaussian_embedding_dim, self.W)]
        self.feature_out_2 = [nn.Linear(self.temporal_embedding_dim + self.thermal_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out_1.append(nn.ReLU())
            self.feature_out_1.append(nn.Linear(self.W,self.W))
            self.feature_out_2.append(nn.ReLU())
            self.feature_out_2.append(nn.Linear(self.W,self.W))
        feature_out_1 = nn.Sequential(*self.feature_out_1)
        feature_out_2 = nn.Sequential(*self.feature_out_2)
        return  \
            feature_out_1, feature_out_2,\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]

        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2

        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        emb = emb.repeat(1,1,N,1).squeeze()

        return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time_1(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if type(pc) == type(None):
            h = torch.cat([h, embeddings], dim=-1)
        else:        
            h = torch.cat([h, pc.get_embedding], dim=-1)

        h = _run_deformation_head(feature_out, h)
        return h

    def query_time_2(self, pts, scales, rotations, time_emb, pc=None, thermal_embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if type(pc) == type(None):
            h = torch.cat([h, thermal_embeddings], dim=-1)
        else:        
            h = torch.cat([h, pc.get_thermal_embedding], dim=-1)

        h = feature_out(h)
        return h

    def deform(self, hidden, pts, scales, rotations, opacity, sh_coefs,
                pos_deform, scales_deform, rotations_deform, opacity_deform, rgb_deform,
                  scale=1., scale_c=1., scale_o=1., coef_s=1.):
        dx, ds, dr, do = _run_deformation_head(pos_deform, hidden), None, None, None
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = _run_deformation_head(scales_deform, hidden)
            scales = scales + ds * scale * coef_s
        if not self.args.no_dr:
            dr = _run_deformation_head(rotations_deform, hidden)
            rotations = rotations + dr * scale
        if not self.args.no_do:
            do = _run_deformation_head(opacity_deform, hidden)
            opacity = opacity + do * scale * scale_o
        if not self.args.no_dc:
            dc = _run_deformation_head(rgb_deform, hidden)
            sh_coefs = sh_coefs + dc.view(-1,16,3) * scale_c
        return pts, scales, rotations, opacity, sh_coefs

    def forward(self, point, scales=None, rotations=None, opacity=None, thermal_opacity=None,
                time_emb=None, cam_no=None, pc=None, embeddings=None, thermal_embeddings=None,
                sh_coefs=None, thermal_sh_coefs=None, iter=None, num_down_emb_c=30, num_down_emb_f=30):
        pts, scales, rotations, opacity, thermal_opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1], thermal_opacity[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig = pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs

        if type(cam_no) == type(None):
            offset = _mean_active_offset(self.offsets)
        else:
            offset = self.offsets[cam_no]
        time_emb += offset

        use_anneal = self.args.use_anneal
        if use_anneal:
            coef = np.clip(iter / 1000, 0, 1)
            coef_c = np.clip((iter - self.args.deform_from_iter) / 1000, 0, 1)
            coef_o = coef_s = coef_c
        else:
            coef = coef_c = coef_o = coef_s = 1

        if self.args.no_coarse_deform:
            pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig
        else:
            hidden_1 = self.query_time_1(pts, scales, rotations, time_emb, pc, embeddings,
                                     sh_coefs, iter, self.feature_out_c_1, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()
            pts_sub_1, scales_sub_1, rotations_sub_1, opacity_sub, sh_coefs_sub = self.deform(hidden_1, pts, scales, rotations, opacity, sh_coefs, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c, coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_1

            # second query_time for thermal embeddings
            hidden_2 = self.query_time_2(pts_sub_1, scales_sub_1, rotations_sub_1, time_emb, pc, thermal_embeddings,
                                     thermal_sh_coefs, iter, self.feature_out_c_2, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()
            pts_sub, scales_sub, rotations_sub, thermal_opacity_sub, thermal_sh_coefs_sub = self.deform(hidden_2, pts_sub_1, scales_sub_1, rotations_sub_1, thermal_opacity, thermal_sh_coefs, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.thermal_opacity_deform_c, self.thermal_deform_c,
                  coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_2, pts_sub_1, scales_sub_1, rotations_sub_1

        if self.args.no_fine_deform:
            pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs = pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub
        else:
            hidden_1 = self.query_time_1(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings,
                                      sh_coefs_sub, iter, self.feature_out_f_1, num_down_emb=num_down_emb_f).float()
            pts_1, scales_1, rotations_1, opacity, sh_coefs = self.deform(hidden_1, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub, \
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f,
                  coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_1

            # second query_time for thermal embeddings
            hidden_2 = self.query_time_2(pts_1, scales_1, rotations_1, time_emb, pc, thermal_embeddings,
                                      thermal_sh_coefs_sub, iter, self.feature_out_f_2, num_down_emb=num_down_emb_f).float()
            pts, scales, rotations, thermal_opacity, thermal_sh_coefs = self.deform(hidden_2, pts_1, scales_1, rotations_1, thermal_opacity_sub, thermal_sh_coefs_sub, \
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.thermal_opacity_deform_f, self.thermal_deform_f,
                  coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_2, pts_1, scales_1, rotations_1
                        
        return pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig))
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list    


class multiemb_thermal_deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(multiemb_thermal_deform_network, self).__init__()
        self.D = D
        self.W = W

        self.args = args
        self.min_embeddings = min_embeddings
        self.max_embeddings = max_embeddings
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        
        # new thermal_embedding_dim
        self.thermal_embedding_dim = args.thermal_embedding_dim

        self.c2f_temporal_iter = args.c2f_temporal_iter

        self.feature_out_c_1, self.feature_out_c_2, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c, self.thermal_deform_c, self.thermal_opacity_deform_c = self.create_net()
        self.feature_out_f_1, self.feature_out_f_2, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f, self.thermal_deform_f, self.thermal_opacity_deform_f = self.create_net()

        if args.zero_temporal:
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))

    # add thermal_embedding_dim
    def create_net(self):
        self.feature_out_1 = [nn.Linear(self.temporal_embedding_dim + self.gaussian_embedding_dim, self.W)]
        self.feature_out_2 = [nn.Linear(self.temporal_embedding_dim + self.thermal_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out_1.append(nn.ReLU())
            self.feature_out_1.append(nn.Linear(self.W,self.W))
            self.feature_out_2.append(nn.ReLU())
            self.feature_out_2.append(nn.Linear(self.W,self.W))
        feature_out_1 = nn.Sequential(*self.feature_out_1)
        feature_out_2 = nn.Sequential(*self.feature_out_2)
        return  \
            feature_out_1, feature_out_2,\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]

        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2

        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        emb = emb.repeat(1,1,N,1).squeeze()

        return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time_1(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if type(pc) == type(None):
            h = torch.cat([h, embeddings], dim=-1)
        else:        
            h = torch.cat([h, pc.get_embedding], dim=-1)

        h = feature_out(h)
        return h

    def query_time_2(self, pts, scales, rotations, time_emb, pc=None, thermal_embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if type(pc) == type(None):
            h = torch.cat([h, thermal_embeddings], dim=-1)
        else:        
            h = torch.cat([h, pc.get_thermal_embedding], dim=-1)

        h = feature_out(h)
        return h

    def deform(self, hidden, pts, scales, rotations, opacity, sh_coefs,
                pos_deform, scales_deform, rotations_deform, opacity_deform, rgb_deform,
                  scale=1., scale_c=1., scale_o=1., coef_s=1.):
        dx, ds, dr, do = pos_deform(hidden), None, None, None
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = scales_deform(hidden)
            scales = scales + ds * scale * coef_s
        if not self.args.no_dr:
            dr = rotations_deform(hidden)
            rotations = rotations + dr * scale
        if not self.args.no_do:
            do = opacity_deform(hidden) 
            opacity = opacity + do * scale * scale_o
        if not self.args.no_dc:
            dc = rgb_deform(hidden) 
            sh_coefs = sh_coefs + dc.view(-1,16,3) * scale_c
        return pts, scales, rotations, opacity, sh_coefs


    def deform_thermal(self, hidden, pts, scales, rotations, thermal_opacity, thermal_sh_coefs,
                        pos_deform, scales_deform, rotations_deform, thermal_deform, thermal_opacity_deform,
                        scale=1., scale_c=1., scale_o=1., coef_s=1., modality_routing=None, thermal_only=False):
        # Geometric deformation for thermal:
        # - When thermal_only=True: always deform geometry (thermal drives everything)
        # - When change_thermal_geo=True: thermal adds residual geometry on top of RGB
        deform_geo = thermal_only or self.args.change_thermal_geo
        if deform_geo:
            dx_th = _run_deformation_head(pos_deform, hidden)
            if thermal_only:
                pts = pts + dx_th * scale
            else:
                dx = apply_modality_residual(torch.zeros_like(dx_th), torch.zeros_like(dx_th), dx_th, modality_routing)
                pts = pts + dx * scale

            if not self.args.no_ds:
                ds = _run_deformation_head(scales_deform, hidden)
                scales = scales + ds * scale * coef_s
            if not self.args.no_dr:
                dr = _run_deformation_head(rotations_deform, hidden)
                rotations = rotations + dr * scale

        if not self.args.no_do:
            thermal_do = _run_deformation_head(thermal_opacity_deform, hidden)
            thermal_opacity = thermal_opacity + thermal_do * scale * scale_o
        if not self.args.no_dc:
            thermal_dc = _run_deformation_head(thermal_deform, hidden)
            thermal_sh_coefs = thermal_sh_coefs + thermal_dc.view(-1,16,3) * scale_c
        return pts, scales, rotations, thermal_opacity, thermal_sh_coefs

    def forward(self, point, scales=None, rotations=None, opacity=None, thermal_opacity=None,
                time_emb=None, cam_no=None, pc=None, embeddings=None, thermal_embeddings=None,
                sh_coefs=None, thermal_sh_coefs=None, iter=None, num_down_emb_c=30,
                num_down_emb_f=30, modality_routing=None, thermal_only=False,
                rgb_only_teacher=False):
        if thermal_only and rgb_only_teacher:
            raise ValueError("thermal_only and rgb_only_teacher are mutually exclusive")
        pts, scales, rotations, opacity, thermal_opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1], thermal_opacity[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig = pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs

        deformation_start_iter = max(int(getattr(self.args, "deformation_start_iter", 0)), 0)
        deformation_ramp_iters = max(int(getattr(self.args, "deformation_ramp_iters", 0)), 0)
        if iter is not None and iter < deformation_start_iter:
            identity_state = (
                pts_orig, scales_orig, rotations_orig, opacity_orig,
                thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig,
            )
            return (
                pts_orig, scales_orig, rotations_orig, opacity_orig,
                thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig,
                (identity_state, identity_state),
            )

        if type(cam_no) == type(None):
            offset = _mean_active_offset(self.offsets)
        else:
            offset = self.offsets[cam_no]
        time_emb += offset

        use_anneal = self.args.use_anneal
        if use_anneal:
            coef = np.clip(iter / 1000, 0, 1)
            coef_c = np.clip((iter - self.args.deform_from_iter) / 1000, 0, 1)
            coef_o = coef_s = coef_c
        else:
            coef = coef_c = coef_o = coef_s = 1

        if iter is not None and deformation_ramp_iters > 0:
            schedule_scale = np.clip(
                (iter - deformation_start_iter) / deformation_ramp_iters, 0, 1
            )
            coef *= schedule_scale
            coef_c *= schedule_scale
            coef_o *= schedule_scale
            coef_s *= schedule_scale

        if self.args.no_coarse_deform:
            pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig
        else:
            if thermal_only:
                # ---- Thermal-only: skip RGB stream, thermal drives geometry + appearance ----
                hidden_2 = self.query_time_2(pts, scales, rotations, time_emb, pc, thermal_embeddings,
                                         thermal_sh_coefs, iter, self.feature_out_c_2, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()
                pts_sub, scales_sub, rotations_sub, thermal_opacity_sub, thermal_sh_coefs_sub = self.deform_thermal(
                    hidden_2, pts, scales, rotations, thermal_opacity, thermal_sh_coefs,
                    self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c,
                    self.thermal_deform_c, self.thermal_opacity_deform_c,
                    coef, coef_c, coef_o, coef_s, modality_routing,
                    thermal_only=True)
                opacity_sub = opacity
                sh_coefs_sub = sh_coefs
                del hidden_2
            else:
                hidden_1 = self.query_time_1(pts, scales, rotations, time_emb, pc, embeddings,
                                         sh_coefs, iter, self.feature_out_c_1, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()
                pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = self.deform(hidden_1, pts, scales, rotations, opacity, sh_coefs,
                    self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c,
                      coef, coef_c, coef_o, coef_s)
                del hidden_1
                if rgb_only_teacher:
                    thermal_opacity_sub = thermal_opacity
                    thermal_sh_coefs_sub = thermal_sh_coefs
                else:
                    hidden_2 = self.query_time_2(pts_sub, scales_sub, rotations_sub, time_emb, pc, thermal_embeddings,
                                             thermal_sh_coefs, iter, self.feature_out_c_2, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()
                    pts_sub, scales_sub, rotations_sub, thermal_opacity_sub, thermal_sh_coefs_sub = self.deform_thermal(
                        hidden_2, pts_sub, scales_sub, rotations_sub, thermal_opacity, thermal_sh_coefs,
                        self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c,
                        self.thermal_deform_c, self.thermal_opacity_deform_c,
                        coef, coef_c, coef_o, coef_s, modality_routing)
                    del hidden_2

        if self.args.no_fine_deform:
            pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs = pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub
        else:
            if thermal_only:
                # ---- Thermal-only fine stage ----
                hidden_2 = self.query_time_2(pts_sub, scales_sub, rotations_sub, time_emb, pc, thermal_embeddings,
                                         thermal_sh_coefs_sub, iter, self.feature_out_f_2, num_down_emb=num_down_emb_f).float()
                pts, scales, rotations, thermal_opacity, thermal_sh_coefs = self.deform_thermal(
                    hidden_2, pts_sub, scales_sub, rotations_sub, thermal_opacity_sub, thermal_sh_coefs_sub,
                    self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f,
                    self.thermal_deform_f, self.thermal_opacity_deform_f,
                    coef, coef_c, coef_o, coef_s, modality_routing,
                    thermal_only=True)
                opacity = opacity_sub
                sh_coefs = sh_coefs_sub
                del hidden_2
            else:
                hidden_1 = self.query_time_1(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings,
                                          sh_coefs_sub, iter, self.feature_out_f_1, num_down_emb=num_down_emb_f).float()
                pts, scales, rotations, opacity, sh_coefs = self.deform(hidden_1, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub, \
                    self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f,
                      coef, coef_c, coef_o, coef_s)
                del hidden_1
                if rgb_only_teacher:
                    thermal_opacity = thermal_opacity_sub
                    thermal_sh_coefs = thermal_sh_coefs_sub
                else:
                    hidden_2 = self.query_time_2(pts, scales, rotations, time_emb, pc, thermal_embeddings,
                                              thermal_sh_coefs_sub, iter, self.feature_out_f_2, num_down_emb=num_down_emb_f).float()
                    pts, scales, rotations, thermal_opacity, thermal_sh_coefs = self.deform_thermal(
                        hidden_2, pts, scales, rotations, thermal_opacity_sub, thermal_sh_coefs_sub,
                        self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f,
                        self.thermal_deform_f, self.thermal_opacity_deform_f,
                        coef, coef_c, coef_o, coef_s, modality_routing)
                    del hidden_2

        return pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig))

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list


class multiemb_MLP_thermal_deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(multiemb_MLP_thermal_deform_network, self).__init__()
        self.D = D
        self.W = W

        self.args = args
        self.min_embeddings = min_embeddings
        self.max_embeddings = max_embeddings
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        
        # new thermal_embedding_dim
        self.thermal_embedding_dim = args.thermal_embedding_dim

        self.c2f_temporal_iter = args.c2f_temporal_iter

        self.feature_out_c_shared, self.feature_out_c_rgb, self.feature_out_c_thermal, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c, self.thermal_deform_c, self.thermal_opacity_deform_c = self.create_net()
        self.feature_out_f_shared, self.feature_out_f_rgb, self.feature_out_f_thermal, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f, self.thermal_deform_f, self.thermal_opacity_deform_f = self.create_net()

        if args.zero_temporal:
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))

    # add thermal_embedding_dim
    def create_net(self):
        # 共享部分：提取几何 & 时间特征
        shared_layers = [
            nn.Linear(self.temporal_embedding_dim, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU()
        ]
        shared_net = nn.Sequential(*shared_layers)

        # RGB 专属部分
        rgb_layers = [
            nn.Linear(self.W + self.gaussian_embedding_dim, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W)
        ]
        rgb_net = nn.Sequential(*rgb_layers)

        # Thermal 专属部分
        thermal_layers = [
            nn.Linear(self.W + self.thermal_embedding_dim, self.W),
            nn.ReLU(),
            nn.Linear(self.W, self.W)
        ]
        thermal_net = nn.Sequential(*thermal_layers)
        return  \
            shared_net, rgb_net, thermal_net, \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))

    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]

        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2

        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        emb = emb.repeat(1,1,N,1).squeeze()

        return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time_1(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30, feature_shared=None):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        h_shared = feature_shared(h)

        if type(pc) == type(None):
            h = torch.cat([h_shared, embeddings], dim=-1)
        else:        
            h = torch.cat([h_shared, pc.get_embedding], dim=-1)

        h = feature_out(h)
        return h

    def query_time_2(self, pts, scales, rotations, time_emb, pc=None, thermal_embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30, feature_shared=None):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        h_shared = feature_shared(h)

        if type(pc) == type(None):
            h = torch.cat([h_shared, thermal_embeddings], dim=-1)
        else:        
            h = torch.cat([h_shared, pc.get_thermal_embedding], dim=-1)

        h = feature_out(h)
        return h

    def deform(self, hidden, pts, scales, rotations, opacity, sh_coefs,
                pos_deform, scales_deform, rotations_deform, opacity_deform, rgb_deform,
                  scale=1., scale_c=1., scale_o=1., coef_s=1.):
        dx, ds, dr, do = pos_deform(hidden), None, None, None
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = scales_deform(hidden)
            scales = scales + ds * scale * coef_s
        if not self.args.no_dr:
            dr = rotations_deform(hidden)
            rotations = rotations + dr * scale
        if not self.args.no_do:
            do = opacity_deform(hidden) 
            opacity = opacity + do * scale * scale_o
        if not self.args.no_dc:
            dc = rgb_deform(hidden) 
            sh_coefs = sh_coefs + dc.view(-1,16,3) * scale_c
        return pts, scales, rotations, opacity, sh_coefs

    # def deform_thermal(self, hidden, thermal_opacity, thermal_sh_coefs,
    #                     thermal_deform, thermal_opacity_deform,
    #                     scale=1., scale_c=1., scale_o=1.):

    #     if not self.args.no_do:
    #         thermal_do = thermal_opacity_deform(hidden)  # ← 新增 thermal 不透明度
    #         thermal_opacity = thermal_opacity + thermal_do * scale * scale_o
    #     if not self.args.no_dc:
    #         thermal_dc = thermal_deform(hidden)
    #         thermal_sh_coefs = thermal_sh_coefs + thermal_dc.view(-1,16,3) * scale_c
    #     return thermal_opacity, thermal_sh_coefs

    def deform_thermal(self, hidden, pts, scales, rotations, thermal_opacity, thermal_sh_coefs,
                        pos_deform, scales_deform, rotations_deform, thermal_deform, thermal_opacity_deform,
                        scale=1., scale_c=1., scale_o=1., coef_s=1.):
        # add geometric deformation for thermal
        dx, ds, dr, do = pos_deform(hidden), None, None, None
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = scales_deform(hidden)
            scales = scales + ds * scale * coef_s
        if not self.args.no_dr:
            dr = rotations_deform(hidden)
            rotations = rotations + dr * scale

        if not self.args.no_do:
            thermal_do = thermal_opacity_deform(hidden)  # ← 新增 thermal 不透明度
            thermal_opacity = thermal_opacity + thermal_do * scale * scale_o
        if not self.args.no_dc:
            thermal_dc = thermal_deform(hidden)
            thermal_sh_coefs = thermal_sh_coefs + thermal_dc.view(-1,16,3) * scale_c
        # return thermal_opacity, thermal_sh_coefs
        return pts, scales, rotations, thermal_opacity, thermal_sh_coefs

    def forward(self, point, scales=None, rotations=None, opacity=None, thermal_opacity=None,
                time_emb=None, cam_no=None, pc=None, embeddings=None, thermal_embeddings=None,
                sh_coefs=None, thermal_sh_coefs=None, iter=None, num_down_emb_c=30, num_down_emb_f=30):
        pts, scales, rotations, opacity, thermal_opacity = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1], thermal_opacity[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig = pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs

        if type(cam_no) == type(None):
            offset = _mean_active_offset(self.offsets)
        else:
            offset = self.offsets[cam_no]
        time_emb += offset

        use_anneal = self.args.use_anneal
        if use_anneal:
            coef = np.clip(iter / 1000, 0, 1)
            coef_c = np.clip((iter - self.args.deform_from_iter) / 1000, 0, 1)
            coef_o = coef_s = coef_c
        else:
            coef = coef_c = coef_o = coef_s = 1

        if self.args.no_coarse_deform:
            pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig
        else:
            hidden_1 = self.query_time_1(pts, scales, rotations, time_emb, pc, embeddings,
                                     sh_coefs, iter, self.feature_out_c_rgb, self.args.use_coarse_temporal_embedding, feature_shared = self.feature_out_c_shared, num_down_emb=num_down_emb_c).float()
            pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub = self.deform(hidden_1, pts, scales, rotations, opacity, sh_coefs,
                self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c,
                  coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_1

            # second query_time for thermal embeddings
            hidden_2 = self.query_time_2(pts_sub, scales_sub, rotations_sub, time_emb, pc, thermal_embeddings,
                                     thermal_sh_coefs, iter, self.feature_out_c_thermal, self.args.use_coarse_temporal_embedding, feature_shared = self.feature_out_f_shared,  num_down_emb=num_down_emb_c).float()
            # thermal_opacity_sub, thermal_sh_coefs_sub = self.deform_thermal(hidden_2, thermal_opacity, thermal_sh_coefs, self.thermal_deform_c, self.thermal_opacity_deform_c, coef, coef_c, coef_o)
            pts_sub, scales_sub, rotations_sub, thermal_opacity_sub, thermal_sh_coefs_sub = self.deform_thermal(hidden_2, pts_sub, scales_sub, rotations_sub, thermal_opacity, thermal_sh_coefs, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.thermal_deform_c, self.thermal_opacity_deform_c,   coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_2

        if self.args.no_fine_deform:
            pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs = pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub
        else:
            hidden_1 = self.query_time_1(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings,
                                      sh_coefs_sub, iter, self.feature_out_f_rgb, feature_shared = self.feature_out_f_shared, num_down_emb=num_down_emb_f).float()
            pts, scales, rotations, opacity, sh_coefs = self.deform(hidden_1, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub, \
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f,
                  coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_1

            # second query_time for thermal embeddings
            hidden_2 = self.query_time_2(pts, scales, rotations, time_emb, pc, thermal_embeddings,
                                      thermal_sh_coefs_sub, iter, self.feature_out_f_thermal, feature_shared = self.feature_out_c_shared, num_down_emb=num_down_emb_f).float()
            # thermal_opacity, thermal_sh_coefs = self.deform_thermal(hidden_2, thermal_opacity_sub, thermal_sh_coefs_sub, self.thermal_deform_f, self.thermal_opacity_deform_f, coef, coef_c, coef_o)
            pts, scales, rotations, thermal_opacity, thermal_sh_coefs = self.deform_thermal(hidden_2, pts, scales, rotations, thermal_opacity, thermal_sh_coefs, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.thermal_deform_f, self.thermal_opacity_deform_f, coef, coef_c, coef_o, coef_s)
            # 释放显存
            del hidden_2
                        
        return pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig))

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list


class multiemb_2opacity_add_ac_deform_network(nn.Module):
    def __init__(self, D=8, W=256, min_embeddings=30, max_embeddings=150, num_frames=300, num_cam=None, args=None,):
        super(multiemb_2opacity_add_ac_deform_network, self).__init__()
        self.D = D
        self.W = W

        self.args = args
        self.min_embeddings = min_embeddings
        self.max_embeddings = max_embeddings
        self.num_frames = num_frames
        self.temporal_embedding_dim = args.temporal_embedding_dim
        self.gaussian_embedding_dim = args.gaussian_embedding_dim
        # new thermal_embedding_dim
        self.thermal_embedding_dim = args.thermal_embedding_dim

        self.c2f_temporal_iter = args.c2f_temporal_iter

        self.feature_out_c_1, self.feature_out_c_2, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.thermal_opacity_deform_c, self.m_rgb_deform_c, self.m_thermal_deform_c, self.rgb_deform_c, self.thermal_deform_c = self.create_net()
        self.feature_out_f_1, self.feature_out_f_2, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.thermal_opacity_deform_f, self.m_rgb_deform_f, self.m_thermal_deform_f, self.rgb_deform_f, self.thermal_deform_f = self.create_net()

        if args.zero_temporal:
            self.weight = torch.nn.Parameter(torch.zeros(max_embeddings, self.temporal_embedding_dim))
        else:
            self.weight = torch.nn.Parameter(torch.normal(0., 0.01/np.sqrt(self.temporal_embedding_dim),size=(max_embeddings, self.temporal_embedding_dim)))
        self.offsets = torch.nn.Parameter(torch.zeros((30, 1)))

    # add thermal_embedding_dim
    def create_net(self):
        self.feature_out_1 = [nn.Linear(self.temporal_embedding_dim + self.gaussian_embedding_dim, self.W)]
        self.feature_out_2 = [nn.Linear(self.temporal_embedding_dim + self.thermal_embedding_dim, self.W)]
        
        for i in range(self.D-1):
            self.feature_out_1.append(nn.ReLU())
            self.feature_out_1.append(nn.Linear(self.W,self.W))
            self.feature_out_2.append(nn.ReLU())
            self.feature_out_2.append(nn.Linear(self.W,self.W))
        feature_out_1 = nn.Sequential(*self.feature_out_1)
        feature_out_2 = nn.Sequential(*self.feature_out_2)
        return  \
            feature_out_1, feature_out_2,\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3*16))


    def get_temporal_embed(self, t, current_num_embeddings, align_corners=True):
        emb_resized = F.interpolate(self.weight[None,None,...], 
                                 size=(current_num_embeddings, self.temporal_embedding_dim), 
                                 mode='bilinear', align_corners=True)
        N, _ = t.shape
        t = t[0,0]

        fdim = self.temporal_embedding_dim
        grid = torch.cat([torch.arange(fdim).cuda().unsqueeze(-1)/(fdim-1), torch.ones(fdim,1).cuda() * t, ], dim=-1)[None,None,...]
        grid = (grid - 0.5) * 2

        emb = F.grid_sample(emb_resized, grid, align_corners=align_corners, mode='bilinear', padding_mode='reflection')
        emb = emb.repeat(1,1,N,1).squeeze()

        return emb
    
    def int_lininterp(self, t, init_val, final_val, until):
        return int(init_val + (final_val - init_val) * min(max(t, 0), until) / until)
    
    def query_time_1(self, pts, scales, rotations, time_emb, pc=None, embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if type(pc) == type(None):
            h = torch.cat([h, embeddings], dim=-1)
        else:        
            h = torch.cat([h, pc.get_embedding], dim=-1)

        h = feature_out(h)
        return h

    def query_time_2(self, pts, scales, rotations, time_emb, pc=None, thermal_embeddings=None, sh_coef=None, 
                    iter=None, feature_out=None, use_coarse_temporal_embedding=False, num_down_emb=30):
        t = time_emb[:,:1]
        if use_coarse_temporal_embedding:
            h = self.get_temporal_embed(t, num_down_emb)
        else:
            if self.args.no_c2f_temporal_embedding:
                h = self.get_temporal_embed(t, self.max_embeddings)
            else:
                h = self.get_temporal_embed(t, self.int_lininterp(iter, num_down_emb, self.max_embeddings, self.c2f_temporal_iter))
    
        if type(pc) == type(None):
            h = torch.cat([h, thermal_embeddings], dim=-1)
        else:        
            h = torch.cat([h, pc.get_thermal_embedding], dim=-1)

        h = feature_out(h)
        return h

    def deform(self, hidden, pts, scales, rotations, opacity, sh_coefs, m_rgb,
                pos_deform, scales_deform, rotations_deform, opacity_deform, rgb_deform, m_rgb_deform,
                  scale=1., scale_c=1., scale_o=1., coef_s=1., scale_m=1.):
        dx, ds, dr, do = pos_deform(hidden), None, None, None
        pts = pts + dx * scale
        
        if not self.args.no_ds:
            ds = scales_deform(hidden)
            scales = scales + ds * scale * coef_s
        if not self.args.no_dr:
            dr = rotations_deform(hidden)
            rotations = rotations + dr * scale
        if not self.args.no_do:
            do = opacity_deform(hidden) 
            opacity = opacity + do * scale * scale_o
        if not self.args.no_dc:
            dc = rgb_deform(hidden) 
            sh_coefs = sh_coefs + dc.view(-1,16,3) * scale_c
        if not self.args.no_dm:
            dm = m_rgb_deform(hidden)
            m_rgb = m_rgb + dm * scale_m
        return pts, scales, rotations, opacity, sh_coefs, m_rgb


    def deform_thermal(self, hidden, pts, scales, rotations, thermal_opacity, thermal_sh_coefs, m_th, 
                        pos_deform, scales_deform, rotations_deform, thermal_opacity_deform, thermal_deform, m_thermal_deform,
                        scale=1., scale_c=1., scale_o=1., coef_s=1., scale_m=1.):
        # add geometric deformation for thermal
        if self.args.change_thermal_geo:
            dx, ds, dr, do = pos_deform(hidden), None, None, None
            pts = pts + dx * scale
            
            if not self.args.no_ds:
                ds = scales_deform(hidden)
                scales = scales + ds * scale * coef_s
            if not self.args.no_dr:
                dr = rotations_deform(hidden)
                rotations = rotations + dr * scale

        if not self.args.no_do:
            thermal_do = thermal_opacity_deform(hidden)  # ← 新增 thermal 不透明度
            thermal_opacity = thermal_opacity + thermal_do * scale * scale_o
        if not self.args.no_dc:
            thermal_dc = thermal_deform(hidden)
            thermal_sh_coefs = thermal_sh_coefs + thermal_dc.view(-1,16,3) * scale_c
        if not self.args.no_dm:
            thermal_dm = m_thermal_deform(hidden)
            m_th = m_th + thermal_dm * scale_m
        # return thermal_opacity, thermal_sh_coefs
        return pts, scales, rotations, thermal_opacity, thermal_sh_coefs, m_th

    def forward(self, point, scales=None, rotations=None, opacity=None, thermal_opacity=None,
                time_emb=None, cam_no=None, pc=None, embeddings=None, thermal_embeddings=None,
                sh_coefs=None, thermal_sh_coefs=None, m_rgb=None, m_th=None, iter=None, num_down_emb_c=30, num_down_emb_f=30):
        pts, scales, rotations, opacity, thermal_opacity, m_rgb, m_th = point[:, :3], scales[:,:3], rotations[:,:4], opacity[:,:1], thermal_opacity[:,:1], m_rgb[:,:1], m_th[:,:1]
        pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig, m_rgb_orig, m_th_orig = pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, m_rgb, m_th

        if type(cam_no) == type(None):
            offset = _mean_active_offset(self.offsets)
        else:
            offset = self.offsets[cam_no]
        time_emb += offset

        use_anneal = self.args.use_anneal
        if use_anneal:
            coef = np.clip(iter / 1000, 0, 1)
            coef_c = np.clip((iter - self.args.deform_from_iter) / 1000, 0, 1)
            coef_o = coef_s = coef_c
            coef_m = coef_c
        else:
            coef = coef_c = coef_o = coef_s = coef_m = 1

        if self.args.no_coarse_deform:
            pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub, m_rgb_sub, m_th_sub = pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig, m_rgb_orig, m_th_orig
        else:
            hidden_1 = self.query_time_1(pts, scales, rotations, time_emb, pc, embeddings,
                                     sh_coefs, iter, self.feature_out_c_1, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()
            pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub, m_rgb_sub = self.deform(hidden_1, pts, scales, rotations, opacity, sh_coefs, m_rgb,
                self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.opacity_deform_c, self.rgb_deform_c, self.m_rgb_deform_c,
                  coef, coef_c, coef_o, coef_s, coef_m)
            # 释放显存
            del hidden_1

            # second query_time for thermal embeddings
            hidden_2 = self.query_time_2(pts_sub, scales_sub, rotations_sub, time_emb, pc, thermal_embeddings,
                                     thermal_sh_coefs, iter, self.feature_out_c_2, self.args.use_coarse_temporal_embedding, num_down_emb=num_down_emb_c).float()
            pts_sub, scales_sub, rotations_sub, thermal_opacity_sub, thermal_sh_coefs_sub, m_th_sub = self.deform_thermal(hidden_2, pts_sub, scales_sub, rotations_sub, thermal_opacity, thermal_sh_coefs, m_th, self.pos_deform_c, self.scales_deform_c, self.rotations_deform_c, self.thermal_opacity_deform_c, self.thermal_deform_c, self.m_thermal_deform_c,
                  coef, coef_c, coef_o, coef_s, coef_m)
            # 释放显存
            del hidden_2

        if self.args.no_fine_deform:
            pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, m_rgb, m_th = pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub, m_rgb_sub, m_th_sub
        else:
            hidden_1 = self.query_time_1(pts_sub, scales_sub, rotations_sub, time_emb, pc, embeddings,
                                      sh_coefs_sub, iter, self.feature_out_f_1, num_down_emb=num_down_emb_f).float()
            pts, scales, rotations, opacity, sh_coefs, m_rgb = self.deform(hidden_1, pts_sub, scales_sub, rotations_sub, opacity_sub, sh_coefs_sub, m_rgb_sub,
                self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.opacity_deform_f, self.rgb_deform_f, self.m_rgb_deform_f,
                  coef, coef_c, coef_o, coef_s, coef_m)
            # 释放显存
            del hidden_1

            # second query_time for thermal embeddings
            hidden_2 = self.query_time_2(pts, scales, rotations, time_emb, pc, thermal_embeddings,
                                      thermal_sh_coefs_sub, iter, self.feature_out_f_2, num_down_emb=num_down_emb_f).float()
            pts, scales, rotations, thermal_opacity, thermal_sh_coefs, m_th = self.deform_thermal(hidden_2, pts, scales, rotations, thermal_opacity_sub, thermal_sh_coefs_sub, m_th_sub, self.pos_deform_f, self.scales_deform_f, self.rotations_deform_f, self.thermal_opacity_deform_f, self.thermal_deform_f, self.m_thermal_deform_f,
                  coef, coef_c, coef_o, coef_s, coef_m)
            # 释放显存
            del hidden_2

        return pts, scales, rotations, opacity, thermal_opacity, sh_coefs, thermal_sh_coefs, m_rgb, m_th, \
            ((pts_sub, scales_sub, rotations_sub, opacity_sub, thermal_opacity_sub, sh_coefs_sub, thermal_sh_coefs_sub, m_rgb_sub, m_th_sub), \
            (pts_orig, scales_orig, rotations_orig, opacity_orig, thermal_opacity_orig, sh_coefs_orig, thermal_sh_coefs_orig, m_rgb_orig, m_th_orig))

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if name != "offsets":
                parameter_list.append(param)
        return parameter_list

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
