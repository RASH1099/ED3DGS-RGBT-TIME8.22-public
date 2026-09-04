import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def build_viewmatrix_from_Tcw(Tcw_12):
    """
    从 (12,) 列优先 Tcw 还原 4x4 world_view_transform (列优先, OpenGL convention).
    Tcw_12 索引: [W00,W10,W20, W01,W11,W21, W02,W12,W22, tx,ty,tz]

    目标格式 (Self-Cali-GS / 3dgs-pose convention):
    列优先存储 = [R | 0; t | 1].t() 即:
      col0=[R00,R10,R20,tx], col1=[R01,R11,R21,ty], col2=[R02,R12,R22,tz], col3=[0,0,0,1]
    """
    M34 = Tcw_12.view(4, 3).t().contiguous()       # (3,4): [R | t]
    R = M34[:, :3].contiguous()                     # (3,3)
    t = M34[:, 3].contiguous()                      # (3,)
    Rt = torch.cat([R, t.unsqueeze(1)], dim=1)      # (3,4): [R | t]
    last_row = torch.tensor([[0., 0., 0., 1.]], device=Tcw_12.device, dtype=Tcw_12.dtype)
    w2c = torch.cat([Rt, last_row], dim=0)           # (4,4): [[R t]; [0 1]]
    return w2c.T.contiguous()                        # 列优先


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           bg_thermal: torch.Tensor, mode="None", scaling_modifier=1.0,
           override_color=None, override_thermal=None, cam_no=None, iter=None,
           train_coarse=False, num_down_emb_c=5, num_down_emb_f=5,
           modality_routing=None, modality_tau=1.0, modality_stage="B",
           thermal_only=False, rgb_only_teacher=False):
    """
    Render the scene using 3dgs-pose rasterizer (single-modal, called twice).

    RGB: uses fixed COLMAP/JSON poses (no delta learning).
    Thermal: uses RGB pose as init + learnable delta extrinsics + learnable intrinsics.

    When thermal_only=True: skip RGB render entirely, thermal drives all geometry+appearance.
    When rgb_only_teacher=True: skip Thermal render entirely; no Thermal branch can affect RGB.
    """
    if thermal_only and rgb_only_teacher:
        raise ValueError("thermal_only and rgb_only_teacher are mutually exclusive")

    # ============================================================
    # 1. Screenspace points (for densification gradients)
    # ============================================================
    screenspace_points_rgb = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                              requires_grad=True, device="cuda") + 0
    screenspace_points_rgb_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                                      requires_grad=True, device="cuda") + 0
    screenspace_points_th = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                             requires_grad=True, device="cuda") + 0
    screenspace_points_th_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                                     requires_grad=True, device="cuda") + 0
    try:
        screenspace_points_rgb.retain_grad()
        screenspace_points_rgb_densify.retain_grad()
        screenspace_points_th.retain_grad()
        screenspace_points_th_densify.retain_grad()
    except:
        pass

    # ============================================================
    # 2. Get poses
    # ============================================================
    # 3dgs-pose rasterizer expects column-major 4x4:
    #   viewmatrix: w2c,  projmatrix: w2c @ K,  intrinsic: K

    # --- RGB: use Camera-stored transforms directly (no LearnPose) ---
    rgb_viewmatrix = viewpoint_camera.world_view_transform          # column-major w2c
    rgb_full_proj = viewpoint_camera.full_proj_transform           # w2c @ K
    rgb_full_proj_intrinsic = rgb_full_proj.detach()
    rgb_intrinsic = viewpoint_camera.projection_matrix             # K
    rgb_campos = viewpoint_camera.camera_center                    # camera position

    # --- Thermal: Camera-level delta (Self-Cali-GS style) + learnable intrinsics ---
    if hasattr(viewpoint_camera, 'has_thermal') and viewpoint_camera.has_thermal:
        thermal_viewmatrix = viewpoint_camera.get_thermal_world_view_transform()
        viewpoint_camera.refresh_thermal_projection()
        thermal_intrinsic = viewpoint_camera.thermal_projection_matrix_learnable
        # Keep the established averaged pose gradient, while routing the exact
        # summed projection gradient only to the learnable intrinsic matrix.
        thermal_full_proj = thermal_viewmatrix @ thermal_intrinsic.detach()
        thermal_full_proj_intrinsic = thermal_viewmatrix.detach() @ thermal_intrinsic
        thermal_campos = thermal_viewmatrix.inverse()[3, :3]
    else:
        thermal_viewmatrix = rgb_viewmatrix
        thermal_full_proj = rgb_full_proj
        thermal_full_proj_intrinsic = rgb_full_proj_intrinsic
        thermal_intrinsic = rgb_intrinsic
        thermal_campos = rgb_campos

    # ============================================================
    # 3. Build rasterization settings (one per modality)
    # ============================================================
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    shift_factors = torch.zeros(3, device="cuda")  # no entrance pupil shift

    # RGB raster settings
    raster_settings_rgb = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=rgb_viewmatrix,
        projmatrix=rgb_full_proj,
        projmatrix_intrinsic=rgb_full_proj_intrinsic,
        intrinsic=rgb_intrinsic,
        sh_degree=pc.active_sh_degree,
        campos=rgb_campos,
        prefiltered=False,
        debug=pipe.debug,
        debug_iter=iter,
    )

    # Thermal raster settings
    if hasattr(viewpoint_camera, 'has_thermal') and viewpoint_camera.has_thermal:
        th_img_h = int(getattr(viewpoint_camera, 'thermal_height',
                               viewpoint_camera.image_height))
        th_img_w = int(getattr(viewpoint_camera, 'thermal_width',
                               viewpoint_camera.image_width))
        effective_tfovx, effective_tfovy = viewpoint_camera.get_thermal_fovs()
        th_tanfovx = torch.tan(effective_tfovx * 0.5)
        th_tanfovy = torch.tan(effective_tfovy * 0.5)
    else:
        th_img_h = int(viewpoint_camera.image_height)
        th_img_w = int(viewpoint_camera.image_width)
        th_tanfovx = tanfovx
        th_tanfovy = tanfovy

    raster_settings_thermal = GaussianRasterizationSettings(
        image_height=th_img_h,
        image_width=th_img_w,
        tanfovx=th_tanfovx,
        tanfovy=th_tanfovy,
        bg=bg_thermal,
        scale_modifier=scaling_modifier,
        viewmatrix=thermal_viewmatrix,
        projmatrix=thermal_full_proj,
        projmatrix_intrinsic=thermal_full_proj_intrinsic,
        intrinsic=thermal_intrinsic,
        sh_degree=pc.active_sh_degree,
        campos=thermal_campos,
        prefiltered=False,
        debug=pipe.debug,
        debug_iter=iter,
    )

    # ============================================================
    # 4. Deformation network forward
    # ============================================================
    means3D = pc.get_xyz
    rgb_time_value = torch.as_tensor(
        viewpoint_camera.time, device=means3D.device, dtype=means3D.dtype)
    rgb_time = rgb_time_value.reshape(1, 1).repeat(means3D.shape[0], 1)
    # Inference is controlled only by the learned model/camera interface.
    # thermal_frame_shift is truth metadata and may be cleared before forward.
    temporal_alignment_enabled = bool(getattr(
        viewpoint_camera, 'temporal_observation_correction_enabled', False))
    if temporal_alignment_enabled:
        thermal_time_value = viewpoint_camera.get_temporal_time().to(
            device=means3D.device, dtype=means3D.dtype)
        thermal_time = thermal_time_value.reshape(1, 1).repeat(means3D.shape[0], 1)
    else:
        thermal_time_value = rgb_time_value
        thermal_time = rgb_time

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation

    opacity = pc._opacity
    thermal_opacity = pc._thermal_opacity
    shs = pc.get_features
    thermal_shs = pc.get_thermal_features

    # Resolve modality routing
    if modality_routing is None:
        if modality_stage == "A":
            s_hard = torch.zeros((pc.get_xyz.shape[0], 3), device=means3D.device)
            s_hard[:, 0] = 1.0
        elif modality_stage == "D":
            s_soft = torch.softmax(pc._logit_modality, dim=-1)
            s_hard = torch.zeros_like(s_soft)
            s_hard.scatter_(1, torch.argmax(s_soft, dim=-1, keepdim=True), 1.0)
        else:
            from utils.modality_routing import compute_modality_routing
            s_hard, _ = compute_modality_routing(pc._logit_modality, tau=modality_tau, hard=True)
    else:
        s_hard = modality_routing

    # Deformation
    means3D_rgb_final, scales_rgb_final, rotations_rgb_final, opacity_rgb_final, \
        thermal_opacity_rgb_final, shs_rgb_final, thermal_shs_rgb_final, extras = pc._deformation(
            means3D, scales, rotations, opacity, thermal_opacity,
            rgb_time, cam_no, pc, None, None, shs, thermal_shs,
            iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f,
            modality_routing=s_hard,
            thermal_only=thermal_only,
            rgb_only_teacher=rgb_only_teacher)

    if rgb_only_teacher:
        scales_rgb_final = pc.scaling_activation(scales_rgb_final)
        rotations_rgb_final = pc.rotation_activation(rotations_rgb_final)
        opacity_rgb_final = pc.opacity_activation(opacity_rgb_final)
        opacity_rgb = opacity_rgb_final * (s_hard[:, 0:1] + s_hard[:, 1:2])
        colors_precomp = None
        rgb_shs = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(
                    pc.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                colors_precomp = torch.clamp_min(
                    eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5,
                    0.0,
                )
            else:
                rgb_shs = shs_rgb_final
        else:
            colors_precomp = override_color
        rasterizer_rgb = GaussianRasterizer(raster_settings=raster_settings_rgb)
        rendered_rgb, radii_rgb, depth_rgb, weights_rgb, mean2D_rgb = rasterizer_rgb(
            means3D=means3D_rgb_final,
            means2D=screenspace_points_rgb,
            means2D_densify=screenspace_points_rgb_densify,
            shift_factors=shift_factors,
            shs=rgb_shs,
            colors_precomp=colors_precomp,
            opacities=opacity_rgb,
            scales=scales_rgb_final,
            rotations=rotations_rgb_final,
            cov3D_precomp=cov3D_precomp,
        )
        radii_th = torch.zeros_like(radii_rgb)
        return {
            "render_thermal": torch.zeros(3, 1, 1, device="cuda"),
            "render_rgb": rendered_rgb,
            "viewspace_points": screenspace_points_rgb,
            "viewspace_points_densify": screenspace_points_rgb_densify,
            "visibility_filter": radii_rgb > 0,
            "radii": radii_rgb,
            "viewspace_points_rgb": screenspace_points_rgb,
            "viewspace_points_th": screenspace_points_th,
            "viewspace_points_densify_rgb": screenspace_points_rgb_densify,
            "viewspace_points_densify_th": screenspace_points_th_densify,
            "visibility_filter_rgb": radii_rgb > 0,
            "visibility_filter_th": radii_th > 0,
            "radii_rgb": radii_rgb,
            "radii_th": radii_th,
            "sh_coefs_final": shs_rgb_final,
            "thermal_sh_coefs_final": None,
            "s_hard": s_hard,
            "extras": {"rgb_only_teacher": True},
            "depth": depth_rgb,
            "weights": weights_rgb,
            "means2D": mean2D_rgb,
            "rgb_only_teacher": True,
        }

    if temporal_alignment_enabled:
        means3D_th_final, scales_th_final, rotations_th_final, opacity_th_rgb_final, \
            thermal_opacity_th_final, shs_th_rgb_final, thermal_shs_th_final, \
            thermal_extras = pc._deformation(
                means3D, scales, rotations, opacity, thermal_opacity,
                thermal_time, cam_no, pc, None, None, shs, thermal_shs,
                iter=iter, num_down_emb_c=num_down_emb_c, num_down_emb_f=num_down_emb_f,
                modality_routing=s_hard,
                thermal_only=thermal_only)
    else:
        means3D_th_final = means3D_rgb_final
        scales_th_final = scales_rgb_final
        rotations_th_final = rotations_rgb_final
        thermal_opacity_th_final = thermal_opacity_rgb_final
        thermal_shs_th_final = thermal_shs_rgb_final
        thermal_extras = extras

    scales_rgb_final = pc.scaling_activation(scales_rgb_final)
    rotations_rgb_final = pc.rotation_activation(rotations_rgb_final)
    opacity_rgb_final = pc.opacity_activation(opacity_rgb_final)
    scales_th_final = pc.scaling_activation(scales_th_final)
    rotations_th_final = pc.rotation_activation(rotations_th_final)
    thermal_opacity_th_final = pc.opacity_activation(thermal_opacity_th_final)

    # Modality masking
    m_rgb = s_hard[:, 0:1] + s_hard[:, 1:2]
    m_th = s_hard[:, 0:1] + s_hard[:, 2:3]
    opacity_rgb = opacity_rgb_final * m_rgb
    opacity_th = thermal_opacity_th_final * m_th

    # ============================================================
    # 5. SH → colors (or use precomp)
    # ============================================================
    colors_precomp = None
    thermal_precomp = None
    rgb_shs = None
    thermal_shs_in = None

    if override_color is None:
        if pipe.convert_SHs_python:
            # RGB SH → precomp
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(
                pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            rgb_shs = shs_rgb_final

        # Thermal SH → precomp
        if (not rgb_only_teacher
                and hasattr(viewpoint_camera, 'has_thermal') and viewpoint_camera.has_thermal):
            if pipe.convert_SHs_python:
                thermal_shs_view = pc.get_thermal_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp_th = (pc.get_xyz - thermal_campos.cuda().repeat(
                    pc.get_thermal_features.shape[0], 1))
                dir_pp_th_normalized = dir_pp_th / dir_pp_th.norm(dim=1, keepdim=True)
                sh2thermal = eval_sh(pc.active_sh_degree, thermal_shs_view, dir_pp_th_normalized)
                thermal_precomp = torch.clamp_min(sh2thermal + 0.5, 0.0)
            else:
                thermal_shs_in = thermal_shs_th_final
    else:
        colors_precomp = override_color
        thermal_precomp = override_thermal

    # ============================================================
    # 6. Dual render — two calls to 3dgs-pose rasterizer
    #    When thermal_only: skip RGB, use tracked viewspace points for thermal
    # ============================================================

    if thermal_only:
        # --- Thermal-only: skip RGB, thermal drives everything ---
        rendered_rgb = torch.zeros(3, 1, 1, device="cuda")
        radii_rgb = torch.zeros(pc.get_xyz.shape[0], device="cuda")
        depth_rgb = None
        weights_rgb = None
        mean2D_rgb = None

        if hasattr(viewpoint_camera, 'has_thermal') and viewpoint_camera.has_thermal:
            rasterizer_th = GaussianRasterizer(raster_settings=raster_settings_thermal)
            rendered_thermal, radii_th, depth_th, weights_th, mean2D_th = rasterizer_th(
                means3D=means3D_th_final,
                means2D=screenspace_points_th,       # grad-tracked for densification
                means2D_densify=screenspace_points_th_densify,
                shift_factors=shift_factors,
                shs=thermal_shs_in,
                colors_precomp=thermal_precomp,
                opacities=opacity_th,
                scales=scales_th_final,
                rotations=rotations_th_final,
                cov3D_precomp=cov3D_precomp,
            )
        else:
            rendered_thermal = rendered_rgb
            radii_th = radii_rgb

        # Use thermal for densification (viewspace_points, radii, visibility)
        vis_filter = radii_th > 0
        viewspace_pts = screenspace_points_th
        viewspace_pts_densify = screenspace_points_th_densify
        radii_out = radii_th

    else:
        # --- Normal dual-modal path ---
        # --- RGB render ---
        rasterizer_rgb = GaussianRasterizer(raster_settings=raster_settings_rgb)

        rendered_rgb, radii_rgb, depth_rgb, weights_rgb, mean2D_rgb = rasterizer_rgb(
            means3D=means3D_rgb_final,
            means2D=screenspace_points_rgb,
            means2D_densify=screenspace_points_rgb_densify,
            shift_factors=shift_factors,
            shs=rgb_shs,
            colors_precomp=colors_precomp,
            opacities=opacity_rgb,
            scales=scales_rgb_final,
            rotations=rotations_rgb_final,
            cov3D_precomp=cov3D_precomp,
        )

        # --- Thermal render ---
        if hasattr(viewpoint_camera, 'has_thermal') and viewpoint_camera.has_thermal:
            rasterizer_th = GaussianRasterizer(raster_settings=raster_settings_thermal)

            rendered_thermal, radii_th, depth_th, weights_th, mean2D_th = rasterizer_th(
                means3D=means3D_th_final,
                means2D=screenspace_points_th,
                means2D_densify=screenspace_points_th_densify,
                shift_factors=shift_factors,
                shs=thermal_shs_in,
                colors_precomp=thermal_precomp,
                opacities=opacity_th,
                scales=scales_th_final,
                rotations=rotations_th_final,
                cov3D_precomp=cov3D_precomp,
            )
        else:
            rendered_thermal = rendered_rgb
            radii_th = radii_rgb
            screenspace_points_th = screenspace_points_rgb
            screenspace_points_th_densify = screenspace_points_rgb_densify

        vis_filter = radii_rgb > 0
        viewspace_pts = screenspace_points_rgb
        viewspace_pts_densify = screenspace_points_rgb_densify
        radii_out = radii_rgb

    if isinstance(extras, dict):
        extras["s_hard"] = s_hard
        extras["thermal_time"] = thermal_time_value
        extras["thermal_deformation"] = thermal_extras

    return {"render_thermal": rendered_thermal,
            "render_rgb": rendered_rgb,
            "renderer_temporal_branch_applied": temporal_alignment_enabled,
            "renderer_rgb_time": rgb_time_value,
            "renderer_thermal_time": thermal_time_value,
            "viewspace_points": viewspace_pts,
            "viewspace_points_densify": viewspace_pts_densify,
            "visibility_filter": vis_filter,
            "radii": radii_out,
            "viewspace_points_rgb": screenspace_points_rgb,
            "viewspace_points_th": screenspace_points_th,
            "viewspace_points_densify_rgb": screenspace_points_rgb_densify,
            "viewspace_points_densify_th": screenspace_points_th_densify,
            "visibility_filter_rgb": radii_rgb > 0,
            "visibility_filter_th": radii_th > 0,
            "radii_rgb": radii_rgb,
            "radii_th": radii_th,
            "sh_coefs_final": shs_rgb_final,
            "thermal_sh_coefs_final": thermal_shs_th_final,
            "s_hard": s_hard,
            "extras": extras,
            "depth": depth_rgb,
            "weights": weights_rgb,
            "means2D": mean2D_rgb,
            "rgb_only_teacher": rgb_only_teacher,
            }
