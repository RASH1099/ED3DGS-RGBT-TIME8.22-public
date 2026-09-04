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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from plyfile import PlyData, PlyElement
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils.loss_utils import l1_loss, ssim
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, background_thermal,hyperparam=None):
    rgb_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_rgb".format(name), "renders")
    rgb_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_rgb".format(name), "gt")
    thermal_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_thermal".format(name), "renders")
    thermal_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "{}_thermal".format(name), "gt")

    makedirs(rgb_render_path, exist_ok=True)
    makedirs(rgb_gts_path, exist_ok=True)
    makedirs(thermal_render_path, exist_ok=True)
    makedirs(thermal_gts_path, exist_ok=True)
    render_images_rgb = []
    render_images_thermal = []
    renderer_time_trace = []

    num_down_emb_c = hyperparam.min_embeddings
    num_down_emb_f = hyperparam.min_embeddings

    count = 0
    total_time = 0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if type(view.original_image) == type(None):
            if name == 'video':
                view.set_image()
            else:
                view.load_image()
        time1 = time()
        render_pkg = render(view, gaussians, pipeline, background, background_thermal,
                           iter=iteration, num_down_emb_c=num_down_emb_c,
                           num_down_emb_f=num_down_emb_f, modality_stage="D")
        rendering_rgb = render_pkg["render_rgb"]
        rendering_thermal = render_pkg["render_thermal"]
        renderer_time_trace.append({
            "image_name": str(view.image_name),
            "temporal_branch_applied": bool(
                render_pkg["renderer_temporal_branch_applied"]),
            "rgb_time": float(
                render_pkg["renderer_rgb_time"].detach().cpu().item()),
            "thermal_time": float(
                render_pkg["renderer_thermal_time"].detach().cpu().item()),
        })

        time2 = time()
        total_time += (time2 - time1)
        render_images_rgb.append(to8b(rendering_rgb).transpose(1,2,0))
        render_images_thermal.append(to8b(rendering_thermal).transpose(1,2,0))

        torchvision.utils.save_image(rendering_rgb, os.path.join(rgb_render_path, '{0:05d}'.format(count) + ".png"))
        torchvision.utils.save_image(rendering_thermal, os.path.join(thermal_render_path, '{0:05d}'.format(count) + ".png"))

        if name in ["train", "test"]:
            gt_rgb = view.original_image[0:3, :, :]
            gt_thermal = view.thermal_image[0:3, :, :]
            torchvision.utils.save_image(gt_rgb, os.path.join(rgb_gts_path, '{0:05d}'.format(count) + ".png"))
            torchvision.utils.save_image(gt_thermal, os.path.join(thermal_gts_path, '{0:05d}'.format(count) + ".png"))
        count +=1

    print("FPS:",(len(views)-1)/total_time)
    return renderer_time_trace


def render_set_test_optimize(model_path, name, iteration, views, gaussians, pipeline,
                              background, background_thermal, hyperparam, opt, args):
    """Render test views with per-view thermal pose + intrinsics optimization.

    Freezes all Gaussian parameters, only optimizes thermal delta (extrinsics)
    and learnable FoV (intrinsics) for each test view independently.
    """
    thermal_render_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                                       "{}_thermal".format(name), "renders")
    thermal_gts_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                                    "{}_thermal".format(name), "gt")
    makedirs(thermal_render_path, exist_ok=True)
    makedirs(thermal_gts_path, exist_ok=True)

    num_down_emb_c = hyperparam.min_embeddings
    num_down_emb_f = hyperparam.min_embeddings

    # Freeze all Gaussian parameters: geometry, SH, opacity, embedding, deformation
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._thermal_dc.requires_grad_(False)
    gaussians._thermal_rest.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._thermal_opacity.requires_grad_(False)
    gaussians._logit_modality.requires_grad_(False)
    gaussians._embedding.requires_grad_(False)
    gaussians._t_embedding.requires_grad_(False)
    for p in gaussians._deformation.parameters():
        p.requires_grad_(False)

    for idx, view in enumerate(tqdm(views, desc="Rendering test (pose opt)")):
        if type(view.original_image) == type(None):
            view.load_image()

        has_th = hasattr(view, 'has_thermal') and view.has_thermal and view.thermal_image is not None

        if not has_th:
            # No thermal data → just render thermal normally
            with torch.no_grad():
                render_pkg = render(view, gaussians, pipeline, background, background_thermal,
                                   iter=iteration, num_down_emb_c=num_down_emb_c,
                                   num_down_emb_f=num_down_emb_f, modality_stage="D",
                                   thermal_only=True)
                rendering_th = render_pkg["render_thermal"]
            torchvision.utils.save_image(rendering_th, os.path.join(thermal_render_path, f'{idx:05d}.png'))
            continue

        # Enable grad only for this view's thermal extrinsics + intrinsics
        view.thermal_delta_quaternion.requires_grad_(True)
        view.thermal_delta_translation.requires_grad_(True)
        view.learnable_tfovx.requires_grad_(True)
        view.learnable_tfovy.requires_grad_(True)

        # Separate LRs: rotation / translation / intrinsics (same as training)
        opt_params = [
            {'params': [view.thermal_delta_quaternion], 'lr': args.test_pose_lr_r},
            {'params': [view.thermal_delta_translation], 'lr': args.test_pose_lr_t},
            {'params': [view.learnable_tfovx, view.learnable_tfovy], 'lr': args.test_pose_lr_fov},
        ]
        optimizer_pose = torch.optim.Adam(opt_params, eps=1e-15)

        gt_thermal = view.thermal_image.cuda()
        best_loss = float('inf')
        best_rendering = None

        num_iter = args.optim_test_pose_iter

        # Use enable_grad since outer render_sets has torch.no_grad()
        pbar = tqdm(total=num_iter, desc=f"View {idx+1}/{len(views)}", leave=False)
        initial_loss = None
        initial_quat = view.thermal_delta_quaternion.data.clone()
        initial_trans = view.thermal_delta_translation.data.clone()
        initial_fovx = view.learnable_tfovx.data.clone()
        initial_fovy = view.learnable_tfovy.data.clone()

        with torch.enable_grad():
            for it in range(num_iter):
                render_pkg = render(view, gaussians, pipeline, background, background_thermal,
                                   iter=iteration, num_down_emb_c=num_down_emb_c,
                                   num_down_emb_f=num_down_emb_f, modality_stage="D",
                                   thermal_only=True)
                rendering_th = render_pkg["render_thermal"]

                loss_l1 = l1_loss(rendering_th, gt_thermal)
                ssim_val = ssim(rendering_th, gt_thermal)[0]
                loss = loss_l1 + args.test_ssim_weight * (1.0 - ssim_val)
                loss.backward()

                # Diagnostic: check gradients on first iteration
                if it == 0:
                    grad_q = view.thermal_delta_quaternion.grad
                    grad_t = view.thermal_delta_translation.grad
                    grad_fx = view.learnable_tfovx.grad
                    grad_fy = view.learnable_tfovy.grad
                    gq_norm = grad_q.norm().item() if grad_q is not None else -1
                    gt_norm = grad_t.norm().item() if grad_t is not None else -1
                    gfx_norm = grad_fx.item() if grad_fx is not None else -1
                    gfy_norm = grad_fy.item() if grad_fy is not None else -1
                    tqdm.write(f"[View {idx}] iter0 grad: dQ={gq_norm:.6f} dT={gt_norm:.6f} "
                              f"dFovX={gfx_norm:.6f} dFovY={gfy_norm:.6f} "
                              f"L1={loss_l1.item():.4f} SSIM={ssim_val.item():.4f} loss={loss.item():.4f}")
                    initial_loss = loss.item()

                optimizer_pose.step()
                optimizer_pose.zero_grad(set_to_none=True)

                with torch.no_grad():
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_rendering = rendering_th.clone().detach()
                    pbar.set_postfix(loss=f"{loss.item():.4f}", best=f"{best_loss:.4f}")
                    pbar.update(1)
        pbar.close()

        # Diagnostic: print delta changes for this view
        dq_change = (view.thermal_delta_quaternion.data - initial_quat).norm().item()
        dt_change = (view.thermal_delta_translation.data - initial_trans).norm().item()
        dfx_change = (view.learnable_tfovx.data - initial_fovx).item()
        dfy_change = (view.learnable_tfovy.data - initial_fovy).item()
        tqdm.write(f"[View {idx}] final: loss {initial_loss:.4f}→{best_loss:.4f} | "
                   f"ΔQ={dq_change:.6f} ΔT={dt_change:.6f} ΔFovX={dfx_change:.6f} ΔFovY={dfy_change:.6f}")

        # Save best rendering and GT
        torchvision.utils.save_image(best_rendering, os.path.join(thermal_render_path, f'{idx:05d}.png'))
        torchvision.utils.save_image(gt_thermal, os.path.join(thermal_gts_path, f'{idx:05d}.png'))

        # Freeze params again
        view.thermal_delta_quaternion.requires_grad_(False)
        view.thermal_delta_translation.requires_grad_(False)
        view.learnable_tfovx.requires_grad_(False)
        view.learnable_tfovy.requires_grad_(False)


def render_sets(dataset : ModelParams, hyperparam, opt, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, args=None):
    # No global torch.no_grad() — let render_set and render_set_test_optimize manage it
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, duration=hyperparam.total_num_frames, loader=dataset.loader, opt=opt)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    bg_thermal = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    background_thermal = torch.tensor(bg_thermal, dtype=torch.float32, device="cuda")

    if not skip_train:
        with torch.no_grad():
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, background_thermal, hyperparam=hyperparam)
    if not skip_test:
        if getattr(args, 'no_test_opt', False):
            with torch.no_grad():
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, background_thermal, hyperparam=hyperparam)
        else:
            render_set_test_optimize(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, background_thermal, hyperparam=hyperparam, opt=opt, args=args)
    # if not skip_video:
    #     render_set(dataset.model_path, "video", scene.loaded_iter, scene.getVideoCameras(), gaussians, pipeline, background, hyperparam=hyperparam)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    # Test-view thermal pose optimization args (matching training LR split)
    parser.add_argument("--optim_test_pose_iter", default=500, type=int,
                        help="Number of iterations for per-view thermal pose optimization")
    parser.add_argument("--test_pose_lr_r", default=0.01, type=float,
                        help="LR for thermal rotation delta (per-view opt)")
    parser.add_argument("--test_pose_lr_t", default=0.02, type=float,
                        help="LR for thermal translation delta (per-view opt)")
    parser.add_argument("--no_test_opt", action="store_true",
                        help="Disable per-view thermal pose optimization for test views")
    parser.add_argument("--test_ssim_weight", default=0.2, type=float,
                        help="SSIM weight in test pose optimization loss (L1 + w*DSSIM)")
    parser.add_argument("--test_pose_lr_fov", default=0.01, type=float,
                        help="LR for thermal FoV (per-view opt)")

    # import sys
    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        # import mmcv
        import mmengine
        from utils.params_utils import merge_hparams
        # config = mmcv.Config.fromfile(args.configs)
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), opt.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args=args)
    # CUDA_VISIBLE_DEVICES=2 python render.py --model_path output/dynerf/coffee_martini_wo_cam13 --skip_train --configs arguments/dynerf/coffee_martini_wo_cam13.py
