from pathlib import Path
import warnings
warnings.filterwarnings("ignore")  
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips  # 这里lpips是一个函数
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

lpips_vgg_model = lpips.LPIPS(net='vgg').cuda()  # VGG版本
lpips_alex_model = lpips.LPIPS(net='alex').cuda()  # Alex版本
lpips_vgg_model.eval()
lpips_alex_model.eval()

# -------------------------- 批量处理函数 --------------------------
def read_images_batch(renders_dir, gt_dir, batch_size=16):
    """Stream matched render/GT image batches."""
    image_names = sorted(name for name in os.listdir(renders_dir) if name.lower().endswith('.png'))
    gt_names = sorted(name for name in os.listdir(gt_dir) if name.lower().endswith('.png'))
    if image_names != gt_names:
        missing_gt = sorted(set(image_names) - set(gt_names))
        missing_render = sorted(set(gt_names) - set(image_names))
        raise ValueError(
            f"Render/GT filename mismatch: missing_gt={missing_gt[:5]}, "
            f"missing_render={missing_render[:5]}"
        )
    if not image_names:
        raise ValueError(f"No PNG images found in {renders_dir}")
    total_images = len(image_names)
    
    for i in range(0, total_images, batch_size):
        batch_names = image_names[i:i+batch_size]
        render_batch = []
        gt_batch = []
        
        for fname in batch_names:
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            
            # 转换为张量并添加到批次
            render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :]
            gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :]
            
            render_batch.append(render_tensor)
            gt_batch.append(gt_tensor)
        
        # 合并批次并移至GPU
        render_batch = torch.cat(render_batch, dim=0).cuda()
        gt_batch = torch.cat(gt_batch, dim=0).cuda()
        
        yield render_batch, gt_batch, batch_names


def evaluate_metrics_batch(renders, gts):
    """批量计算评估指标"""
    batch_size = renders.size(0)
    ssims = []
    psnrs = []
    lpips_vggs = []
    lpips_alexs = []
    
    with torch.no_grad():
        for i in range(batch_size):
            ssims.append(ssim(renders[i:i+1], gts[i:i+1])[0])
            psnrs.append(psnr(renders[i:i+1], gts[i:i+1]))

        # Match the official E-D3DGS metric protocol: keep PNG tensors in [0, 1]
        # and use LPIPS' default normalize=False behavior.
        lpips_vgg_batch = lpips_vgg_model(renders, gts).reshape(-1)
        lpips_alex_batch = lpips_alex_model(renders, gts).reshape(-1)
        lpips_vggs = lpips_vgg_batch.cpu().tolist()
        lpips_alexs = lpips_alex_batch.cpu().tolist()

    return ssims, psnrs, lpips_vggs, lpips_alexs


def evaluate_images(model_paths, test_type="rgb", batch_size=16):
    """评估RGB或热成像图像"""
    results_file = f"summary_{test_type}.json"
    per_view_file = f"per_view_{test_type}.json"

    full_dict = {}
    per_view_dict = {}

    print(f"\n开始评估{test_type.upper()}图像...")

    for scene_dir in model_paths:
        print(f"场景: {scene_dir}")
        scene_path = Path(scene_dir)
        test_dir = scene_path / "test"
        split_name = test_dir.name  # "train" or "test"
        test_dir_name = f"{split_name}_{test_type}"
        
        if not test_dir.exists():
            print(f"警告: {test_dir} 不存在，跳过此场景")
            continue
        
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        
        for method in os.listdir(test_dir):
            print(f"方法: {method}")
            
            method_dir = test_dir / method
            method_dir = method_dir / test_dir_name
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            
            if not (gt_dir.exists() and renders_dir.exists()):
                print(f"警告: {gt_dir} 或 {renders_dir} 不存在，跳过此方法")
                continue
            
            # 读取图像批次
            all_ssims = []
            all_psnrs = []
            all_lpips_vggs = []
            all_lpips_alexs = []
            all_names = []
            
            # 处理每个批次
            batches = read_images_batch(renders_dir, gt_dir, batch_size)
            for render_batch, gt_batch, batch_names in tqdm(batches, desc="批次处理进度"):
                ssims, psnrs, lpips_vggs, lpips_alexs = evaluate_metrics_batch(render_batch, gt_batch)
                
                all_ssims.extend([s.item() for s in ssims])
                all_psnrs.extend(psnrs)
                all_lpips_vggs.extend(lpips_vggs)
                all_lpips_alexs.extend(lpips_alexs)
                all_psnrs = [p.item() if isinstance(p, torch.Tensor) else p for p in all_psnrs]
                all_lpips_vggs = [v.item() if isinstance(v, torch.Tensor) else v for v in all_lpips_vggs]
                all_lpips_alexs = [a.item() if isinstance(a, torch.Tensor) else a for a in all_lpips_alexs]                
                all_names.extend(batch_names)
                
                # 释放GPU内存
                del render_batch, gt_batch
                torch.cuda.empty_cache()
            
            # 在处理完所有批次后，将结果转为张量再求平均
            avg_ssim = torch.tensor(all_ssims).mean().item()
            avg_psnr = torch.tensor(all_psnrs).mean().item()
            avg_lpips_vgg = torch.tensor(all_lpips_vggs).mean().item()  # 直接用张量求平均，更稳定
            avg_lpips_alex = torch.tensor(all_lpips_alexs).mean().item()
            
            print(f"场景: {scene_dir}")
            print(f"SSIM : {avg_ssim:>12.5f}")
            print(f"PSNR : {avg_psnr:>12.5f}")
            print(f"LPIPS_VGG: {avg_lpips_vgg:>12.5f}")
            print(f"LPIPS_ALEX: {avg_lpips_alex:>12.5f}\n")
            
            # 更新结果字典
            full_dict[scene_dir][method] = {
                "SSIM": avg_ssim,
                "PSNR": avg_psnr,
                "LPIPS_VGG": avg_lpips_vgg,
                "LPIPS_ALEX": avg_lpips_alex
            }
            
            per_view_dict[scene_dir][method] = {
                "SSIM": {name: ssim for ssim, name in zip(all_ssims, all_names)},
                "PSNR": {name: psnr for psnr, name in zip(all_psnrs, all_names)},
                "LPIPS_VGG": {name: lp for lp, name in zip(all_lpips_vggs, all_names)},
                "LPIPS_ALEX": {name: lp for lp, name in zip(all_lpips_alexs, all_names)}
            }
        
        # 保存结果
        with open(scene_dir + f"/{results_file}", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + f"/{per_view_file}", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
    
    return full_dict, per_view_dict

if __name__ == "__main__":
    # 设置GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    
    # 命令行参数解析
    parser = ArgumentParser(description="渲染图像评估脚本")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, help="模型路径列表")
    parser.add_argument('--batch_size', type=int, default=32, help="批处理大小")
    parser.add_argument('--rgb', action='store_true', help="评估RGB图像")
    parser.add_argument('--thermal', action='store_true', help="评估热成像图像")
    
    args = parser.parse_args()
    
    # 如果未指定评估类型，默认同时评估RGB和热成像
    evaluate_rgb = args.rgb or (not args.rgb and not args.thermal)
    evaluate_thermal = args.thermal or (not args.rgb and not args.thermal)
    
    if evaluate_rgb:
        evaluate_images(args.model_paths, test_type="rgb", batch_size=args.batch_size)
    
    if evaluate_thermal:
        evaluate_images(args.model_paths, test_type="thermal", batch_size=args.batch_size)
