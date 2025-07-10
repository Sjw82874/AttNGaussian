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

from pathlib import Path  #用于处理路径
import os
from PIL import Image  #打开、保存图像文件
import torch
import torchvision.transforms.functional as tf  #图像变换
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import gc

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())  #转换为张量，且只取RGB通道
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)  #记录文件名
    return renders, gts, image_names


def evaluate(model_paths):  #output/xxx/artistic/xxx
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            train_dir = Path(scene_dir) / "train"  #每个场景下必须有"train"文件夹

            for method in os.listdir(train_dir):
                print("Method:", method)
                method_dir = train_dir / method  #train文件夹下具体的风格图编号

                method_results = {"SSIM": None, "PSNR": None, "LPIPS": None, "per_view": {}}
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "render"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")
                method_results["SSIM"] = torch.tensor(ssims).mean().item()
                method_results["PSNR"] = torch.tensor(psnrs).mean().item()
                method_results["LPIPS"] = torch.tensor(lpipss).mean().item()
                method_results["per_view"] = {
                    name:{
                            "SSIM": ssim_val.item(),
                            "PSNR": psnr_val.item(),
                            "LPIPS": lpips_val.item()
                        } for ssim_val, psnr_val, lpips_val, name in zip(ssims, psnrs, lpipss, image_names)
                    }
                
                with open(method_dir / "results.json", 'w') as fp:  # 保存到json文件
                    json.dump({
                        "SSIM": method_results["SSIM"],
                        "PSNR": method_results["PSNR"],
                        "LPIPS": method_results["LPIPS"]
                    }, fp, indent=2)
                with open(method_dir / "per_view.json", 'w') as fp:
                    json.dump(method_results["per_view"], fp, indent=2)
                
                del renders, gts, ssims, psnrs, lpipss  # 释放显存
                torch.cuda.empty_cache()
                gc.collect()
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)