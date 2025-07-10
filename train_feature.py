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
import torch
from random import randint
from utils.loss_utils import l1_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
from scene.VGG import VGGEncoder  #VGG编码器，特征提取

def training(dataset, opt, pipe, ply_path, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    vgg_encoder = VGGEncoder().cuda()
    # load the rgb reconstructed gaussians ply file
    scene = Scene(dataset, gaussians, load_path=ply_path, vgg_encoder=vgg_encoder)  #加载前一步训练的RGB高斯模型
    gaussians.training_setup_feature(opt)

    bg_color = [1]*32 if dataset.white_background else [0]*32
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Feature training", bar_format='{l_bar}{r_bar}')
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #特征渲染，返回渲染特征、相关视点、可见过滤器和半径
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, feature_linear=gaussians.feature_linear)
        rendered_feature = render_pkg["render"]

        # Loss
        gt_feature = viewpoint_cam.vgg_features  #真实特征
        loss = l1_loss(rendered_feature, gt_feature)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log
            tb_writer.add_scalar('train_loss/l1_loss', loss.item(), iteration)  #仅由L1损失控制

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
    # Save model
    os.makedirs(args.model_path + "/chkpnt", exist_ok = True)
    torch.save(gaussians.capture(is_feature_model=True), args.model_path + "/chkpnt" + "/feature.pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)

    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])  #没用到
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])  #保存7000与30000轮时的结果
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default='default')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]
    
    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "feature", args.exp_name)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.ply_path, args.debug_from)

    # All done
    print("\nFeature training complete.")
