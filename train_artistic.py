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
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader  #构建数据加载器
from torchvision import datasets  #加载图像数据集
from torchvision.transforms.functional import resize  #调整图像大小
import torchvision.transforms as T  #数据处理及变换
import numpy as np
from scene.VGG import VGGEncoder, normalize_vgg  #VGG编码器及归一化
from utils.loss_utils import cal_adain_style_loss, cal_mse_content_loss, cal_attn_local_loss

# 加载图像并变换后构建数据加载器
def getDataLoader(dataset_path, batch_size, sampler, image_side_length=256, num_workers=2):
    transform = T.Compose([
                T.Resize(size=(image_side_length*2, image_side_length*2)),  #将图像扩大2倍
                T.RandomCrop(image_side_length),  #随机裁剪回原大小
                T.ToTensor(),  #转换为张量
            ])

    train_dataset = datasets.ImageFolder(dataset_path, transform=transform)  #加载图像
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler(len(train_dataset)), num_workers=num_workers)

    return dataloader

def InfiniteSampler(n):  #循环采样器
    # i = 0
    i = n - 1
    order = np.random.permutation(n)  #打乱order序列
    while True:
        yield order[i]  #创建生成器，返回order[i]的值，下次调用时从下一句开始执行
        i += 1
        if i >= n:
            np.random.seed()  #生成随机种子，使每次打乱后的序列不同
            order = np.random.permutation(n)  #每遍历完一轮则打乱一次
            i = 0

class InfiniteSamplerWrapper(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):  #模拟无限数据流
        return 2 ** 31

def training(dataset, opt, pipe, ckpt_path, decoder_path, style_weight, content_preserve):
    opt.iterations = 50_000 if not decoder_path else 15_000  #默认50_000
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    # load the feature reconstructed gaussians ckpt file
    scene = Scene(dataset, gaussians, load_path=ckpt_path)  #加载前一步生成的特征高斯模型
    vgg_encoder = VGGEncoder().cuda()

    # compute the final vgg features for each point, and init pointnet decoder
    gaussians.training_setup_style(opt, decoder_path)  #设置高斯的风格化配置及初始化风格解码器

    # init wikiart dataset
    style_loader = getDataLoader(args.wikiartdir, batch_size=1, sampler=InfiniteSamplerWrapper,  #一次训练一张图
                    image_side_length=256, num_workers=4)
    style_iter = iter(style_loader)  #循环采样wikiart中的图像

    bg_color = [1]*3 if dataset.white_background else [0]*3
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Artistic training", bar_format='{l_bar}{r_bar}')
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # content preserve training
        if content_preserve and iteration % 7 == 0:  #内容保持训练
            decoded_rgb = gaussians.decoder(gaussians.final_vgg_features.detach()) # [N, 3]  对点云VGG特征解码
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=decoded_rgb)
            rendered_rgb = render_pkg["render"] # [3, H, W]  渲染的RGB图像
            gt_image = viewpoint_cam.original_image.cuda() # [3, H, W]
            loss = l1_loss(gt_image, rendered_rgb)
            loss.backward()

            iter_end.record()
            if iteration % 10 == 0:
                progress_bar.update(10)
            tb_writer.add_scalar('train_loss/content_preserve', loss.item(), iteration)  #仅由L1损失控制
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            continue


        # get style_img, this style_img has NOT been normalized according to the pretrained VGGmodel
        style_img = next(style_iter)[0].cuda()
        gt_image = viewpoint_cam.original_image.cuda() # [3, H, W]

        # Render
        with torch.no_grad():
            style_img_features = vgg_encoder(normalize_vgg(style_img))
            gt_image_features = vgg_encoder(normalize_vgg(gt_image.unsqueeze(0)))  #确保输入为[1, C, H, W]格式

        tranfered_features = gaussians.style_transfer(  #仅使用relu3_1层对点云的VGG特征进行风格迁移
            gaussians.final_vgg_features.detach(), # point cloud features [N, C]， 不计算梯度
            style_img_features.relu3_1,
        )

        # decoder the features of points to rgb
        decoded_rgb = gaussians.decoder(tranfered_features) # [N, 3]  对风格化的点云VGG特征解码

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_color=decoded_rgb)
        rendered_rgb = render_pkg["render"] # [3, H, W]
        
        rendered_rgb_features = vgg_encoder(normalize_vgg(rendered_rgb.unsqueeze(0)))  #渲染的风格化图特征

        # content loss and style loss
        content_loss = 0.
        for gt_feature, image_feature in zip(gt_image_features, rendered_rgb_features):  #累加内容损失
            content_loss += cal_mse_content_loss(torch.nn.functional.instance_norm(gt_feature), torch.nn.functional.instance_norm(image_feature))
        style_loss = 0.
        for style_feature, image_feature in zip(style_img_features, rendered_rgb_features):  #累加风格损失
            style_loss += cal_adain_style_loss(style_feature, image_feature)
        local_loss = 0.  #累加局部损失
        local_loss += cal_attn_local_loss(gt_image_features.relu2_1, style_img_features.relu2_1, rendered_rgb_features.relu2_1)
        local_loss += cal_attn_local_loss(gt_image_features.relu3_1, style_img_features.relu3_1, rendered_rgb_features.relu3_1)
        local_loss += cal_attn_local_loss(gt_image_features.relu4_1, style_img_features.relu4_1, rendered_rgb_features.relu4_1)

        local_weight = 3.
        loss = content_loss + style_loss * style_weight + local_loss * local_weight
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
            if iteration % 200 == 0:  #每200轮记录当前各项损失
                tb_writer.add_scalar('train_loss/content_loss', content_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/style_loss', style_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/local_loss', local_loss.item(), iteration)
            if iteration % 400 == 0:  #每400轮添加合成图像用于检视效果
                style_img = resize(style_img, (128, 128))  #调整风格图像大小为128x128
                rendered_rgb.clamp_(0, 1)  #将风格化图像中的像素值限制在[0,1]
                rendered_rgb[:, -128:, -128:] = style_img.squeeze(0)  #将风格图像覆盖到风格化图像的右下角
                tb_writer.add_image('stylized_img', rendered_rgb.clamp(0,1), iteration, dataformats='CHW')

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
    # Save model
    os.makedirs(args.model_path + "/chkpnt", exist_ok = True)
    torch.save(gaussians.capture(is_style_model=True), args.model_path + "/chkpnt" + "/gaussians.pth")

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
    parser.add_argument("--ckpt_path", type=str, required=True)  #前一步的特征高斯的路径
    parser.add_argument("--decoder_path", type=str, default=None)  #另一个已训练好的风格化高斯模型
    parser.add_argument("--rendering_mode", type=str, default="rgb", choices=["rgb", "feature"])
    parser.add_argument("--wikiartdir", type=str, default="datasets/wikiart/images")
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument("--style_weight", type=float, default=10.)
    parser.add_argument("--content_preserve", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]
    
    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "artistic", args.exp_name)
    print("Optimizing " + args.model_path + (' with content_preserve' if args.content_preserve else ''))

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.ckpt_path, args.decoder_path, args.style_weight, args.content_preserve)

    # All done
    print("\nArtistic training complete.")