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

import os  #文件读写、创建目录等
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui  #渲染图像
import sys  #处理命令行参数
from scene import Scene, GaussianModel  #3D重建场景、渲染高斯体
from utils.general_utils import safe_state  #设置程序状态
import uuid  #生成唯一标识符
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace  #解析命令行参数、存储命令行参数
from arguments import ModelParams, PipelineParams, OptimizationParams  #模型、流程、优化参数
try:
    from torch.utils.tensorboard import SummaryWriter  #训练过程中记录日志
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0  #起始迭代次数
    tb_writer = prepare_output_and_logger(dataset)  #准备输出目录、设置日志记录
    gaussians = GaussianModel(dataset.sh_degree)  #初始化高斯
    scene = Scene(dataset, gaussians)  #创建对应场景
    gaussians.training_setup_reconstruction(opt)  #设置高斯的重建配置
    if checkpoint:  #检查点
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  #选择背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  #转换为张量

    #记录每次迭代的开始、结束时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None  #存储相机视角
    ema_loss_for_log = 0.0  #记录指数加权平均损失，用于损失曲线
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Reconstruction training")
    first_iter += 1  #从下一轮开始训练
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)  #根据迭代次数更新学习率

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()  #从场景中获取相机视角
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))  #随机选择

        # Render
        if (iteration - 1) == debug_from:  #如果迭代数为调试起始点，启动调试模式
            pipe.debug = True

        #选择随机背景或预设背景
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #图像渲染，返回渲染图像、相关视点、可见过滤器和半径
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()  #原始图像
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))  #根据比重计算总损失，L1损失占80%
        loss.backward()

        iter_end.record()

        with torch.no_grad():  #推理阶段，禁用梯度计算
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log  #更新累积的加权总损失
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})  #每10轮一更新
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练过程中的信息
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):  #保存该次迭代
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:  #稠密化未结束(15_000)
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])  #更新高斯模型的最大半径，用于剪枝
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)  #记录稠密化信息
                # 稠密化开始(500)且满足稠密化周期(100)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None  #根据透明度重置周期来设置尺寸阈值
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)  #根据相机视野范围进行稠密化、根据阈值进行剪枝
                # 满足透明化重置周期(3_000)/稠密化开始时背景为白，则重置透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()  #优化器更新
                gaussians.optimizer.zero_grad(set_to_none = True)  #清空梯度

            if (iteration in checkpoint_iterations):  #保存检查点
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):  #存储日志及配置信息 
    if not args.model_path:  #若未指定模型保存路径
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])  #自动生成唯一保存路径
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))  #将所有命令行参数存入cfg_args文件

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

#记录训练统计信息
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:  #将每个迭代的L1损失、总损失、迭代时间记录到tensorboard
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:  #进行验证，每隔一定迭代次数进行
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},   #使用测试相机
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})  #使用部分训练相机

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']): #对每个视角进行渲染
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:  #第一次则记录真实图像
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()  #累加损失的平方
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])  #计算均方误差
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:  #记录场景的透明度直方图和高斯体总数
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)  #默认不调试
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  #默认不异常检测
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])  #默认不验证
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--exp_name", type=str, default='default')  #样例名默认为default
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)  #添加总迭代次数
    
    if args.source_path[-1] == '/':
        args.source_path = args.source_path[:-1]

    #设置输出路径
    args.model_path = os.path.join("./output", os.path.basename(args.source_path), "reconstruction", args.exp_name)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  #异常检测
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nReconstruction complete.")
