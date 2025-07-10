import os
from PIL import Image
import torch
import torchvision
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import numpy as np
from softmax_splatting import softsplat
from RAFT import RAFT, InputPadder

# 初始化RAFT模型
def initialize_raft():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir,"RAFT/models/raft-things.pth")
    args = Namespace(model=model_path, small=False, mixed_precision=False, alternate_corr=False)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model = model.module.cuda().eval()
    return model

# 反向光流变换
backwarp_tenGrid = {}
def backwarp(tenIn, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

def main(model_path):  # output/xxx/artistic/xxx/train/xx/render
    raft_model = initialize_raft()
    
    # 读取图像
    image_dir = model_path
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    images = []
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    for f in image_files:
        img = Image.open(os.path.join(image_dir, f)).convert('RGB')
        images.append(transform(img).unsqueeze(0).cuda())
    n = len(images)

    # 计算指标
    results = {'short': {'lpips': [], 'rmse': []}, 'long': {'lpips': [], 'rmse': []}}
    
    # 生成图像对
    short_pairs = [(i, i+1) for i in range(n-1)]
    long_pairs = [(i, i+5) for i in range(n-5)]
    
    # 短距离计算
    for i, j in tqdm(short_pairs, desc='Processing short-range'):
        with torch.no_grad():
            img1, img2 = images[i], images[j]
            padder = InputPadder(img1.shape)
            img1_pad, img2_pad = padder.pad(img1, img2)
            _, flow = raft_model(img1_pad, img2_pad, iters=32, test_mode=True)
            tenMetric = torch.nn.functional.l1_loss(input=img1_pad, target=backwarp(tenIn=img2_pad, tenFlow=flow), reduction='none').mean([1], True)
            warped = softsplat(tenIn=img1_pad, tenFlow=flow, tenMetric=(-10 * tenMetric).clip(-10, 10), strMode='soft')
            mask =  torch.where(warped.sum(1, keepdim=True) == 0.,0.,1.)

            lpips_val = lpips(warped, img2_pad * mask, net_type='vgg').item()
            rmse_val = torch.sqrt(torch.nn.functional.mse_loss(warped, img2_pad * mask, reduction='mean')).item()
            results['short']['lpips'].append(lpips_val)
            results['short']['rmse'].append(rmse_val)
    
    # 长距离计算
    for i, j in tqdm(long_pairs, desc='Processing long-range'):
        with torch.no_grad():
            img1, img2 = images[i], images[j]
            padder = InputPadder(img1.shape)
            img1_pad, img2_pad = padder.pad(img1, img2)
            _, flow = raft_model(img1_pad, img2_pad, iters=32, test_mode=True)
            tenMetric = torch.nn.functional.l1_loss(input=img1_pad, target=backwarp(tenIn=img2_pad, tenFlow=flow), reduction='none').mean([1], True)
            warped = softsplat(tenIn=img1_pad, tenFlow=flow, tenMetric=(-10 * tenMetric).clip(-10, 10), strMode='soft')
            mask =  torch.where(warped.sum(1, keepdim=True) == 0.,0.,1.)

            lpips_val = lpips(warped, img2_pad * mask, net_type='vgg').item()
            rmse_val = torch.sqrt(torch.nn.functional.mse_loss(warped, img2_pad * mask, reduction='mean')).item()
            results['long']['lpips'].append(lpips_val)
            results['long']['rmse'].append(rmse_val)
    
    # 计算均值
    avg_short_lpips = np.mean(results['short']['lpips'])
    avg_long_lpips = np.mean(results['long']['lpips'])
    avg_short_rmse = np.mean(results['short']['rmse'])
    avg_long_rmse = np.mean(results['long']['rmse'])
    # 计算最小值
    min_short_lpips = np.min(results['short']['lpips'])
    min_long_lpips = np.min(results['long']['lpips'])
    min_short_rmse = np.min(results['short']['rmse'])
    min_long_rmse = np.min(results['long']['rmse'])
    
    # 保存结果
    output = {
        'short_range': {
            'lpips_values': results['short']['lpips'],
            'average_lpips': avg_short_lpips,
            'min_lpips': min_short_lpips,

            'rmse_values': results['short']['rmse'],
            'average_rmse': avg_short_rmse,
            'min_rmse': min_short_rmse
        },
        'long_range': {
            'lpips_values': results['long']['lpips'],
            'average_lpips': avg_long_lpips,
            'min_lpips': min_long_lpips,
            
            'rmse_values': results['long']['rmse'],
            'average_rmse': avg_long_rmse,
            'min_rmse': min_long_rmse
        }
    }
    
    with open(os.path.join(model_path, 'multi-view metrics.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"average short-range LPIPS: {avg_short_lpips:.7f}")
    print(f"min short-range LPIPS: {min_short_lpips:.7f}")
    print(f"average long-range LPIPS: {avg_long_lpips:.7f}")
    print(f"min long-range LPIPS: {min_long_lpips:.7f}")
    print()
    print(f"average short-range RMSE: {avg_short_rmse:.7f}")
    print(f"min short-range RMSE: {min_short_rmse:.7f}")
    print(f"average long-range RMSE: {avg_long_rmse:.7f}")
    print(f"min long-range RMSE: {min_long_rmse:.7f}")

if __name__ == '__main__':
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_path', '-m', required=True, type=str)
    args = parser.parse_args()
    main(args.model_path)