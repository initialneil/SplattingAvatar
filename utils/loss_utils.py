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

import torch
from torch import nn
import torch.nn.functional as thf
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = thf.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = thf.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = thf.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = thf.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = thf.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# based on https://github.com/frozoul/4K-NeRF/blob/main/lib/utils.py#L144
class LPIPS(nn.Module):
    def __init__(self, eval=True):
        super().__init__()
        self.__LPIPS__ = {}
        self.eval = eval

    def init_lpips(self, net_name, device):
        assert net_name in ['alex', 'vgg']
        import lpips
        if self.eval:
            print(f'init_lpips: lpips_{net_name} [eval]')
            return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)
        else:
            print(f'init_lpips: lpips_{net_name}')
            return lpips.LPIPS(net=net_name, version='0.1').to(device)
    
    # require input: BCHW
    def forward(self, inputs, targets, device=None, net_name='alex'):
        if not device:
            device = inputs.device
            
        if net_name not in self.__LPIPS__:
            self.__LPIPS__[net_name] = self.init_lpips(net_name, device)

        if self.eval:
            return self.__LPIPS__[net_name](targets, inputs, normalize=True).item()
        else:
            return self.__LPIPS__[net_name](targets, inputs, normalize=True)

# dynamic 3dgs
def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()

def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()

def l2_loss_v2(x, y, reduce='mean'):
    if reduce == 'mean':
        return torch.sqrt(((x - y) ** 2).sum(-1) + 1e-20).mean()
    else:
        return torch.sqrt(((x - y) ** 2).sum(-1) + 1e-20)

def _quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def quat_mult(q1, q2):
    if len(q1.shape) == 2:
        return _quat_mult(q1, q2)
    
    input_shape = q1.shape
    q = _quat_mult(q1.reshape(-1, 4), q2.reshape(-1, 4))
    return q.view(*input_shape[:-1], 4)
