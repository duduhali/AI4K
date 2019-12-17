import torch.nn as nn
import torch
import math

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bias=True):
        m = []
        for _ in range(int(math.log(scale, 2))):
            m.append(conv(n_feats, 4 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(2))
            m.append(nn.ReLU(True))
        super(Upsampler, self).__init__(*m)

