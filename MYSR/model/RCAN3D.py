from model import common
import torch
import torchvision
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(in_channels, out_channels, (1,kernel_size,kernel_size),padding=(0,kernel_size//2,kernel_size//2), bias=bias)

# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, n_frames,channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d((n_frames,1,1))
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_frames, n_feat, kernel_size, reduction, bias=True):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0: modules_body.append(nn.ReLU(True))
        modules_body.append(CALayer(n_frames,n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_frames, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [ RCAB(conv, n_frames, n_feat, kernel_size, reduction, bias=True,) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Channel Attention Network (RCAN)
class RCAN3D(nn.Module):
    def __init__(self, args):
        super(RCAN3D, self).__init__()
        n_frames = args.n_frames
        n_resgroups = args.n_resgroups      #10
        n_res_blocks = args.n_res_blocks    #20
        n_feats = args.n_feats              #64
        kernel_size = 3
        reduction = args.reduction          #16
        scale = args.scale
        act = nn.ReLU(True)
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std) #rgb_range 255
        
        # define head module
        modules_head = [default_conv(args.n_colors, n_feats, kernel_size)] #n_colors 3

        # define body module
        modules_body = [
            ResidualGroup(
                default_conv, n_frames, n_feats, kernel_size, reduction, n_resblocks=n_res_blocks) \
            for _ in range(n_resgroups)]

        modules_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(default_conv, scale, n_feats),
            default_conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

