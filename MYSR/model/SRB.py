import torch
import torchvision
import torch.nn as nn
from model import common
import torch.nn.functional as F


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        # print(x.shape, y.shape)  # torch.Size([1, 64, 128, 128]) torch.Size([1, 64, 1, 1])
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True)):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True))
                        for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

#https://blog.csdn.net/Wayne2019/article/details/79946799
class Block3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act, n_sub, frist_stride,layer=3):
        super(Block3D, self).__init__()
        the_padding = kernel_size//2
        self.head = nn.Conv3d(in_channels, out_channels, (1+n_sub,kernel_size,kernel_size), stride=(1,frist_stride,frist_stride), padding=(0,the_padding,the_padding), bias=True)
        body = []
        if layer<=1:
            raise Exception("Number of layers must be greater than 2")
        for _ in range(layer-1):
            body.append(act)
            body.append(nn.Conv3d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=True))
        self.body = nn.Sequential(*body)
    def forward(self, x):
        #B, C, N, H, W
        x = self.head(x)
        x = self.body(x)
        return x

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act, n_sub, frist_stride,layer=3):
        super(UpBlock3D, self).__init__()
        the_padding = kernel_size//2
        self.out_channels = out_channels
        self.n_sub = n_sub
        self.frist_stride = frist_stride
        self.head = nn.ConvTranspose3d(in_channels, out_channels, (1+n_sub,kernel_size,kernel_size), stride=(1,frist_stride,frist_stride), padding=(0,the_padding,the_padding), bias=True)
        body = []
        if layer<=1:
            raise Exception("Number of layers must be greater than 2")
        for _ in range(layer - 1):
            body.append(act)
            body.append(nn.Conv3d(out_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=True))
        self.body = nn.Sequential(*body)
    def forward(self, x):
        B, _, N, H, W = x.size()
        x = self.head(x,output_size=(B, self.out_channels, N+self.n_sub, H * self.frist_stride, W * self.frist_stride))
        x = self.body(x)
        return x

class Block3DGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_frames, act):
        super(Block3DGroup, self).__init__()
        self.n_frames = n_frames
        channels = 16
        self.a1 = Block3D(in_channels, channels, kernel_size, act, n_sub=0, frist_stride=1)
        self.b1 = UpBlock3D(channels, out_channels, kernel_size, act, n_sub=0, frist_stride=1)

        self.a2 = Block3D(channels, channels * 2, kernel_size, act, n_sub=1, frist_stride=2)
        self.b2 = UpBlock3D(channels * 2, channels, kernel_size, act, n_sub=1, frist_stride=2)

        channels = channels * 2
        self.a3 = Block3D(channels, channels * 2, kernel_size, act, n_sub=1, frist_stride=2)
        self.b3 = UpBlock3D(channels * 2, channels, kernel_size, act, n_sub=1, frist_stride=2)

        channels = channels * 2
        self.a4 = Block3D(channels, channels * 2, kernel_size, act, n_sub=1, frist_stride=2)
        self.b4 = UpBlock3D(channels * 2, channels, kernel_size, act, n_sub=1, frist_stride=2)

        channels = channels * 2
        self.a5 = Block3D(channels, channels * 2, kernel_size, act, n_sub=1, frist_stride=2)
        self.b5 = UpBlock3D(channels * 2, channels, kernel_size, act, n_sub=1, frist_stride=2)
        #A1 in*16,5,128,128
        #B1 out, 5,128,128

        #A2 in*32, 4, 64, 64
        #B2 in*16, 5, 128,128

        #A3 in*64, 3, 32, 32
        #B3 in*32, 4, 64,64

        #A4 in*128, 2, 16, 16
        #B4 in*64, 3, 64,64

        #A5 in*256, 1, 8, 8
        #B5 in*128, 2, 16,16
    def forward(self, x):
        a1 = self.a1(x)
        a2 = self.a2(a1)
        a3 = self.a3(a2)
        a4 = self.a4(a3)
        a5 = self.a5(a4)

        b5 = self.b5(a5)
        b4 = self.b4(b5 + a4)
        b3 = self.b3(b4 + a3)
        b2 = self.b2(b3 + a2)
        b1 = self.b1(b2 + a1)

        return b1

class ResidualConv3D(nn.Module):
    def __init__(self, n_groups_3d, n_colors, n_feats, kernel_size, n_frames, act):
        super(ResidualConv3D, self).__init__()
        self.head = Block3D(n_colors, 64, kernel_size, act, n_sub=0, frist_stride=1,layer=2)

        modules_body = [Block3DGroup(64, 64, kernel_size=kernel_size, n_frames=n_frames, act=act) for _ in range(n_groups_3d)]
        self.body = nn.Sequential(*modules_body)

        self.tail = Block3D(64, n_feats, kernel_size, act, n_sub=n_frames - 1, frist_stride=1, layer=2)
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

# Residual Channel Attention Network (RCAN)
class SRB(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRB, self).__init__()
        n_resgroups = args.n_resgroups  # 10
        n_res_blocks = args.n_res_blocks  # 20
        n_feats = args.n_feats  # 64
        kernel_size = 3
        reduction = args.reduction  # 16
        scale = args.scale
        n_colors = args.n_colors
        rgb_range = args.rgb_range  # rgb_range 255
        n_frames = args.n_frames
        self.center = n_frames//2
        n_groups_3d = args.n_groups_3d

        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]  # n_colors 3
        self.head = nn.Sequential(*modules_head)
        self.residual_conv3d = ResidualConv3D(n_groups_3d,n_colors, n_feats, kernel_size, n_frames, act)


        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_res_blocks) \
            for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, n_img):
        B, N, C, H, W = n_img.size()  # N video frames
        x_center = n_img[:, self.center, :, :, :].contiguous()  # 把tensor变成在内存中连续分布的形式。
        x_center = self.head(x_center)


        n_img = n_img.view(B*N, C, H, W)
        n_img = self.sub_mean(n_img)

        n_img = n_img.view(B, N, C, H, W)
        #B, N, C, H, W -> B, C, N, H, W
        n_img = n_img.permute(0, 2, 1, 3, 4).contiguous()#把tensor变成在内存中连续分布的形式。

        img = self.residual_conv3d(n_img)  #B, C, 1, H, W
        img = img.view(B, -1, H, W)

        x = img+x_center
        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


