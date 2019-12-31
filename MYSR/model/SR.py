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

#define tail module
class Tail(nn.Module):
    def __init__(self,conv ,n_feats, n_colors, kernel_size, patch_size,bias=True):
        super(Tail, self).__init__()
        m = []
        for _ in range(2):
            m.append(conv(n_feats, 4 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(2))
            m.append(nn.ReLU(True))
        m.append(conv(n_feats, n_colors, kernel_size))
        self.body = nn.Sequential(*m)
    def forward(self, x):
        x = self.body(x)
        return x

class Tail2(nn.Module):
    def __init__(self,conv, n_feats, n_colors, kernel_size,patch_size):
        super(Tail2, self).__init__()
        self.convt1 = nn.ConvTranspose2d(n_feats, n_feats, 3, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(n_feats, n_feats, 3, stride=2, padding=1)
        self.conv = conv(n_feats, n_colors, kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.shape1 = (-1, n_feats, patch_size*2, patch_size*2)
        self.shape2 = (-1, n_feats, patch_size*4, patch_size*4)
    def forward(self, x):
        x = self.convt1(x, output_size=self.shape1)
        x = self.relu(x)
        x = self.convt2(x, output_size=self.shape2)
        x = self.relu(x)
        x = self.conv(x)
        return x

class MeanShift(nn.Conv2d):
    def __init__(self, d_weight = (1., 1., 1.),d_bias=(0,0,0)):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.FloatTensor([[d_weight[0], 0, 0], [0, d_weight[1], 0], [0, 0, d_weight[2]]]).view(3, 3, 1, 1)
        self.bias.data = torch.Tensor([d_bias[0],d_bias[1],d_bias[2]])
        for p in self.parameters():
            p.requires_grad = False

# Residual Channel Attention Network (RCAN)
class SR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SR, self).__init__()
        patch_size = args.patch_size
        n_resgroups = args.n_resgroups  # 10
        n_res_blocks = args.n_res_blocks  # 20
        n_feats = args.n_feats  # 64
        kernel_size = 3
        reduction = args.reduction  # 16
        scale = args.scale
        act = nn.ReLU(True)

        # [0,255] -> [-1,1]
        self.sub_mean = MeanShift(d_weight=(2/255.,2/255.,2/255.), d_bias=(-1,-1,-1))

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]  # n_colors 3

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_res_blocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        #[-1,1] -> [0,255]
        self.add_mean = nn.Sequential(
            MeanShift(d_bias=(1,1,1)),
            MeanShift(d_weight=(255/2.,255/2., 255/2.))
        )

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = Tail(conv, n_feats, args.n_colors, kernel_size,patch_size)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


