import torch
import torchvision
import torch.nn as nn
from model import common

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class SRA(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(SRA, self).__init__()
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 4
        n_colors = 3
        rgb_range = 255

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(rgb_range)  # rgb_range 255
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        frame_b1_head = [conv(n_colors, n_feats, kernel_size)]
        frame_main_head = [conv(n_colors, n_feats, kernel_size)]
        frame_f1_head = [conv(n_colors, n_feats, kernel_size)]
        self.b1_head = nn.Sequential(*frame_b1_head)
        self.main_head = nn.Sequential(*frame_main_head)
        self.f1_head = nn.Sequential(*frame_f1_head)

        #M
        frame_b1_m = []
        frame_f1_m = []
        for _ in range(2):
            frame_b1_m.append(conv(n_feats, n_feats, kernel_size))
            frame_f1_m.append(conv(n_feats, n_feats, kernel_size))

            frame_b1_m.append(nn.ReLU(inplace=True))
            frame_f1_m.append(nn.ReLU(inplace=True))
        self.b1_m = nn.Sequential(*frame_b1_m)
        self.f1_m = nn.Sequential(*frame_f1_m)

        #S
        frame_b1_s = []
        frame_f1_s = []
        for _ in range(2):
            frame_b1_s.append(RCAB(conv, n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True)))
            frame_f1_s.append(RCAB(conv, n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True)))
        self.b1_s = nn.Sequential(*frame_b1_s)
        self.f1_s = nn.Sequential(*frame_f1_s)

        # T
        frame_b1_t = []
        frame_f1_t = []
        for _ in range(2):
            frame_b1_t.append(RCAB(conv, n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True)))
            frame_f1_t.append(RCAB(conv, n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True)))
        self.b1_t = nn.Sequential(*frame_b1_t)
        self.f1_t = nn.Sequential(*frame_f1_t)


        modules_x = []
        for _ in range(10):
            modules_x.append(RCAB(conv, n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True)))
            modules_x.append(conv(n_feats, n_feats, kernel_size))
        self.main_x = nn.Sequential(*modules_x)



        modules_middle = []
        for _ in range(20):
            modules_middle.append(RCAB(conv, n_feats, kernel_size, reduction, bias=True, act=nn.ReLU(True)))
            modules_middle.append(conv(n_feats, n_feats, kernel_size))
        self.middle = nn.Sequential(*modules_middle)

        modules_tail = [
            common.Upsampler(conv, scale, n_feats),
            conv(n_feats, n_colors, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, b,x,f):
        x_b1, x_main, x_f1 = self.sub_mean(b), self.sub_mean(x), self.sub_mean(f)

        x_b1 = self.b1_head(x_b1)
        x_main = self.main_head(x_main)
        x_f1 = self.f1_head(x_f1)

        #M
        m_b1 = x_b1+x_main
        m_f1 = x_f1 + x_main
        m_b1 = self.b1_m(m_b1) + x_b1
        m_f1 = self.f1_m(m_f1) + x_f1

        #S
        s_b1 = m_b1 + x_main
        s_f1 = m_f1 + x_main
        s_b1 = self.b1_s(s_b1) + m_b1
        s_f1 = self.f1_s(s_f1) + m_f1

        #T
        t_b1 = self.b1_t(s_b1) + s_b1
        t_f1 = self.f1_t(s_f1) + s_f1


        x_main = self.main_x(x_main) + x_main

        res = x_main+t_b1+t_f1
        res = self.middle(res)
        res += x_main

        x = self.tail(res)
        x = self.add_mean(x)

        return x