import torch
import torch.nn as nn

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
        channels = in_channels*16
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

m = Block3D(3,16,3,nn.ReLU(True),n_sub=4,frist_stride=1,layer=2)

up_m = UpBlock3D(64,3,3,nn.ReLU(True),n_sub=0,frist_stride=1)

g = Block3DGroup(3, 3, kernel_size=3, n_frames=5, act=nn.ReLU(True))
print(g)
input = torch.randn(2, 3, 5, 32, 32)
output = g(input)
print(output.shape)





