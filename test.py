import argparse
from model.rcan import RCAN

parser = argparse.ArgumentParser()
parser.add_argument('--img-path', type=str, default='')
parser.add_argument('--batch-size', type=int, default='5',help='Works when entering a directory')
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--outputs-dir', default='outputs', type=str)
parser.add_argument("--resume", type=str, default='')
parser.add_argument('--gpus', type=int, default=2)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--n_resgroups', type=int, default=10,help='number of residual groups')
parser.add_argument("--n_res_blocks", type=int, default=20)
parser.add_argument("--n_feats", type=int, default=64)
parser.add_argument('--reduction', type=int, default=16,help='number of feature maps reduction')
parser.add_argument('--scale', default=4, type=int)
parser.add_argument('--rgb_range', type=int, default=255,help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,help='number of color channels to use')
parser.add_argument('--res_scale', type=float, default=0.1,help='residual scaling')
args = parser.parse_args()


# model = RCAN(args)
# print(model)


import torch
import torch.nn as nn
from torch.autograd import Variable

x1 = Variable(torch.linspace(0, 11,12).type(torch.FloatTensor)).view(3,4).unsqueeze_(0)
x2 = Variable(torch.linspace(0, 11,12).type(torch.FloatTensor)).view(3,4).unsqueeze_(0)
x3 = Variable(torch.linspace(0, 11,12).type(torch.FloatTensor)).view(3,4).unsqueeze_(0)
x = torch.cat((x1,x2,x3), 0).unsqueeze_(0)
print(x.size())
#torch.Size([1, 3, 3, 4])

print(x)
m = nn.Conv2d(3, 2, 1, padding=0, bias=True)
output = m(x)
print(output.shape)
print(output)
print(m.bias)
