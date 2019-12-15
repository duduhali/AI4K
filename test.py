import argparse
from RCAN_PyTorch.model.rcan import RCAN
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--img-path', type=str, default='')
parser.add_argument('--batch-size', type=int, default='1',help='Works when entering a directory')
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


model = RCAN(args)
# print(model)
x=torch.autograd.Variable(torch.rand(1,3,32,32)) #随便定义一个输入


#解决：RecursionError: maximum recursion depth exceeded while calling a Python object
# import sys
# sys.setrecursionlimit(1000000)
#
# y = model(x)
# g = make_dot(y)
# g.view()

from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='logs', comment='rcan')
with writer:
    writer.add_graph(model, (x,))

