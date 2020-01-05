import torch
import argparse
from model.EDVR import EDVR

parser = argparse.ArgumentParser()
# model
parser.add_argument('--n_frames', default=5, type=int)
parser.add_argument('--nf', default=64, type=int)
parser.add_argument('--groups', default=8, type=int)
parser.add_argument('--front_RBs', default=5, type=int)
parser.add_argument('--back_RBs', default=10, type=int)
parser.add_argument('--predeblur', default=False, type=bool)  # 是否使用滤波
parser.add_argument('--w_TSA', default=True, type=bool)  # 是否使用TSA模块

args = parser.parse_args()


model = EDVR(nf=args.nf, nframes=args.n_frames, groups=args.groups, front_RBs=args.front_RBs,
                           back_RBs=args.back_RBs, center=args.center, predeblur=args.predeblur, w_TSA=args.w_TSA)
print(model)








