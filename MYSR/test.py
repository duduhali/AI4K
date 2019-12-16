import torch
import argparse
from model.SR import SR


parser = argparse.ArgumentParser()


args = parser.parse_args()

args.n_resgroups = 1
args.n_res_blocks = 2

args.scale = 4
args.n_feats = 64
args.reduction = 16
args.rgb_range = 255
args.n_colors = 3
args.res_scale = 0.1 #模型中未使用


model = SR(args)
print(model)








