import argparse


parser = argparse.ArgumentParser()
# model parameter
parser.add_argument('--scale', default=4, type=int)
parser.add_argument('--patch_size', default=64, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--step_batch_size', default=1, type=int)
parser.add_argument('--workers', default=32, type=int)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument("--n_res_blocks", type=int, default=20)
parser.add_argument("--n_feats", type=int, default=64)
parser.add_argument("--step", type=int, default=2)
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--res_scale', type=float, default=0.1,
                    help='residual scaling')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

# path
parser.add_argument('--data-lr', type=str, metavar='PATH',default='train_lr')
parser.add_argument('--data-hr', type=str, metavar='PATH',default='train_hr')
parser.add_argument('--logs-dir', type=str, default='logs')

# check point
parser.add_argument("--resume", default='checkpoint', type=str)
parser.add_argument("--checkpoint", default='checkpoint', type=str)
parser.add_argument('--print_freq', default=100, type=int)

args = parser.parse_args()

from model.rcan import RCAN
model = RCAN(args)

# from model.edsr import EDSR
# model = EDSR(args)
print(model)

