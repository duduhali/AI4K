import argparse
import os
import cv2
from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from model.rcan import RCAN
from dataloder import EvalDataset

def eval_path(args):
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    device_ids = list(range(args.gpus))
    model = RCAN(args)
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            model.load_state_dict(new_state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model.eval()
    data_set = EvalDataset(args.img_path)
    eval_loader = DataLoader(data_set, batch_size=args.batch_size,num_workers=args.workers)

    with tqdm(total=(len(data_set) - len(data_set) % args.batch_size)) as t:
        for data in eval_loader:
            inputs,names = data
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = model(inputs).data.float().cpu().clamp_(0, 255).numpy()
            for img,file in zip(outputs,names):
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
                img = img.round()
                cv2.imwrite('{0}/{1}'.format(args.outputs_dir, file), img)
            t.update(len(names))



if __name__ == '__main__':
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

    eval_path(args)



    #python3 eval.py --img-path /home/ubuntu/img_540p --resume checkpoint/model_epoch_16_rcan.pth --batch-size 8 --workers 16 --outputs-dir outputs
