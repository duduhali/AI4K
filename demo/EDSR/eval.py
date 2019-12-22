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
from glob import glob
from model.edsr import EDSR
from dataloder import EvalDataset

def eval_path(args):
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    device_ids = list(range(args.gpus))
    model = EDSR(args)
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    if args.resume:
        if os.path.isdir(args.resume):
            #获取目录中最后一个
            pth_list = sorted( glob(os.path.join(args.resume, '*.pth')) )
            if len(pth_list)>0:
                args.resume = pth_list[-1]
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            model.load_state_dict(new_state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model.eval()

    file_names = sorted(os.listdir(args.test_lr))
    lr_list = []
    for one in file_names:
        dst_dir = os.path.join(args.outputs_dir,one)
        if os.path.exists(dst_dir) and len(os.listdir(dst_dir)) == 100:
            continue
        lr_tmp = sorted(glob(os.path.join(args.test_lr, one, '*.png')))
        lr_list.extend(lr_tmp)

    data_set = EvalDataset(lr_list)
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

                arr = file.split('/')
                dst_dir = os.path.join(args.outputs_dir,arr[-2])
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_name = os.path.join(dst_dir,arr[-1])

                cv2.imwrite(dst_name, img)
            t.update(len(names))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_lr', type=str, default='test_lr')
    parser.add_argument('--batch-size', type=int, default='8',help='Works when entering a directory')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--outputs-dir', default='output_img', type=str)

    parser.add_argument("--resume", default='checkpoint', type=str)

    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)

    parser.add_argument('--n_resblocks', type=int, default=20,
                        help='number of residual blocks')
    parser.add_argument('--n_resgroups', type=int, default=10,help='number of residual groups')
    parser.add_argument("--n_feats", type=int, default=64)
    parser.add_argument('--reduction', type=int, default=16,help='number of feature maps reduction')
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--rgb_range', type=int, default=255,help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,help='number of color channels to use')
    parser.add_argument('--res_scale', type=float, default=0.1,help='residual scaling')
    args = parser.parse_args()

    eval_path(args)



    #python3 eval.py


    #python3 eval.py  --test_lr ../test_lr  --outputs-dir ../output_img
    #python3 eval.py --test_lr ../test_lr  --outputs-dir ../output_img  --batch-size 4 --workers 4