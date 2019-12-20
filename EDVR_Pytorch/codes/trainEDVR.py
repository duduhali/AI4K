import os
import random
from glob import glob
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.parallel import DataParallel
from dataloderREDS import DatasetLoader
from models.loss import CharbonnierLoss
import models.archs.EDVR_arch as EDVR_arch

def main(args):
    #### create dataloader
    print("===> Loading datasets")
    file_name = sorted(os.listdir(args.data_lr))
    lr_list = []
    hr_list = []
    for one in file_name:
        lr_tmp = sorted(glob(os.path.join(args.data_lr, one, '*.png')))
        lr_list.extend(lr_tmp)
        hr_tmp = sorted(glob(os.path.join(args.data_hr, one, '*.png')))
        if len(hr_tmp) != 100:
            print(one)
        hr_list.extend(hr_tmp)

    data_set = DatasetLoader(lr_list, hr_list, size_w=args.size_w, size_h=args.size_h,
                             scale=args.scale, n_frames=args.n_frames, interval_list=args.interval_list)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                              pin_memory=False, drop_last=True)


    #### random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')

    print("===> Building model")
    #### create model
    # network_G['predeblur'] = True  # ** 是否使用一个预编码层，它的作用是对输入 HxW 经过下采样得到 H/4xW/4 的feature，以便符合后面的网络
    # network_G['HR_in'] = True  # ** 很重要！！只要你的输入与输出是同样分辨率，就要求设置为true
    # network_G['w_TSA'] = True  # ** 是否使用TSA模块
    model = EDVR_arch.EDVR(nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10,
                           center=None, predeblur=False, HR_in=False, w_TSA=True)

    model = model.to(device)
    model = DataParallel(model)
    print(model)
    model.train()


    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)


    optimizer_G = torch.optim.Adam(optim_params, lr=4e-4, weight_decay=0, betas=(0.9, 0.99))
    criterion = CharbonnierLoss().to(device)
    #### resume training
    start_epoch = 0


    #### training
    for epoch in range(start_epoch, args.epochs):
        for i, data in enumerate(train_loader):
            #### training
            var_L = data['LQs'].to(device)
            real_H = data['GT'].to(device)


            fake_H = model(var_L)
            loss = criterion(fake_H, real_H)

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            print(epoch,i,loss.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataloader
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--size_w', default=128, type=int)
    parser.add_argument('--size_h', default=128, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='/home/yons/data/train5/lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='/home/yons/data/train5/hr_small')
    parser.add_argument('--workers', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1], type=int, nargs='+')

    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()


    main(args)

