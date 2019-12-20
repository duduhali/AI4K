import os
import random
from glob import glob
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn
from dataloderREDS import DatasetLoader
from models.loss import CharbonnierLoss
import models.archs.EDVR_arch as EDVR_arch
from tqdm import tqdm
from utils.my_utils import AverageMeter,psnr_cal
from collections import OrderedDict


def one_epoch_train_tqdm(model, optimizer, criterion, data_len, train_loader, epoch, epochs, batch_size, lr):
    model.train()
    losses = AverageMeter()
    psnrs = AverageMeter()
    with tqdm(total=(data_len -  data_len%batch_size)) as t:
        t.set_description('epoch:{}/{} lr={}'.format(epoch, epochs - 1, lr))

        for data in train_loader:
            data_x = data['LRs'].cuda()
            data_y = data['HR'].cuda()

            pred = model(data_x)
            # pix loss
            loss = criterion(pred, data_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = pred.cpu()
            pred = pred.detach().numpy().astype(np.float32)

            data_y = data_y.cpu()
            data_y = data_y.numpy().astype(np.float32)

            psnr = psnr_cal(pred, data_y)
            losses.update(loss.item())
            psnrs.update(psnr)

            t.set_postfix(loss='Loss: {losses.val:.3f} ({losses.avg:.3f})'
                               ' PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                          .format(losses=losses, psnrs=psnrs))

            t.update(batch_size)
    return losses, psnrs

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

    data_set = DatasetLoader(lr_list, hr_list, size_w=args.size_w, size_h=args.size_h, scale=args.scale,
                             n_frames=args.n_frames, interval_list=args.interval_list, border_mode=args.border_mode,
                             random_reverse=args.random_reverse)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                              pin_memory=False, drop_last=True)

    #### random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    #cudnn.deterministic = True

    device_ids = list(range(1))

    print("===> Building model")
    #### create model
    # network_G['predeblur'] = True  # ** 是否使用一个预编码层，它的作用是对输入 HxW 经过下采样得到 H/4xW/4 的feature，以便符合后面的网络
    # network_G['HR_in'] = True  # ** 很重要！！只要你的输入与输出是同样分辨率，就要求设置为true
    # network_G['w_TSA'] = True  # ** 是否使用TSA模块
    model = EDVR_arch.EDVR(nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10,
                           center=None, predeblur=True, HR_in=True, w_TSA=True)
    criterion = CharbonnierLoss()
    print("===> Setting GPU")
    model = DataParallel(model,device_ids=device_ids)
    model = model.cuda()
    criterion = criterion.cuda()

    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay,
                                 betas=(args.beta1, args.beta2))

    print(model)

    start_epoch = args.start_epoch
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isdir(args.resume):
            # 获取目录中最后一个
            pth_list = sorted(glob(os.path.join(args.resume, '*.pth')))
            if len(pth_list) > 0:
                args.resume = pth_list[-1]
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            start_epoch = checkpoint['epoch'] + 1
            state_dict = checkpoint['state_dict']

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            model.load_state_dict(new_state_dict)

            # 如果文件中有lr，则不用启动参数
            args.lr = checkpoint.get('lr', args.lr)

        if args.start_epoch != 0:
            # 如果设置了 start_epoch 则不用checkpoint中的epoch参数
            start_epoch = args.start_epoch


    #### training
    print("===> Training")
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)
        losses, psnrs = one_epoch_train_tqdm(model, optimizer, criterion, len(data_set), train_loader, epoch, args.epochs,
                                             args.batch_size, optimizer.param_groups[0]["lr"])

        # save model
        if epoch %9 != 0:
            continue

        model_out_path = os.path.join(args.checkpoint,
                                      "model_epoch_%04d_reds_loss_%.3f_psnr_%.3f.pth" % (epoch, losses.avg, psnrs.avg))
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
        torch.save({
            'state_dict': model.module.state_dict(),
            "epoch": epoch,
            'lr': optimizer.param_groups[0]["lr"]
        }, model_out_path)


def adjust_lr(opt, epoch):
    scale = 0.1
    if epoch in [200, 300, 350]:
    # if epoch in [40, 60, 70]:
        args.lr *= scale
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataloader
    parser.add_argument('--size_w', default=128, type=int)
    parser.add_argument('--size_h', default=128, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='/home/yons/data/train5/lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='/home/yons/data/train5/hr_small')
    parser.add_argument('--workers', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--scale', default=1, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1], type=int, nargs='+')
    parser.add_argument('--border_mode', default=True, type=bool)
    parser.add_argument('--random_reverse', default=True, type=bool)

    #train
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--start_epoch", default=0, type=int)

    #model
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)


    # check point
    parser.add_argument("--resume", default='checkpoint', type=str)
    parser.add_argument("--checkpoint", default='checkpoint', type=str)
    parser.add_argument('--print_freq', default=100, type=int)

    args = parser.parse_args()


    main(args)

