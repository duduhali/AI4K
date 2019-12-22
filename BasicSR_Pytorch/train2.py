import os
import torch.nn as nn
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import create_dataloader, create_dataset
from models import create_model
from torch.nn.parallel import DataParallel
from utils.my_utils import AverageMeter,psnr_cal_0_1
from collections import OrderedDict
from tqdm import tqdm
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch

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

            psnr = psnr_cal_0_1(pred, data_y)
            # 3 : channel      255: data_y range [0,1]
            mean_loss = loss.item() * 255 / (args.batch_size * 3 * ((args.size_w * args.scale)*(args.size_h * args.scale)))
            losses.update(mean_loss)
            psnrs.update(psnr)

            t.set_postfix(loss='Loss: {losses.val:.3f} ({losses.avg:.3f})'
                               ' PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                          .format(losses=losses, psnrs=psnrs))

            t.update(batch_size)
    return losses, psnrs

def main():

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True
    #cudnn.deterministic = True

    device_ids = list(range(1))

    #### create dataloader
    data_set = create_dataset(dataset_opt)
    train_loader = create_dataloader(data_set, dataset_opt, opt, train_sampler)


    #### create model
    model = create_model(opt)
    netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                nf=opt_net['nf'], nb=opt_net['nb'])
    netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    netG.train()
    netD.train()

    criterion = nn.L1Loss()
    print("===> Setting GPU")
    model = DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    criterion = criterion.cuda()

    print(model)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=1e-8)
    start_epoch = 0
    #### training
    for epoch in range(start_epoch, args.epochs):
        #### update learning rate
        adjust_lr(optimizer, epoch)

        #### training
        losses, psnrs = one_epoch_train_tqdm(model, optimizer, criterion, len(data_set), train_loader, epoch,
                                             args.epochs,
                                             args.batch_size, optimizer.param_groups[0]["lr"])

        # if epoch %9 != 0:
        #     continue
        model_out_path = os.path.join(args.checkpoint, "model_epoch_%04d_loss_%.3f_psnr_%.3f.pth" %
                                      (epoch, losses.avg, psnrs.avg))
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
    parser.add_argument('--size_w', default=64, type=int)
    parser.add_argument('--size_h', default=64, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='/home/yons/data/train_lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='/home/yons/data/train_hr')
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1], type=int, nargs='+')  # 序列取值间隔
    parser.add_argument('--random_reverse', default=True, type=bool)  # 是否随机反转序列
    parser.add_argument('--border_mode', default=True, type=bool)
    parser.add_argument('--center', default=0, type=int)  # 序列中和目标帧对应的帧的位置
    # model
    parser.add_argument('--nf', default=64, type=int)
    parser.add_argument('--groups', default=8, type=int)
    parser.add_argument('--front_RBs', default=5, type=int)
    parser.add_argument('--back_RBs', default=40, type=int)
    parser.add_argument('--predeblur', default=True, type=bool)  # 是否使用滤波
    parser.add_argument('--HR_in', default=False, type=bool)  # 很重要！！输入与输出是同样分辨率，就要求设置为true
    parser.add_argument('--w_TSA', default=True, type=bool)  # 是否使用TSA模块

    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--use_current_lr', type=float, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)

    # train
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    # check point
    parser.add_argument("--resume", default='weights', type=str)
    parser.add_argument("--checkpoint", default='weights', type=str)
    parser.add_argument('--print_freq', default=100, type=int)

    args = parser.parse_args()
    main(args)
