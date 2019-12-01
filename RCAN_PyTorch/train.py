import argparse
import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model.rcan import RCAN
from dataloder import  DatasetLoader
from utils import  AverageMeter,psnr_cal


def main(arg):
    print("===> Loading datasets")
    lr_list = glob.glob(os.path.join(args.data_lr, '*'))
    hr_list = glob.glob(os.path.join(args.data_hr, '*'))
    data_set = DatasetLoader(lr_list, hr_list, arg.patch_size, arg.scale)
    train_loader = DataLoader(data_set, batch_size=arg.batch_size, num_workers=arg.workers, shuffle=True,
                              pin_memory=True, drop_last=True)


    print("===> Building model")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    device_ids = list(range(args.gpus))
    model = RCAN(arg)
    criterion = nn.L1Loss(reduction='sum')

    print("===> Setting GPU")
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if arg.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(arg.resume))
            checkpoint = torch.load(arg.resume)
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            model.load_state_dict(new_state_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=arg.lr, weight_decay=arg.weight_decay, betas=(0.9, 0.999), eps=1e-08)

    print("===> Training")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)
        model.train()
        losses = AverageMeter()
        psnrs = AverageMeter()
        with tqdm(total=(len(data_set) - len(data_set) % args.batch_size)) as t:
            t.set_description('epoch:{}/{} lr={}'.format(epoch, args.epochs - 1, optimizer.param_groups[0]["lr"]))

            for data in train_loader:
                data_x, data_y = Variable(data[0]), Variable(data[1], requires_grad=False)

                data_x = data_x.type(torch.FloatTensor)
                data_y = data_y.type(torch.FloatTensor)

                data_x = data_x.cuda()
                data_y = data_y.cuda()

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
                mean_loss = loss.item() / (args.batch_size * args.n_colors * ((args.patch_size * args.scale) ** 2))
                losses.update(mean_loss)
                psnrs.update(psnr)

                t.set_postfix(loss='Loss: {losses.val:.3f} ({losses.avg:.3f})'
                                   ' PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                              .format(losses=losses, psnrs=psnrs))

                t.update(len(data[0]))


        # save model
        model_out_path = os.path.join(args.checkpoint,"model_epoch_{}_rcan.pth".format(epoch))
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
        torch.save(model.module.state_dict(), model_out_path)


def adjust_lr(opt, epoch):
    scale = 0.1
    if epoch in [30, 45, 60]:
        args.lr *= 0.1
        print('Change lr to {}'.format(args.lr))
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] * scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--step_batch_size', default=1, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument('--epochs', type=int, default=400)
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
    parser.add_argument('--data-lr', type=str, metavar='PATH',default='/home/ubuntu/img_lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH',default='/home/ubuntu/img_hr')

    # check point
    parser.add_argument("--resume", default='', type=str)
    parser.add_argument("--checkpoint", default='checkpoint', type=str)

    args = parser.parse_args()
    main(args)

    #python3 train.py --data-lr img_lr --data-hr img_hr --batch_size 32 --workers 16 --gpus 2 --resume checkpoint/model_epoch_10_rcan.pth --start_epoch 11

    #python3 train.py --data-lr img_lr --data-hr img_hr --batch_size 80 --workers 16 --gpus 2  --resume checkpoint/model_epoch_12_rcan.pth --start_epoch 13

    #C:\Python37\python  train.py --data-lr J:/AI+4K/pngs/X4  --data-hr J:/AI+4K/pngs/gt  --batch_size 4 --workers 4
