import os
import random
from glob import glob
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn
from dataloder import DatasetLoader
from models.loss import CharbonnierLoss
import models.archs.EDVR_arch as EDVR_arch
from tqdm import tqdm
from my_utils import AverageMeter,psnr_cal_0_1
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

def one_epoch_train_logger(model,optimizer,criterion,data_len,train_loader,epoch,epochs,batch_size,lr):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    psnrs = AverageMeter()
    end = time.time()
    for iteration, data in enumerate(train_loader):
        data_time.update(time.time() - end)

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
        mean_loss = loss.item() * 255 / (
                args.batch_size * 3 * ((args.size_w * args.scale) * (args.size_h * args.scale)))
        losses.update(mean_loss)
        psnrs.update(psnr)

        batch_time.update(time.time() - end)

        end = time.time()
        use_time = batch_time.sum
        time_h = use_time//3600
        time_m = (use_time-time_h*3600)//60
        show_time =  '%d:%d:%d'%(time_h,time_m,use_time%60)

        if iteration % args.print_freq == 0:
            print('Epoch:[{0}/{1}][{2}/{3}]  lr={4}\t {5} \t'
                  'data_time: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'batch_time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss: {losses.val:.3f} ({losses.avg:.3f})\t'
                  'PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                  .format(epoch, epochs, iteration, data_len // batch_size, lr, show_time,
                          data_time=data_time,batch_time=batch_time,  losses=losses, psnrs=psnrs))

    return losses,psnrs

def main(args):
    print("===> Loading datasets")
    data_set = DatasetLoader(args.data_lr, args.data_hr, size_w=args.size_w, size_h=args.size_h, scale=args.scale,
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

    print("===> Building model")
    #### create model
    model = EDVR_arch.EDVR(nf=args.nf, nframes=args.n_frames, groups=args.groups, front_RBs=args.front_RBs,
                           back_RBs=args.back_RBs,
                           center=args.center, predeblur=args.predeblur, HR_in=args.HR_in, w_TSA=args.w_TSA)
    criterion = CharbonnierLoss()
    print("===> Setting GPU")
    gups = args.gpus if args.gpus!=0 else torch.cuda.device_count()
    device_ids = list(range(gups))
    model = DataParallel(model,device_ids=device_ids)
    model = model.cuda()
    criterion = criterion.cuda()

    # print(model)

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


        # 如果设置了 start_epoch 则不用checkpoint中的epoch参数
        start_epoch = args.start_epoch if args.start_epoch != 0 else start_epoch

    #如果use_current_lr大于0 测代替作为lr
    args.lr = args.use_current_lr if args.use_current_lr > 0 else args.lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2),eps=1e-8)

    #### training
    print("===> Training")
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)
        if args.use_tqdm == 1:
            losses, psnrs = one_epoch_train_tqdm(model, optimizer, criterion, len(data_set), train_loader, epoch, args.epochs,
                                             args.batch_size, optimizer.param_groups[0]["lr"])
        else:
            losses, psnrs = one_epoch_train_logger(model, optimizer, criterion, len(data_set), train_loader, epoch,
                                                 args.epochs,
                                                 args.batch_size, optimizer.param_groups[0]["lr"])

        # save model
        # if epoch %9 != 0:
        #     continue

        model_out_path = os.path.join(args.checkpoint,"model_epoch_%04d_edvr_loss_%.3f_psnr_%.3f.pth" %
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
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument('--size_w', default=64, type=int)
    parser.add_argument('--size_h', default=64, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='./../train_lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='./../train_hr')
    parser.add_argument('--workers', default=32, type=int)
    parser.add_argument('--batch_size', default=56, type=int)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--n_frames', default=5, type=int)
    parser.add_argument('--interval_list', default=[1], type=int, nargs='+')

    parser.add_argument('--random_reverse', default=True, type=bool)

    parser.add_argument('--border_mode', default=True, type=bool)
    parser.add_argument('--center', default=0, type=int)

    # model
    parser.add_argument('--nf', default=64, type=int)
    parser.add_argument('--groups', default=8, type=int)
    parser.add_argument('--front_RBs', default=5, type=int)
    parser.add_argument('--back_RBs', default=10, type=int)
    parser.add_argument('--predeblur', default=False, type=bool)  # 是否使用滤波
    parser.add_argument('--HR_in', default=False, type=bool)  # 很重要！！输入与输出是同样分辨率，就要求设置为true
    parser.add_argument('--w_TSA', default=True, type=bool)  # 是否使用TSA模块


    #train
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument("--start_epoch", default=0, type=int)

    #model
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_current_lr', type=float, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)


    # check point
    parser.add_argument("--resume", default='checkpoint', type=str)
    parser.add_argument("--checkpoint", default='checkpoint', type=str)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--use_tqdm', default=0, type=int)
    args = parser.parse_args()


    main(args)

    # nohup python3 trainEDVR.py>> output.log 2>&1 &
    # ps -aux|grep trainEDVR.py
    # pgrep python3 | xargs kill -s 9
    # python3 trainEDVR.py
    # python3 trainEDVR.py --use_tqdm 1 --batch_size 256

    # nvidia-smi