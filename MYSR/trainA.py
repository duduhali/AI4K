import argparse
import os
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model.SRA import SRA
from dataloderA import  DatasetLoader
from utils import  AverageMeter,psnr_cal_0_255
import sys
import time


def one_epoch_train_tqdm(model,optimizer,criterion,data_len,train_loader,epoch,epochs,batch_size,lr):
    model.train()
    losses = AverageMeter()
    psnrs = AverageMeter()
    with tqdm(total=(data_len -  data_len%batch_size)) as t:
        t.set_description('epoch:{}/{} lr={}'.format(epoch, epochs - 1, lr))

        for data in train_loader:
            data_x_b = Variable(data[0])
            data_x = Variable(data[1])
            data_x_f = Variable(data[2])
            data_y = Variable(data[3], requires_grad=False)

            data_x_b = data_x_b.type(torch.FloatTensor)
            data_x = data_x.type(torch.FloatTensor)
            data_x_f = data_x_f.type(torch.FloatTensor)
            data_y = data_y.type(torch.FloatTensor)

            data_x_b = data_x_b.cuda()
            data_x = data_x.cuda()
            data_x_f = data_x_f.cuda()
            data_y = data_y.cuda()

            pred = model(data_x_b,data_x,data_x_f)
            # pix loss
            loss = criterion(pred, data_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = pred.cpu()
            pred = pred.detach().numpy().astype(np.float32)

            data_y = data_y.cpu()
            data_y = data_y.numpy().astype(np.float32)

            psnr = psnr_cal_0_255(pred, data_y)
            mean_loss = loss.item() / (args.batch_size * args.n_colors * ((args.patch_size * args.scale) ** 2))
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
        psnr = psnr_cal_0_255(pred, data_y)
        mean_loss = loss.item() / (args.batch_size * args.n_colors * ((args.patch_size * args.scale) ** 2))

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
    file_name = sorted(os.listdir(args.data_lr))
    lr_list = []
    hr_list = []
    for one in file_name:
        lr_tmp = sorted( glob(os.path.join(args.data_lr, one,'*.png')) )
        lr_list.extend(lr_tmp)
        hr_tmp = sorted( glob(os.path.join(args.data_hr, one,'*.png')) )
        if len(hr_tmp) != 100:
            print(one)
        hr_list.extend(hr_tmp)


    # lr_list = glob(os.path.join(args.data_lr, '*'))
    # hr_list = glob(os.path.join(args.data_hr, '*'))
    lr_list = lr_list[0:max_index]
    hr_list = hr_list[0:max_index]


    data_set = DatasetLoader(lr_list, hr_list, args.patch_size)
    data_len = len(data_set)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                              pin_memory=True, drop_last=True)


    print("===> Building model")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    device_ids = list(range(args.gpus))
    model = SRA()
    criterion = nn.L1Loss(reduction='sum')

    print("===> Setting GPU")
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    criterion = criterion.cuda()

    start_epoch = args.start_epoch
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isdir(args.resume):
            #获取目录中最后一个
            pth_list = sorted( glob(os.path.join(args.resume, '*.pth')) )
            if len(pth_list)>0:
                args.resume = pth_list[-1]
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            start_epoch = checkpoint['epoch']+1
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                namekey = 'module.' + k  # remove `module.`
                new_state_dict[namekey] = v
            model.load_state_dict(new_state_dict)

            #如果文件中有lr，则不用启动参数
            args.lr = checkpoint.get('lr', args.lr)

    if args.start_epoch != 0:
        #如果设置了 start_epoch 则不用checkpoint中的epoch参数
        start_epoch = args.start_epoch


    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-08)

    # record = []
    print("===> Training")
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)

        losses, psnrs = one_epoch_train_tqdm(model, optimizer, criterion, data_len, train_loader, epoch, args.epochs, args.batch_size, optimizer.param_groups[0]["lr"])


        # lr = optimizer.param_groups[0]["lr"]
        # the_lr = 1e-2
        # lr_len = 2
        # while lr + (1e-9) < the_lr:
        #     the_lr *= 0.1
        #     lr_len += 1
        # record.append([losses.avg,psnrs.avg,lr_len])


        # save model
        if epoch+1 != args.epochs:
            continue

        model_out_path = os.path.join(args.checkpoint,"model_epoch_%04d_sra_loss_%.3f_psnr_%.3f.pth"%(epoch,losses.avg,psnrs.avg) )
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
        torch.save({
            'state_dict': model.module.state_dict(),
            "epoch": epoch,
            'lr':optimizer.param_groups[0]["lr"]
        }, model_out_path)


    # import matplotlib.pyplot as plt
    # # 绘制模型的训练误差曲线
    # plt.figure(figsize=(10, 7))
    # plt.plot([i[0] for i in record], label='loss')
    # plt.plot([i[1] for i in record], label='psnr')
    # plt.plot([i[2] for i in record], label='lr')
    # # plt.xlabel('Batchs')
    # # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

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
    # model parameter
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--step_batch_size', default=1, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument('--epochs', type=int, default=400)


    # path
    parser.add_argument('--data-lr', type=str, metavar='PATH',default='J:/train_lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH',default='J:/train_hr')
    # parser.add_argument('--data-lr', type=str, metavar='PATH', default='../train_lr')
    # parser.add_argument('--data-hr', type=str, metavar='PATH', default='../train_hr')

    parser.add_argument('--logs-dir', type=str, default='logs')

    # check point
    parser.add_argument("--resume", default='checkpoint', type=str)
    parser.add_argument("--checkpoint", default='checkpoint', type=str)
    parser.add_argument('--print_freq', default=100, type=int)

    args = parser.parse_args()
    args.n_colors = 3
    args.scale = 4



    args.epochs = 80
    args.batch_size = 4
    args.workers = 4
    args.resume = ''
    max_index = 200


    main(args)

    # nohup python3 train.py>> output.log 2>&1 &
    # ps -aux|grep train.py
    # pgrep python3 | xargs kill -s 9



    #python3 train.py



    # nvidia-smi
