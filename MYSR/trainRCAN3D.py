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

from model.RCAN3D import RCAN3D
from dataloderRCAN3D import  DatasetLoader
from utils import  AverageMeter,psnr_cal_0_255


def one_epoch_train_tqdm(model,optimizer,criterion,data_len,train_loader,epoch,epochs,batch_size,lr,n_frames):
    model.train()
    losses = AverageMeter()
    psnrs = AverageMeter()
    with tqdm(total=(data_len -  data_len%batch_size)) as t:
        t.set_description('epoch:{}/{} lr={}'.format(epoch, epochs - 1, lr))

        for data in train_loader:
            data_x = Variable(data['LRs'])
            data_y = Variable(data['HRs'], requires_grad=False)

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
            pred = pred.permute(0, 2, 1, 3, 4)
            b, n, c, w, h = pred.size()
            pred = pred.view(b * n, c, w, h)
            pred = pred.detach().numpy().astype(np.float32)

            data_y = data_y.cpu()
            data_y = data_y.permute(0,2, 1, 3, 4)
            b,n,c,w,h = data_y.size()
            data_y = data_y.view(b*n,c,w,h)
            data_y = data_y.numpy().astype(np.float32)

            psnr = psnr_cal_0_255(pred, data_y)
            mean_loss = loss.item() / (n_frames*args.batch_size * args.n_colors * ((args.size_w * args.scale)*(args.size_h * args.scale)))
            losses.update(mean_loss)
            psnrs.update(psnr)

            t.set_postfix(loss='Loss: {losses.val:.3f} ({losses.avg:.3f})'
                               ' PNSR: {psnrs.val:.3f} ({psnrs.avg:.3f})'
                          .format(losses=losses, psnrs=psnrs))

            t.update(batch_size)
    return losses, psnrs


def main(args):
    data_set = DatasetLoader(args.data_lr, args.data_hr, size_w=args.size_w, size_h=args.size_h, scale=args.scale,
                             frame_interval=args.frame_interval, border_mode=args.border_mode,
                             random_reverse=args.random_reverse)
    data_len = len(data_set)
    train_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                              pin_memory=False, drop_last=True)


    print("===> Building model")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    device_ids = list(range(args.gpus))
    model = RCAN3D(args)
    print(model)
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

        losses, psnrs = one_epoch_train_tqdm(model, optimizer, criterion, data_len, train_loader, epoch, args.epochs,
                                             args.batch_size, optimizer.param_groups[0]["lr"],len(args.frame_interval))


        # lr = optimizer.param_groups[0]["lr"]
        # the_lr = 1e-2
        # lr_len = 2
        # while lr + (1e-9) < the_lr:
        #     the_lr *= 0.1
        #     lr_len += 1
        # record.append([losses.avg,psnrs.avg,lr_len])

        with open(args.log, 'a') as f:
            f.write("epoch: %d/%d    loss:%.3f    psnr:%.3f\n"%(epoch, args.epochs, losses.avg, psnrs.avg))
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
    #dataset
    parser.add_argument('--size_w', default=64, type=int)
    parser.add_argument('--size_h', default=64, type=int)
    parser.add_argument('--data-lr', type=str, metavar='PATH', default='J:/2file/train_lr')
    parser.add_argument('--data-hr', type=str, metavar='PATH', default='J:/2file/train_hr')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--frame_interval', default=[0, 1, 3, 5, 7], type=int, nargs='+')
    parser.add_argument('--border_mode', default=True, type=bool)
    parser.add_argument('--random_reverse', default=True, type=bool)

    # model parameter
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument("--start_epoch", default=0, type=int)

    args = parser.parse_args()

    args.epochs = 50
    args.batch_size = 1
    args.workers = 4
    args.resume = ''
    args.checkpoint = 'checkpoint'
    args.log = 'log/RCAN3D.txt'

    args.n_colors = 3
    args.scale = 4
    args.n_resgroups = 10
    args.n_res_blocks =20
    args.n_feats = 64
    args.reduction = 16
    args.rgb_range = 255

    main(args)

    # nohup python3 train.py>> output.log 2>&1 &
    # ps -aux|grep train.py
    # pgrep python3 | xargs kill -s 9



    #python3 train.py



    # nvidia-smi
