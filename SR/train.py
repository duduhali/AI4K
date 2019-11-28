import os,glob
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize
import torchnet.meter as meter
import copy
from PIL import Image

from data_utils import DatasetFromFolder,TrainDataset, EvalDataset
from check_utils import calc_psnr


def eval(model,val_loader):
    model.eval()
    epoch_psnr = meter.AverageValueMeter()

    for data in val_loader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)
        preds = preds.cpu()
        labels = labels.cpu()
        epoch_psnr.add(calc_psnr(preds, labels), len(labels))

    psnr_avg = epoch_psnr.value()[0]
    print('eval psnr: {:.4f}'.format(psnr_avg))
    return psnr_avg


def hand_png_file(args,input_trans,target_trans):
    input_files = glob.glob(os.path.join(args.input, "*"))
    target_files = glob.glob(os.path.join(args.target, "*"))

    if len(input_files) != len(target_files):
        raise Exception('两边的文件数量不相等', len(input_files), len(target_files))
    input_files = np.array(input_files)
    target_files = np.array(target_files)
    train_input = input_files[0:int(len(input_files) * args.train_val_ratio)]
    train_target = target_files[0:int(len(target_files) * args.train_val_ratio)]
    val_input = input_files[int(len(input_files) * args.train_val_ratio):]
    val_target = target_files[int(len(target_files) * args.train_val_ratio):]

    print(len(train_input), len(train_target), len(val_input), len(val_target))
    train_set = DatasetFromFolder(train_input, train_target,input_transform=input_trans,target_transform=target_trans)
    val_set = DatasetFromFolder(val_input, val_target,input_transform=input_trans,target_transform=target_trans)
    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batch_size,
                              drop_last=True, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=args.num_workers, batch_size=args.batch_size, drop_last=True,
                            shuffle=True)
    return train_set,val_set,train_loader,val_loader

def train_espcn(args,train_set, train_loader, val_loader):
    from model.ESPCN import ESPCN
    model = ESPCN(upscale_factor=args.scale)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    if os.path.isfile(args.model_pth):
        checkpoint = torch.load(args.model_pth)
        args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])

    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        epoch_losses = meter.AverageValueMeter()
        with tqdm(total=(len(train_set) - len(train_set) % args.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_loader:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.add(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_postfix(loss='{:.8f}'.format(epoch_losses.value()[0]))
                t.update(len(inputs))
        if epoch % 10 == 9:
            torch.save({'state_dict': model.state_dict(), "epoch": epoch},
                       "{}/epoch_{}.pth".format(args.outputs_dir, epoch))

        eval(model, val_loader)

def train_srcnn(args,train_set, train_loader, val_loader):
    from model.SRCNN import SRCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    if os.path.isfile(args.model_pth):
        checkpoint = torch.load(args.model_pth)
        args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        epoch_losses = meter.AverageValueMeter()
        with tqdm(total=(len(train_set) - len(train_set) % args.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))
            for data in train_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                loss = criterion(preds, labels)
                epoch_losses.add(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.8f}'.format(epoch_losses.value()[0]))
                t.update(len(inputs))

        if epoch % 10 == 9:
            torch.save({'state_dict': model.state_dict(), "epoch": epoch},
                       "{}/epoch_{}.pth".format(args.outputs_dir, epoch))

        psnr_avg = eval(model, val_loader)
        if psnr_avg > best_psnr:
            best_epoch = epoch
            best_psnr = psnr_avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.4f}'.format(best_epoch, best_psnr))
    torch.save({'state_dict': best_weights, "epoch": best_epoch},"{}/best.pth".format(args.outputs_dir))

def train_vdsr(args,train_set, train_loader, val_loader):
    from model.VDSR import VDSR
    pass


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    if args.model == 'ESPCN':
        input_trans = Compose([transforms.ToTensor()])
        target_trans = transforms.ToTensor()
        train_set, val_set, train_loader, val_loader = hand_png_file(args,input_trans,target_trans)

        train_espcn(args, train_set, train_loader, val_loader)
    elif args.model == 'SRCNN':
        # input_trans = Compose([Resize((216, 384), interpolation=Image.BICUBIC),transforms.ToTensor()])
        # target_trans = transforms.ToTensor()
        # train_set, val_set, train_loader, val_loader = hand_png_file(args,input_trans,target_trans)

        train_set = TrainDataset(args.train_file)
        val_set = EvalDataset(args.eval_file)
        train_loader = DataLoader(dataset=train_set,num_workers=args.num_workers, batch_size=args.batch_size,drop_last=True, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=1)

        train_srcnn(args, train_set, train_loader, val_loader)

    elif args.model == 'VDSR':
        train_set = TrainDataset(args.train_file)
        val_set = EvalDataset(args.eval_file)
        train_loader = DataLoader(dataset=train_set,num_workers=args.num_workers, batch_size=args.batch_size,drop_last=True, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=1)

        train_vdsr(args, train_set, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--train-val-ratio', type=float, default=1)

    parser.add_argument('--train-file', type=str, default='')
    parser.add_argument('--eval-file', type=str, default='')

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--outputs-dir', type=str, default='weights')
    parser.add_argument('--model-pth', type=str, default='')
    parser.add_argument('--start-epoch', type=int, default=0)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, args.model)
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    main(args)

    #python3 train.py --model ESPCN  --input E:/test/cut_pngs/X4 --target E:/test/cut_pngs/gt --batch-size 4 --train-val-ratio 0.8 --num-workers 2 --num-epochs 10
    #python3 train.py --model ESPCN  --input E:/test/cut_pngs/X4 --target E:/test/cut_pngs/gt --train-val-ratio 0.6 --num-epochs 20 --model-pth weights/ESPCN/epoch_9.pth

    #python3 train.py --model SRCNN  --input E:/test/cut_pngs/X4 --target E:/test/cut_pngs/gt --batch-size 2 --train-val-ratio 0.6 --num-workers 2 --num-epochs 5
    #python3 train.py --model SRCNN  --train-file E:/test/train.hdf5 --eval-file E:/test/eval.hdf5 --batch-size 2 --num-workers 2 --num-epochs 5

    #python3 train.py --model SRCNN  --train-file /home/uftp/pngs_cut20.hdf5 --eval-file /home/uftp/pngs_cut20_eval.hdf5 --batch-size 256 --num-workers 4 --num-epochs 400

