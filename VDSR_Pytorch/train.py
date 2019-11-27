import os,glob
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchnet.meter as meter
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize
from VDSR_Pytorch.model import VDSR
from VDSR_Pytorch.data_utils import DatasetFromFolder,calc_psnr

def adjust_learning_rate(epoch):
    return lr * (0.1 ** (epoch // step))


EPOCHS = 100
BATCH_SIZE = 8
num_workers = 0

momentum = 0.9
weight_decay = 1e-4
lr = 0.1
step = 8
start_epoch = 0
clip = 0.4
model_pth = ''
# model_pth = 'checkpoint/model_epoch_9.pth'
cuda = torch.cuda.is_available()
if __name__ == "__main__":
    # model = VDSR()
    # print(model.parameters)

    #准备数据
    LR = 'E:/pngs_cut20/X4'  # (960, 540)
    HR = 'E:/pngs_cut20/gt'  # (3840, 2160)  3840/960 = 2160/540 = 4
    L_files = glob.glob(os.path.join(LR, "*"))
    H_files = glob.glob(os.path.join(HR, "*"))
    # L_files = L_files[0:100]   #####################
    # H_files = H_files[0:100]   #####################

    L_files = np.array(L_files)
    H_files = np.array(H_files)
    train_val_scale = 0.9
    train_L = L_files[0:int(len(L_files) * train_val_scale)]
    train_H = H_files[0:int(len(H_files) * train_val_scale)]
    val_L = L_files[int(len(L_files) * train_val_scale):]
    val_H = H_files[int(len(H_files) * train_val_scale):]

    train_set = DatasetFromFolder(train_L, train_H, input_transform=Compose(
        [Resize((216, 384), interpolation=Image.BICUBIC), transforms.ToTensor()]),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder(val_L, val_H, input_transform=Compose(
        [Resize((216, 384), interpolation=Image.BICUBIC), transforms.ToTensor()]),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=BATCH_SIZE, drop_last=True,shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=BATCH_SIZE, drop_last=True,shuffle=False)

    #准备模型
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True

    model = VDSR()
    criterion = nn.MSELoss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

    if os.path.isfile(model_pth):
        checkpoint = torch.load(model_pth)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])

    #开始训练
    for epoch in range(start_epoch,EPOCHS):
        model.train()

        lr = adjust_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        epoch_losses = meter.AverageValueMeter()
        with tqdm(total=(len(train_set) - len(train_set) % BATCH_SIZE)) as t:
            t.set_description('epoch:{}/{}, lr={}'.format(epoch, EPOCHS - 1,optimizer.param_groups[0]["lr"]))
            for data in train_loader:
                inputs, labels = data
                if cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.add(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                t.set_postfix(loss='{:.8f}'.format(epoch_losses.value()[0]))
                t.update(len(inputs))

        model.eval()
        epoch_psnr = meter.AverageValueMeter()

        for data in val_loader:
            inputs, labels = data
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.add(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.value()[0]))

        # if (epoch+1) % 5 == 0:
        torch.save({'state_dict': model.state_dict(), "epoch": epoch},
                       "checkpoint/epoch_{}_psnr_{:.2f}.pth".format(epoch,epoch_psnr.value()[0]))

