import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm
import glob,os

from ESPCN_Pytorch.data_utils import DatasetFromFolder
from ESPCN_Pytorch.model import Net
from ESPCN_Pytorch.psnrmeter import PSNRMeter
from ESPCN_Pytorch.utils import AverageMeter, calc_psnr


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='Train Super Resolution')
    # parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    # parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
    # opt = parser.parse_args()
    # UPSCALE_FACTOR = opt.upscale_factor
    # NUM_EPOCHS = opt.num_epochs
    UPSCALE_FACTOR = 4
    EPOCHS = 100
    BATCH_SIZE = 256
    num_workers = 4
    LR = 'J:/AI+4K/pngs_cut20/X4'  # (960, 540)
    HR = 'J:/AI+4K/pngs_cut20/gt'  # (3840, 2160)  3840/960 = 2160/540 = 4
    L_files = glob.glob(os.path.join(LR, "*"))
    H_files = glob.glob(os.path.join(HR, "*"))
    # L_files = L_files[0:1024]
    # H_files = H_files[0:1024]

    L_files = np.array(L_files)
    H_files = np.array(H_files)

    train_val_scale = 0.8
    train_L = L_files[0:int(len(L_files) * train_val_scale)]
    train_H = H_files[0:int(len(H_files) * train_val_scale)]
    val_L = L_files[int(len(L_files) * train_val_scale):]
    val_H = H_files[int(len(H_files) * train_val_scale):]

    train_set = DatasetFromFolder(train_L,train_H, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder(val_L,val_H,input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=BATCH_SIZE, shuffle=False)

    model = Net(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_set) - len(train_set) % BATCH_SIZE)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, EPOCHS - 1))

            for data in train_loader:
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        if epoch % 10 == 9:
            torch.save(model.state_dict(), 'epochs/epoch_%d_.pth' % epoch)

        model.eval()
        epoch_psnr = AverageMeter()

        for data in val_loader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))