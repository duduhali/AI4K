import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import torch.backends.cudnn as cudnn
from tqdm import tqdm

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=9, padding=4)
        # self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=9, padding=4)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=7, padding=3)

    def forward(self, img):
        out = F.relu(self.conv1(img))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        return out

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

transform_data = transforms.Compose([  # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), #会把 0-255  压缩到  0-1
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # [0,1] -> [-1,1]
                                    ])
def transform_file(input_batch_file,output_batch_file):
    imgLR = []
    imgHR = []
    for input_file,output_file in zip(input_batch_file,output_batch_file):
        img1 = Image.open(input_file).convert('RGB')
        img2 = Image.open(output_file).convert('RGB')
        img1 = img1.resize((img2.width, img2.height), Image.BICUBIC)
        # 旋转
        # angle = random.randrange(0, 90)   #角度
        # print('angle',angle)
        # img1 = img1.rotate(angle)
        # img2 = img2.rotate(angle)
        # 剪切
        # oneWidth, oneHeight = int(sizeH[0]/widthNum), int(sizeH[1]/heightNum)
        # startWidth, startHeight = random.randrange(0, sizeH[0]-oneWidth), random.randrange(0, sizeH[1]-oneHeight)
        # # print('startWidth, startHeight',startWidth, startHeight)
        # x1,y1,x2,y2 = startWidth, startHeight, startWidth+oneWidth, startHeight+oneHeight
        # img1 = img1.crop((x1,y1,x2,y2))
        # img2 = img2.crop((x1,y1,x2,y2))

        # img1 = img1.crop((oneWidth, oneHeight, 2*oneWidth, 2*oneHeight))
        # img2 = img2.crop((oneWidth, oneHeight, 2*oneWidth, 2*oneHeight))

        # 水平反转
        if random.choice([0, 1]) == 1:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        # 垂直反转
        if random.choice([0, 0, 1]) == 1:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

        imgLR.append(transform_data(img1).numpy())
        imgHR.append(transform_data(img2).numpy())

    imgLR = torch.FloatTensor(imgLR)
    imgHR = torch.FloatTensor(imgHR)
    # print(imgLR.shape) #torch.Size([4, 3, 2160, 3840])
    # print(imgHR.shape)
    return imgLR,imgHR


EPOCHS = 100
BATCH_SIZE = 32
widthNum, heightNum = 8, 8  # 剪切，随机取的一块为原图像的几分之一
sizeL = (960, 540)
sizeH = (3840, 2160)
learn_rate = 0.0001
if __name__ == '__main__':
    # LR = 'J:/AI+4K/pngs/X4'  #(960, 540)
    # HR = 'J:/AI+4K/pngs/gt'  #(3840, 2160)  3840/960 = 2160/540 = 4
    LR = 'J:/AI+4K/pngs_cut20/X4'  # (960, 540)
    HR = 'J:/AI+4K/pngs_cut20/gt'  # (3840, 2160)  3840/960 = 2160/540 = 4
    L_files = glob.glob(os.path.join(LR, "*"))
    H_files = glob.glob(os.path.join(HR, "*"))
    # L_files = L_files[0:160]
    # H_files = H_files[0:160]

    L_files = np.array(L_files)
    H_files = np.array(H_files)

    train_val_scale = 0.8
    train_L = L_files[0:int(len(L_files) * train_val_scale)]
    train_H = H_files[0:int(len(H_files) * train_val_scale)]
    val_L = L_files[int(len(L_files) * train_val_scale):]
    val_H = H_files[int(len(H_files) * train_val_scale):]

    print('train', len(train_L), len(train_H))
    print('val', len(val_L), len(val_H))

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    loss_func = nn.MSELoss()
    optimizer = opt.Adam(model.parameters(), lr=learn_rate)

    train_loss = []
    val_psnr = []
    for epoch in range(EPOCHS):
        index_list = list(range(len(train_L)))
        np.random.shuffle(index_list)
        the_train_L = train_L[index_list]  # 打乱顺序后的数据
        the_train_H = train_H[index_list]

        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(the_train_L) - len(the_train_L) % BATCH_SIZE)) as t:
            t.set_description('epoch:{}/{}'.format(epoch, EPOCHS - 1))
            for index in range(int(len(train_L) / BATCH_SIZE)):
                inputs = the_train_L[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]  # 不足一个BATCH_SIZE的取不到
                lables = the_train_H[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]

                inputs, lables = transform_file(inputs, lables)

                inputs = inputs.to(device)
                lables = lables.to(device)

                out = model(inputs)
                loss = loss_func(out, lables)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
        train_loss.append(epoch_losses.avg)
        if epoch % 10 == 9:
            torch.save({'state_dict': model.state_dict()}, 'SRCNN_weights/epoch_{}_.pth'.format(epoch))

        # 验证
        model.eval()
        epoch_psnr = AverageMeter()
        for index in range(int(len(val_L) / BATCH_SIZE)):
            inputs = the_train_L[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]  # 不足一个BATCH_SIZE的取不到
            lables = the_train_H[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]

            inputs, lables = transform_file(inputs, lables)

            inputs = inputs.to(device)
            lables = lables.to(device)
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(preds, lables), len(inputs))
        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        val_psnr.append(epoch_psnr.avg)

    # 预测曲线
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, label='loss')
    # plt.plot(X,y1,color='r',linestyle='--',linewidth=2,alpha=0.5,label='sin 函数')
    # plt.plot(X,y2,color='y',linestyle='-',linewidth=2,label='cos 函数')
    plt.xlabel('Steps')
    plt.ylabel('loss')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(val_psnr, label='psnr')
    plt.xlabel('Steps')
    plt.ylabel('psnr')
    plt.show()

