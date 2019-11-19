from torch import nn
import torch
import torch.optim as opt
import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        # 对YCrCb颜色空间中的Y通道进行重建
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

lr = 1e-4 #1e-4 = 0.0001
use_cuda = torch.cuda.is_available()
model = SRCNN()
criterion = nn.MSELoss()
# optimizer = opt.Adam(model.parameters(), lr = lr)
optimizer = opt.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr*0.1}
    ], lr=lr)
if use_cuda :
    model = model.cuda()
    criterion = criterion.cuda()


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))

#处理图片并写入hdf5文件，减小IO瓶颈的影响
def prepareData():
    png_path="E:/test/pngs"
    h5_file = h5py.File('pngs.hdf5', 'w')
    widthNum,heightNum = 2, 2  # 切成几块
    bigWidth, bigHeight = 3840, 2160
    # smallWidth, smallHeight = 960, 540
    oneWidth, oneHeight = int(bigWidth/widthNum), int(bigHeight/heightNum)

    lr_patches = []
    hr_patches = []
    for l_file,h_file in zip( glob.glob(os.path.join(png_path,'X4/*.png')), glob.glob(os.path.join(png_path,'gt/*.png')) ):
        lr = Image.open(l_file).convert('RGB')
        hr = Image.open(h_file).convert('RGB')
        lr = lr.resize((hr.width,hr.height), resample=Image.BICUBIC)

        lr = np.array(lr).astype(np.float32)
        hr = np.array(hr).astype(np.float32)
        lr = convert_rgb_to_y(lr)
        hr = convert_rgb_to_y(hr)

        for j in range(heightNum):
            for i in range(widthNum):
                #分割图片，没有变换
                one1 = lr[j*oneHeight:(j+1)*oneHeight, i*oneWidth:(i+1)*oneWidth]
                one2 = hr[j * oneHeight:(j + 1) * oneHeight, i * oneWidth:(i + 1) * oneWidth]
                lr_patches.append(one1)
                hr_patches.append(one2)
        # print(lr.shape) #(2160, 3840)
        # plt.imshow(lr)
        # plt.show()
        # break
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    print(lr_patches.shape)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()

# 从视频写入hdf5文件，100个视频文件时会内存不足崩溃
def prepareVideo():
    import cv2
    video_path = "E:/AI+4K/videos"
    h5_file = h5py.File('E:/AI+4K/pngs.hdf5', 'w')
    widthNum, heightNum = 2, 2  # 切成几块
    bigWidth, bigHeight = 3840, 2160
    oneWidth, oneHeight = int(bigWidth / widthNum), int(bigHeight / heightNum)

    lr_patches = []
    hr_patches = []
    for lr_file,hr_file in zip(glob.glob(video_path+'/X4/*'),glob.glob(video_path+'/gt/*')):
        print(lr_file,hr_file)
        lr_cap = cv2.VideoCapture(lr_file)
        hr_cap = cv2.VideoCapture(hr_file)
        i=0
        s = 7
        while True:
            flag, lr_frame = lr_cap.read()
            _, hr_frame = hr_cap.read()
            if i%s==0:
                lr = Image.fromarray(cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                hr = Image.fromarray(cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)).convert('RGB')
                lr = lr.resize((hr.width, hr.height), resample=Image.BICUBIC)

                lr = np.array(lr).astype(np.float32)
                hr = np.array(hr).astype(np.float32)
                lr = convert_rgb_to_y(lr)
                hr = convert_rgb_to_y(hr)

                for j in range(heightNum):
                    for i in range(widthNum):
                        # 分割图片，没有变换
                        lr_part = lr[j * oneHeight:(j + 1) * oneHeight, i * oneWidth:(i + 1) * oneWidth]
                        hr_part = hr[j * oneHeight:(j + 1) * oneHeight, i * oneWidth:(i + 1) * oneWidth]
                        lr_patches.append(lr_part)
                        hr_patches.append(hr_part)

            i += 1
            if flag==False:
                break
        lr_cap.release()
        hr_cap.release()

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    print(lr_patches.shape,hr_patches.shape)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()

class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # np.expand_dims是为了：(1080, 1920) ->(1, 1080, 1920)
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

def train():
    train_dataset = TrainDataset('pngs.hdf5')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)

    for data in train_dataloader:
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        print(inputs.shape, labels.shape)
        preds = model(inputs)
        loss = criterion(preds, labels)
        # epoch_losses.update(loss.item(), len(inputs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    # prepareData()
    # prepareVideo()
    train()



