from PIL import ImageFilter as IF
import torchvision.datasets as datasets
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


LR = 'E:/AI+4K/pngs/X4'  #(960, 540)
HR = 'E:/AI+4K/pngs/gt'  #(3840, 2160)  3840/960 = 2160/540 = 4
L_files = glob.glob(os.path.join(LR,"*"))
H_files = glob.glob(os.path.join(HR,"*"))
L_files = np.array(L_files)
H_files = np.array(H_files)
print(len(L_files))
print(len(H_files))


class SRCNN(nn.Module):
	def __init__(self):
		super(SRCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3);
		self.conv2 = nn.Conv2d(64, 32, kernel_size=9, padding=4);
		self.conv3 = nn.Conv2d(32, 32, kernel_size=9, padding=4);
		self.conv4 = nn.Conv2d(32, 3, kernel_size=7, padding=3);
	def forward(self, img):
		out = F.relu(self.conv1(img))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))
		out = self.conv4(out)
		return out


EPOCHS = 4
BATCH_SIZE = 32
sizeL = (960, 540)
sizeH = (3840, 2160)
learn_rate = 0.0001
use_cuda = torch.cuda.is_available()
# use_cuda = False

srcnn = SRCNN()
loss_func = nn.MSELoss()
optimizer = opt.Adam(srcnn.parameters(), lr = learn_rate)
if use_cuda :
	srcnn.cuda()
	loss_func = loss_func.cuda()

def train(i, imgLR, imgHR):
    optimizer.zero_grad()
    out_model = srcnn(imgLR)
    loss = loss_func(out_model, imgHR)
    loss.backward()
    optimizer.step()
    print(loss)

transform_data = transforms.Compose([  # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), #会把 0-255  压缩到  0-1
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])
def transform_file(input_batch_file,output_batch_file):
    imgLR = []
    imgHR = []
    # print((960/4, 540/4)) #(240.0, 135.0)
    # print((960 / 2, 540 / 2))  # (480.0, 270.0)
    the_size1 = (480,270)
    # print((3840 / 2, 2160 / 2))  # (1920.0, 1080.0)
    the_size2 = (1920, 1080)
    for input_file,output_file in zip(input_batch_file,output_batch_file):
        # print(input_file,output_file)
        img = Image.open(input_file)
        # img = img.resize(sizeH, Image.BICUBIC) #双三次插值
        img = img.resize(the_size1, Image.BICUBIC) # test
        imgLR.append(transform_data(img).numpy())

        img2 = Image.open(output_file)
        img2 = img2.resize(the_size2, Image.BICUBIC)  #  test
        imgHR.append(transform_data(img2).numpy())

    imgLR = torch.FloatTensor(imgLR)
    imgHR = torch.FloatTensor(imgHR)
    print(imgLR.shape) #torch.Size([4, 3, 2160, 3840])
    print(imgHR.shape)
    return imgLR,imgHR

record = []
for epoch in range(EPOCHS):
    index_list = list(range(len(L_files)))
    np.random.shuffle(index_list)
    the_L_files = L_files[index_list]  #打乱顺序后的数据
    the_H_files = H_files[index_list]

    loss_epoch = []
    for index in range(int(len(L_files) / BATCH_SIZE)):
        input_batch_file = the_L_files[index * BATCH_SIZE : (index + 1) * BATCH_SIZE] #不足一个BATCH_SIZE的取不到
        output_batch_file = the_H_files[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]

        imgLR, imgHR = transform_file(input_batch_file, output_batch_file)
        if use_cuda:
            imgLR = imgLR.cuda()
            imgHR = imgLR.cuda()
        train(index,imgLR, imgHR)

    #     loss_epoch.append(loss/BATCH_SIZE)
    #     print(loss_epoch)
    #     print(np.mean(loss_epoch))
    # the_lost = np.mean(loss_epoch)
    # print('epoch {0} loss {1}'.format(epoch, the_lost))
    # record.append(the_lost)

torch.save(srcnn,'srcnn.mdl')
# srcnn = torch.load('srcnn.mdl')

# 预测曲线
# plt.figure(figsize = (10, 7))
# plt.plot(record) #record记载了每一个打印周期记录的训练和校验数据集上的准确度
# plt.xlabel('Steps')
# plt.ylabel('loss')
# plt.show()

