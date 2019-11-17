from PIL import ImageFilter as IF
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch


transform_data = transforms.Compose(
    [transforms.ToTensor(),   # 0-255  -> 0-1
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #(（0,1）-0.5）/0.5=(-1,1)

upscale_factor = 4
def transform(img):
    #中心裁剪   # (750, 468)  ->  (748, 468)
	crop = transforms.CenterCrop( (int(img.size[1]/upscale_factor)*upscale_factor,
                                   int(img.size[0]/upscale_factor)*upscale_factor) )
	img = crop(img) #高分辨率图片

    #生成低分辨率图片
	out = img.filter(IF.GaussianBlur(1.3))#高斯模糊
	out = out.resize((int(out.size[0]/upscale_factor), int(out.size[1]/upscale_factor))) #缩小4倍
	out = out.resize((int(out.size[0]*upscale_factor), int(out.size[1]*upscale_factor))) #放大4倍
	return transform_data(out), transform_data(img)

trainSet = datasets.ImageFolder(train_path, transform=transform, target_transform=None)
testSet = datasets.ImageFolder(test_path, transform=transform, target_transform=None)

class SRCNN(nn.Module):
	def __init__(self):
		super(SRCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3);
		self.conv2 = nn.Conv2d(64, 32, kernel_size=9, padding=4);
		self.conv3 = nn.Conv2d(32, 32, kernel_size=9, padding=4);
		self.conv4 = nn.Conv2d(32, 3, kernel_size=7, padding=3);
		#self.relu  = nn.ReLU();
	def forward(self, img):
		out = F.relu(self.conv1(img))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))
		out = self.conv4(out)
		return out

srcnn = SRCNN()
loss_func = nn.MSELoss()

use_cuda = torch.cuda.is_available()
if use_cuda :
	srcnn.cuda()
	loss_func = loss_func.cuda()

learn_rate = 0.0001
optimizer = opt.Adam(srcnn.parameters(), lr = learn_rate)


def train(epoch, trainSet):
    epoch_loss = 0
    for itr, data in enumerate(trainSet):
        imgs, label = data
        imgLR, imgHR = imgs
        imgLR.unsqueeze_(0)
        imgHR.unsqueeze_(0)

        if use_cuda:
            imgLR = imgLR.cuda()
            imgHR = imgLR.cuda()

        optimizer.zero_grad()
        out_model = srcnn(imgLR)
        loss = loss_func(out_model, imgHR)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(trainSet)))

for epoch in range(20):
	train(epoch, trainSet)