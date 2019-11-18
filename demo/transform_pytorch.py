#读取数据并做裁剪和变换

import torchvision.transforms as transforms
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random

transform_data = transforms.Compose([  # transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), #会把 0-255  压缩到  0-1
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

def transform_file(input_batch_file,output_batch_file):
    widthNum, heightNum = 2, 2  # 切成几块
    imgLR = []
    imgHR = []
    for input_file,output_file in zip(input_batch_file,output_batch_file):
        img1 = Image.open(input_file).resize((3840, 2160), Image.BICUBIC)
        img2 = Image.open(output_file)
        # 旋转
        angle = random.randrange(0, 90)   #角度
        print('angle',angle)
        img1 = img1.rotate(angle)
        img2 = img2.rotate(angle)
        # 剪切
        oneWidth, oneHeight = int(3840/widthNum), int(2160/heightNum)
        startWidth, startHeight = random.randrange(0, 3840-oneWidth), random.randrange(0, 2160-oneHeight)
        print('startWidth, startHeight',startWidth, startHeight)
        x1,y1,x2,y2 = startWidth, startHeight, startWidth+oneWidth, startHeight+oneHeight
        img1 = img1.crop((x1,y1,x2,y2))
        img2 = img2.crop((x1,y1,x2,y2))

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

if __name__ == '__main__':
    input_batch_file = ['E:/test/one_L.png']
    output_batch_file = ['E:/test/one_H.png']
    imgLR, imgHR = transform_file(input_batch_file, output_batch_file)

    lr = imgLR[0]
    hr = imgHR[0]

    lr_img = lr.data.numpy().transpose((1, 2, 0))
    print(lr_img.shape)
    plt.imshow(lr_img)
    plt.show()

    hr_img = hr.data.numpy().transpose((1, 2, 0))
    print(hr_img.shape)
    plt.imshow(hr_img)
    plt.show()



