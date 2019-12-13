#放大图像，然后替换Y通道
import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

l_file = 'E:/test/one_L.png'
h_file = 'E:/test/one_H.png'
# def calc_psnr(img1, img2):
#     return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
# lr = Image.open(l_file).convert('RGB')
# hr = Image.open(h_file).convert('RGB')
# lr = lr.resize((hr.width,hr.height), resample=Image.BICUBIC)
# lr = np.array(lr).astype(np.float32)
# hr = np.array(hr).astype(np.float32)
# print(calc_psnr(torch.tensor(lr),torch.tensor(hr))) #峰值信噪比



l_img = cv2.imread(l_file)
h_img = cv2.imread(h_file)
l_img = cv2.resize(l_img, (h_img.shape[1], h_img.shape[0]), interpolation=cv2.INTER_CUBIC)
print(l_img.shape)
# plt.imshow(l_img) #低
# plt.show()
# plt.imshow(h_img) #高
# plt.show()
l_img = cv2.cvtColor(l_img, cv2.COLOR_BGR2YCrCb)
l_y,l_cr,l_cb = l_img[:,:,0],l_img[:,:,1],l_img[:,:,2]
h_y,h_cr,h_cb = h_img[:,:,0],h_img[:,:,1],h_img[:,:,2]
test = np.array([h_y,l_cr,l_cb]).transpose([1, 2, 0])
test = cv2.cvtColor(test, cv2.COLOR_YCrCb2BGR)
print(test.shape)
plt.imshow(test)  # Y高 Cr低 Cb低
plt.show()

#效果并不好