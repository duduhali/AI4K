#修改y通道的值，使其和目标的均值相同
import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

l_file = 'J:/a1_big.png'
h_file = 'J:/a2.png'
img1 = Image.open(l_file)
img2 = Image.open(h_file)
img1_y, img1_cb, img1_cr = img1.convert('RGB').split()
img2_y, img2_cb, img2_cr = img2.convert('RGB').split()

img1_y, img1_cb, img1_cr = np.asarray(img1_y),np.asarray(img1_cb),np.asarray(img1_cr)
img2_y, img2_cb, img2_cr = np.asarray(img2_y),np.asarray(img2_cb),np.asarray(img2_cr)
print(img1_y.shape,img2_y.shape)
mean1_y,mean1_cb,mean1_cr = np.mean(img1_y),np.mean(img1_cb),np.mean(img1_cr)
mean2_y,mean2_cb,mean2_cr = np.mean(img2_y),np.mean(img2_cb),np.mean(img2_cr)
print(mean1_y,mean2_y)
print(mean1_cb,mean2_cb)
print(mean1_cr,mean2_cr)


x,y = mean1_y,mean2_y
img1_y = img1_y*(y/x)
print(np.mean(img1_y))

x,y = mean1_cb,mean2_cb
img1_cb = img1_cb*(y/x)
print(np.mean(img1_cb))

x,y = mean1_cr,mean2_cr
img1_cr = img1_cr*(y/x)
print(np.mean(img1_cr))


img1_y = Image.fromarray(np.uint8(img1_y), mode='L')
img1_cb = Image.fromarray(np.uint8(img1_cb), mode='L')
img1_cr = Image.fromarray(np.uint8(img1_cr), mode='L')
out_img = Image.merge('RGB', [img1_y, img1_cb, img1_cr])
out_img.save("J:/a1_mul_rgb.png")




