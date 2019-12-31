#放大图像，然后替换Y通道
import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

# l_file = 'J:/c1.png'
# h_file = 'J:/c2.png'
# img1 = Image.open(l_file)
# img2 = Image.open(h_file)
# img1 = img1.resize(img2.size, Image.BICUBIC)
# img1.save("J:/c1_big.png")
# img1_y, img1_cb, img1_cr = img1.convert('YCbCr').split()
# img2_y, img2_cb, img2_cr = img2.convert('YCbCr').split()
# out_img = Image.merge('YCbCr', [img2_y, img1_cb, img1_cr]).convert('RGB')
# out_img.save("J:/c3.png")


def psnr_np_0_255(y_true,y_pred):
    mse = np.mean((y_true / 1. - y_pred / 1.) ** 2)
    if mse < 1.0e-10:
        return 45
    return 10 * np.log10(255.0 * 255.0 / mse)
#
print('替换y通道',psnr_np_0_255(np.asarray(Image.open('J:/a3.png').convert('YCbCr')),np.asarray(Image.open('J:/a2.png').convert('YCbCr'))))
print('只放大',psnr_np_0_255(np.asarray(Image.open('J:/a1_big.png').convert('YCbCr')),np.asarray(Image.open('J:/a2.png').convert('YCbCr'))))
print('均值y通道',psnr_np_0_255(np.asarray(Image.open('J:/a1_big_mul.png').convert('YCbCr')),np.asarray(Image.open('J:/a2.png').convert('YCbCr'))))
print('eval',psnr_np_0_255(np.asarray(Image.open('J:/eval.png').convert('YCbCr')),np.asarray(Image.open('J:/a2.png').convert('YCbCr'))))

#
# print('替换y通道',psnr_np_0_255(np.asarray(Image.open('J:/b3.png').convert('YCbCr')),np.asarray(Image.open('J:/b2.png').convert('YCbCr'))))
# print('只放大',psnr_np_0_255(np.asarray(Image.open('J:/b1_big.png').convert('YCbCr')),np.asarray(Image.open('J:/b2.png').convert('YCbCr'))))
# print('均值y通道',psnr_np_0_255(np.asarray(Image.open('J:/b1_big_mul.png').convert('YCbCr')),np.asarray(Image.open('J:/b2.png').convert('YCbCr'))))

# print('替换y通道',psnr_np_0_255(np.asarray(Image.open('J:/c3.png').convert('YCbCr')),np.asarray(Image.open('J:/c2.png').convert('YCbCr'))))
# print('只放大',psnr_np_0_255(np.asarray(Image.open('J:/c1_big.png').convert('YCbCr')),np.asarray(Image.open('J:/c2.png').convert('YCbCr'))))
# print('均值y通道',psnr_np_0_255(np.asarray(Image.open('J:/c1_big_mul.png').convert('YCbCr')),np.asarray(Image.open('J:/c2.png').convert('YCbCr'))))

# def SSIMnp(y_true , y_pred):
#     u_true,u_pred = np.mean(y_true),np.mean(y_pred)
#     var_true,var_pred = np.var(y_true),np.var(y_pred)
#     std_true,std_pred = np.sqrt(var_true),np.sqrt(var_pred)
#     c1,c2  = np.square(0.01*7), np.square(0.03*7)
#     ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
#     denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
#     return ssim / denom
# print(SSIMnp(img1,img2))