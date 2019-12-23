import cv2
import numpy as np
import math
import random

def horizontal_flip(image, axis):
    if axis != 2:
        image = cv2.flip(image, axis)
    return image

def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def read_img(path):
    """read image by cv2
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def psnr_cal_0_255(pred, gt):
    batch = pred.shape[0]
    psnr = 0
    for i in range(batch):
        pr = pred[i]
        hd = gt[i]
        mse = np.mean((pr / 1. - hd / 1.) ** 2)
        if mse < 1.0e-10:
            psnr = psnr + 45
            continue
        psnr = psnr + 10 * np.log10(255 * 255 / mse)
    return psnr / (batch)


def psnr_cal_0_1(pred, gt):
    batch = pred.shape[0]
    psnr = 0
    for i in range(batch):
        pr = pred[i]
        hd = gt[i]
        mse = np.mean((pr / 1. - hd / 1.) ** 2)
        psnr = psnr + 10* np.log10(1. / mse)
    return psnr / (batch)
