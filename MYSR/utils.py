import cv2
import numpy as np
import math


def horizontal_flip(image, axis):
    if axis != 2:
        image = cv2.flip(image, axis)
    return image

def getFileName(lr_file):
    # lr_file = '/aaa/bbb/1050345\\050.png'
    lr_file = lr_file.replace('\\', '/')

    file_name = lr_file.split('/')[-1]
    name = file_name.split('.')[0]

    x = int(name)
    b = x - 1 if x != 1 else x + 1
    f = x + 1 if x != 100 else x - 1

    b_file = lr_file.replace(file_name, '%03d.png' % b)
    f_file = lr_file.replace(file_name, '%03d.png' % f)

    return  b_file,f_file

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


# def psnr_cal(pred, gt):
#     batch = pred.shape[0]
#     psnr = 0
#     for i in range(batch):
#         for j in range(3):
#             pr = pred[i, j, :, :]
#             hd = gt[i, j, :, :]
#
#             imdff = pr - hd
#             rmse = math.sqrt(np.mean(imdff ** 2))
#             if rmse == 0:
#                 psnr = psnr + 45
#                 continue
#             psnr = psnr + 20 * math.log10(255.0 / rmse)
#     return psnr / (batch*3)

def psnr_cal(pred, gt):
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

