import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import utils as util
import os
import os.path as osp
from glob import glob

class DatasetLoader(Dataset):
    def __init__(self, data_lr, data_hr, patch_size, scale=4):
        super(DatasetLoader, self).__init__()
        file_name = sorted(os.listdir(data_lr))
        lr_list = []
        hr_list = []
        for one in file_name:
            lr_tmp = sorted(glob(osp.join(data_lr, one, '*.png')))
            lr_list.extend(lr_tmp)
            hr_tmp = sorted(glob(osp.join(data_hr, one, '*.png')))
            if len(hr_tmp) != 100:
                print(one)
            hr_list.extend(hr_tmp)

        self.lr_list = lr_list
        self.hr_list = hr_list
        self.patch_size = patch_size
        self.scale = scale

    def __getitem__(self, index):
        try:
            lr_file = self.lr_list[index]
            hr_file = self.hr_list[index]

            # get the GT image (as the center frame)
            hr_data = cv2.imread(hr_file)
            hr_data = cv2.cvtColor(hr_data, cv2.COLOR_BGR2YCrCb)
            hr_data = np.array(hr_data)
            hr_data = hr_data.astype(np.float32)
            height, width, channel = hr_data.shape
            gt_size = self.patch_size * self.scale

            # get LR image
            lr_data = cv2.imread(lr_file)
            lr_data = cv2.cvtColor(lr_data, cv2.COLOR_BGR2YCrCb)
            lr_data = np.array(lr_data)
            lr_data = lr_data.astype(np.float32)

            # randomly crop
            lr_height = height // self.scale
            lr_width = width // self.scale

            rnd_h = random.randint(0, max(0, lr_height - self.patch_size))
            rnd_w = random.randint(0, max(0, lr_width - self.patch_size))
            img_lr = lr_data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            rnd_h_hr, rnd_w_hr = int(rnd_h * self.scale), int(rnd_w * self.scale)
            img_gt = hr_data[rnd_h_hr:rnd_h_hr + gt_size, rnd_w_hr:rnd_w_hr + gt_size, :]

            # augmentation - flip, rotate
            # axis1 = np.random.randint(low=-1, high=3)
            # img_lr = util.horizontal_flip(img_lr, axis=axis1)
            # img_gt = util.horizontal_flip(img_gt, axis=axis1)

            img_lr,img_gt = util.augment([img_lr,img_gt], hflip=True, rot=True)




            # HWC to CHW, numpy to tensor
            img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
            img_lr = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lr, (2, 0, 1)))).float()
            #可以这样认为，ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        except Exception as e:
            random_sum = random.randrange(0, self.__len__())
            # print(index,random_sum,'出现异常',e.__str__())
            return self.__getitem__(random_sum)
        return img_lr, img_gt

    def __len__(self):
        return len(self.lr_list)



class EvalDataset(Dataset):
    def __init__(self, test_lr,outputs_dir):
        super(EvalDataset, self).__init__()
        file_names = sorted(os.listdir(test_lr))
        lr_list = []
        for one in file_names:
            dst_dir = os.path.join(outputs_dir, one)
            if os.path.exists(dst_dir) and len(os.listdir(dst_dir)) == 100:
                continue
            lr_tmp = sorted(glob(os.path.join(test_lr, one, '*.png')))
            lr_list.extend(lr_tmp)

        self.test_lr = lr_list

    def __getitem__(self, idx):
        img_file = self.test_lr[idx]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        img = img * 1.0
        # HWC to CHW : (2, 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        return img,img_file

    def __len__(self):
        return len(self.test_lr)