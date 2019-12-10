import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import horizontal_flip
import os

class DatasetLoader(Dataset):
    def __init__(self, lr_list, hr_list, patch_size, scale):
        super(DatasetLoader, self).__init__()
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
            hr_data = cv2.cvtColor(hr_data, cv2.COLOR_BGR2RGB)
            hr_data = np.array(hr_data)
            hr_data = hr_data.astype(np.float32)
            height, width, channel = hr_data.shape
            gt_size = self.patch_size * self.scale

            # get LR image
            lr_data = cv2.imread(lr_file)
            lr_data = cv2.cvtColor(lr_data, cv2.COLOR_BGR2RGB)
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
            axis1 = np.random.randint(low=-1, high=3)
            img_lr = horizontal_flip(img_lr, axis=axis1)
            img_gt = horizontal_flip(img_gt, axis=axis1)

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

class DatasetLoaderWithHR(Dataset):
    def __init__(self, lr_list, fake_list, hr_list, patch_size, scale):
        super(DatasetLoaderWithHR, self).__init__()
        self.lr_list = lr_list
        self.fake_list = fake_list
        self.hr_list = hr_list
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return len(self.lr_list)

    def __getitem__(self, index):
        while True:
            lr_file = self.lr_list[index]
            hr_file = self.hr_list[index]
            fake_file = self.fake_list[index]

            try:
                # get the GT image (as the center frame)
                hr_data = cv2.imread(hr_file)
                hr_data = cv2.cvtColor(hr_data, cv2.COLOR_BGR2RGB)
                hr_data = np.array(hr_data)
                hr_data = hr_data.astype(np.float32)
                height, width, channel = hr_data.shape
                gt_size = self.patch_size * self.scale

                # get the Fake GT image
                fake_data = cv2.imread(fake_file)
                fake_data = cv2.cvtColor(fake_data, cv2.COLOR_BGR2RGB)
                fake_data = np.array(fake_data)
                fake_data = fake_data.astype(np.float32)

                # get LR image
                lr_data = cv2.imread(lr_file)
                lr_data = cv2.cvtColor(lr_data, cv2.COLOR_BGR2RGB)
                lr_data = np.array(lr_data)
                lr_data = lr_data.astype(np.float32)
            except Exception:
                print(fake_file)
                index += 1
                continue

            # randomly crop
            lr_height = height // self.scale
            lr_width = width // self.scale

            rnd_h = random.randint(0, max(0, lr_height - self.patch_size))
            rnd_w = random.randint(0, max(0, lr_width - self.patch_size))
            img_lr = lr_data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            rnd_h_hr, rnd_w_hr = int(rnd_h * self.scale), int(rnd_w * self.scale)
            img_gt = hr_data[rnd_h_hr:rnd_h_hr + gt_size, rnd_w_hr:rnd_w_hr + gt_size, :]
            img_fake = fake_data[rnd_h_hr:rnd_h_hr + gt_size, rnd_w_hr:rnd_w_hr + gt_size, :]

            # augmentation - flip, rotate
            axis1 = np.random.randint(low=-1, high=3)
            img_lr = horizontal_flip(img_lr, axis=axis1)
            img_gt = horizontal_flip(img_gt, axis=axis1)
            img_fake = horizontal_flip(img_fake, axis=axis1)

            # HWC to CHW, numpy to tensor
            img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
            img_lr = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lr, (2, 0, 1)))).float()
            img_fake = torch.from_numpy(np.ascontiguousarray(np.transpose(img_fake, (2, 0, 1)))).float()
            return img_lr, img_fake, img_gt


class EvalDataset(Dataset):
    def __init__(self, img_path):
        super(EvalDataset, self).__init__()
        self.files_list = os.listdir(img_path)
        self.img_path = img_path

    def __getitem__(self, idx):
        img_file = self.files_list[idx]
        img = cv2.imread(os.path.join(self.img_path,img_file), cv2.IMREAD_COLOR)
        img = img * 1.0
        # BGR -> RGB : [2, 1, 0]     HWC to CHW : (2, 0, 1)
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        return img,img_file

    def __len__(self):
        return len(self.files_list)