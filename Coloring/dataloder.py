import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


def horizontal_flip(image, axis):
    if axis != 2:
        image = cv2.flip(image, axis)
    return image

class DatasetLoader(Dataset):
    def __init__(self, data_list,patch_size):
        super(DatasetLoader, self).__init__()
        self.data_list = data_list
        self.patch_size = patch_size
    def __getitem__(self, index):
        data_file = self.data_list[index]
        img = cv2.imread(data_file)
        try:
            height, width, channel = img.shape
        except Exception as e:
            return self.__getitem__(random.randrange(0, self.__len__()))

        if height<self.patch_size or width<self.patch_size:
            random_sum = random.randrange(0, self.__len__())
            return self.__getitem__(random_sum)

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

        img_lab = np.array(img_lab, dtype=np.float32)
        img_rgb = np.array(img_rgb, dtype=np.float32)

        rnd_h = random.randint(0, max(0, height - self.patch_size))
        rnd_w = random.randint(0, max(0, width - self.patch_size))
        img_lab = img_lab[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        img_rgb = img_rgb[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

        # augmentation - flip, rotate
        axis1 = np.random.randint(low=-1, high=3)
        img_lab = horizontal_flip(img_lab, axis=axis1)
        img_rgb = horizontal_flip(img_rgb, axis=axis1)


        # HWC to CHW, numpy to tensor
        img_lab = np.transpose(img_lab, (2, 0, 1))
        img_rgb = np.transpose(img_rgb, (2, 0, 1))

        img_l = img_lab[0,:,:]
        img_l = img_l[np.newaxis,:]
        img_ab = img_lab[1:, :, :]
        img_ab = img_ab/128
        img_l = torch.from_numpy(np.ascontiguousarray(img_l))
        img_ab = torch.from_numpy(np.ascontiguousarray(img_ab))
        img_rgb = torch.from_numpy(np.ascontiguousarray(img_rgb))

        return img_l,img_ab,img_rgb
    def __len__(self):
        return len(self.data_list)

class EvalDatasetLoader(Dataset):
    def __init__(self, data_list):
        super(EvalDatasetLoader, self).__init__()
        self.data_list = data_list
    def __getitem__(self, index):
        data_file = self.data_list[index]
        img = cv2.imread(data_file)

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)

        img_lab = np.array(img_lab, dtype=np.float32)
        img_rgb = np.array(img_rgb, dtype=np.float32)

        # HWC to CHW, numpy to tensor
        img_lab = np.transpose(img_lab, (2, 0, 1))
        img_rgb = np.transpose(img_rgb, (2, 0, 1))

        img_l = img_lab[0,:,:]
        img_l = img_l[np.newaxis,:]
        img_ab = img_lab[1:, :, :]
        img_ab = img_ab/128
        img_l = torch.from_numpy(np.ascontiguousarray(img_l))
        img_ab = torch.from_numpy(np.ascontiguousarray(img_ab))
        img_rgb = torch.from_numpy(np.ascontiguousarray(img_rgb))

        return img_l,img_ab,img_rgb
    def __len__(self):
        return len(self.data_list)