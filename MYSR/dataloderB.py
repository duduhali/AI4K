import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import utils as util
import os.path as osp
from glob import glob
import os


def read_img(file):
    img = cv2.imread(file)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def getNeighbor(center_frame_idx, n_frames, interval):
    min_frame_id = 1
    max_frame_id = 100
    half = n_frames // 2
    neighbor_list = list(range(center_frame_idx - half * interval, center_frame_idx +half * interval + 1, interval))
    if neighbor_list[0]<min_frame_id:
        for i in range(half):
            neighbor_list[i] = neighbor_list[-1 - i]
    if neighbor_list[-1] > max_frame_id:
        for i in range(half):
            neighbor_list[half + 1 + i] = neighbor_list[half - 1 - i]


    return neighbor_list

class DatasetLoader(Dataset):
    def __init__(self, data_lr, data_hr, patch_size, scale,n_frames,interval_list,random_reverse):
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
        self.n_frames = n_frames
        self.interval_list = interval_list
        self.random_reverse = random_reverse

    def __getitem__(self, index):
        try:
            lr_file = self.lr_list[index]
            hr_file = self.hr_list[index]
            interval = random.choice(self.interval_list)
            lr_path,lr_name = util.get_file_name(lr_file)
            hr_path, _ = util.get_file_name(hr_file)
            center_frame_idx = int(lr_name)
            neighbor_list = getNeighbor(center_frame_idx,self.n_frames,interval)
            if self.random_reverse and random.random() < 0.5:
                neighbor_list.reverse()
            file_text = '{:03d}.png'.format(center_frame_idx)

            #### get lr images
            lr_data_list = []
            for v in neighbor_list:
                lr_data_path = osp.join(lr_path, '{:03d}.png'.format(v))
                lr_data_list.append(read_img(lr_data_path))

            #### get hr images
            hr_data_path = osp.join(hr_path, file_text)
            hr_data = read_img(hr_data_path)


            # randomly crop
            height, width, channel = hr_data.shape
            hr_size_w,hr_size_h = self.patch_size * self.scale, self.patch_size * self.scale
            lr_height = height // self.scale
            lr_width = width // self.scale

            rnd_h = random.randint(0, max(0, lr_height - self.patch_size))
            rnd_w = random.randint(0, max(0, lr_width - self.patch_size))
            img_lr_list = [one_data[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :] for one_data in lr_data_list]

            rnd_h_hr, rnd_w_hr = int(rnd_h * self.scale), int(rnd_w * self.scale)
            img_hr = hr_data[rnd_h_hr:rnd_h_hr + hr_size_h, rnd_w_hr:rnd_w_hr + hr_size_w, :]


            # augmentation - flip, rotate
            img_lr_list.append(img_hr)
            rlt = util.augment(img_lr_list, hflip=True, rot=True)
            img_lr_list = rlt[0:-1]
            img_hr = rlt[-1]

            # stack lr images to NHWC, N is the frame number
            img_lrs = np.stack(img_lr_list, axis=0)

            # HWC to CHW, numpy to tensor
            img_hr = torch.from_numpy(np.ascontiguousarray(np.transpose(img_hr, (2, 0, 1)))).float()
            # img_lrs = [torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float() for img in img_lr_list]
            img_lrs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lrs, (0, 3, 1, 2)))).float()
        except Exception as e:
            random_sum = random.randrange(0, self.__len__())
            return self.__getitem__(random_sum)

        return {'LRs': img_lrs, 'HR': img_hr}

    def __len__(self):
        return len(self.lr_list)



class EvalDataset(Dataset):
    def __init__(self, test_lr):
        super(EvalDataset, self).__init__()
        self.test_lr = test_lr

    def __getitem__(self, idx):
        img_file = self.test_lr[idx]
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        img = img * 1.0
        # BGR -> RGB : [2, 1, 0]     HWC to CHW : (2, 0, 1)
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        return img,img_file

    def __len__(self):
        return len(self.test_lr)