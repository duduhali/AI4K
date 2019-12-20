import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils import my_utils as util
import os.path as osp

class DatasetLoader(Dataset):
    def __init__(self, lr_list, hr_list, size_w, size_h, scale,n_frames,interval_list,border_mode,random_reverse=False):
        super(DatasetLoader, self).__init__()
        self.lr_list = lr_list
        self.hr_list = hr_list
        self.size_w = size_w
        self.size_h = size_h
        self.scale = scale
        self.interval_list = interval_list
        self.n_frames = n_frames
        self.half_N_frames = n_frames // 2
        self.border_mode = border_mode
        self.random_reverse = random_reverse

    def _get_file_name(self,file):
        # file = '/aaa/bbb/1050345\\050.png'  return /aaa/bbb/1050345 050
        file = file.replace('\\', '/')
        d_list = file.rsplit('/', maxsplit=1)
        path = d_list[0]
        file_text = d_list[-1]
        name = file_text.split('.')[0]

        return path, name

    def __getitem__(self, index):
        try:
            lr_file = self.lr_list[index]
            hr_file = self.hr_list[index]
            interval = random.choice(self.interval_list)
            lr_path,lr_name = self._get_file_name(lr_file)
            hr_path, _ = self._get_file_name(hr_file)
            center_frame_idx = int(lr_name)

            if self.border_mode:
                direction = 1  # 1: forward; 0: backward
                if self.random_reverse and random.random() < 0.5:
                    direction = random.choice([0, 1])
                if center_frame_idx + interval * (self.n_frames - 1) > 99:
                    direction = 0
                elif center_frame_idx - interval * (self.n_frames - 1) < 0:
                    direction = 1
                # get the neighbor list
                if direction == 1:
                    neighbor_list = list(
                        range(center_frame_idx, center_frame_idx + interval * self.n_frames, interval))
                else:
                    neighbor_list = list(
                        range(center_frame_idx, center_frame_idx - interval * self.n_frames, -interval))
                file_text = '{:03d}.png'.format(neighbor_list[0])
            else:
                while (center_frame_idx + self.half_N_frames * interval > 99) or (center_frame_idx - self.half_N_frames * interval < 0):
                    center_frame_idx = random.randint(0, 99)
                # get the neighbor list
                neighbor_list = list(range(center_frame_idx - self.half_N_frames * interval,
                                           center_frame_idx + self.half_N_frames * interval + 1, interval))
                if self.random_reverse and random.random() < 0.5:
                    neighbor_list.reverse()
                file_text = '{:03d}.png'.format(neighbor_list[self.half_N_frames])

            #### get the hr image (as the center frame)
            hr_data_path = osp.join(hr_path, file_text)
            hr_data = util.read_img(hr_data_path)

            #### get lr images
            lr_data_list = []
            for v in neighbor_list:
                lr_data_path = osp.join(lr_path, '{:03d}.png'.format(v))
                lr_data = util.read_img(lr_data_path)
                lr_data_list.append(lr_data)

            # randomly crop
            height, width, channel = hr_data.shape
            hr_size_w,hr_size_h = self.size_w * self.scale, self.size_h * self.scale
            lr_height = height // self.scale
            lr_width = width // self.scale

            rnd_h = random.randint(0, max(0, lr_height - self.size_h))
            rnd_w = random.randint(0, max(0, lr_width - self.size_w))
            img_lr_list = [one_data[rnd_h:rnd_h + self.size_h, rnd_w:rnd_w + self.size_w, :] for one_data in lr_data_list]

            rnd_h_hr, rnd_w_hr = int(rnd_h * self.scale), int(rnd_w * self.scale)
            img_hr = hr_data[rnd_h_hr:rnd_h_hr + hr_size_h, rnd_w_hr:rnd_w_hr + hr_size_w, :]


            # augmentation - flip, rotate
            img_lr_list.append(img_hr)
            rlt = util.augment(img_lr_list, hflip=True, rot=True)
            img_lr_list = rlt[0:-1]
            img_hr = rlt[-1]

            # stack lr images to NHWC, N is the frame number
            img_lrs = np.stack(img_lr_list, axis=0)


            # BGR to RGB,
            img_hr = img_hr[:, :, [2, 1, 0]]
            img_lrs = img_lrs[:, :, :, [2, 1, 0]]

            #HWC to CHW, numpy to tensor
            img_hr = torch.from_numpy(np.ascontiguousarray(np.transpose(img_hr, (2, 0, 1)))).float()
            img_lrs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lrs,(0, 3, 1, 2)))).float()

        except Exception as e:
            raise e
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