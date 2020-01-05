import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import utils as util
import os.path as osp
from glob import glob
import os


def get_file_name(file):
    # file = '/aaa/bbb/1050345\\050.png'  return /aaa/bbb/1050345 050
    file = file.replace('\\', '/')
    d_list = file.rsplit('/', maxsplit=1)
    path = d_list[0]
    file_text = d_list[-1]
    name = file_text.split('.')[0]
    return path, name

class DatasetLoader(Dataset):
    def __init__(self, data_lr, data_hr, size_w, size_h, scale,frame_interval,border_mode,random_reverse=False):
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
        self.size_w = size_w
        self.size_h = size_h
        self.scale = scale
        self.frame_interval = frame_interval
        self.n_frames = len(frame_interval)
        self.half_N_frames = self.n_frames // 2
        self.border_mode = border_mode
        self.random_reverse = random_reverse

        self.sub_mean = util.MeanShift()

    def _read_img(self,file):
        img = cv2.imread(file)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def __getitem__(self, index):
        try:
            lr_file = self.lr_list[index]
            hr_file = self.hr_list[index]
            lr_path,lr_name = util.get_file_name(lr_file)
            hr_path, _ = util.get_file_name(hr_file)
            center_frame_idx = int(lr_name)

            if self.border_mode:
                direction = 1  # 1: forward; 0: backward
                if self.random_reverse and random.random() < 0.5:
                    direction = random.choice([0, 1])
                if center_frame_idx + sum(self.frame_interval) > 99:
                    direction = 0
                elif center_frame_idx - sum(self.frame_interval) < 0:
                    direction = 1
                # get the neighbor list
                if direction == 1:
                    neighbor_list = [center_frame_idx + sum(self.frame_interval[0:i+1]) for i in range(len(self.frame_interval))]
                else:
                    neighbor_list = [center_frame_idx - sum(self.frame_interval[0:i + 1]) for i in
                                     range(len(self.frame_interval))]
            else:
                while (center_frame_idx+sum(self.frame_interval[self.half_N_frames:]) > 99) or (center_frame_idx-sum(self.frame_interval[:self.half_N_frames+1]) < 0):
                    center_frame_idx = random.randint(0, 99)
                # get the neighbor list
                neighbor_list = []
                i = center_frame_idx
                for x in self.frame_interval[0:self.half_N_frames]:
                    i -= x
                    neighbor_list.append(i)
                neighbor_list.reverse()
                neighbor_list.append(center_frame_idx)
                i = center_frame_idx
                for x in self.frame_interval[self.half_N_frames + 1:]:
                    i += x
                    neighbor_list.append(i)

                if self.random_reverse and random.random() < 0.5:
                    neighbor_list.reverse()

            #### get lr images
            lr_data_list = []
            for v in neighbor_list:
                lr_data_path = osp.join(lr_path, '{:03d}.png'.format(v))
                lr_data_list.append(self._read_img(lr_data_path))

            #### get hr images
            hr_data_list = []
            for v in neighbor_list:
                hr_data_path = osp.join(hr_path, '{:03d}.png'.format(v))
                hr_data_list.append(self._read_img(hr_data_path))

            # randomly crop
            height, width, channel = hr_data_list[0].shape
            hr_size_w,hr_size_h = self.size_w * self.scale, self.size_h * self.scale
            lr_height = height // self.scale
            lr_width = width // self.scale

            rnd_h = random.randint(0, max(0, lr_height - self.size_h))
            rnd_w = random.randint(0, max(0, lr_width - self.size_w))
            img_lr_list = [one_data[rnd_h:rnd_h + self.size_h, rnd_w:rnd_w + self.size_w, :] for one_data in lr_data_list]

            rnd_h_hr, rnd_w_hr = int(rnd_h * self.scale), int(rnd_w * self.scale)
            img_hr_list = [one_data[rnd_h_hr:rnd_h_hr + hr_size_h, rnd_w_hr:rnd_w_hr + hr_size_w, :] for one_data in hr_data_list]


            # augmentation - flip, rotate
            img_lr_list.extend(img_hr_list)
            rlt = util.augment(img_lr_list, hflip=True, rot=True)
            img_lr_list = rlt[0:self.n_frames]
            img_hr_list = rlt[self.n_frames:]

            # # stack lr images to NHWC, N is the frame number
            img_lrs = np.stack(img_lr_list, axis=0)
            img_hrs = np.stack(img_hr_list, axis=0)

            #HWC to CHW, numpy to tensor
            img_lrs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lrs, (0, 3, 1, 2)))).float()
            img_hrs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_hrs,(0, 3, 1, 2)))).float()

            # RGB mean for DIV2K
            img_lrs = self.sub_mean(img_lrs)
            img_hrs = self.sub_mean(img_hrs)

            #NCHW to CNHW
            img_lrs = img_lrs.permute(1, 0, 2, 3).contiguous()
            img_hrs = img_hrs.permute(1, 0, 2, 3).contiguous()
            # contiguous()  # 把tensor变成在内存中连续分布的形式。
            # 判断是否contiguous用torch.Tensor.is_contiguous()函数。
        except Exception as e:
            random_sum = random.randrange(0, self.__len__())
            return self.__getitem__(random_sum)

        return {'LRs': img_lrs, 'HRs': img_hrs}

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