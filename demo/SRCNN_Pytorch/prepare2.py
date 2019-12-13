"""
Author  : Xu fuyong
Time    : created by 2019/7/17 19:32

"""
import argparse
import glob
import h5py
import numpy as np
import PIL.Image as Image
from SRCNN_Pytorch.utils import convert_rgb_to_y


def train():
    gt = "J:/AI+4K/pngs_cut20/gt/*"
    X4 = 'J:/AI+4K/pngs_cut20/X4/*'
    h5_file = h5py.File('J:/AI+4K/pngs_cut20.hdf5', 'w')
    lr_patches = []
    hr_patches = []
    i = 0
    for l_file, h_file in zip(glob.glob(X4),glob.glob(gt)):
        if i%7==0:
            lr = Image.open(l_file).convert('RGB')
            hr = Image.open(h_file).convert('RGB')
            lr = lr.resize((hr.width, hr.height), resample=Image.BICUBIC)

            lr = np.array(lr).astype(np.float32)
            hr = np.array(hr).astype(np.float32)
            lr = convert_rgb_to_y(lr)
            hr = convert_rgb_to_y(hr)
            lr_patches.append(lr)
            hr_patches.append(hr)
        i += 1
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    print(lr_patches.shape)

    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()


def eval():
    gt = "J:/AI+4K/pngs_cut20/gt/*"
    X4 = 'J:/AI+4K/pngs_cut20/X4/*'
    h5_file = h5py.File('J:/AI+4K/pngs_cut20_eval.hdf5', 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    i = 0
    index = 0
    for l_file, h_file in zip(glob.glob(X4), glob.glob(gt)):
        if (i+3) % 70 == 0:
            lr = Image.open(l_file).convert('RGB')
            hr = Image.open(h_file).convert('RGB')
            lr = lr.resize((hr.width, hr.height), resample=Image.BICUBIC)

            lr = np.array(lr).astype(np.float32)
            hr = np.array(hr).astype(np.float32)
            lr = convert_rgb_to_y(lr)
            hr = convert_rgb_to_y(hr)
            lr_group.create_dataset(str(index), data=lr)
            hr_group.create_dataset(str(index), data=hr)
            index += 1
        i += 1

    h5_file.close()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--images-dir', type=str, required=True)
    # parser.add_argument('--output-path', type=str, required=True)
    # parser.add_argument('--patch-size', type=int, default=33)
    # parser.add_argument('--stride', type=int, default=14)
    # parser.add_argument('--scale', type=int, default=2)
    # parser.add_argument('--eval', action='store_true')
    # args = parser.parse_args()

    # if not args.eval:
    #     train(args)
    # else:
    #     eval(args)

    # train()
    eval()