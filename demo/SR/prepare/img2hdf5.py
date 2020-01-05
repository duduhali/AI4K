#图片 转 hdf5

import argparse
import glob
import h5py
import numpy as np
import PIL.Image as Image
import os

def train(args):
    h5_file = h5py.File(args.h5_file, 'w')
    lr_patches = []
    hr_patches = []
    i = 0
    for l_file, h_file in zip(glob.glob(os.path.join(args.input, "*")),glob.glob(os.path.join(args.target, "*"))):
        if (i+args.start)%args.space==0:
            lr = Image.open(l_file)
            hr = Image.open(h_file)
            lr = lr.resize((hr.width, hr.height), resample=Image.BICUBIC)

            image, _, _ = lr.convert('YCbCr').split()
            target, _, _ = hr.convert('YCbCr').split()

            lr = np.array(image).astype(np.float32)
            hr = np.array(target).astype(np.float32)
            lr_patches.append(lr)
            hr_patches.append(hr)
        i += 1
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    print(lr_patches.shape)

    h5_file.create_dataset('input', data=lr_patches)
    h5_file.create_dataset('target', data=hr_patches)
    h5_file.close()

def eval(args):
    h5_file = h5py.File(args.h5_file, 'w')
    lr_group = h5_file.create_group('input')
    hr_group = h5_file.create_group('target')

    i = 0
    index = 0
    for l_file, h_file in zip(glob.glob(os.path.join(args.input, "*")), glob.glob(os.path.join(args.target, "*"))):
        if (i+3) % 70 == 0:
            lr = Image.open(l_file)
            hr = Image.open(h_file)
            lr = lr.resize((hr.width, hr.height), resample=Image.BICUBIC)

            image, _, _ = lr.convert('YCbCr').split()
            target, _, _ = hr.convert('YCbCr').split()

            lr = np.array(image).astype(np.float32)
            hr = np.array(target).astype(np.float32)

            lr_group.create_dataset(str(index), data=lr)
            hr_group.create_dataset(str(index), data=hr)
            index += 1
        i += 1
    h5_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--h5-file', type=str, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--space', type=int, required=True)
    parser.add_argument('--eval', type=int, default=0)
    args = parser.parse_args()

    if args.eval == 0:
        train(args)
    else:
        eval(args)

    #python3 prepare/img2hdf5.py --input E:/test/cut_pngs/X4 --target E:/test/cut_pngs/gt --h5-file E:/test/train.hdf5 --space 4
    #python3 prepare/img2hdf5.py --input E:/test/cut_pngs/X4 --target E:/test/cut_pngs/gt --h5-file E:/test/eval.hdf5 --space 7 --start 2 --eval 1
